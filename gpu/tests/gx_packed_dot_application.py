"""Independent bit-value oracle for packed dots executed through actual GX."""
import ctypes as C
import itertools
import os
import random

V, U, Z = C.c_void_p, C.c_uint, C.c_size_t
lib = C.CDLL(os.environ['GX_TEST_LIBRARY'], mode=C.RTLD_GLOBAL)
marks = C.CDLL(os.environ['GX_EXPLAIN_GPU_LIBRARY'], mode=C.RTLD_GLOBAL)
marks.drperf_gpu_region_begin.argtypes = [C.c_char_p, Z, C.POINTER(C.c_char_p), C.POINTER(C.c_int64)]
marks.drperf_gpu_region_begin.restype = C.c_uint64
marks.drperf_gpu_region_end.argtypes = [C.c_uint64]


def call(name, *args):
    rc = getattr(lib, name)(*args)
    if rc:
        raise RuntimeError((name, rc))


def signed_field(value, bits, signed):
    return value - (1 << bits) if signed and value >= (1 << (bits-1)) else value


def oracle(a, b, c, bits, high, sign_a, sign_b):
    # Decode byte strings independently of the translator's shift/mask helper.
    aa, bb = a.to_bytes(4, 'little'), b.to_bytes(4, 'little')
    width = bits // 8
    pairs = zip((aa[i:i+width] for i in range(0, 4, width)),
                bb[2:4] if high else bb)
    total = c
    for left, right in pairs:
        total += signed_field(int.from_bytes(left, 'little'), bits, sign_a) * signed_field(right, 8, sign_b)
    return total & 0xffffffff


variants, entries = [], []
for bits, high in [(16, False), (16, True), (8, False)]:
    for sign_a, sign_b, masked in itertools.product((False, True), repeat=3):
        at, bt = ('s32' if sign_a else 'u32'), ('s32' if sign_b else 'u32')
        op = f'dp2a.{"hi" if high else "lo"}.{at}.{bt}' if bits == 16 else f'dp4a.{at}.{bt}'
        name = op.replace('.', '_') + ('_masked' if masked else '_all')
        predicate = 'and.b32 %r6,%r0,1; setp.eq.u32 %p1,%r6,0; @%p1 ' if masked else ''
        entries.append(f'''
.visible .entry {name}(.param .u64 input, .param .u64 output, .param .u32 n) {{
 .reg .b32 %r<8>; .reg .b64 %d<5>; .reg .pred %p0,%p1;
 ld.param.u64 %d0,[input]; ld.param.u64 %d1,[output]; ld.param.u32 %r7,[n];
 mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x; mad.lo.u32 %r0,%r1,32,%r0;
 setp.ge.u32 %p0,%r0,%r7; @%p0 bra done;
 mul.wide.u32 %d2,%r0,12; add.u64 %d2,%d0,%d2;
 ld.global.u32 %r3,[%d2]; ld.global.u32 %r4,[%d2+4]; ld.global.u32 %r5,[%d2+8];
 {predicate}{op} %r3,%r3,%r4,%r5;
 mul.wide.u32 %d3,%r0,4; add.u64 %d3,%d1,%d3; st.global.u32 [%d3],%r3;
done: ret;
}}
''')
        variants.append((name, bits, high, sign_a, sign_b, masked))

ptx = ('.version 8.0\n.target sm_80\n.address_size 64\n' + ''.join(entries)).encode()
call('cuInit', 0)
ctx = V(); call('cuCtxCreate_v2', C.byref(ctx), U(0), 0)
module = V(); call('cuModuleLoadData', C.byref(module), C.c_char_p(ptx))
edges = [(0x00020001, 0x04030201, 5), (0, 0, 0xffffffff),
         (0xffffffff, 0xffffffff, 0xffffffff), (0x80008000, 0x80808080, 0x80000000),
         (0x7fff7fff, 0x7f7f7f7f, 0x7fffffff), (0xff00ff00, 0x00ff80ff, 0xfffffffe)]
rng = random.Random(20261028)
vectors = edges + [tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(65-len(edges))]
guard = 0xdeadbeef
checked = 0
for name, bits, high, sign_a, sign_b, masked in variants:
    fn = V(); call('cuModuleGetFunction', C.byref(fn), module, name.encode())
    for n in (1, 17, 65):
        raw = (U*(3*n+8))(*([guard]*4 + [v for row in vectors[:n] for v in row] + [guard]*4))
        out = (U*(n+8))(*([guard]*(n+8)))
        original = bytes(raw)
        ip, op = C.c_uint64(), C.c_uint64()
        call('cuMemAlloc_v2', C.byref(ip), Z(C.sizeof(raw)))
        call('cuMemAlloc_v2', C.byref(op), Z(C.sizeof(out)))
        call('cuMemcpyHtoD_v2', ip, raw, Z(C.sizeof(raw)))
        call('cuMemcpyHtoD_v2', op, out, Z(C.sizeof(out)))
        input_ptr, output_ptr, size = C.c_uint64(ip.value+16), C.c_uint64(op.value+16), U(n)
        params = (V*3)(*(C.cast(C.byref(v), V) for v in (input_ptr, output_ptr, size)))
        active = (n+1)//2 if masked else n
        token = marks.drperf_gpu_region_begin(name.encode(), 2,
                    (C.c_char_p*2)(b'lanes', b'active'), (C.c_int64*2)(n, active))
        call('cuLaunchKernel', fn, U((n+31)//32), U(1), U(1), U(32), U(1), U(1), U(0), None, params, None)
        marks.drperf_gpu_region_end(token)
        call('cuCtxSynchronize')
        call('cuMemcpyDtoH_v2', out, op, Z(C.sizeof(out)))
        call('cuMemcpyDtoH_v2', raw, ip, Z(C.sizeof(raw)))
        expected = [a if masked and i%2 else oracle(a,b,c,bits,high,sign_a,sign_b)
                    for i,(a,b,c) in enumerate(vectors[:n])]
        assert list(out)[4:-4] == expected, (name,n,list(out)[4:-4],expected)
        assert list(out)[:4] == list(out)[-4:] == [guard]*4 and bytes(raw) == original
        checked += n
        call('cuMemFree_v2', ip); call('cuMemFree_v2', op)
call('cuModuleUnload', module); call('cuCtxDestroy_v2', ctx)
print(f'PASS: {len(variants)*3} GX launches; {checked} packed-dot lane results; signedness, halves, overflow, aliasing, predication, tails and guards')
