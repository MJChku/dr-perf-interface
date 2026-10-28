"""CPU-only GX execution oracle for scalar 128-bit shared-memory accounting."""
import ctypes as C
import os

V = C.c_void_p
U = C.c_uint
Z = C.c_size_t
lib = C.CDLL(os.environ['GX_TEST_LIBRARY'], mode=C.RTLD_GLOBAL)
marks = C.CDLL(os.environ['GX_EXPLAIN_GPU_LIBRARY'], mode=C.RTLD_GLOBAL)
marks.drperf_gpu_region_begin.argtypes = [C.c_char_p, Z, C.POINTER(C.c_char_p), C.POINTER(C.c_int64)]
marks.drperf_gpu_region_begin.restype = C.c_uint64
marks.drperf_gpu_region_end.argtypes = [C.c_uint64]


def call(name, *args):
    rc = getattr(lib, name)(*args)
    if rc:
        raise RuntimeError((name, rc))


ptx = b'''.version 8.7
.target sm_90
.address_size 64
.visible .entry wide_shared(.param .u64 output) {
 .reg .b32 %r<8>;
 .reg .b64 %d<3>;
 .reg .b128 %q;
 .shared .align 16 .b8 source[16];
 .shared .align 16 .b8 copied[16];
 ld.param.u64 %d0,[output];
 mov.u32 %r0,%ctaid.x;
 add.u32 %r1,%r0,1;
 add.u32 %r2,%r0,2;
 add.u32 %r3,%r0,3;
 add.u32 %r4,%r0,4;
 st.shared.v4.u32 [source],{%r1,%r2,%r3,%r4};
 ld.shared.b128 %q,[source];
 st.shared.b128 [copied],%q;
 ld.shared.v4.u32 {%r1,%r2,%r3,%r4},[copied];
 mul.wide.u32 %d1,%r0,16;
 add.u64 %d2,%d0,%d1;
 st.global.v4.u32 [%d2],{%r1,%r2,%r3,%r4};
 ret;
}
'''
call('cuInit', 0)
ctx = V()
call('cuCtxCreate_v2', C.byref(ctx), U(0), 0)
module = V()
call('cuModuleLoadData', C.byref(module), C.c_char_p(ptx))
fn = V()
call('cuModuleGetFunction', C.byref(fn), module, b'wide_shared')
for n in (1, 2, 4):
    ptr = C.c_uint64()
    call('cuMemAlloc_v2', C.byref(ptr), Z(16*n))
    args = (V*1)(C.cast(C.byref(ptr), V))
    token = marks.drperf_gpu_region_begin(b'wide_shared', 1,
        (C.c_char_p*1)(b'n'), (C.c_int64*1)(n))
    call('cuLaunchKernel', fn, U(n), U(1), U(1), U(1), U(1), U(1), U(0), None, args, None)
    marks.drperf_gpu_region_end(token)
    call('cuCtxSynchronize')
    output = (U*(4*n))()
    call('cuMemcpyDtoH_v2', output, ptr, Z(16*n))
    assert list(output) == [i+j for i in range(n) for j in (1, 2, 3, 4)], list(output)
    call('cuMemFree_v2', ptr)
call('cuModuleUnload', module)
call('cuCtxDestroy_v2', ctx)
print('PASS: all 128 shared-memory bits survived the GX roundtrip')
