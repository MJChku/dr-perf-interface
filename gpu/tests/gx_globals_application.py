"""Real GX regression: host-populated PTX constant arrays retain size and counts."""
import ctypes as C
import os

V, U, Z = C.c_void_p, C.c_uint, C.c_size_t
cuda = C.CDLL(os.environ['GX_TEST_LIBRARY'], mode=C.RTLD_GLOBAL)
marks = C.CDLL(os.environ['GX_EXPLAIN_GPU_LIBRARY'], mode=C.RTLD_GLOBAL)
marks.drperf_gpu_region_begin.argtypes = [C.c_char_p, Z, C.POINTER(C.c_char_p), C.POINTER(C.c_int64)]
marks.drperf_gpu_region_begin.restype = C.c_uint64
marks.drperf_gpu_region_end.argtypes = [C.c_uint64]


def call(name, *args):
    rc = getattr(cuda, name)(*args)
    if rc:
        raise RuntimeError((name, rc))


ptx = b'''.version 8.7
.target sm_90
.address_size 64
.const .align 16 .b8 coefficients [ 12 ]; .visible .global .align 8 .u32 scratch[ 3 ];
.const .u16 scalar;
.global .v2 .u64 vector
 [ 1
 ];
.visible .entry constants(.param .u64 output) {
 .reg .b32 %r<3>;
 .reg .b64 %d<5>;
 ld.param.u64 %d0,[output];
 mov.u32 %r0,%tid.x;
 mul.wide.u32 %d1,%r0,4;
 mov.u64 %d2,coefficients;
 add.u64 %d2,%d2,%d1;
 ld.const.u32 %r1,[%d2];
 mov.u64 %d3,scratch;
 add.u64 %d3,%d3,%d1;
 ld.global.u32 %r2,[%d3];
 add.u32 %r1,%r1,%r2;
 add.u64 %d4,%d0,%d1;
 st.global.u32 [%d4],%r1;
 ret;
}
'''
call('cuInit', 0)
context = V()
call('cuCtxCreate_v2', C.byref(context), U(0), 0)
modules = []
for index in range(2):
    module, function = V(), V()
    call('cuModuleLoadData', C.byref(module), C.c_char_p(ptx))
    call('cuModuleGetFunction', C.byref(function), module, b'constants')
    symbols = {}
    for name, extent, alignment in [('coefficients', 12, 16), ('scratch', 12, 8),
                                     ('scalar', 2, 2), ('vector', 16, 16)]:
        address, size = C.c_uint64(), Z()
        call('cuModuleGetGlobal_v2', C.byref(address), C.byref(size), module, name.encode())
        assert size.value == extent and address.value % alignment == 0, (name, size.value, address.value)
        before = (C.c_ubyte*extent)(*([255]*extent))
        call('cuMemcpyDtoH_v2', before, address, Z(extent))
        assert bytes(before) == bytes(extent), (name, list(before))
        written = (C.c_ubyte*extent)(*range(index+1, index+1+extent))
        call('cuMemcpyHtoD_v2', address, written, Z(extent))
        call('cuMemcpyDtoH_v2', before, address, Z(extent))
        assert bytes(before) == bytes(written), name
        symbols[name] = address
    if modules:
        assert all(address.value != modules[0][2][name].value for name, address in symbols.items())
    for name in (b'absent', b'constants__gxptx'):
        address, size = C.c_uint64(), Z()
        assert cuda.cuModuleGetGlobal_v2(C.byref(address), C.byref(size), module, name) != 0
    call('cuMemcpyHtoD_v2', symbols['scratch'], (U*3)(1, 2, 3), Z(12))
    modules.append((module, function, symbols))

output = C.c_uint64()
call('cuMemAlloc_v2', C.byref(output), Z(12))
args = (V*1)(C.cast(C.byref(output), V))
for shift in (0, 100):
    # Update both modules before launching either. Distinct addresses alone do
    # not prove each translated kernel references its own module's storage.
    for index, (_, _, symbols) in enumerate(modules):
        values = (U*3)(10+shift+1000*index, 20+shift+1000*index, 30+shift+1000*index)
        call('cuMemcpyHtoD_v2', symbols['coefficients'], values, Z(12))
    for index, (_, function, symbols) in enumerate(modules):
        for n in (1, 2, 3):
            call('cuMemcpyHtoD_v2', output, (U*3)(999, 999, 999), Z(12))
            token = marks.drperf_gpu_region_begin(b'constants', 1, (C.c_char_p*1)(b'n'), (C.c_int64*1)(n))
            call('cuLaunchKernel', function, U(1), U(1), U(1), U(n), U(1), U(1), U(0), None, args, None)
            marks.drperf_gpu_region_end(token)
            call('cuCtxSynchronize')
            actual = (U*3)()
            call('cuMemcpyDtoH_v2', actual, output, Z(12))
            assert list(actual) == [11*(i+1)+shift+1000*index if i < n else 999 for i in range(3)], list(actual)
        actual = (U*3)()
        call('cuMemcpyDtoH_v2', actual, symbols['coefficients'], Z(12))
        assert list(actual) == [10+shift+1000*index, 20+shift+1000*index, 30+shift+1000*index]
call('cuMemFree_v2', output)
for module, _, _ in modules:
    call('cuModuleUnload', module)
call('cuCtxDestroy_v2', context)
print('GX globals: scalar/vector/array extents, alignment, updates, module isolation and outputs PASS')
