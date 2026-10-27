"""Ordinary CUDA driver application used by the GX integration regression.

Only region markers are DrPerf APIs; every allocation, copy, launch, stream,
barrier, and graph replay goes through GX's normal CUDA implementation.
"""
import ctypes as C
import os
import threading
import sys

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

def begin(name, **pcvs):
    return marks.drperf_gpu_region_begin(name.encode(), len(pcvs),
        (C.c_char_p * len(pcvs))(*(x.encode() for x in pcvs)),
        (C.c_int64 * len(pcvs))(*pcvs.values()))

def end(token):
    marks.drperf_gpu_region_end(token)

if os.environ.get('GX_TEST_PERFMARK'):
    cpu = C.CDLL(os.environ['GX_TEST_PERFMARK'])
    cpu.perfmark_begin_v.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_int64)]
    cpu.perfmark_end.argtypes = [C.c_char_p]
    def begin(name, **pcvs):
        cpu.perfmark_begin_v(name.encode(), len(pcvs),
            (C.c_char_p*len(pcvs))(*(x.encode() for x in pcvs)),
            (C.c_int64*len(pcvs))(*pcvs.values()))
        return name
    def end(token):
        cpu.perfmark_end(token.encode())

ptx = b'''.version 8.0
.target sm_80
.address_size 64
.visible .entry affine(.param .u64 data,.param .u32 n) {
 .reg .b32 %r<5>; .reg .b64 %d<3>; .reg .f32 %f<2>; .reg .pred %p;
 ld.param.u64 %d0,[data]; ld.param.u32 %r0,[n];
 mov.u32 %r1,%tid.x; mov.u32 %r2,%ctaid.x;
 mad.lo.u32 %r1,%r2,32,%r1;
 bar.sync 0;
 setp.ge.u32 %p,%r1,%r0; @%p bra end;
 mul.wide.u32 %d1,%r1,4; add.u64 %d1,%d0,%d1;
 ld.global.f32 %f0,[%d1];
 fma.rn.f32 %f1,%f0,0f40000000,0f3f800000;
 st.global.f32 [%d1],%f1;
end: ret;
}
'''
call('cuInit', 0)
ctx = V(); call('cuCtxCreate_v2', C.byref(ctx), U(0), 0)
module = V(); call('cuModuleLoadData', C.byref(module), C.c_char_p(ptx))
fn = V(); call('cuModuleGetFunction', C.byref(fn), module, b'affine')
streams = [V(), V()]
for stream in streams:
    call('cuStreamCreate', C.byref(stream), U(1))


def allocate(n):
    ptr = C.c_uint64(); call('cuMemAlloc_v2', C.byref(ptr), Z(n*4))
    host = (C.c_float*n)(*([1]*n))
    call('cuMemcpyHtoD_v2', ptr, host, Z(n*4))
    return ptr


def launch(ptr, n, stream):
    size = U(n)
    args = (V*2)(C.cast(C.byref(ptr), V), C.cast(C.byref(size), V))
    call('cuLaunchKernel', fn, U((n+31)//32), U(1), U(1), U(32), U(1), U(1), U(0), stream, args, None)

if os.environ.get('GX_TEST_SKIP'):
    ptr = allocate(13)
    r = begin('skipped', n=13)
    launch(ptr, 13, streams[0])
    end(r)
    call('cuStreamSynchronize', streams[0])
    call('cuMemFree_v2', ptr)
    for stream in streams: call('cuStreamDestroy_v2', stream)
    call('cuModuleUnload', module); call('cuCtxDestroy_v2', ctx)
    print('PASS: timing-only launch recorded as unmeasured')
    sys.exit(0)

# Two streams, repeated states, nested regions; close before async execution.
for repeat in range(2):
    for n in (5, 13, 32, 47):
        ptr = allocate(n)
        gate = threading.Event()
        callback_type = C.CFUNCTYPE(None, V)
        callback = callback_type(lambda _: gate.wait(30))
        call('cuLaunchHostFunc', streams[0], callback, None)
        parent = begin('request', n=n)
        child = begin('transform', n=n)
        launch(ptr, n, streams[0])
        launch(ptr, n, streams[0])
        end(child)
        end(parent)
        gate.set()  # Both markers have ended before either kernel can execute.
        call('cuStreamSynchronize', streams[0])
        out = (C.c_float*n)(); call('cuMemcpyDtoH_v2', out, ptr, Z(n*4))
        assert list(out) == [7]*n
        call('cuMemFree_v2', ptr)

# Graph capture must NOT bind observations to the capture region. Each replay
# must bind freshly to its submitting region, and duplicate PCV states pool.
ptr = allocate(13)
call('cuStreamBeginCapture_v2', streams[1], 0)
launch(ptr, 13, streams[1])
graph = V(); call('cuStreamEndCapture', streams[1], C.byref(graph))
exe = V(); call('cuGraphInstantiateWithFlags', C.byref(exe), graph, C.c_uint64(0))
for repeats in (1, 2, 3, 2):
    r = begin('replay', repeats=repeats)
    for _ in range(repeats):
        call('cuGraphLaunch', exe, streams[1])
    end(r)
call('cuStreamSynchronize', streams[1])
out = (C.c_float*13)(); call('cuMemcpyDtoH_v2', out, ptr, Z(52))
assert list(out) == [511]*13
call('cuGraphExecDestroy', exe); call('cuGraphDestroy', graph); call('cuMemFree_v2', ptr)

# Distinct modules exporting the same kernel name retain distinct block IDs.
module2 = V()
ptx2 = ptx.replace(b' fma.rn.f32', b' ld.global.f32 %f0,[%d1];\n fma.rn.f32')
call('cuModuleLoadData', C.byref(module2), C.c_char_p(ptx2))
fn2 = V(); call('cuModuleGetFunction', C.byref(fn2), module2, b'affine')
original_fn = fn
for n in (7,19):
    ptr = allocate(n)
    r = begin('same_symbol', n=n)
    launch(ptr, n, streams[0])
    fn = fn2
    launch(ptr, n, streams[0])
    fn = original_fn
    end(r)
    call('cuStreamSynchronize', streams[0])
    out = (C.c_float*n)(); call('cuMemcpyDtoH_v2', out, ptr, Z(n*4))
    assert list(out) == [7]*n
    call('cuMemFree_v2', ptr)
call('cuModuleUnload', module2)

# Cross-thread submitters: PCVs and nested region tokens must not bleed.
def worker(n, stream):
    call('cuCtxSetCurrent', ctx)
    ptr = allocate(n)
    r = begin('threaded', n=n)
    launch(ptr, n, stream)
    end(r)
    call('cuStreamSynchronize', stream)
    out = (C.c_float*n)(); call('cuMemcpyDtoH_v2', out, ptr, Z(n*4))
    assert list(out) == [3]*n
    call('cuMemFree_v2', ptr)
errors = []
def guarded(n, stream):
    try: worker(n, stream)
    except BaseException as e: errors.append(e)
threads = [threading.Thread(target=guarded,args=(n,s)) for n,s in zip((7,29), streams)]
for t in threads: t.start()
for t in threads: t.join()
assert not errors, errors
for stream in streams: call('cuStreamDestroy_v2', stream)
call('cuModuleUnload', module)
call('cuCtxDestroy_v2', ctx)
print('PASS: CUDA launches, numerical outputs, nested regions, graph replay, multiple streams and submitter threads')
