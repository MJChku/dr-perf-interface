"""Ordinary CUDA app: async-copy numerics and PCV sweep for GX capture tests."""
import ctypes as C
import os
from pathlib import Path
import perfmark

V,U,Z=C.c_void_p,C.c_uint,C.c_size_t
lib=C.CDLL(None)


def call(name,*args):
    code=getattr(lib,name)(*args)
    if code:raise RuntimeError((name,code))


call('cuInit',0)
ctx=V();call('cuCtxCreate_v2',C.byref(ctx),U(0),0)
module=V();call('cuModuleLoadData',C.byref(module),C.c_char_p(Path(__file__).with_name('gx_async_copy.ptx').read_bytes()))
stream=V();call('cuStreamCreate',C.byref(stream),U(1))
entries=['simple_copy','bulk_g2s']
if os.environ.get('DRPERF_TEST_UNMODELED_BULK'): entries.append('bulk_s2g')
for entry in entries:
    function=V();call('cuModuleGetFunction',C.byref(function),module,entry.encode())
    for size in [16,32,64,32]:
        n=U(size)
        src=(C.c_uint32*(size//4))(*(0xf0100011+i for i in range(size//4)))
        guard=(C.c_uint32*(size//4+4))(*([0xa5a5a5a5]*(size//4+4)))
        source=C.c_uint64();dest=C.c_uint64()
        call('cuMemAlloc_v2',C.byref(source),Z(size))
        call('cuMemAlloc_v2',C.byref(dest),Z(size+16))
        call('cuMemcpyHtoD_v2',source,src,Z(size))
        call('cuMemcpyHtoD_v2',dest,guard,Z(size+16))
        args=(V*3)(*(C.cast(C.byref(x),V) for x in [source,dest,n]))
        with perfmark.region(entry,bytes=size):
            call('cuLaunchKernel',function,U(1),U(1),U(1),U(1),U(1),U(1),U(0),stream,args,None)
        call('cuStreamSynchronize',stream)
        call('cuMemcpyDtoH_v2',guard,dest,Z(size+16))
        assert list(guard)[:size//4]==list(src)
        assert list(guard)[size//4:]==[0xa5a5a5a5]*4
        call('cuMemFree_v2',source);call('cuMemFree_v2',dest)
call('cuStreamDestroy_v2',stream);call('cuModuleUnload',module);call('cuCtxDestroy_v2',ctx)
print('PASS: simple and bulk async G2S copies, four invocations each, output guards')
