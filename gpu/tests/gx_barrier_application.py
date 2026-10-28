"""Shared-memory oracle for CTA spelling aliases and warp-scope accounting."""
import ctypes as C
import os

FORMS = [('bar_sync', 'bar.sync 0', 32),
         ('bar_cta_sync', 'bar.cta.sync 0', 32),
         ('barrier_sync', 'barrier.sync 0', 32),
         ('barrier_cta_sync', 'barrier.cta.sync 0', 32),
         ('barrier_aligned', 'barrier.sync.aligned 0', 32),
         ('barrier_cta_aligned', 'barrier.cta.sync.aligned 0', 32),
         ('warp_sync', 'bar.warp.sync 0xffffffff', 1)]


def ptx():
    text = '.version 8.0\n.target sm_90\n.address_size 64\n'
    for name, instruction, peer in FORMS:
        text += f'''
.visible .entry {name}(.param .u64 output) {{
 .reg .b32 %r<6>;
 .reg .b64 %d<4>;
 .shared .align 16 .b8 smem[256];
 ld.param.u64 %d0,[output]; mov.u32 %r0,%tid.x; mov.u32 %r1,%ctaid.x;
 mad.lo.u32 %r2,%r1,64,%r0; mul.wide.u32 %d1,%r0,4;
 mov.u64 %d2,smem; add.u64 %d1,%d2,%d1; st.shared.u32 [%d1],%r2;
 {instruction};
 xor.b32 %r3,%r0,{peer}; mul.wide.u32 %d1,%r3,4; add.u64 %d1,%d2,%d1;
 ld.shared.u32 %r4,[%d1]; mul.wide.u32 %d3,%r2,4; add.u64 %d3,%d0,%d3;
 st.global.u32 [%d3],%r4; ret;
}}
'''
    return text


def main():
    V,U,Z=C.c_void_p,C.c_uint,C.c_size_t
    lib=C.CDLL(os.environ['GX_TEST_LIBRARY'],mode=C.RTLD_GLOBAL)
    marks=C.CDLL(os.environ['GX_EXPLAIN_GPU_LIBRARY'],mode=C.RTLD_GLOBAL)
    marks.drperf_gpu_region_begin.argtypes=[C.c_char_p,Z,C.POINTER(C.c_char_p),C.POINTER(C.c_int64)]
    marks.drperf_gpu_region_begin.restype=C.c_uint64
    marks.drperf_gpu_region_end.argtypes=[C.c_uint64]
    def call(name,*args):
        rc=getattr(lib,name)(*args)
        if rc:raise RuntimeError((name,rc))
    call('cuInit',0);ctx=V();call('cuCtxCreate_v2',C.byref(ctx),U(0),0)
    module=V();call('cuModuleLoadData',C.byref(module),C.c_char_p(ptx().encode()))
    for name,_,peer in FORMS:
        fn=V();call('cuModuleGetFunction',C.byref(fn),module,name.encode())
        for blocks in (1,3):
            n=64*blocks;out=(U*(n+8))(*([0xdeadbeef]*(n+8)));ptr=C.c_uint64()
            call('cuMemAlloc_v2',C.byref(ptr),Z(C.sizeof(out)))
            call('cuMemcpyHtoD_v2',ptr,out,Z(C.sizeof(out)))
            base=C.c_uint64(ptr.value+16);params=(V*1)(C.cast(C.byref(base),V))
            token=marks.drperf_gpu_region_begin(name.encode(),1,(C.c_char_p*1)(b'blocks'),(C.c_int64*1)(blocks))
            call('cuLaunchKernel',fn,U(blocks),U(1),U(1),U(64),U(1),U(1),U(0),None,params,None)
            marks.drperf_gpu_region_end(token);call('cuCtxSynchronize')
            call('cuMemcpyDtoH_v2',out,ptr,Z(C.sizeof(out)))
            assert list(out)[4:-4]==[i^peer for i in range(n)],name
            assert list(out)[:4]==list(out)[-4:]==[0xdeadbeef]*4
            call('cuMemFree_v2',ptr)
    call('cuModuleUnload',module);call('cuCtxDestroy_v2',ctx)
    print('PASS: 14 launches, six CTA aliases and one warp barrier; shared-memory peers and guards exact')


if __name__=='__main__':main()
