"""Real CPU-only GX scalar/shuffle bit regression; no benchmark-case dependency."""
import ctypes as C
import os
from pathlib import Path
from contextlib import contextmanager
api=C.CDLL(os.environ['GX_TEST_LIBRARY'],mode=C.RTLD_GLOBAL)
P,U,Z=C.c_void_p,C.c_uint,C.c_size_t
marks=C.CDLL(os.environ['GX_EXPLAIN_GPU_LIBRARY'],mode=C.RTLD_GLOBAL)
marks.drperf_gpu_region_begin.argtypes=[C.c_char_p,Z,C.POINTER(C.c_char_p),C.POINTER(C.c_int64)]
marks.drperf_gpu_region_begin.restype=C.c_uint64
marks.drperf_gpu_region_end.argtypes=[C.c_uint64]
@contextmanager
def region(name,*,n):
 token=marks.drperf_gpu_region_begin(name.encode(),1,(C.c_char_p*1)(b'n'),(C.c_int64*1)(n))
 try:yield
 finally:marks.drperf_gpu_region_end(token)
# Input bits, denominator bits, expected plain bits, expected FTZ bits.
# None means any NaN payload, as specified by PTX.
cases=[(0x3f800000,0x40000000,0x3f000000,0x3f000000),
 (0x7e800000,0x7e800000,0x3f800000,0x3f800000),
 (0x7f7fffff,0x7e800001,0,0),
 (0xff7fffff,0x7f7fffff,0x80000000,0x80000000),
 (0x3f800000,0xff7fffff,0x80000000,0x80000000),
 (0xbf800000,0xff7fffff,0,0),
 (0x7f800000,0x7e800001,None,None),
 (0xff800000,0xff7fffff,None,None),
 (0x7fc01234,0x7f7fffff,None,None),
 (1,1,0x3f800000,None),
 (0x00800000,0x40000000,0x00400000,0),
 (0x80800000,0x40000000,0x80400000,0x80000000),
 (0x3f800000,0x80000001,0xff800000,0xff800000),
 (0x80000001,0x3f800000,0x80000001,0x80000000),
 (0,0x80000000,None,None),
 (0x3f800000,0x80000000,0xff800000,0xff800000),
 (0xbf800000,0x7f800000,0x80000000,0x80000000),
 (0x7f800000,0xc0000000,0xff800000,0xff800000),
 (0x7f800000,0x7f800000,None,None),
 (0x3f800000,0x7fc00001,None,None)]
def call(n,*a):
 r=getattr(api,n)(*a)
 if r:raise RuntimeError((n,r))
def alloc(a):
 p=C.c_uint64();call('cuMemAlloc_v2',C.byref(p),Z(C.sizeof(a)));call('cuMemcpyHtoD_v2',p,a,Z(C.sizeof(a)));return p
call('cuInit',0);ctx=P();call('cuCtxCreate_v2',C.byref(ctx),U(0),0);m=P();call('cuModuleLoadData',C.byref(m),C.c_char_p(Path(__file__).with_name('gx_scalar_bits.ptx').read_bytes()));fn=P();call('cuModuleGetFunction',C.byref(fn),m,b'div_cases');checks=[]
for n in [17,20,32]:
 cs=[cases[i%len(cases)] for i in range(n)];a=(U*n)(*(x[0] for x in cs));b=(U*n)(*(x[1] for x in cs));o=(U*(2*n+4))(*([0xdeadbeef]*(2*n+4)));da,db,do=[alloc(x) for x in [a,b,o]];v=U(n);params=(P*4)(*(C.cast(C.byref(x),P) for x in [da,db,do,v]))
 with region('div_approx_compatibility',n=n):call('cuLaunchKernel',fn,U(1),U(1),U(1),U(32),U(1),U(1),U(0),None,params,None)
 call('cuCtxSynchronize');call('cuMemcpyDtoH_v2',o,do,Z(C.sizeof(o)))
 for i,c in enumerate(cs):
  for j,expected in enumerate(c[2:]):
   actual=o[2*i+j];assert ((actual&0x7fffffff)>0x7f800000 if expected is None else actual==expected),(i,j,hex(actual),expected)
 assert list(o)[-4:]==[0xdeadbeef]*4
 checks.append({'n':n,'numerical':True,'guard':True,'observed_bits':[int(x) for x in o]})
 for ptr in [da,db,do]:call('cuMemFree_v2',ptr)
call('cuModuleGetFunction',C.byref(fn),m,b'shfl_cases')
patterns=[0x3fc00000,0xbfa00000,0,0x80000000,0x7fc01234,0xffc05678,0x7f801234,0xff801234,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f800000,0xff800000]
for n in [32,64]:
 values=[patterns[(i+3*(i//32))%len(patterns)] for i in range(n)];a=(U*n)(*values);o=(U*(12*n+4))(*([0xdeadbeef]*(12*n+4)));da,do=alloc(a),alloc(o);params=(P*2)(*(C.cast(C.byref(x),P) for x in [da,do]))
 with region('shfl_float_compatibility',n=n):call('cuLaunchKernel',fn,U(1),U(1),U(1),U(n),U(1),U(1),U(0),None,params,None)
 call('cuCtxSynchronize');call('cuMemcpyDtoH_v2',o,do,Z(C.sizeof(o)))
 for i in range(n):
  lane=i%32;base=i-lane;src=values[i];up=values[i-1] if lane else src;down=values[i+1] if lane<31 else src;mx=src&0x80000000 if (src&0x7fffffff)<0x00800000 else src
  expected=[values[base],values[base],values[base+(lane^1)],values[base+(lane^1)],up,up,down,down,values[base],src,mx,3+(4 if lane else 0)+(8 if lane<31 else 0)]
  for j,e in enumerate(expected):
   actual=o[12*i+j]
   assert ((actual&0x7fffffff)>0x7f800000 if j==10 and (e&0x7fffffff)>0x7f800000 else actual==e),(i,j,hex(actual),hex(e))
 assert list(o)[-4:]==[0xdeadbeef]*4
 checks.append({'shuffle_n':n,'ctas':1,'threads_per_cta':n,'numerical':True,'guard':True,'modes':['idx','bfly','up','down'],'paired_unpaired':True,'source_alias':True,'predicated_off_preserved':True,'floating_payloads_preserved':True})
 for ptr in [da,do]:call('cuMemFree_v2',ptr)
call('cuModuleUnload',m)
call('cuCtxDestroy_v2',ctx)
print('PASS: approximate division, max FTZ and raw float shuffles')
