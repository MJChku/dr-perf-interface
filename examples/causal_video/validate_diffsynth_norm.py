"""Exercise actual patched classes without importing optional model integrations."""
import ast, copy, json, statistics, time
from collections import OrderedDict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import argparse
p=argparse.ArgumentParser();p.add_argument('tree',type=Path);root=p.parse_args().tree
g=dict(torch=torch,copy=copy,OrderedDict=OrderedDict,Union=__import__('typing').Union,nn=nn,F=F)
for relative,names in [('diffsynth/core/vram/layers.py',{'AutoTorchModule','AutoWrappedModule','AutoWrappedStatelessNorm'}),('diffsynth/models/wan_video_dit.py',{'RMSNorm'})]:
 tree=ast.parse((root/relative).read_text());tree.body=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in names];tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0));ast.fix_missing_locations(tree);exec(compile(tree,relative,'exec'),g)
wrapper=object.__new__(g['AutoWrappedStatelessNorm'])
def candidate(m,dtype):return wrapper.cast_to(m,dtype,'cpu')
def reference(m,dtype):return copy.deepcopy(m).to(dtype=dtype,device='cpu')
# Registry subclasses and custom copy protocols must retain deepcopy semantics.
class TaggedDict(dict):
    pass
m=nn.LayerNorm(16);m.extra=TaggedDict();m.extra.notes=[1]
a,b=candidate(m,torch.float64),reference(m,torch.float64)
assert type(a.extra) is type(b.extra) is TaggedDict
a.extra.notes.append(2);assert m.extra.notes==[1]
m=nn.LayerNorm(16);m.extra=m._buffers
a,b=candidate(m,torch.float64),reference(m,torch.float64)
assert a.extra is a._buffers and b.extra is b._buffers and a.extra is not m.extra
class CustomCopy(nn.LayerNorm):
    def __deepcopy__(self, memo):
        result=nn.LayerNorm(self.normalized_shape,eps=self.eps*2)
        result.load_state_dict(self.state_dict())
        return result
m=CustomCopy(16);a,b=candidate(m,torch.float64),reference(m,torch.float64)
assert a.eps==b.eps and type(a) is type(b) is nn.LayerNorm

torch.set_num_threads(1);checks=[]
for kind in ['layer_affine','layer_noaffine','rms','torch_rms']:
 for dtype in [torch.float32,torch.float64,torch.bfloat16]:
  torch.manual_seed(1)
  m=nn.LayerNorm(128,elementwise_affine=kind=='layer_affine') if kind.startswith('layer') else g['RMSNorm'](128)
  if kind=='torch_rms':m.use_torch_norm=True
  a,b=candidate(m,dtype),reference(m,dtype);x=torch.randn(2,3,128).to(dtype)
  torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
  for p,q in zip(a.parameters(),m.parameters()):assert p.data_ptr()!=q.data_ptr() and p.dtype==dtype and q.dtype==torch.float32
  # Modifying the temporary registry must not modify the source registry.
  a.register_buffer('test',torch.ones(1));assert 'test' not in m._buffers
  checks.append((kind,str(dtype)))
m=g['RMSNorm'](128);m.weight=nn.Parameter(torch.randn(256)[::2]);a,b=candidate(m,torch.float64),reference(m,torch.float64);torch.testing.assert_close(a.weight,b.weight,rtol=0,atol=0);assert a.weight.stride()==b.weight.stride()
m=nn.LayerNorm(16);m.extra={'items':[1]};a=candidate(m,torch.float64);a.extra['items'].append(2);assert m.extra=={'items':[1]}
m=nn.LayerNorm(16);m.register_buffer('counter',torch.ones(1));a=candidate(m,torch.float64);assert a.counter.dtype==torch.float64 and a.counter.data_ptr()!=m.counter.data_ptr()
m=nn.LayerNorm(16);seen=[];m.register_forward_hook(lambda mod,args,out:seen.append(mod));a=candidate(m,torch.float64);a(torch.randn(1,16,dtype=torch.float64));assert seen==[a]
m=nn.LayerNorm(16);a,b=candidate(m,torch.float64),reference(m,torch.float64);x=torch.randn(3,16,dtype=torch.float64);a(x).square().sum().backward();b(x).square().sum().backward();torch.testing.assert_close(a.weight.grad,b.weight.grad,rtol=0,atol=0);assert m.weight.grad is None
class Custom(nn.LayerNorm):
 def _apply(self,fn,recurse=True):
  self.eps*=2
  return super()._apply(fn,recurse)
m=Custom(16);a,b=candidate(m,torch.float64),reference(m,torch.float64);assert a.eps==b.eps
results=[]
for kind,m in [('layer_noaffine',nn.LayerNorm(1536,elementwise_affine=False)),('rms',g['RMSNorm'](1536))]:
 row={'kind':kind,'copies':2000}
 for label,fn in [('before',reference),('after',candidate)]:
  ts=[]
  for repeat in range(5):
   t=time.perf_counter()
   for _ in range(row['copies']):fn(m,torch.bfloat16)
   ts.append(time.perf_counter()-t)
  row[label]=statistics.median(ts)
 results.append(row)
print(json.dumps({'checks':'exact outputs, independent parameters/registries, noncontiguous weights, mutable-state/hooks/buffers/custom-conversion fallbacks and autograd passed','output_cases':checks,'scope':'CPU copy microbench only, not end-to-end','timing':results},indent=2))
