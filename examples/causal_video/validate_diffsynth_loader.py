"""Compare upstream/patched DiskMap values, layouts, and copy isolation."""
import ast,inspect,json,os,tempfile,zipfile
from pathlib import Path
import torch
from safetensors import safe_open
import argparse
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);args=p.parse_args()
classes=[]
for path in [args.baseline/'diffsynth/core/vram/disk_map.py',args.candidate/'diffsynth/core/vram/disk_map.py']:
 tree=ast.parse(path.read_text());tree.body=[x for x in tree.body if isinstance(x,ast.ClassDef)];g=dict(torch=torch,os=os,inspect=inspect,zipfile=zipfile,safe_open=safe_open);exec(compile(tree,str(path),'exec'),g);classes.append(g['DiskMap'])
checks=[]
with tempfile.TemporaryDirectory() as tmp:
 for zip_format in [True,False]:
  data={'dense':torch.randn(3,5),'transpose':torch.randn(7,5).T,'slice':torch.arange(30,dtype=torch.float32)[::3],'expand':torch.randn(3,1).expand(3,6),'channels_last':torch.randn(2,3,4,5).contiguous(memory_format=torch.channels_last),'scalar':torch.tensor(2.5),'integer':torch.arange(7),'bf16':torch.randn(20).bfloat16()}
  path=Path(tmp)/'weights.pth';torch.save(data,path,_use_new_zipfile_serialization=zip_format)
  for dtype in [None,torch.float32,torch.bfloat16,torch.float64]:
   before,after=[cls(str(path),'cpu',torch_dtype=dtype,buffer_size=100) for cls in classes]
   assert list(before)==list(after)
   for name in data:
    a,b=before[name],after[name];torch.testing.assert_close(a,b,rtol=0,atol=0);assert a.stride()==b.stride(),(name,a.stride(),b.stride());b.fill_(0);torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)
   checks.append({'zip':zip_format,'dtype':str(dtype),'tensors':len(data)})
print(json.dumps({'checks':'exact values/layouts and repeated-read mutation isolation passed','cases':checks},indent=2))
