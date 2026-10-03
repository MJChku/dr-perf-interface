"""CPU oracle for slicing the normalization input; no model or timing claim."""
import argparse,ast,json,subprocess,tempfile,types
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tree',type=Path,required=True,help='Pinned StreamDiffusionV2 checkout; may already have the candidate applied.')
p.add_argument('--patch',type=Path,default=Path(__file__).with_name('cpu_patches')/'streamdiffusionv2_input_window.patch')
a=p.parse_args();source=a.tree/'streamdiffusionv2/pipeline.py'
original=subprocess.check_output(['git','-C',str(a.tree),'show','HEAD:streamdiffusionv2/pipeline.py'],text=True)
with tempfile.TemporaryDirectory(prefix='sdv2-oracle-') as folder:
    target=Path(folder)/'streamdiffusionv2/pipeline.py';target.parent.mkdir();target.write_text(original)
    subprocess.run(['patch','--quiet','--batch','-p1','--directory',folder,'--input',str(a.patch.resolve())],check=True)
    candidate=target.read_text()
noise_source=(a.tree/'streamv2v/inference.py').read_text()
noise_node=next(n for n in ast.parse(noise_source).body if isinstance(n,ast.FunctionDef) and n.name=='compute_noise_scale_and_step')

def load(text):
 tree=ast.parse(text)
 normalize=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_normalize_video_tensor')
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StreamDiffusionV2Pipeline')
 encode=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode_chunk')
 encode.decorator_list=[]
 for fn in [normalize,encode,noise_node]:
  fn.returns=None
  for arg in fn.args.posonlyargs+fn.args.args+fn.args.kwonlyargs:arg.annotation=None
 ns=dict(torch=torch,Path=Path,EncodedChunk=lambda **kw:types.SimpleNamespace(**kw))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[normalize,noise_node,encode],type_ignores=[])),str(source),'exec'),ns)
 sizes=[];normal=ns['_normalize_video_tensor']
 def track(video,**kw):sizes.append(video.numel());return normal(video,**kw)
 ns['_normalize_video_tensor']=track
 return ns['encode_chunk'],sizes
before,bsizes=load(original);after,asizes=load(candidate)
torch.set_num_threads(1);torch.manual_seed(0);cases=0
for ndim in [4,5]:
 for dtype in [torch.float32,torch.bfloat16,torch.float16]:
  for frames in [9,17,33,65]:
   for chunk_size in [4,8]:
    for noncontiguous in [False,True]:
     x=torch.randn((2,3,frames,6,10) if ndim==5 else (3,frames,6,10),dtype=dtype)
     if noncontiguous:x=x.transpose(-1,-2)
     fake=types.SimpleNamespace(height=x.shape[-2],width=x.shape[-1],device=torch.device('cpu'),noise_scale=.8,chunk_size=chunk_size,pipeline_manager=types.SimpleNamespace(_timed_stream_encode=lambda frames:torch.arange(24,dtype=torch.float32).reshape(1,2,3,2,2)))
     for end in range(chunk_size+1,frames+1,chunk_size):
      start=0 if end==chunk_size+1 else end-chunk_size
      chunk=types.SimpleNamespace(start_idx=start,end_idx=end,current_start=start,current_end=end,frames=x[...,start:end,:,:])
      torch.manual_seed(11);a=before(fake,x,chunk,previous_noise_scale=.76,initial_noise_scale=.8)
      torch.manual_seed(11);b=after(fake,x,chunk,previous_noise_scale=.76,initial_noise_scale=.8)
      assert a.noise_scale==b.noise_scale and a.current_step==b.current_step
      assert torch.equal(a.noisy_latents,b.noisy_latents)
      cases+=1
# First chunk retains complete normalization; later calls normalize a 5-frame window.
print(json.dumps(dict(passed=True,comparisons=cases,baseline_normalized_elements=sum(bsizes),candidate_normalized_elements=sum(asizes),scope='Actual encode_chunk and normalize/noise functions extracted from source; CPU values, seeded latent stub; full-model/GPU validation separate.'),indent=2))
