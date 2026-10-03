"""Outlines native Generator API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'prompt':'Return a 64 digit integer:','regex':'[0-9]{64}','batch_size':4,'max_new_tokens':96,'do_sample':False,'use_cache':True,'seed':42}
reference=json.loads(a.controls.read_text())if a.controls else None
if reference:assert reference['request']==spec
ref_arrays=dict(np.load(a.control_arrays or a.controls.with_suffix('.npz')))if reference else {}
ref_tensors={k:torch.from_numpy(v)for k,v in ref_arrays.items()}
torch.set_num_threads(1);torch.set_num_interop_threads(1)
original={name:getattr(torch.Tensor,name)for name in ('item','__bool__','__int__','__index__','tolist','cpu','to','__getitem__')}
pending_hooks={}
class Tape:
 def __init__(self):self.active=False;self.records=[];self.arrays={}
 def start(self,index):
  self.active=not a.plain;self.index=index;self.events=[];self.cursor=0
  self.selected=reference['requests'][min(index,len(reference['requests'])-1)]if reference else None
 def finish(self):
  self.active=False
  if self.selected is not None and not a.plain:assert self.cursor==len(self.selected),(self.cursor,len(self.selected))
  self.records.append(self.events)
 def exchange(self,tag,value=None,tensor=None):
  if self.selected is not None:
   expected=self.selected[self.cursor];assert expected['tag']==tag,('control path',self.cursor,tag,expected['tag'])
   if tensor is not None:
    key=expected['array'];target=ref_tensors[key]
    if physical:assert torch.equal(tensor,target),('native CPU array mismatch',tag)
    result=target
   else:
    if physical:assert value==expected['value'],('native scalar mismatch',tag,value,expected['value'])
    result=expected['value']
   event=expected
  else:
   assert physical
   if tensor is not None:
    key=f'r{self.index}_e{self.cursor}';self.arrays[key]=tensor.numpy().copy();event={'tag':tag,'array':key};result=tensor
   else:event={'tag':tag,'value':value};result=value
  self.events.append(event);self.cursor+=1;return result

tape=Tape()
def tag(name,t,frame):return [name,'runner.py' if frame.f_code.co_filename==__file__ else Path(frame.f_code.co_filename).name,frame.f_code.co_name,list(t.shape),str(t.dtype)]
for name in ('item','__bool__','__int__','__index__','tolist'):
 def scalar(t,*args,_name=name,**kwargs):
  if not tape.active or t.device.type!='cuda':return original[_name](t,*args,**kwargs)
  frame=sys._getframe(1);value=original[_name](t,*args,**kwargs)
  return tape.exchange(tag(_name,t,frame),value if physical else None)
 pending_hooks[name]=scalar
def cpu(t,*args,**kwargs):
 result=original['cpu'](t,*args,**kwargs)
 if not tape.active or t.device.type!='cuda':return result
 frame=sys._getframe(1);key=tag('cpu',t,frame)
 # Preloaded native host values keep post-GPU CPU work (including the original
 # watermark) on real inputs. Preserve the original D2H copy and stream wait.
 if physical:return tape.exchange(key,tensor=result)
 expected=tape.selected[tape.cursor];assert expected['tag']==key,(key,expected['tag'])
 return tape.exchange(key,tensor=ref_tensors[expected['array']])
pending_hooks['cpu']=cpu
def getitem(t,index):
 result=original['__getitem__'](t,index)
 if tape.active and isinstance(index,torch.Tensor) and index.dtype==torch.bool and index.device.type=='cuda':
  frame=sys._getframe(1);shape=tape.exchange(tag('boolean_index_shape',t,frame),list(result.shape)if physical else None)
  if not physical and list(result.shape)!=shape:result=t.new_empty(shape)
 return result
pending_hooks['__getitem__']=getitem
# Preserve explicit GPU-to-CPU length transfers used by packed-RNN helpers.
def to_cpu(t,*args,**kwargs):
 result=original['to'](t,*args,**kwargs)
 if not tape.active or t.device.type!='cuda' or result.device.type!='cpu':return result
 frame=sys._getframe(1);key=tag('to_cpu',t,frame)
 if physical:return tape.exchange(key,tensor=result)
 expected=tape.selected[tape.cursor];assert expected['tag']==key,(key,expected['tag'])
 return tape.exchange(key,tensor=ref_tensors[expected['array']])
pending_hooks['to']=to_cpu
original_arange=torch.arange
original_stft=torch.stft;original_istft=torch.istft;original_irfft=torch.fft.irfft
fft_adapters=[]
def arange(*args,**kwargs):
 tensors=[x for x in args if isinstance(x,torch.Tensor)and x.device.type=='cuda']
 if tape.active and tensors:
  key=tag('arange_shape',tensors[0],sys._getframe(1))
  if physical:
   result=original_arange(*args,**kwargs);tape.exchange(key,list(result.shape));return result
  assert len(args)==1
  original['item'](tensors[0]) # retain scalar read/copy/wait
  shape=tape.exchange(key);assert len(shape)==1
  return original_arange(shape[0],**kwargs)
 return original_arange(*args,**kwargs)
pending_arange=arange
fft_adapters=[];shape_adapters=[]
from transformers import AutoModelForCausalLM,AutoTokenizer
import outlines
root=Path(outlines.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'request':spec,'scope':'Original Outlines Generator.batch with outlines_core regex backend and pretrained SmolLM2-135M BF16 SDPA. Four identical prompts, constrained64digits, max96newtokens,greedy,seed42. Includes prompt tokenization, model decoding, CPU grammar and token-mask processing, output decoding; excludes model/grammar compilation and file I/O.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','transformers','numpy','outlines_core')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def seed():torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
save();seed();start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(str(a.model),local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left';network=AutoModelForCausalLM.from_pretrained(str(a.model),local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();wrapper=outlines.from_transformers(network,tokenizer);model=outlines.Generator(wrapper,outlines.regex(spec['regex']),backend='outlines_core')
# This workload needs no indexing/arange shape adapter; leave those methods
# original so the outlines_core compiled mask kernel sees ordinary operators.
pending_hooks.pop('__getitem__',None)
for hook_name,hook in pending_hooks.items():setattr(torch.Tensor,hook_name,hook)
torch.arange=original_arange
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters=sum(p.numel()for p in network.parameters()),model_dtype=str(next(network.parameters()).dtype));save();print('OUTLINES_LOADED',report['load_host_seconds'],flush=True)
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 torch.arange=original_arange

def generate(index):
 seed();tape.start(index);out=model.batch([spec['prompt']]*spec['batch_size'],max_new_tokens=spec['max_new_tokens'],do_sample=spec['do_sample'],use_cache=spec['use_cache'],pad_token_id=tokenizer.pad_token_id);torch.cuda.synchronize();return out

def check(out):
 import re
 assert isinstance(out,list) and len(out)==spec['batch_size']
 row={'batch_size':len(out),'lengths':[len(x)for x in out],'control_count':tape.cursor}
 if physical:
  assert all(re.fullmatch(spec['regex'],x)for x in out),out
  row.update(finite=True,outputs=out,output_sha256=hashlib.sha256(json.dumps(out,ensure_ascii=False).encode()).hexdigest())
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('OUTLINES_WARMUP_DONE',flush=True)
rt=ctypes.CDLL(None)
if a.role=='partial_sync':rt.gxvm_adopt_now();rt.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
if a.role=='cpu-profile':rt.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
for i in range(a.repetitions):
 if a.role=='partial_sync':rt.gxvm_timeline_mark(i+1,0)
 start=time.perf_counter();out=generate(a.warmup+i);elapsed=time.perf_counter()-start
 if a.role=='partial_sync':rt.gxvm_timeline_mark(i+1,1)
 tape.finish();report['iterations'].append(check(out)|{'iteration':i+1,'application_clock_seconds':elapsed});save()
if a.role=='cpu-profile':assert rt.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
if reference is None and not a.plain:
 (a.output/'controls.json').write_text(json.dumps({'request':spec,'requests':tape.records},indent=2)+'\n');np.savez_compressed(a.output/'controls.npz',**tape.arrays)
report.update(phase='completed',fft_adapters=fft_adapters,shape_adapters=shape_adapters,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save();print('OUTLINES_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
