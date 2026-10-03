"""Chatterbox Turbo public API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'text':'Hello. This is a test of speech generation.','audio_prompt_path':None,'temperature':0.8,'top_k':1000,'top_p':0.95,'repetition_penalty':1.2,'seed':42}
reference=json.loads(a.controls.read_text())if a.controls else None
if reference:assert reference['request']==spec
ref_arrays=dict(np.load(a.control_arrays or a.controls.with_suffix('.npz')))if reference else {}
ref_tensors={k:torch.from_numpy(v)for k,v in ref_arrays.items()}
torch.set_num_threads(1);torch.set_num_interop_threads(1)
original={name:getattr(torch.Tensor,name)for name in ('item','__bool__','__int__','__index__','cpu','__getitem__')}
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
def tag(name,t,frame):return [name,Path(frame.f_code.co_filename).name,frame.f_code.co_name,list(t.shape),str(t.dtype)]
for name in ('item','__bool__','__int__','__index__'):
 def scalar(t,*args,_name=name,**kwargs):
  if not tape.active or t.device.type!='cuda':return original[_name](t,*args,**kwargs)
  frame=sys._getframe(1);value=original[_name](t,*args,**kwargs)
  return tape.exchange(tag(_name,t,frame),value if physical else None)
 setattr(torch.Tensor,name,scalar)
def cpu(t,*args,**kwargs):
 result=original['cpu'](t,*args,**kwargs)
 if not tape.active or t.device.type!='cuda':return result
 frame=sys._getframe(1);key=tag('cpu',t,frame)
 # Preloaded native host values keep post-GPU CPU work (including the original
 # watermark) on real inputs. Preserve the original D2H copy and stream wait.
 if physical:return tape.exchange(key,tensor=result)
 expected=tape.selected[tape.cursor];assert expected['tag']==key,(key,expected['tag'])
 return tape.exchange(key,tensor=ref_tensors[expected['array']])
torch.Tensor.cpu=cpu
def getitem(t,index):
 result=original['__getitem__'](t,index)
 if tape.active and isinstance(index,torch.Tensor) and index.dtype==torch.bool and index.device.type=='cuda':
  frame=sys._getframe(1);shape=tape.exchange(tag('boolean_index_shape',t,frame),list(result.shape)if physical else None)
  if not physical and list(result.shape)!=shape:result=t.new_empty(shape)
 return result
torch.Tensor.__getitem__=getitem
from chatterbox.tts_turbo import ChatterboxTurboTTS
import chatterbox
from chatterbox.models.s3gen.hifigan import HiFTGenerator
fft_adapters=[]
original_stft=HiFTGenerator._stft;original_istft=HiFTGenerator._istft
def stft_shape(self,x):
 assert self.istft_params=={'n_fft':16,'hop_len':4} and x.ndim==2
 shape=(x.shape[0],9,1+x.shape[-1]//4)
 if physical:
  out=original_stft(self,x);assert all(tuple(y.shape)==shape for y in out)
 else:out=(x.new_empty(shape),x.new_empty(shape))
 fft_adapters.append({'operation':'stft','input':list(x.shape),'output':list(shape),'native_shape_checked':physical})
 return out
def istft_shape(self,magnitude,phase):
 assert self.istft_params=={'n_fft':16,'hop_len':4} and magnitude.ndim==3 and magnitude.shape[1]==9 and magnitude.shape==phase.shape
 shape=(magnitude.shape[0],4*(magnitude.shape[-1]-1))
 if physical:
  out=original_istft(self,magnitude,phase);assert tuple(out.shape)==shape
 else:out=magnitude.new_empty(shape)
 fft_adapters.append({'operation':'istft','input':list(magnitude.shape),'output':list(shape),'native_shape_checked':physical})
 return out
HiFTGenerator._stft=stft_shape;HiFTGenerator._istft=istft_shape
root=Path(chatterbox.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'request':spec,'scope':'Public ChatterboxTurboTTS.from_local and generate using built-in voice. Original tokenizer, autoregressive loop, meanflow decoder and CPU Perth watermark; no reference encoding or file encoding.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','torchaudio','transformers','numpy','librosa','chatterbox-tts','resemble-perth','diffusers')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save();torch.manual_seed(42);start=time.perf_counter();model=ChatterboxTurboTTS.from_local(a.model,device='cuda')
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters=sum(p.numel()for m in (model.t3,model.s3gen,model.ve)for p in m.parameters()));save();print('CHATTERBOX_LOADED',report['load_host_seconds'],flush=True)
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 HiFTGenerator._stft=original_stft;HiFTGenerator._istft=original_istft
def generate(index):
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
 tape.start(index);out=model.generate(**{k:v for k,v in spec.items()if k!='seed'});torch.cuda.synchronize();return out
def check(out):
 w=out.numpy();assert w.ndim==2 and w.shape[0]==1 and w.shape[1]>0
 row={'sample_rate':model.sr,'samples':w.shape[1],'dtype':str(w.dtype),'control_count':tape.cursor}
 if physical:assert np.isfinite(w).all();row.update(finite=True,output_sha256=hashlib.sha256(w.tobytes()).hexdigest(),minimum=float(w.min()),maximum=float(w.max()))
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('CHATTERBOX_WARMUP_DONE',flush=True)
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
report.update(phase='completed',fft_adapters=fft_adapters,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save();print('CHATTERBOX_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
