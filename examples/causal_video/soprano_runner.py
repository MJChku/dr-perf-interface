"""Soprano public API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'text':'Hello. This is a test of speech generation.','temperature':0.0,'top_p':0.95,'repetition_penalty':1.2,'retries':0,'seed':42}
reference=json.loads(a.controls.read_text())if a.controls else None
if reference:assert reference['request']==spec
ref_arrays=dict(np.load(a.control_arrays or a.controls.with_suffix('.npz')))if reference else {}
ref_tensors={k:torch.from_numpy(v)for k,v in ref_arrays.items()}
torch.set_num_threads(1);torch.set_num_interop_threads(1)
original={name:getattr(torch.Tensor,name)for name in ('item','__bool__','__int__','__index__','cpu','to','__getitem__')}
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
# Preserve explicit GPU-to-CPU length transfers used by packed-RNN helpers.
def to_cpu(t,*args,**kwargs):
 result=original['to'](t,*args,**kwargs)
 if not tape.active or t.device.type!='cuda' or result.device.type!='cpu':return result
 frame=sys._getframe(1);key=tag('to_cpu',t,frame)
 if physical:return tape.exchange(key,tensor=result)
 expected=tape.selected[tape.cursor];assert expected['tag']==key,(key,expected['tag'])
 return tape.exchange(key,tensor=ref_tensors[expected['array']])
torch.Tensor.to=to_cpu
from soprano import SopranoTTS
import soprano
fft_adapters=[]
original_istft=torch.istft
def istft(input,*args,**kwargs):
 if not tape.active:return original_istft(input,*args,**kwargs)
 key=tag('istft_shape',input,sys._getframe(1))
 assert input.ndim==3 and len(args)>=3 and tuple(args[:3])==(2048,512,2048) and kwargs.get('center') is True
 if physical or not a.fft_shape_adapter:
  out=original_istft(input,*args,**kwargs);info={'shape':list(out.shape),'dtype':str(out.dtype)};tape.exchange(key,info)
 else:
  info=tape.exchange(key);out=torch.empty(info['shape'],device=input.device,dtype=getattr(torch,info['dtype'].split('.')[-1]))
 fft_adapters.append({'operation':'istft','input':list(input.shape),'output':info,'native_shape_checked':physical})
 return out
torch.istft=istft
root=Path(soprano.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'fft_shape_adapter':a.fft_shape_adapter,'request':spec,'scope':'Original SopranoTTS.infer with official Transformers backend, public text preprocessing, autoregressive generation, Vocos decoder and CPU waveform return. Constructor warmup retained outside measured requests. No file encoding or playback. Default LMDeploy-preferred auto backend is not measured.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','torchaudio','transformers','numpy','inflect','Unidecode')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def seed():torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 torch.istft=original_istft
save();seed();tape.start(0);start=time.perf_counter();model=SopranoTTS(backend='transformers',device='cuda',model_path=str(a.model));torch.cuda.synchronize();tape.finish()
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,bootstrap_controls=tape.cursor,parameters=sum(p.numel()for p in model.pipeline.model.parameters()),decoder_parameters=sum(p.numel()for p in model.decoder.parameters()),model_dtype=str(next(model.pipeline.model.parameters()).dtype),decoder_dtype=str(next(model.decoder.parameters()).dtype),model_attention_implementation=model.pipeline.model.config._attn_implementation);save();print('SOPRANO_LOADED',report['load_host_seconds'],flush=True)
def generate(index):
 seed();tape.start(index+1);out=model.infer(**{k:v for k,v in spec.items()if k!='seed'});torch.cuda.synchronize();return out
def check(out):
 w=out.numpy();assert w.ndim==1 and len(w)>0
 row={'sample_rate':32000,'samples':len(w),'dtype':str(w.dtype),'control_count':tape.cursor}
 if physical:
  assert np.isfinite(w).all();row.update(finite=True,output_sha256=hashlib.sha256(w.tobytes()).hexdigest(),minimum=float(w.min()),maximum=float(w.max()))
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('SOPRANO_WARMUP_DONE',flush=True)
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
save();print('SOPRANO_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
