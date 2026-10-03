"""F5-TTS public API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'reference_text':'Some call me nature, others call me mother nature.','text':'This is a test of speech generation.','steps':32,'cfg_strength':2.0,'sway_sampling_coef':-1.0,'speed':1.0,'seed':42}
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
 # waveform and spectrogram return) on real inputs. Preserve the original D2H copy and stream wait.
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
original_pad=torch.nn.functional.pad;original_randn=torch.randn
shape_adapters=[]
def pad(input,pad,*args,**kwargs):
 positions=[i for i,v in enumerate(pad)if isinstance(v,torch.Tensor)and v.device.type=='cuda']
 if not tape.active or not positions:return original_pad(input,pad,*args,**kwargs)
 key=tag('pad_shape',input,sys._getframe(1))
 if physical:
  result=original_pad(input,pad,*args,**kwargs);concrete=list(pad)
  for i in positions:
   assert not isinstance(pad[i^1],torch.Tensor)
   axis=input.ndim-1-i//2;concrete[i]=result.shape[axis]-input.shape[axis]-pad[i^1]
  shape=tape.exchange(key,{'padding':concrete,'output':list(result.shape)})
 else:
  for i in positions:original['item'](pad[i]) # original implicit scalar-copy/wait
  shape=tape.exchange(key)
  for i,v in enumerate(pad):
   if i not in positions:assert v==shape['padding'][i]
  result=original_pad(input,shape['padding'],*args,**kwargs)
 assert list(result.shape)==shape['output']
 shape_adapters.append({'operation':'pad','input':list(input.shape),'output':shape,'native_shape_checked':physical})
 return result
def randn(*args,**kwargs):
 positions=[i for i,v in enumerate(args)if isinstance(v,torch.Tensor)and v.device.type=='cuda']
 if not tape.active or not positions:return original_randn(*args,**kwargs)
 key=tag('randn_shape',args[positions[0]],sys._getframe(1))
 if physical:
  result=original_randn(*args,**kwargs);shape=tape.exchange(key,list(result.shape))
 else:
  for i in positions:original['item'](args[i])
  shape=tape.exchange(key);concrete=list(args)
  for i in positions:concrete[i]=shape[i]
  result=original_randn(*concrete,**kwargs)
 assert list(result.shape)==shape
 shape_adapters.append({'operation':'randn','output':shape,'native_shape_checked':physical})
 return result
torch.nn.functional.pad=pad;torch.randn=randn
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
torch.arange=arange
for fft_name,fft_original in [('stft',original_stft),('istft',original_istft),('irfft',original_irfft)]:
 def fft(*args,_name=fft_name,_fn=fft_original,**kwargs):
  x=args[0]if args else kwargs['input']
  if not tape.active or x.device.type!='cuda':return _fn(*args,**kwargs)
  key=tag(_name+'_shape',x,sys._getframe(1))
  if physical or not a.fft_shape_adapter:
   result=_fn(*args,**kwargs)
   if physical:shape=tape.exchange(key,{'shape':list(result.shape),'dtype':str(result.dtype)})
   else:shape=tape.exchange(key)
   assert list(result.shape)==shape['shape']and str(result.dtype)==shape['dtype']
  else:
   shape=tape.exchange(key);result=torch.empty(shape['shape'],dtype=getattr(torch,shape['dtype'].split('.')[-1]),device=x.device)
  fft_adapters.append({'operation':_name,'input':list(x.shape),'output':shape,'native_shape_checked':physical})
  return result
 if fft_name=='irfft':torch.fft.irfft=fft
 else:setattr(torch,fft_name,fft)
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
from f5_tts.api import F5TTS
import f5_tts,torchdiffeq
solver_root=Path(torchdiffeq.__file__).parent
root=Path(sys.modules['f5_tts.api'].__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'fft_shape_adapter':a.fft_shape_adapter,'request':spec,'scope':'Original F5TTS.infer with supplied reference transcript, original reference preprocessing/cache, English text conversion, one generation chunk,32Euler steps, CFG2, sway-1, Vocos decode and CPU return. No ASR or output file encoding.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'solver_source_sha256':{str(f.relative_to(solver_root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in solver_root.rglob('*.py')},'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','torchaudio','transformers','numpy','f5-tts','torchdiffeq','x-transformers','vocos','torchcodec')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save();torch.manual_seed(42);start=time.perf_counter()
model=F5TTS(model='F5TTS_v1_Base',ckpt_file=str(a.model/'F5-TTS/F5TTS_v1_Base/model_1250000.safetensors'),vocab_file=str(a.model/'F5-TTS/F5TTS_v1_Base/vocab.txt'),vocoder_local_path=str(a.model/'vocos-mel-24khz'),device='cuda')
ref_file=Path(str(files('f5_tts').joinpath('infer/examples/basic/basic_ref_en.wav')))
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters=sum(p.numel()for p in model.ema_model.parameters()),model_dtype=str(next(model.ema_model.parameters()).dtype),vocoder_dtype=str(next(model.vocoder.parameters()).dtype),reference_sha256=hashlib.sha256(ref_file.read_bytes()).hexdigest());save();print('F5_LOADED',report['load_host_seconds'],flush=True)
rt=ctypes.CDLL(None);timing=False;current_iteration=0;worker_requests=0;original_submit=ThreadPoolExecutor.submit
# Original API creates a worker for each text chunk. Mark its entire callback
# separately: mode2 clocks across caller/worker threads are not synchronized.
def submit(pool,fn,*args,**kwargs):
 if fn.__name__!='infer_single_process':return original_submit(pool,fn,*args,**kwargs)
 def invoke():
  global worker_requests
  worker_requests+=1
  if timing:rt.gxvm_timeline_mark(current_iteration,0)
  result=fn(*args,**kwargs)
  if timing:rt.gxvm_timeline_mark(current_iteration,1)
  return result
 return original_submit(pool,invoke)
ThreadPoolExecutor.submit=submit
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 torch.nn.functional.pad=original_pad;torch.randn=original_randn
 torch.arange=original_arange;torch.stft=original_stft;torch.istft=original_istft;torch.fft.irfft=original_irfft
 ThreadPoolExecutor.submit=original_submit
def generate(index):
 global worker_requests
 worker_requests=0;tape.start(index)
 out=model.infer(ref_file=str(ref_file),ref_text=spec['reference_text'],gen_text=spec['text'],nfe_step=spec['steps'],cfg_strength=spec['cfg_strength'],sway_sampling_coef=spec['sway_sampling_coef'],speed=spec['speed'],seed=spec['seed'],show_info=lambda *args:None,progress=None)
 torch.cuda.synchronize()
 if not a.plain:assert worker_requests==1,worker_requests
 return out
def check(out):
 w,sr,mel=out;assert w.ndim==1 and len(w)>0 and sr==24000
 row={'sample_rate':sr,'samples':len(w),'dtype':str(w.dtype),'mel_shape':list(mel.shape),'control_count':tape.cursor}
 if physical:
  assert np.isfinite(w).all()and np.isfinite(mel).all();row.update(finite=True,output_sha256=hashlib.sha256(w.tobytes()).hexdigest(),mel_sha256=hashlib.sha256(mel.tobytes()).hexdigest(),minimum=float(w.min()),maximum=float(w.max()))
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('F5_WARMUP_DONE',flush=True)
if a.role=='partial_sync':
 rt.gxvm_adopt_now();rt.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int];timing=True
if a.role=='cpu-profile':rt.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
for i in range(a.repetitions):
 current_iteration=i+1
 if timing:rt.gxvm_timeline_mark(1000+current_iteration,0)
 start=time.perf_counter();out=generate(a.warmup+i);elapsed=time.perf_counter()-start
 if timing:rt.gxvm_timeline_mark(1000+current_iteration,1)
 tape.finish();report['iterations'].append(check(out)|{'iteration':i+1,'application_clock_seconds':elapsed});save()
if a.role=='cpu-profile':assert rt.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
if reference is None and not a.plain:
 (a.output/'controls.json').write_text(json.dumps({'request':spec,'requests':tape.records},indent=2)+'\n');np.savez_compressed(a.output/'controls.npz',**tape.arrays)
report.update(phase='completed',fft_adapters=fft_adapters,shape_adapters=shape_adapters,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save();print('F5_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
