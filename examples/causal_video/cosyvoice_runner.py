"""CosyVoice2 original text-streaming API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--speaker',type=Path);p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'text_chunks':['收到好友从远方寄来的生日礼物，','那份意外的惊喜与深深的祝福','让我心中充满了甜蜜的快乐，','笑容如花儿般绽放。'],'prompt_text':'希望你以后能够做的比我还好呦。','stream_audio':False,'fp16':True,'seed':42,'cached_speaker':True,'sampling':'original RAS top_p0.8 top_k25'}
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

import threading
from cosyvoice.cli.cosyvoice import CosyVoice2
import cosyvoice
root=Path(cosyvoice.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'request':spec,'scope':'Original CosyVoice2.inference_zero_shot text generator from official example, cached reference speaker, batched audio output. Native Qwen2,flow,HiFT pretrained weights,fp16,original RAS sampling. Includes text tokenization,threaded autoregressive generation,flow/vocoder,output transfer,request-end allocator clear/sync; excludes load,prompt speaker extraction,file encoding. Existing Torch stack,CPU ONNX provider for one-time reference preparation.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','transformers','numpy','onnxruntime','HyperPyYAML')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def seed():torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
save();seed();start=time.perf_counter();model=CosyVoice2(str(a.model),load_jit=False,load_trt=False,load_vllm=False,fp16=True)
if a.speaker:
 speaker=torch.load(a.speaker,map_location='cpu',weights_only=True)
 model.frontend.spk2info['gx_reference']={k:v.to(model.frontend.device)if isinstance(v,torch.Tensor)else v for k,v in speaker.items()}
else:
 assert physical
 prompt=root.parent/'asset/zero_shot_prompt.wav'
 model.add_zero_shot_spk(spec['prompt_text'],str(prompt),'gx_reference')
 speaker={k:v.cpu()if isinstance(v,torch.Tensor)else v for k,v in model.frontend.spk2info['gx_reference'].items()}
 torch.save(speaker,a.output/'speaker.pt')
# Hooks are installed only after construction and cached speaker preparation.
# Overriding __getitem__ changes PySequence_Check for scalar tensors and breaks
# the original torch.tensor(list_of_gpu_scalars) constructor. Preserve it.
pending_hooks.pop("__getitem__",None)
for name,fn in pending_hooks.items():setattr(torch.Tensor,name,fn)
torch.arange=pending_arange
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters={name:sum(p.numel()for p in getattr(model.model,name).parameters())for name in ('llm','flow','hift')},speech_tokenizer_providers=model.frontend.speech_tokenizer_session.get_providers());save();print('COSYVOICE_LOADED',report['load_host_seconds'],flush=True)
rt=ctypes.CDLL(None);timing=False;current_iteration=0;thread_errors=[]
def thread_error(args):thread_errors.append(str(args.exc_value));sys.__excepthook__(args.exc_type,args.exc_value,args.exc_traceback)
threading.excepthook=thread_error
original_llm_job=model.model.llm_job
worker_windows=[]
def llm_job(*args,**kwargs):
 if timing:rt.gxvm_timeline_mark(current_iteration,0)
 start=time.perf_counter()
 try:return original_llm_job(*args,**kwargs)
 finally:
  worker_windows.append(time.perf_counter()-start)
  if timing:rt.gxvm_timeline_mark(current_iteration,1)
model.model.llm_job=llm_job
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 torch.arange=original_arange;torch.stft=original_stft;torch.istft=original_istft;torch.fft.irfft=original_irfft
 model.model.llm_job=original_llm_job

def generate(index):
 seed();tape.start(index)
 def text_chunks():yield from spec['text_chunks']
 outputs=list(model.inference_zero_shot(text_chunks(),spec['prompt_text'],'',zero_shot_spk_id='gx_reference',stream=False,text_frontend=False))
 assert not thread_errors,thread_errors
 torch.cuda.synchronize();return outputs

def check(out):
 assert len(out)==1
 audio=out[0]['tts_speech'];row={'samples':audio.numel(),'sample_rate':model.sample_rate,'control_count':tape.cursor}
 if physical:
  array=audio.detach().float().cpu().numpy();assert np.isfinite(array).all();row.update(finite=True,output_sha256=hashlib.sha256(array.tobytes()).hexdigest())
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('COSYVOICE_WARMUP_DONE',flush=True)
if a.role=='partial_sync':rt.gxvm_adopt_now();rt.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int];timing=True
if a.role=='cpu-profile':rt.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
for i in range(a.repetitions):
 current_iteration=i+1
 if timing:rt.gxvm_timeline_mark(1001+i,0)
 start=time.perf_counter();out=generate(a.warmup+i);elapsed=time.perf_counter()-start
 if timing:rt.gxvm_timeline_mark(1001+i,1)
 tape.finish();report['iterations'].append(check(out)|{'iteration':i+1,'application_clock_seconds':elapsed});save()
if a.role=='cpu-profile':assert rt.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
if reference is None and not a.plain:
 (a.output/'controls.json').write_text(json.dumps({'request':spec,'requests':tape.records},indent=2)+'\n');np.savez_compressed(a.output/'controls.npz',**tape.arrays)
report.update(phase='completed',fft_adapters=fft_adapters,shape_adapters=shape_adapters,worker_windows=worker_windows,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save();print('COSYVOICE_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
