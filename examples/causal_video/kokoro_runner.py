"""Kokoro public API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--prepare-inference',action='store_true');p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'text':'Hello. This is a test of speech generation.','voice':'af_heart','speed':1.0,'seed':42}
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
original_arange=torch.arange;original_repeat=torch.repeat_interleave
def arange(*args,**kwargs):
 result=original_arange(*args,**kwargs)
 tensors=[x for x in args if isinstance(x,torch.Tensor)and x.device.type=='cuda']
 if tape.active and tensors:
  key=tag('arange_shape',tensors[0],sys._getframe(1));shape=tape.exchange(key,list(result.shape)if physical else None)
  if not physical and list(result.shape)!=shape:
   assert len(args)==1 and not kwargs and len(shape)==1
   result=original_arange(shape[0],device=result.device,dtype=result.dtype)
 return result
def repeat_interleave(*args,**kwargs):
 controlled=tape.active and len(args)>1 and isinstance(args[1],torch.Tensor)and args[1].device.type=='cuda'
 if controlled and not physical:
  # GX does not compute predicted durations. Supplying the calibrated output
  # size avoids reading arbitrary values for allocation. The original native
  # path is unchanged; its shape-discovery work is omitted and audited.
  shape=tape.exchange(tag('repeat_shape',args[1],sys._getframe(1)))
  assert len(shape)==1 and 'output_size' not in kwargs
  return original_repeat(*args,**kwargs,output_size=shape[0])
 result=original_repeat(*args,**kwargs)
 if controlled:tape.exchange(tag('repeat_shape',args[1],sys._getframe(1)),list(result.shape))
 return result
torch.arange=arange;torch.repeat_interleave=repeat_interleave
from kokoro import KModel,KPipeline
import kokoro
from kokoro.istftnet import TorchSTFT
fft_adapters=[]
original_stft=TorchSTFT.transform;original_istft=TorchSTFT.inverse
# Original native FFT calls validate the fixed model's geometry. Optional GX
# shape adapters omit FFT-related work; launch audits must quantify omissions.
def stft(self,x):
 assert (self.filter_length,self.hop_length,self.win_length)==(20,5,20) and x.ndim==2
 expected=(x.shape[0],11,1+x.shape[-1]//5)
 out=original_stft(self,x)if physical or not a.fft_shape_adapter else (x.new_empty(expected),x.new_empty(expected))
 assert all(tuple(y.shape)==expected for y in out)
 fft_adapters.append({'operation':'stft','input':list(x.shape),'output':[list(y.shape)for y in out],'filter_length':self.filter_length,'hop_length':self.hop_length,'win_length':self.win_length,'native_shape_checked':physical})
 return out
def istft(self,magnitude,phase):
 assert (self.filter_length,self.hop_length,self.win_length)==(20,5,20) and magnitude.ndim==3 and magnitude.shape[1]==11 and magnitude.shape==phase.shape
 expected=(magnitude.shape[0],1,5*(magnitude.shape[-1]-1))
 out=original_istft(self,magnitude,phase)if physical or not a.fft_shape_adapter else magnitude.new_empty(expected)
 assert tuple(out.shape)==expected
 fft_adapters.append({'operation':'istft','input':list(magnitude.shape),'output':list(out.shape),'filter_length':self.filter_length,'hop_length':self.hop_length,'win_length':self.win_length,'native_shape_checked':physical})
 return out
TorchSTFT.transform=stft;TorchSTFT.inverse=istft
root=Path(kokoro.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'prepare_inference':a.prepare_inference,'fft_shape_adapter':a.fft_shape_adapter,'request':spec,'scope':'Public Kokoro KPipeline using local pretrained KModel and af_heart voice. Original English G2P, model duration prediction, packed RNNs, decoder, CPU waveform/duration return and word timestamps; no file encoding.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','torchaudio','transformers','numpy','kokoro','misaki','spacy','en-core-web-sm')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save();torch.manual_seed(42);start=time.perf_counter();model=KModel(repo_id='hexgrad/Kokoro-82M',config=str(a.model/'config.json'),model=str(a.model/'kokoro-v1_0.pth')).to('cuda').eval()
if a.prepare_inference:model.prepare_for_inference()
pipeline=KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M',model=model);voice=pipeline.load_voice(str(a.model/'voices/af_heart.pt'))
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters=sum(p.numel()for p in model.parameters()));save();print('KOKORO_LOADED',report['load_host_seconds'],flush=True)
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 TorchSTFT.transform=original_stft;TorchSTFT.inverse=original_istft
 torch.arange=original_arange;torch.repeat_interleave=original_repeat
def generate(index):
 torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
 tape.start(index);out=list(pipeline(spec['text'],voice=voice,speed=spec['speed']));torch.cuda.synchronize();return out
def check(out):
 assert len(out)==1,len(out)
 result=out[0];w=result.audio.numpy();assert w.ndim==1 and len(w)>0
 row={'sample_rate':24000,'samples':len(w),'dtype':str(w.dtype),'control_count':tape.cursor,'phonemes':result.phonemes,'graphemes':result.graphemes,'predicted_durations':result.pred_dur.tolist(),'timestamps':[[x.text,x.start_ts,x.end_ts]for x in result.tokens]}
 if physical:
  assert np.isfinite(w).all();row.update(finite=True,output_sha256=hashlib.sha256(w.tobytes()).hexdigest(),minimum=float(w.min()),maximum=float(w.max()))
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('KOKORO_WARMUP_DONE',flush=True)
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
save();print('KOKORO_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
