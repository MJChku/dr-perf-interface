"""MeloTTS public API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'text':'Hello. This is a test of speech generation.','speaker_id':0,'sdp_ratio':0.2,'noise_scale':0.6,'noise_scale_w':0.8,'speed':1.0,'seed':42}
reference=json.loads(a.controls.read_text())if a.controls else None
if reference:assert reference['request']==spec
ref_arrays=dict(np.load(a.control_arrays or a.controls.with_suffix('.npz')))if reference else {}
ref_tensors={k:torch.from_numpy(v)for k,v in ref_arrays.items()}
torch.set_num_threads(1);torch.set_num_interop_threads(1)
original={name:getattr(torch.Tensor,name)for name in ('item','__bool__','__int__','__index__','cpu','to','__getitem__','__setitem__')}
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
def boolean_mask(index):
 parts=index if isinstance(index,tuple)else(index,)
 masks=[(i,x)for i,x in enumerate(parts)if isinstance(x,torch.Tensor)and x.dtype==torch.bool and x.device.type=='cuda']
 return parts,masks

def fixed_index(index,count):
 parts,masks=boolean_mask(index);assert len(masks)==1
 position,mask=masks[0];assert position==0
 # Empty integer-coordinate buffers supply only the calibrated cardinality.
 # GX predicts gathers/scatters, without executing numerical indexing. This
 # omits GPU nonzero/cardinality work; launch audits quantify that omission.
 coords=tuple(torch.empty((count,),dtype=torch.int64,device=mask.device)for _ in range(mask.ndim))
 return coords+tuple(parts[1:])

def getitem(t,index):
 parts,masks=boolean_mask(index)
 if not tape.active or not masks:return original['__getitem__'](t,index)
 frame=sys._getframe(1);key=tag('boolean_index_shape',t,frame)
 if physical:
  result=original['__getitem__'](t,index);tape.exchange(key,list(result.shape));return result
 shape=tape.exchange(key);result=original['__getitem__'](t,fixed_index(index,shape[0]));assert list(result.shape)==shape,(list(result.shape),shape)
 return result

def setitem(t,index,value):
 parts,masks=boolean_mask(index)
 if not tape.active or not masks:return original['__setitem__'](t,index,value)
 assert len(masks)==1;frame=sys._getframe(1);key=tag('boolean_setitem_count',t,frame)
 count=int(original['cpu'](masks[0][1]).numpy().sum())if physical else None
 count=tape.exchange(key,count)
 return original['__setitem__'](t,index if physical else fixed_index(index,count),value)
torch.Tensor.__getitem__=getitem;torch.Tensor.__setitem__=setitem
# Preserve explicit GPU-to-CPU length transfers used by packed-RNN helpers.
def to_cpu(t,*args,**kwargs):
 result=original['to'](t,*args,**kwargs)
 if not tape.active or t.device.type!='cuda' or result.device.type!='cpu':return result
 frame=sys._getframe(1);key=tag('to_cpu',t,frame)
 if physical:return tape.exchange(key,tensor=result)
 expected=tape.selected[tape.cursor];assert expected['tag']==key,(key,expected['tag'])
 return tape.exchange(key,tensor=ref_tensors[expected['array']])
torch.Tensor.to=to_cpu
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
fft_adapters=[];shape_adapters=[]
import torchaudio
"""Resolve only explicitly pinned HF repo aliases to local snapshots at load time."""
import os
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForMaskedLM
model_cache_root=Path(os.environ['HF_HOME'])/'hub'
allowed={'bert-base-uncased','bert-base-multilingual-uncased','tohoku-nlp/bert-base-japanese-v3','kykim/bert-kor-base','dbmdz/bert-base-french-europeana-cased','dccuchile/bert-base-spanish-wwm-uncased'}
for cls in (AutoTokenizer,AutoModelForMaskedLM):
 loader_original=cls.from_pretrained
 def local(cls,name,*args,_original=loader_original,**kwargs):
  if name in allowed:
   folder=model_cache_root/('models--'+name.replace('/','--'));rev=(folder/'refs/main').read_text().strip();name=str(folder/'snapshots'/rev);kwargs['local_files_only']=True
  return _original(name,*args,**kwargs)
 cls.from_pretrained=classmethod(local)

from melo.api import TTS
import melo
root=Path(melo.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'request':spec,'scope':'Original MeloTTS TTS.tts_to_file, English-v3, pretrained FP32 synthesizer plus English BERT, public defaults, resident models. Includes text/phoneme processing, BERT features, synthesis, original empty_cache and CPU audio concatenation. Returns array; no file encoding.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','torchaudio','transformers','numpy','g2p-en','nltk','numba','librosa')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def seed():torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
save();seed();start=time.perf_counter();model=TTS(language='EN_NEWEST',device='cuda',config_path=str(a.model/'config.json'),ckpt_path=str(a.model/'checkpoint.pth'))
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters=sum(p.numel()for p in model.parameters()),model_dtype=str(next(model.parameters()).dtype));save();print('MELO_LOADED',report['load_host_seconds'],flush=True)
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 torch.arange=original_arange

def generate(index):
 seed();tape.start(index);out=model.tts_to_file(quiet=True,output_path=None,**{k:v for k,v in spec.items()if k!='seed'});torch.cuda.synchronize();return out

def check(out):
 row={'sample_rate':model.hps.data.sampling_rate,'samples':out.size,'shape':list(out.shape),'dtype':str(out.dtype),'control_count':tape.cursor}
 if physical:
  assert np.isfinite(out).all();row.update(finite=True,output_sha256=hashlib.sha256(out.tobytes()).hexdigest(),minimum=float(out.min()),maximum=float(out.max()))
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('MELO_WARMUP_DONE',flush=True)
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
save();print('MELO_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
