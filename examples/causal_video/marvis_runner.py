"""Marvis native Generator API and recorded GPU-control replay for GX."""
import argparse,ctypes,hashlib,json,os,random,resource,sys,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=['gpu-profile','native','emu','cpu-profile','partial_sync'],required=True);p.add_argument('--controls',type=Path);p.add_argument('--control-arrays',type=Path);p.add_argument('--warmup',type=int,default=2);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--plain',action='store_true');p.add_argument('--host-cache-positions',action='store_true');p.add_argument('--fft-shape-adapter',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('gpu-profile','native');assert physical or a.controls;assert not a.plain or a.role=='native'
spec={'text':'Hello.','speaker':0,'max_audio_length_ms':2560,'temperature':0.9,'topk':50,'seed':42}
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
import torchaudio
from safetensors.torch import load_file
from transformers import AutoTokenizer
from marvis_tts.models import Model,ModelArgs
from marvis_tts import generator as generator_module
from marvis_tts import utils as marvis_utils
# Resolve only this fixed workload's pinned tokenizer/codec at load time.
tokenizer_loader=AutoTokenizer.from_pretrained
AutoTokenizer.from_pretrained=classmethod(lambda cls,name,*args,**kwargs:tokenizer_loader(str(a.model/'smollm-tokenizer')if name=='HuggingFaceTB/SmolLM2-135M'else name,*args,**kwargs))
def pinned_codec(repo,filename,*args,**kwargs):
 assert repo=='kyutai/moshiko-pytorch-bf16'and filename=='tokenizer-e351c8d8-checkpoint125.safetensors';return str(a.model/'mimi'/filename)
generator_module.hf_hub_download=pinned_codec
root=Path(generator_module.__file__).parent
report={'phase':'loading','role':a.role,'plain':a.plain,'request':spec,'scope':'Original Marvis native Generator.generate, no reference context, Hello., max32audioframes/2.56seconds, temperature0.9 topk50 seed42. BF16 backbone/depth models (supported create_model default), original FP32 Mimi. Includes text processing, original diagnostic print, all frame/codebook sampling, Mimi decode and output D2H; no file encoding. Pinned safetensors checkpoint loaded strictly into native model, without a Transformers conversion.','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},'versions':{n:version(n)for n in ('torch','torchaudio','transformers','numpy','torchtune','torchao','moshi')},'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,'warmups':[],'iterations':[]}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
def seed():torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
save();seed();start=time.perf_counter();tokenizer=marvis_utils.load_smollm2_tokenizer();args=ModelArgs(backbone_flavor='llama-250M',decoder_flavor='llama-60M',text_vocab_size=tokenizer.vocab_size,audio_vocab_size=2051,audio_num_codebooks=32);network=Model(args).to(device='cuda',dtype=torch.bfloat16);network.load_state_dict(load_file(str(a.model/'model.safetensors')),strict=True);network.eval();model=generator_module.Generator(network,text_tokenizer=tokenizer,device='cuda')
tracked_caches=0
if a.host_cache_positions:
 for cache in network.modules():
  if hasattr(cache,'enable_host_position_tracking'):
   cache.enable_host_position_tracking(initial_size=0);tracked_caches+=1
 assert tracked_caches==10,tracked_caches
report['host_position_tracking']=a.host_cache_positions;report['tracked_caches']=tracked_caches
kv_source=Path(sys.modules['torchtune.modules.kv_cache'].__file__);report['kv_cache_source_sha256']=hashlib.sha256(kv_source.read_bytes()).hexdigest()
for hook_name,hook in pending_hooks.items():setattr(torch.Tensor,hook_name,hook)
torch.arange=pending_arange
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,parameters=sum(p.numel()for p in network.parameters()),codec_parameters=sum(p.numel()for p in model._audio_tokenizer.parameters()),model_dtype=str(next(network.parameters()).dtype),codec_dtype=str(next(model._audio_tokenizer.parameters()).dtype));save();print('MARVIS_LOADED',report['load_host_seconds'],flush=True)
if a.plain:
 for name,fn in original.items():setattr(torch.Tensor,name,fn)
 torch.arange=original_arange

def generate(index):
 seed();tape.start(index);out=model.generate(context=[],**{k:v for k,v in spec.items()if k!='seed'});out=out.cpu();torch.cuda.synchronize();return out

def check(out):
 wave=out.float().numpy();row={'sample_rate':model.sample_rate,'samples':out.numel(),'shape':list(out.shape),'dtype':str(wave.dtype),'control_count':tape.cursor}
 if physical:
  assert np.isfinite(wave).all();row.update(finite=True,output_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),minimum=float(wave.min()),maximum=float(wave.max()))
 return row
for i in range(a.warmup):
 out=generate(i);tape.finish();report['warmups'].append(check(out));save()
print('MARVIS_WARMUP_DONE',flush=True)
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
save();print('MARVIS_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
