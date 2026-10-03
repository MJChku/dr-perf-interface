"""Public TeleFuser ABot interactive inference, with checked GX control shadows."""
import argparse,ctypes,functools,hashlib,inspect,json,os,sys,time,weakref
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['tree','checkpoint','image','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','drperf','partial_sync'],default='emu')
p.add_argument('--blocks',type=int,default=12);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=1)
p.add_argument('--native-original-controls',action='store_true')
a=p.parse_args()
assert not a.native_original_controls or a.role=='native'
for name in ['tree','checkpoint','image','output']:setattr(a,name,getattr(a,name).resolve())
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output);sys.path.insert(0,str(a.tree))
import numpy as np
import torch
from PIL import Image
torch.set_num_threads(1);torch.set_grad_enabled(False)
native=a.role in ['native','gpu-profile'];assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==native
report=dict(role=a.role,blocks=a.blocks,height=480,width=832,seed=42,control_latent_frames=3,warmup=a.warmup,repetitions=a.repetitions,
 prompt='A smooth first-person exploration through a vivid natural landscape.',torch=torch.__version__,
 harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),image_sha256=hashlib.sha256(a.image.read_bytes()).hexdigest(),
 source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((a.tree/'telefuser').rglob('*.py'))},
 generation_wall_seconds=[],warmup_wall_seconds=[],outputs=[],control_audit=dict(cursor_reads=0,cursor_substitutions=0,rope_reads=0,rope_substitutions=0,scheduler_copies=0,scheduler_substitutions=0),
 adaptations=['Public loader and unmodified source; mmap checkpoint setup outside measurement.',
 'Single-session eager path: original GPU cursor reads and fills retained; GX uses host shadow of known writes; native asserts equality.',
 'RoPE arange bounds: original GPU arange/reductions/scalar reads retained; GX substitutes known range bounds; native asserts equality.',
 'Scheduler timestep D2H retained; GX substitutes known four-step schedule on CPU; native asserts equality.'])
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
original_load=torch.load
def mmap_load(file,*args,**kwargs):
 if isinstance(file,(str,Path)) and str(file).endswith(('.pt','.pth')):kwargs.update(mmap=True,weights_only=True)
 return original_load(file,*args,**kwargs)
torch.load=mmap_load
from examples.abot_world._loader import get_pipeline
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.denoising import ABotWorldDenoisingStage
from telefuser.models import abot_world_dit as dit
from telefuser.schedulers.flow_match import FlowMatchScheduler

# Shadows carry only host-known control metadata, never model activations.
shadows={};ranges={};rope_expected=[];scheduler_expected=[]
original_new=ABotWorldDenoisingStage._new_cache
@functools.wraps(original_new)
def new_cache(self,*args,**kwargs):
 caches,cross=original_new(self,*args,**kwargs)
 for c in caches:shadows[id(c)]={'global_end_index':0,'local_end_index':0}
 return caches,cross
ABotWorldDenoisingStage._new_cache=new_cache
original_cursor=dit.CausalWanSelfAttention._cursor
original_set=dit.CausalWanSelfAttention._set_cursor
def cursor(cache,name):
 observed=original_cursor(cache,name);expected=shadows[id(cache)][name]
 report['control_audit']['cursor_reads']+=1
 if native:assert observed==expected,(observed,expected)
 elif observed!=expected:report['control_audit']['cursor_substitutions']+=1
 return expected
def set_cursor(cache,name,value):
 original_set(cache,name,value);shadows[id(cache)][name]=value
dit.CausalWanSelfAttention._cursor=staticmethod(cursor)
dit.CausalWanSelfAttention._set_cursor=staticmethod(set_cursor)
original_arange=torch.arange
def arange(*args,**kwargs):
 result=original_arange(*args,**kwargs)
 if result.device.type=='cuda' and result.dtype==torch.int64 and 1<=len(args)<=2 and all(isinstance(v,int) for v in args):
  start,end=(0,args[0]) if len(args)==1 else args
  key=id(result);ranges[key]=(weakref.ref(result,lambda ref:ranges.pop(key,None)),start,end)
 return result
torch.arange=arange
original_rope=dit._rope_apply
@functools.wraps(original_rope)
def rope(x,grid_size,freqs,frame_indices):
 ref,start,end=ranges[id(frame_indices)];assert ref() is frame_indices and end>start
 assert not rope_expected;rope_expected.extend([start,end-1])
 try:
  result=original_rope(x,grid_size,freqs,frame_indices)
  assert not rope_expected
  return result
 finally:rope_expected.clear()
dit._rope_apply=rope
original_item=torch.Tensor.item
def item(self,*args,**kwargs):
 observed=original_item(self,*args,**kwargs)
 if rope_expected and self.device.type=='cuda':
  expected=rope_expected.pop(0);report['control_audit']['rope_reads']+=1
  if native:assert observed==expected,(observed,expected)
  elif observed!=expected:report['control_audit']['rope_substitutions']+=1
  return expected
 return observed
torch.Tensor.item=item
original_denoise=ABotWorldDenoisingStage._denoise_block
@functools.wraps(original_denoise)
def denoise(self,*args,**kwargs):
 assert not scheduler_expected
 bound=inspect.signature(original_denoise).bind(self,*args,**kwargs)
 schedule=self._official_denoising_timesteps(bound.arguments['scheduler'])
 scheduler_expected.extend(schedule[1:].tolist())
 try:
  result=original_denoise(self,*args,**kwargs);assert not scheduler_expected;return result
 finally:scheduler_expected.clear()
ABotWorldDenoisingStage._denoise_block=denoise
original_cpu=torch.Tensor.cpu;in_add_noise=False
def cpu(self,*args,**kwargs):
 observed=original_cpu(self,*args,**kwargs)
 if in_add_noise and self.device.type=='cuda' and self.numel()==1:
  expected=scheduler_expected.pop(0);value=original_item(observed);report['control_audit']['scheduler_copies']+=1
  if native:assert value==expected,(value,expected)
  elif value!=expected:report['control_audit']['scheduler_substitutions']+=1
  if not native:return torch.full_like(observed,expected)
 return observed
torch.Tensor.cpu=cpu
original_noise=FlowMatchScheduler.add_noise
@functools.wraps(original_noise)
def add_noise(self,*args,**kwargs):
 global in_add_noise
 assert not in_add_noise;in_add_noise=True
 try:return original_noise(self,*args,**kwargs)
 finally:in_add_noise=False
FlowMatchScheduler.add_noise=add_noise

start=time.perf_counter();print('TELEFUSER_LOADING',flush=True)
pipe=get_pipeline(a.checkpoint,height=480,width=832,latent_frames=31,device_id=0,pipeline_class=ABotWorldInteractivePipeline)
pipe.preload_models();torch.cuda.synchronize();report['load_wall_seconds']=time.perf_counter()-start
report['attention_config']=str(pipe.denoise_stage.model_runtime_config.attention_config)
assert not pipe.denoise_stage._cuda_graph_enabled
image=Image.open(a.image).convert('RGB')
_,cpu_prompt_mask=pipe.text_encoding_stage.tokenizer(report['prompt'],return_mask=True,add_special_tokens=True)
prompt_lengths=cpu_prompt_mask.gt(0).sum(dim=1).long().tolist()
assert len(prompt_lengths)==1
report['prompt_tokens']=prompt_lengths[0]
report['control_audit'].update(prompt_index_reads=0,prompt_index_substitutions=0)
report['adaptations'].append('Prompt-padding scalar index read retained; GX uses CPU tokenizer mask length; native asserts equality.')
in_prompt=False
original_index=torch.Tensor.__index__
def tensor_index(self):
 observed=original_index(self)
 if in_prompt and self.device.type=='cuda':
  expected=prompt_lengths[0];report['control_audit']['prompt_index_reads']+=1
  if native:assert observed==expected,(observed,expected)
  elif observed!=expected:report['control_audit']['prompt_index_substitutions']+=1
  return expected
 return observed
torch.Tensor.__index__=tensor_index
text_class=type(pipe.text_encoding_stage);original_encode=text_class.encode_prompt
@functools.wraps(original_encode)
def encode_prompt(self,prompt):
 global in_prompt
 assert prompt==report['prompt'] and not in_prompt;in_prompt=True
 try:return original_encode(self,prompt)
 finally:in_prompt=False
text_class.encode_prompt=encode_prompt
if a.native_original_controls:
 ABotWorldDenoisingStage._new_cache=original_new
 dit.CausalWanSelfAttention._cursor=staticmethod(original_cursor)
 dit.CausalWanSelfAttention._set_cursor=staticmethod(original_set)
 torch.arange=original_arange;dit._rope_apply=original_rope;torch.Tensor.item=original_item
 ABotWorldDenoisingStage._denoise_block=original_denoise
 torch.Tensor.cpu=original_cpu;FlowMatchScheduler.add_noise=original_noise
 torch.Tensor.__index__=original_index;text_class.encode_prompt=original_encode
 report['adaptations']=['Public source controls restored for native validation; mmap checkpoint setup only.']
report['native_original_controls']=a.native_original_controls
save();print('TELEFUSER_LOADED',report['load_wall_seconds'],flush=True)

def install_marks():
 import perfmark
 def wrap(owner,method,name,features):
  orig=getattr(owner,method);sig=inspect.signature(orig)
  @functools.wraps(orig)
  def marked(*args,**kwargs):
   b=sig.bind(*args,**kwargs);b.apply_defaults()
   with perfmark.region(name,**features(b.arguments)):return orig(*args,**kwargs)
  setattr(owner,method,marked)
 wrap(ABotWorldInteractivePipeline,'create_interactive_session','abot_session',lambda b:{'pixels':480*832})
 wrap(ABotWorldInteractivePipeline,'generate_next_block','abot_block',lambda b:{'start':b['session'].next_latent_frame,'frames':b['control_latent_frames']})
 wrap(dit.CausalWanSelfAttention,'_update_cache','abot_cache',lambda b:{'tokens':b['key'].shape[1],'capacity':b['cache']['k'].shape[1]})
 wrap(dit,'_rope_apply','abot_rope',lambda b:{'tokens':b['x'].shape[1]})
 wrap(dit.ABotWorldDiT,'forward','abot_dit',lambda b:{'frames':b['x'].shape[2]})
 wrap(FlowMatchScheduler,'add_noise','abot_noise',lambda b:{'elements':b['original_samples'].numel()})

runtime=ctypes.CDLL(None);iteration=0
with torch.inference_mode():
 for cycle in range(a.warmup+a.repetitions):
  measuring=cycle>=a.warmup
  if cycle==a.warmup:
   if a.role=='drperf':install_marks()
   if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
   elif a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
   elif a.role=='partial_sync':runtime.gxvm_adopt_now();runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
   torch.cuda.reset_peak_memory_stats()
  if measuring and a.role=='partial_sync':iteration+=1;runtime.gxvm_timeline_mark(iteration,0)
  start=time.perf_counter();session=pipe.create_interactive_session(image,report['prompt'],seed=42);torch.cuda.synchronize();seconds=time.perf_counter()-start
  if measuring and a.role=='partial_sync':runtime.gxvm_timeline_mark(iteration,1)
  timings=report['generation_wall_seconds' if measuring else 'warmup_wall_seconds'];timings.append(dict(phase='session',cycle=cycle,seconds=seconds))
  for block in range(a.blocks):
   if measuring and a.role=='partial_sync':iteration+=1;runtime.gxvm_timeline_mark(iteration,0)
   start=time.perf_counter();frames=pipe.generate_next_block(session,actions={'W':True},control_latent_frames=3);torch.cuda.synchronize();seconds=time.perf_counter()-start
   if measuring and a.role=='partial_sync':runtime.gxvm_timeline_mark(iteration,1)
   timings.append(dict(phase='block',block=block,cycle=cycle,seconds=seconds,metrics=dict(pipe._last_stage_metrics)))
   assert len(frames)>0
   assert session.next_latent_frame==(block+1)*3
   assert all(f.size==(832,480) for f in frames)
   result=dict(cycle=cycle,block=block,frames=len(frames),size=list(frames[0].size),next_latent_frame=session.next_latent_frame)
   if native:result['sha256']=hashlib.sha256(b''.join(np.asarray(f).tobytes() for f in frames)).hexdigest()
   if measuring:report['outputs'].append(result)
   save();print('TELEFUSER_BLOCK',cycle,block,seconds,flush=True)
  pipe.close_interactive_session(session);shadows.clear()
 if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
 elif a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save();pipe.close();print('TELEFUSER_PASS',flush=True)
