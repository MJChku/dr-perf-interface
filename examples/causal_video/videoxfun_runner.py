"""Real VideoX-Fun Wan1.3B, fixed offload policy and complete decoded requests."""
import argparse,ctypes,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from videox_fun.models import AutoencoderKLWan,AutoTokenizer,WanT5EncoderModel,WanTransformer3DModel
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.fp8_optimization import replace_parameters_by_name
import videox_fun

class EulerFromStart(FlowMatchEulerDiscreteScheduler):
 def set_timesteps(self,*args,**kwargs):
  super().set_timesteps(*args,**kwargs)
  # Full generation always begins at schedule entry0; avoid GPU-derived lookup
  # under skipped arithmetic. Native and GX use exactly the same schedule.
  self.set_begin_index(0)
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=17);p.add_argument('--steps',type=int,default=8);p.add_argument('--height',type=int,default=256);p.add_argument('--width',type=int,default=448);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=2);p.add_argument('--role',choices=['emu','cpu-profile','partial_sync','native','gpu-profile'],default='emu');p.add_argument('--offload',choices=['sequential','model','resident'],default='sequential');p.add_argument('--profile-cpu',action='store_true');p.add_argument('--async-offload',action='store_true');p.add_argument('--allocator-gib',type=float,default=4.0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output);torch.set_num_threads(1);torch.set_num_interop_threads(1)
physical=a.role in ['native','gpu-profile'];assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==physical
root=Path(videox_fun.__file__).parent
report={'config':vars(a)|{'source':str(a.source),'output':str(a.output)},'torch':torch.__version__,'phase':'loading','scheduler':'Supported Flow Euler, shift3; full generation starts at schedule entry0 in both GX and native','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*.py'))}}
import accelerate
accel_root=Path(accelerate.__file__).parent
report['accelerate_version']=accelerate.__version__
report['accelerate_sha256']={str(f.relative_to(accel_root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(accel_root.rglob('*.py'))}
report['allocator_limit_bytes']=int(a.allocator_gib*1024**3)
torch.cuda.set_per_process_memory_fraction(report['allocator_limit_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
save(a.output/'report.json',report);start=time.perf_counter();cfg=OmegaConf.load(a.source/'config/wan2.1/wan_civitai.yaml');dtype=torch.bfloat16
transformer=WanTransformer3DModel.from_pretrained(a.model,transformer_additional_kwargs=OmegaConf.to_container(cfg.transformer_additional_kwargs),low_cpu_mem_usage=True,torch_dtype=dtype)
vae=AutoencoderKLWan.from_pretrained(str(Path(a.model)/'Wan2.1_VAE.pth'),additional_kwargs=OmegaConf.to_container(cfg.vae_kwargs)).to(dtype)
text=WanT5EncoderModel.from_pretrained(str(Path(a.model)/'models_t5_umt5-xxl-enc-bf16.pth'),additional_kwargs=OmegaConf.to_container(cfg.text_encoder_kwargs),low_cpu_mem_usage=True,torch_dtype=dtype)
tokenizer=AutoTokenizer.from_pretrained(str(Path(a.model)/'google/umt5-xxl'))
report['parameters']={n:sum(p.numel() for p in m.parameters()) for n,m in [('transformer',transformer),('vae',vae),('text_encoder',text)]}
for m in [transformer,vae,text]:assert all(p.device.type!='meta' for p in m.parameters())
pipe=WanPipeline(transformer=transformer,vae=vae,text_encoder=text,tokenizer=tokenizer,scheduler=EulerFromStart(num_train_timesteps=1000,shift=3.0))
if a.offload=='sequential':
 replace_parameters_by_name(transformer,['modulation'],device='cuda:0');transformer.freqs=transformer.freqs.to('cuda:0');pipe.enable_sequential_cpu_offload(device='cuda:0')
elif a.offload=='model':pipe.enable_model_cpu_offload(device='cuda:0')
else:pipe.to('cuda:0')
from accelerate.hooks import AlignDevicesHook,SequentialHook
def configure_async(hook):
 if isinstance(hook,AlignDevicesHook):
  if hook.offload and isinstance(hook.execution_device,(str,torch.device)) and torch.device(hook.execution_device).type=='cuda':
   hook.non_blocking=a.async_offload
   hook.clear_cache=not a.async_offload
   return 1
 if isinstance(hook,SequentialHook):return sum(configure_async(h) for h in hook.hooks)
 return 0
report['configured_offload_hooks']=sum(configure_async(m._hf_hook) for model in [transformer,vae,text] for m in model.modules() if hasattr(m,'_hf_hook'))
assert report['configured_offload_hooks']>0
# A large transient embedding allocation must be released before small live
# outputs can split and pin its cached segment. Keep normal layer reuse.
report['embedding_cache_release_hooks']=0
if a.async_offload:
 def release_embedding_cache(module,args,output):
  torch.cuda.empty_cache()
 for model in [transformer,vae,text]:
  for module in model.modules():
   if isinstance(module,torch.nn.Embedding) and hasattr(module,'_hf_hook') and module.weight.numel()*module.weight.element_size()>report['allocator_limit_bytes']/2:
    module.register_forward_hook(release_embedding_cache,always_call=True)
    report['embedding_cache_release_hooks']+=1

report.update(load_seconds=time.perf_counter()-start,phase='loaded');save(a.output/'report.json',report)
def generate():
 try:
  return pipe(prompt='A cat walks on the grass, realistic style',negative_prompt='',height=a.height,width=a.width,num_frames=a.frames,num_inference_steps=a.steps,guidance_scale=5.0,generator=torch.Generator(device='cuda').manual_seed(42),output_type='pil',shift=3)
 finally:
  # Release unused reservation at each request boundary, including failure.
  # All weights continue to be offloaded immediately by their original hooks.
  torch.cuda.empty_cache()
for _ in range(a.warmup):generate()
torch.cuda.synchronize();print('VIDEOXFUN_WARMUP_DONE',flush=True);runtime=ctypes.CDLL(None)
if a.role=='partial_sync':runtime.gxvm_adopt_now();runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
if a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
if a.profile_cpu:
 import yappi
 yappi.set_clock_type('cpu');yappi.start(builtins=True)
report['iterations']=[]
for i in range(a.repetitions):
 if a.role=='partial_sync':runtime.gxvm_timeline_mark(i+1,0)
 start=time.perf_counter();result=generate();torch.cuda.synchronize();seconds=time.perf_counter()-start
 if a.role=='partial_sync':runtime.gxvm_timeline_mark(i+1,1)
 row={'iteration':i+1,'application_clock_seconds':seconds}
 if physical:
  pixels=np.asarray(result.videos);row.update(output_shape=list(pixels.shape),output_dtype=str(pixels.dtype),output_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),finite=bool(np.isfinite(pixels).all()));assert row['finite']
 report['iterations'].append(row)
if a.profile_cpu:
 yappi.stop();stats=yappi.get_func_stats();stats.save(str(a.output/'cpu.pstats'),type='pstat')
 with (a.output/'cpu.txt').open('w') as f:
  yappi.get_thread_stats().print_all(out=f);stats.sort('tsub').print_all(out=f,columns={0:('name',130),1:('ncall',12),2:('tsub',12),3:('ttot',12)})
if a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
report.update(phase='completed',peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved());save(a.output/'report.json',report);print('VIDEOXFUN_RESULT',json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
