"""Run the public SkyReels-V2 diffusion-forcing pipeline under native CUDA or GX."""
import argparse,ctypes,functools,hashlib,inspect,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['tree','base','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','drperf','partial_sync'],default='emu')
p.add_argument('--frames',type=int,nargs='+',default=[33,65,97]);p.add_argument('--steps',type=int,default=30)
p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=1)
a=p.parse_args()
for name in ['tree','base','checkpoint','output']:setattr(a,name,getattr(a,name).resolve())
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output);sys.path.insert(0,str(a.tree))
import numpy as np
import torch,diffusers,transformers
from safetensors.torch import load_file
torch.set_num_threads(1);torch.set_grad_enabled(False)
native=a.role in ['native','gpu-profile'];assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==native
report=dict(role=a.role,frames=a.frames,steps=a.steps,height=544,width=960,seed=42,guidance_scale=6.0,shift=8.0,ar_step=0,offload=False,
 prompt='A woman in a leather jacket and sunglasses riding a vintage motorcycle through a desert highway at sunset, her hair blowing wildly in the wind as the motorcycle kicks up dust, with the golden sun casting long shadows across the barren landscape.',
 torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__,warmup=a.warmup,repetitions=a.repetitions,
 harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((a.tree/'skyreels_v2_infer').rglob('*.py'))},
 generation_wall_seconds=[],warmup_wall_seconds=[],outputs=[],control_audit=dict(scheduler_reads=0,scheduler_mismatches=0,solve_info_reads=0,solve_info_mismatches=0),
 adaptations=['Meta/mmap BF16 setup and strict transformer state loading, outside measured inference.',
 'All frame schedulers start at the unique first timestep; original GPU nonzero and scalar read retained, GX substitutes the known zero index; native asserts it.',
 'UniPC order-two solve_ex retains the scalar status read; native asserts success, GX substitutes success for the nonsingular schedule matrix.'])
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
original_load=torch.load
def mmap_load(file,*args,**kwargs):
 if isinstance(file,(str,Path)) and str(file).endswith(('.pt','.pth')):kwargs.update(mmap=True,weights_only=True)
 return original_load(file,*args,**kwargs)
torch.load=mmap_load
from skyreels_v2_infer.modules import t5,transformer
from skyreels_v2_infer.modules.tokenizers import HuggingfaceTokenizer
from skyreels_v2_infer.pipelines import diffusion_forcing_pipeline as pipeline
from skyreels_v2_infer.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler

def efficient_text(checkpoint_path=None,tokenizer_path=None,text_len=512,shard_fn=None):
 assert shard_fn is None
 obj=t5.T5EncoderModel.__new__(t5.T5EncoderModel);torch.nn.Module.__init__(obj)
 obj.text_len=text_len;obj.checkpoint_path=checkpoint_path;obj.tokenizer_path=tokenizer_path
 obj.model=t5.umt5_xxl(encoder_only=True,return_tokenizer=False,dtype=torch.bfloat16,device='meta')
 obj.model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'),strict=True,assign=True)
 obj.model.eval().requires_grad_(False);obj.tokenizer=HuggingfaceTokenizer(name=tokenizer_path,seq_len=text_len,clean='whitespace')
 return obj
import skyreels_v2_infer.modules as modules
modules.T5EncoderModel=efficient_text

def strict_transformer(path,device='cuda',weight_dtype=torch.bfloat16):
 # Construct on CPU so non-buffer RoPE tables remain real; no random parameters survive loading.
 model=transformer.WanModel.from_config(str(Path(path)/'config.json')).to(weight_dtype)
 state=load_file(str(Path(path)/'model.safetensors'));model.load_state_dict(state,strict=True);del state
 return model.eval().requires_grad_(False).to(device)
pipeline.get_transformer=strict_transformer

def checked_index(self,timestep,schedule_timesteps=None):
 assert self.step_index is None and self.begin_index is None and self.num_inference_steps==a.steps
 if schedule_timesteps is None:schedule_timesteps=self.timesteps
 indices=(schedule_timesteps==timestep).nonzero()
 if not native:indices.resize_(1,1)
 pos=1 if len(indices)>1 else 0
 observed=indices[pos].item();report['control_audit']['scheduler_reads']+=1
 if native:assert observed==0,observed
 elif observed!=0:report['control_audit']['scheduler_mismatches']+=1
 return 0
FlowUniPCMultistepScheduler.index_for_timestep=checked_index
original_solve=torch.linalg.solve
def checked_solve(A,B,*,left=True,out=None):
 if A.device.type!='cuda' or tuple(A.shape)!=(2,2) or tuple(B.shape)!=(2,) or out is not None:return original_solve(A,B,left=left,out=out)
 result,info=torch.linalg.solve_ex(A,B,left=left,check_errors=False)
 observed=info.item();report['control_audit']['solve_info_reads']+=1
 if native:assert observed==0,observed
 elif observed!=0:report['control_audit']['solve_info_mismatches']+=1
 return result
torch.linalg.solve=checked_solve
start=time.perf_counter();print('SKYREELS_LOADING',flush=True)
pipe=pipeline.DiffusionForcingPipeline(str(a.base),dit_path=str(a.checkpoint),device='cuda',weight_dtype=torch.bfloat16,offload=False)
assert not pipe.transformer.enable_teacache and not pipe.transformer.flag_causal_attention
report['model_config']=dict(pipe.transformer.config);torch.cuda.synchronize();report['load_wall_seconds']=time.perf_counter()-start
save();print('SKYREELS_LOADED',report['load_wall_seconds'],flush=True)
def generate(frames):
 generator=torch.Generator(device='cuda').manual_seed(42)
 out=pipe(prompt=report['prompt'],height=544,width=960,num_frames=frames,base_num_frames=frames,num_inference_steps=a.steps,guidance_scale=6.0,shift=8.0,ar_step=0,generator=generator,fps=24)
 torch.cuda.synchronize();return out[0]
def install_marks():
 import perfmark
 def wrap(owner,method,name,features):
  orig=getattr(owner,method);sig=inspect.signature(orig)
  @functools.wraps(orig)
  def marked(*args,**kwargs):
   b=sig.bind(*args,**kwargs);b.apply_defaults()
   with perfmark.region(name,**features(b.arguments)):return orig(*args,**kwargs)
  setattr(owner,method,marked)
 wrap(pipeline.DiffusionForcingPipeline,'__call__','sky_generate',lambda b:{'frames':b['num_frames'],'steps':b['num_inference_steps']})
 wrap(pipeline.DiffusionForcingPipeline,'generate_timestep_matrix','sky_schedule',lambda b:{'frames':b['num_frames'],'steps':len(b['step_template'])})
 wrap(FlowUniPCMultistepScheduler,'step','sky_step',lambda b:{'elements':b['sample'].numel(),'step':b['self'].step_index or 0})
 wrap(transformer.WanModel,'forward','sky_dit',lambda b:{'latent_frames':b['x'].shape[2]})
 wrap(transformer.WanSelfAttention,'forward','sky_attention',lambda b:{'tokens':b['x'].shape[1]})
 wrap(t5.T5EncoderModel,'encode','sky_text',lambda b:{'chars':len(b['texts'])})
with torch.inference_mode():
 for r in range(a.warmup):
  for frames in a.frames:
   start=time.perf_counter();out=generate(frames);seconds=time.perf_counter()-start;del out
   report['warmup_wall_seconds'].append(dict(frames=frames,seconds=seconds));save();print('SKYREELS_WARMUP',frames,seconds,flush=True)
 if a.role=='drperf':install_marks()
 runtime=ctypes.CDLL(None)
 if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
 elif a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
 elif a.role=='partial_sync':runtime.gxvm_adopt_now();runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
 torch.cuda.reset_peak_memory_stats();iteration=0
 for r in range(a.repetitions):
  for frames in a.frames:
   iteration+=1
   if a.role=='partial_sync':runtime.gxvm_timeline_mark(iteration,0)
   start=time.perf_counter();out=generate(frames);seconds=time.perf_counter()-start
   if a.role=='partial_sync':runtime.gxvm_timeline_mark(iteration,1)
   report['generation_wall_seconds'].append(dict(frames=frames,seconds=seconds));result=dict(input_frames=frames,shape=list(out.shape))
   if native:result.update(finite=bool(np.isfinite(out).all()),sha256=hashlib.sha256(out.tobytes()).hexdigest());assert result['finite']
   report['outputs'].append(result);save();print('SKYREELS_GENERATED',frames,seconds,flush=True)
 if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
 elif a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save();print('SKYREELS_PASS',flush=True)
