"""Bounded bring-up of Wan2GP's public pipeline under GX."""
import argparse, ctypes, hashlib, importlib, json, os, resource, sys, time, types
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('source','model','output'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--imports-only',action='store_true')
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','partial_sync'],default='emu')
p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=2)
p.add_argument('--frames',type=int,default=17);p.add_argument('--steps',type=int,default=8)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
os.chdir(a.source);sys.path.insert(0,str(a.source))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
# Select the public Wan pipeline without eagerly registering unrelated Ovi and
# diffusion-forcing pipelines from models.wan.__init__.
for name,path in [('models',a.source/'models'),('models.wan',a.source/'models/wan'),('models.wan.scail',a.source/'models/wan/scail')]:
    pkg=types.ModuleType(name);pkg.__path__=[str(path)];sys.modules[name]=pkg
from models.wan.any2video import WanAny2V
from models.wan.configs import WAN_CONFIGS
from mmgp import offload
from shared import attention
print('WAN2GP_IMPORTS_PASS',torch.__version__,flush=True)
physical=a.role in ('native','gpu-profile')
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==physical
report={'mmgp_offload_sha256':hashlib.sha256(Path(offload.__file__).read_bytes()).hexdigest(),'skip_empty_unload':os.environ.get('MMGP_SKIP_EMPTY_UNLOAD')=='1','phase':'imports_completed','torch':torch.__version__,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'control_checks':[],'source_sha256':{str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in a.source.rglob('*.py')},'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
if a.imports_only:raise SystemExit(0)
from models.wan.modules.t5 import T5EncoderModel
from shared.utils.phase_progress import text_encoding_progress
def encode(self,texts,device):
    ids,mask=self.tokenizer(texts,return_mask=True,add_special_tokens=True)
    # MMGP sets the default device to CUDA. Read tokenizer metadata as plain
    # Python lists, independently of tensors and emulated GPU arithmetic.
    cleaned=[self.tokenizer._clean(t) if self.tokenizer.clean else t for t in texts]
    metadata=self.tokenizer.tokenizer(cleaned,return_tensors=None,padding='max_length',truncation=True,max_length=self.text_len,add_special_tokens=True)
    expected=[sum(v>0 for v in row) for row in metadata.attention_mask]
    assert all(0<v<=self.text_len for v in expected),expected
    ids=ids.to(device);mask=mask.to(device);seq_lens=mask.gt(0).sum(dim=1).long()
    with text_encoding_progress(self.model.blocks,prompt_count=len(texts)):
        context=self.model(ids,mask)
    observed=[int(v) for v in seq_lens]
    if physical:assert observed==expected,(observed,expected)
    report['control_checks'].append({'kind':'text_lengths','expected':expected,'observed':observed if physical else None})
    return [u[:v] for u,v in zip(context,expected)]
T5EncoderModel.__call__=encode
# Euler constructs its CPU schedule from CUDA scalar reads. Preserve those
# operations and reads, including index lookup and step arithmetic. Supply
# only the CPU-known index and scalar control value after each lookup; native
# execution verifies every supplied value.
from shared.utils.euler_scheduler import EulerScheduler, _timestep_transform
import numpy as np
old_set=EulerScheduler.set_timesteps
old_index=EulerScheduler._timestep_to_index
def set_timesteps(self,num_inference_steps,device=None,shift=5.0):
    result=old_set(self,num_inference_steps,device=device,shift=shift)
    cpu=[torch.tensor([t],device='cpu') for t in np.linspace(self.num_train_timesteps,1,num_inference_steps,dtype=np.float32)]
    expected=torch.tensor([_timestep_transform(t,shift,self.num_train_timesteps) for t in cpu],device='cpu') if self.use_timestep_transform else torch.tensor(cpu+[torch.tensor([0.],device='cpu')],device='cpu')
    self._gx_cpu_timesteps=expected;self._gx_step_cursor=0
    report['control_checks'].append({'kind':'euler_schedule','verification':'Every returned index and scalar checked on native in step lookup','steps':num_inference_steps})
    return result
def timestep_to_index(self,timestep):
    observed_index,observed_t=old_index(self,timestep)
    index=self._gx_step_cursor;self._gx_step_cursor+=1
    assert index<len(self._gx_cpu_timesteps)
    value=float(self._gx_cpu_timesteps[index])
    if physical:assert observed_index==index and abs(observed_t-value)<0.001,(observed_index,index,observed_t,value)
    report['control_checks'].append({'kind':'euler_index','index':index,'native_checked':physical})
    return index,value
EulerScheduler.set_timesteps=set_timesteps
EulerScheduler._timestep_to_index=timestep_to_index
from shared.utils import files_locator as fl
fl.set_checkpoints_paths([str(a.model),str(a.model/'google')])
definition={'config_file':str(a.source/'models/wan/configs/t2v_1.3B.json'),'VAE_URLs':str(a.model/'Wan2.1_VAE.pth'),'text_encoder_folder':str(a.model/'google/umt5-xxl')}
os.chdir(a.output)
start=time.perf_counter()
pipeline=WanAny2V(config=WAN_CONFIGS['t2v-1.3B'],checkpoint_dir=str(a.model),model_filename=[str(a.model/'diffusion_pytorch_model.safetensors')],model_type='t2v_1.3B',base_model_type='t2v_1.3B',model_def=definition,text_encoder_filename=str(a.model/'models_t5_umt5-xxl-enc-bf16.pth'),dtype=torch.bfloat16,VAE_dtype=torch.bfloat16,quantizeTransformer=False)
pipeline._interrupt=False
pipe={'transformer':pipeline.model,'text_encoder':pipeline.text_encoder.model,'vae':pipeline.vae.model}
offloadobj=offload.profile(pipe,profile_no=2,quantizeTransformer=False,extraModelsToQuantize=[],verboseLevel=1)
offload.shared_state['_attention']='flash'
report.update(phase='loaded',load_seconds=time.perf_counter()-start,parameter_counts={k:sum(t.numel() for t in m.parameters()) for k,m in pipe.items()},scope='WanAny2V.generate including warmed prompt-cache policy, sampling, VAE decode and uint8 CPU output; no UI or file encoding. MMGP profile 2, quantization disabled, CPU offload retained.')
save();print('WAN2GP_LOADED',report['parameter_counts'],flush=True)
def generate():
    return pipeline.generate(input_prompt='A cat walks on the grass, realistic style',n_prompt='',width=448,height=256,frame_num=a.frames,sampling_steps=a.steps,sample_solver='euler',guide_scale=5.0,shift=3.0,seed=42,cfg_star_switch=False,cfg_zero_step=0,model_type='t2v_1.3B',offloadobj=offloadobj,loras_slists={k:[] for k in ('phase1','phase2','phase3','shared')},callback=lambda *args,**kwargs:None,set_header_text=lambda *args,**kwargs:None)['x']
for _ in range(a.warmup):generate()
torch.cuda.synchronize();print('WAN2GP_WARMUP_DONE',flush=True)
runtime=ctypes.CDLL(None)
if a.role=='partial_sync':runtime.gxvm_adopt_now();runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
if a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
report['iterations']=[]
for i in range(a.repetitions):
    if a.role=='partial_sync':runtime.gxvm_timeline_mark(i+1,0)
    start=time.perf_counter();video=generate();torch.cuda.synchronize();seconds=time.perf_counter()-start
    if a.role=='partial_sync':runtime.gxvm_timeline_mark(i+1,1)
    row={'iteration':i+1,'application_clock_seconds':seconds,'shape':list(video.shape),'dtype':str(video.dtype)}
    if physical:
        value=video.detach().cpu().contiguous().numpy()
        row.update(output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),minimum=int(value.min()),maximum=int(value.max()),note='Quantized uint8 output; finiteness before quantization is not established by this hash.')
    report['iterations'].append(row)
if a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
report.update(phase='completed',peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save();print('WAN2GP_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
