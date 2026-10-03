"""Headless execution of the public WanVideoWrapper nodes on pretrained Wan1.3B."""
import argparse,ctypes,hashlib,importlib,json,os,sys,time,types
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['wrapper','comfy','model','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','partial_sync'],default='emu')
p.add_argument('--frames',type=int,default=17);p.add_argument('--height',type=int,default=256);p.add_argument('--width',type=int,default=448);p.add_argument('--steps',type=int,default=8);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=2);p.add_argument('--imports-only',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
sys.path.insert(0,str(a.comfy))
# Load the used public node modules without registering unrelated UI extensions.
pkg=types.ModuleType('wanwrapper');pkg.__path__=[str(a.wrapper)];sys.modules['wanwrapper']=pkg
import torch
physical=a.role in ['native','gpu-profile'];assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==physical
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
# Match the context used by ComfyUI execution.py for public node execution.
node_inference_context=torch.inference_mode();node_inference_context.__enter__()
# The sampler resets Torch's peaks after each denoising phase. Preserve the
# maximum across those resets, so the report covers T5/DiT as well as the VAE.
all_peaks={'allocated':0,'reserved':0}
original_reset_peak=torch.cuda.reset_peak_memory_stats
def preserve_peaks(device=None):
 all_peaks['allocated']=max(all_peaks['allocated'],torch.cuda.max_memory_allocated(device))
 all_peaks['reserved']=max(all_peaks['reserved'],torch.cuda.max_memory_reserved(device))
def reset_peak(device=None):
 preserve_peaks(device);return original_reset_peak(device)
torch.cuda.reset_peak_memory_stats=reset_peak
import comfy.cli_args
from comfy.cli_args import LatentPreviewMethod
comfy.cli_args.args.preview_method=LatentPreviewMethod.NoPreviews
# This headless workload excludes the UI server and previews. The upstream
# preview module nevertheless reads PromptServer.instance at import time.
# Fail if any inference path actually tries to use the absent UI, rather than
# silently replacing inference work or launching a web server for the harness.
class NoUIServer:
 def __getattr__(self,name):raise RuntimeError('Headless run unexpectedly used UI server: '+name)
ui_module=types.ModuleType('server');ui_module.PromptServer=types.SimpleNamespace(instance=NoUIServer());sys.modules['server']=ui_module
import folder_paths
for key in ['diffusion_models','text_encoders','vae']:folder_paths.add_model_folder_path(key,str(a.model))
loading=importlib.import_module('wanwrapper.nodes_model_loading');nodes=importlib.import_module('wanwrapper.nodes');sampling=importlib.import_module('wanwrapper.nodes_sampler')
from wanwrapper.wanvideo.modules.t5 import T5EncoderModel
from wanwrapper.wanvideo.schedulers import get_scheduler
report={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'phase':'imports','torch':torch.__version__,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(a.wrapper)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.wrapper.rglob('*.py'))},'control_checks':[],'node_inference_mode':True,'scope':'Public model-loading, text-encoding, sampling and decoding nodes; excludes UI server, graph queue, previews and video encoding. Any UI-server use fails explicitly.'}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save();print('WANWRAPPER_IMPORTS_PASS',flush=True)
if a.imports_only:raise SystemExit(0)
# Preserve GPU mask/count operations and scalar reads, while giving GX the
# CPU-known lengths needed for actual Python slices. Hardware checks each value.
def encode(self,texts,device):
 ids,mask=self.tokenizer(texts,return_mask=True,add_special_tokens=True)
 expected=mask.gt(0).sum(dim=1).long().tolist()
 ids=ids.to(device);mask=mask.to(device);seq_lens=mask.gt(0).sum(dim=1).long();context=self.model(ids,mask)
 print('prompt token count:',seq_lens)
 observed=[int(v) for v in seq_lens]
 if physical:assert observed==expected,(observed,expected)
 report['control_checks'].append({'kind':'text_lengths','expected':expected,'observed':observed if physical else None})
 return [u[:v] for u,v in zip(context,expected)]
T5EncoderModel.__call__=encode
start=time.perf_counter()
model=loading.WanVideoModelLoader().loadmodel(model='diffusion_pytorch_model.safetensors',base_precision='bf16',load_device='offload_device',quantization='disabled',attention_mode='flash_attn_2')[0]
vae=loading.WanVideoVAELoader().loadmodel(model_name='Wan2.1_VAE.pth',precision='bf16')[0]
t5=loading.LoadWanVideoT5TextEncoder().loadmodel(model_name='models_t5_umt5-xxl-enc-bf16.pth',precision='bf16',load_device='offload_device',quantization='disabled')[0]
report.update(phase='loaded',load_seconds=time.perf_counter()-start,
 initialized_parameter_counts={'dit':sum(x.numel() for x in model.model.diffusion_model.parameters()),'vae':sum(x.numel() for x in vae.parameters()),'t5':sum(x.numel() for x in t5['model'].model.parameters())},
 comfy_source_sha256={str(f.relative_to(a.comfy)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.comfy.rglob('*.py'))})
save();print('WANWRAPPER_INITIALIZED',json.dumps(report['initialized_parameter_counts']),flush=True)
def generate():
 text=nodes.WanVideoTextEncode().process(positive_prompt='A cat walks on the grass, realistic style',negative_prompt='',t5=t5,force_offload=True,use_disk_cache=False,device='gpu')[0]
 empty=nodes.WanVideoEmptyEmbeds().process(num_frames=a.frames,width=a.width,height=a.height)[0]
 scheduler,timesteps,_,_=get_scheduler('euler',a.steps,0,-1,3.0,torch.device('cuda:0'),1536)
 # Supported Euler start-index API, shared by GX and native. Full generation
 # starts at index0; this avoids a device-data-dependent schedule lookup in GX.
 scheduler.set_begin_index(0)
 scheduler_input={'sample_scheduler':scheduler,'timesteps':timesteps,'start_step':0}
 samples=sampling.WanVideoSampler().process(model=model,image_embeds=empty,shift=3.0,steps=a.steps,cfg=5.0,seed=42,scheduler=scheduler_input,riflex_freq_index=0,text_embeds=text,force_offload=True,batched_cfg=False,rope_function='comfy')[0]
 return nodes.WanVideoDecode().decode(vae=vae,samples=samples,enable_vae_tiling=False,tile_x=272,tile_y=272,tile_stride_x=144,tile_stride_y=128)[0]
for _ in range(a.warmup):generate()
report['parameter_counts']={'dit':sum(x.numel() for x in model.model.diffusion_model.parameters()),'vae':sum(x.numel() for x in vae.parameters()),'t5':sum(x.numel() for x in t5['model'].model.parameters())}
# The loader initially constructs meta modules; the first sampling call loads
# their checkpoint tensors, including replacing placeholder parameter shapes.
assert report['parameter_counts']['dit']==1418996800,report['parameter_counts']
save()
torch.cuda.synchronize();print('WANWRAPPER_WARMUP_DONE',flush=True);runtime=ctypes.CDLL(None)
if a.role=='partial_sync':runtime.gxvm_adopt_now();runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
if a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
report['iterations']=[]
for i in range(a.repetitions):
 if a.role=='partial_sync':runtime.gxvm_timeline_mark(i+1,0)
 start=time.perf_counter();video=generate();torch.cuda.synchronize();seconds=time.perf_counter()-start
 if a.role=='partial_sync':runtime.gxvm_timeline_mark(i+1,1)
 row={'iteration':i+1,'application_clock_seconds':seconds,'shape':list(video.shape)}
 if physical:
  value=video.detach().cpu().float().contiguous().numpy();import numpy as np
  row.update(finite=bool(np.isfinite(value).all()),output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),minimum=float(value.min()),maximum=float(value.max()));assert row['finite']
 report['iterations'].append(row)
if a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
preserve_peaks()
cache_entries=[v for mod in model.model.diffusion_model.modules() for v in getattr(mod,'_wan_cpu_cast_cache',{}).values()]
report['cpu_cast_cache_bytes']=sum(v[3].numel()*v[3].element_size() for v in cache_entries)
import resource
report['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
report.update(phase='completed',peak_gpu_bytes=all_peaks['allocated'],peak_reserved_gpu_bytes=all_peaks['reserved'],peak_scope='Maximum across framework-internal peak resets; includes loading, warmup and measured requests.');save();print('WANWRAPPER_RESULT',json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','comfy_source_sha256']}),flush=True)
# Dispose the patcher while ComfyUI's module globals are still available.
del model
import gc
gc.collect()
