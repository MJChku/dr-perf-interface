"""Real Matrix-Game 3 pipeline; file encoding replaced by an output capture.
GPU-value control adapters retain original GPU operations and audit native results.
"""
import argparse
import ctypes
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

p=argparse.ArgumentParser()
p.add_argument('--tree',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','drperf','partial_sync','gpu_only_partial_sync'],default='emu')
p.add_argument('--iterations',type=int,default=2)
p.add_argument('--warmup',type=int,default=1)
p.add_argument('--repetitions',type=int,default=1)
p.add_argument('--compile-vae',action=argparse.BooleanOptionalAction,default=True)
p.add_argument('--int8',action=argparse.BooleanOptionalAction,default=True)
a=p.parse_args();assert a.iterations>=1 and a.repetitions>0 and a.warmup>=0
a.output.mkdir(parents=True,exist_ok=False)
a.tree=a.tree.resolve();a.checkpoint=a.checkpoint.resolve();a.output=a.output.resolve()
sys.path.insert(0,str(a.tree));os.chdir(a.output)
import numpy as np
import torch
import diffusers
import transformers
from PIL import Image
from pipeline import inference_pipeline as pipeline
from wan.configs import WAN_CONFIGS
from wan.modules import t5
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.misc import set_seed
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
torch.set_num_threads(1);torch.set_grad_enabled(False)
# Do not silently compare an eager fallback against a compiled decoder.
torch._dynamo.config.suppress_errors=False
native=a.role in ('native','gpu-profile')
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==native
prompt=(a.tree/'demo_images/001/prompt.txt').read_text().strip()
report=dict(role=a.role,iterations=a.iterations,frames=57+40*(a.iterations-1),height=704,width=1280,
            steps=3,seed=42,prompt=prompt,int8=a.int8,compile_vae=a.compile_vae,fa_version='2',offload=False,
            torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__,
            warmup=a.warmup,repetitions=a.repetitions,generation_wall_seconds=[],warmup_wall_seconds=[],
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest()
                           for d in ['pipeline','utils','wan'] for f in sorted((a.tree/d).rglob('*.py'))},
            control_audit=dict(text_reads=0,text_mismatches=0,scheduler_reads=0,scheduler_mismatches=0,solve_info_reads=0,solve_info_mismatches=0),
            adaptations=['Compiler errors are fatal in the harness; public source silently falls back. Successful compiled execution is unchanged.',
                         'Meta/strict mmap BF16 T5 loading outside inference; same tensor values as public loader.',
                         'File encoder/overlay replaced by uint8 output capture; original output CPU conversions retained; exit suppressed.',
                         'CPU-known tokenizer lengths substitute GX skipped reduction after original GPU read.',
                         'UniPC two-by-two solve uses solve_ex plus the original scalar info read; GX substitutes success for the analytically nonsingular schedule matrix, native asserts info zero.',
                         'Initial UniPC timestep index is known to be zero for this three-step schedule; comparison/nonzero/item retained; native result asserted.'])
def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
# Bound setup peak memory, without changing inference or stored parameter values.
def efficient_t5(self,text_len,dtype=torch.bfloat16,device='cpu',checkpoint_path=None,tokenizer_path=None,shard_fn=None):
    assert shard_fn is None and dtype==torch.bfloat16
    self.text_len=text_len;self.dtype=dtype;self.device=device
    self.checkpoint_path=checkpoint_path;self.tokenizer_path=tokenizer_path
    self.model=t5.umt5_xxl(encoder_only=True,return_tokenizer=False,dtype=dtype,device='meta').eval().requires_grad_(False)
    state=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=True)
    assert all(v.dtype==torch.bfloat16 for v in state.values())
    self.model.load_state_dict(state,strict=True,assign=True)
    assert not any(v.is_meta for v in self.model.state_dict().values())
    self.model.to(device)
    self.tokenizer=t5.HuggingfaceTokenizer(name=tokenizer_path,seq_len=text_len,clean='whitespace')
t5.T5EncoderModel.__init__=efficient_t5

def checked_text(self,texts,device):
    ids,mask=self.tokenizer(texts,return_mask=True,add_special_tokens=True)
    lengths=mask.gt(0).sum(dim=1).long().tolist()
    ids=ids.to(device);mask=mask.to(device)
    seq_lens=mask.gt(0).sum(dim=1).long();context=self.model(ids,mask)
    for u,v,length in zip(context,seq_lens,lengths):
        observed=v.__index__();report['control_audit']['text_reads']+=1
        if native: assert observed==length,(observed,length)
        elif observed!=length: report['control_audit']['text_mismatches']+=1
        u[length:]=0.0
    return [u for u,v in zip(context,seq_lens)]
t5.T5EncoderModel.__call__=checked_text

def checked_index(self,timestep,schedule_timesteps=None):
    assert self.step_index is None and self.begin_index is None and self.num_inference_steps==3
    if schedule_timesteps is None: schedule_timesteps=self.timesteps
    # Public generate() starts from timesteps[0], and this schedule has unique times.
    indices=(schedule_timesteps==timestep).nonzero()
    if not native: indices.resize_(1,1)
    pos=1 if len(indices)>1 else 0
    observed=indices[pos].item();report['control_audit']['scheduler_reads']+=1
    if native: assert observed==0,observed
    elif observed!=0:report['control_audit']['scheduler_mismatches']+=1
    return 0
FlowUniPCMultistepScheduler.index_for_timestep=checked_index
# The 3-step UniPC corrector matrix is [[1,1],[rk,1]], rk<0 for its
# increasing log-SNR schedule. It is nonsingular. GPU info writes are skipped by GX.
original_solve=torch.linalg.solve
def checked_solve(A,B,*,left=True,out=None):
    if A.device.type!='cuda' or tuple(A.shape)!=(2,2) or tuple(B.shape)!=(2,) or out is not None:
        return original_solve(A,B,left=left,out=out)
    result,info=torch.linalg.solve_ex(A,B,left=left,check_errors=False)
    observed=info.item();report['control_audit']['solve_info_reads']+=1
    if native:assert observed==0,observed
    elif observed!=0:report['control_audit']['solve_info_mismatches']+=1
    return result
torch.linalg.solve=checked_solve
captured={}
def capture(video,*args,**kwargs):
    assert video.dtype==np.uint8 and video.ndim==4
    captured['video']=video
pipeline.process_video=capture;pipeline.exit=lambda:None
args=SimpleNamespace(ckpt_dir=str(a.checkpoint),output_dir=str(a.output),size='704*1280',save_name='video',
    num_iterations=a.iterations,vae_type='mg_lightvae_v2',lightvae_pruning_rate=None,use_async_vae=False,
    compile_vae=a.compile_vae,use_int8=a.int8,verify_quant=False,fa_version='2')
start=time.perf_counter();print('MATRIX_GAME_LOADING',flush=True)
set_seed(42)
pipe=pipeline.MatrixGame3Pipeline(config=WAN_CONFIGS['matrix_game3'],checkpoint_dir=str(a.checkpoint),
    device_id=0,rank=0,t5_fsdp=False,dit_fsdp=False,use_sp=False,t5_cpu=False,init_on_cpu=True,
    convert_model_dtype=False,args=args,fa_version='2',use_base_model=False)
torch.cuda.synchronize();report['load_wall_seconds']=time.perf_counter()-start
report['model_config']=dict(pipe.model.config)
report['model_parameters']=sum(v.numel() for v in pipe.model.parameters())
report['model_buffers']=sum(v.numel() for v in pipe.model.buffers())
report['initial_free_bytes']=torch.cuda.mem_get_info()[0]
save();print('MATRIX_GAME_LOADED',report['load_wall_seconds'],flush=True)
image=Image.open(a.tree/'demo_images/001/image.png').convert('RGB')
# Audits stay outside marked regions except for unavoidable control adapters.
model_calls=[];decode_shapes=[]
original_model=pipe.model.forward
@functools.wraps(original_model)
def model_audit(*args,**kwargs):
    bound=inspect.signature(original_model).bind(*args,**kwargs);bound.apply_defaults()
    x=bound.arguments['x'];mem=bound.arguments['x_memory']
    model_calls.append(dict(latents=list(x.shape),memory=None if mem is None else list(mem.shape)))
    return original_model(*args,**kwargs)
pipe.model.forward=model_audit
original_decode=pipe.vae.stream_decode
@functools.wraps(original_decode)
def decode_audit(z,*args,**kwargs):
    decode_shapes.append(list(z.shape))
    result=original_decode(z,*args,**kwargs)
    captured['last_latent']=z
    captured['last_float_video']=result[0]
    return result
pipe.vae.stream_decode=decode_audit

def install_drperf():
    import perfmark
    from wan.modules import model
    regions=[]
    def wrap(owner,method,name,features):
        original=getattr(owner,method);signature=inspect.signature(original)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            bound=signature.bind(*args,**kwargs);bound.apply_defaults()
            with perfmark.region(name,**features(bound.arguments)):
                return original(*args,**kwargs)
        setattr(owner,method,measured);regions.append(name)
    wrap(pipe,'generate','mg_generate',lambda b:{'frames':report['frames']})
    wrap(t5.T5EncoderModel,'__call__','mg_text',lambda b:{'chars':sum(map(len,b['texts']))})
    wrap(pipe.model,'forward','mg_model',lambda b:{'frames':b['x'].shape[2],'memory_frames':0 if b['x_memory'] is None else b['x_memory'].shape[2]})
    wrap(model.WanAttentionBlock,'forward','mg_block',lambda b:{'tokens':b['x'].shape[1]})
    wrap(model.Int8Linear,'forward','mg_int8',lambda b:{'elements':b['x'].numel(),'out_features':b['self'].out_features})
    wrap(pipe.vae,'encode','mg_encode',lambda b:{'pixels':b['videos'][0].numel()})
    wrap(pipe.vae,'stream_decode','mg_decode',lambda b:{'frames':b['z'].shape[2]})
    wrap(pipeline,'select_memory_idx_fov','mg_memory_select',lambda b:{'candidate_reference_pairs':max(0,b['current_start_frame_idx']-1)*len(b['selected_index_base'])})
    wrap(pipeline,'get_data','mg_input',lambda b:{'frames':b['num_frames']})
    return regions

def generate():
    captured.clear();set_seed(42)
    pipe.generate(prompt,image,max_area=704*1280,shift=5.0,num_inference_steps=3,guide_scale=5.0,seed=42,use_base_model=False,args=args)
    torch.cuda.synchronize();assert 'video' in captured
    return captured.pop('video')
with torch.inference_mode():
    for i in range(a.warmup):
        start=time.perf_counter();video=generate();report['warmup_wall_seconds'].append(time.perf_counter()-start)
        del video;save();print('MATRIX_GAME_WARMUP',i,report['warmup_wall_seconds'][-1],flush=True)
    # Compilation is finished; quiesce its helper pools before late instrumentation.
    # Cached compiled callables remain intact. This is setup, outside measurement.
    from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
    report['threads_before_compile_shutdown']=len(list(Path('/proc/self/task').iterdir()))
    shutdown_compile_workers()
    if AsyncCompile.pool.cache_info().currsize:
        AsyncCompile.pool().shutdown(wait=True)
        AsyncCompile.pool.cache_clear()
    torch.cuda.synchronize()
    report['threads_after_compile_shutdown']=len(list(Path('/proc/self/task').iterdir()))
    save();print('MATRIX_GAME_COMPILER_QUIESCED',report['threads_before_compile_shutdown'],report['threads_after_compile_shutdown'],flush=True)
    if a.role=='drperf':report['drperf_regions']=install_drperf()
    runtime=ctypes.CDLL(None)
    if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
    elif a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
    elif a.role in ('partial_sync','gpu_only_partial_sync'):
        (runtime.gxvm_adopt_now if a.role=='partial_sync' else runtime.gxvm_start)()
        runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.repetitions):
        if a.role in ('partial_sync','gpu_only_partial_sync'):runtime.gxvm_timeline_mark(i+1,0)
        start=time.perf_counter();video=generate();report['generation_wall_seconds'].append(time.perf_counter()-start)
        if a.role in ('partial_sync','gpu_only_partial_sync'):runtime.gxvm_timeline_mark(i+1,1)
        save();print('MATRIX_GAME_GENERATED',i,report['generation_wall_seconds'][-1],flush=True)
    if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
    elif a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
report.update(output_shape=list(video.shape),model_calls=model_calls,decode_shapes=decode_shapes,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
assert len(model_calls)==(a.warmup+a.repetitions)*a.iterations*3
assert list(video.shape)==[report['frames'],704,1280,3]
if native:
    report['output_uint8_sha256']=hashlib.sha256(video.tobytes()).hexdigest()
    for key in ('last_latent','last_float_video'):
        tensor=captured[key].detach().cpu().float().contiguous()
        report[key+'_finite']=bool(torch.isfinite(tensor).all());assert report[key+'_finite']
        report[key+'_sha256']=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        del tensor
save();print('MATRIX_GAME_PASS',a.output,flush=True)
