import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

def install_drperf(pipe):
    import functools
    import inspect
    import perfmark
    from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
    from wan.modules import causal_model, attention, t5
    regions = []
    def wrap(owner, method, name, features):
        original = getattr(owner,method)
        signature = inspect.signature(original)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            bound = signature.bind(*args,**kwargs)
            bound.apply_defaults()
            with perfmark.region(name,**features(bound.arguments)):
                return original(*args,**kwargs)
        setattr(owner,method,measured)
        regions.append(name)
    wrap(pipe,'inference','cf_generate',lambda a:{'frames':a['noise'].shape[1]})
    wrap(WanTextEncoder,'forward','cf_text',lambda a:{'chars':sum(map(len,a['text_prompts']))})
    wrap(WanDiffusionWrapper,'forward','cf_model',lambda a:{'start':int(a['current_start'] or 0),'frames':a['noisy_image_or_video'].shape[1]})
    wrap(WanVAEWrapper,'decode_to_pixel','cf_decode',lambda a:{'frames':a['latent'].shape[1]})
    wrap(causal_model.CausalWanAttentionBlock,'forward','cf_block',lambda a:{'tokens':a['x'].shape[1],'start':int(a['current_start'])})
    wrap(causal_model.CausalWanSelfAttention,'forward','cf_self_attn',lambda a:{'tokens':a['x'].shape[1],'start':int(a['current_start'])})
    wrap(causal_model,'causal_rope_apply','cf_rope',lambda a:{'tokens':a['x'].shape[1],'start_frame':int(a['start_frame'])})
    wrap(attention,'flash_attention','cf_flash_dispatch',lambda a:{'q_tokens':a['q'].shape[1],'k_tokens':a['k'].shape[1]})
    wrap(t5.T5Attention,'forward','cf_t5_attn',lambda a:{'tokens':a['x'].shape[1]})
    return regions

p = argparse.ArgumentParser()
p.add_argument('--tree', type=Path, required=True)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--role', choices=['emu','native','gpu-profile','cpu-profile','drperf','partial_sync','gpu_only_partial_sync'], default='emu')
p.add_argument('--frames', type=int, default=21)
p.add_argument('--warmup', type=int, default=1)
p.add_argument('--repetitions', type=int, default=1)
p.add_argument('--audit-controls', action='store_true')
p.add_argument('--memory-efficient-load', action='store_true')
p.add_argument('--low-memory', action=argparse.BooleanOptionalAction, default=None)
a = p.parse_args()
assert 1 <= a.frames <= 21 and a.warmup >= 0 and a.repetitions > 0
a.output.mkdir(parents=True,exist_ok=False)
a.tree = a.tree.resolve(); a.base = a.base.resolve(); a.checkpoint = a.checkpoint.resolve()
sys.path.insert(0,str(a.tree))
os.chdir(a.output)
Path('wan_models').mkdir()
Path('wan_models/Wan2.1-T2V-1.3B').symlink_to(a.base, target_is_directory=True)
import torch
import diffusers
import transformers
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
torch.set_num_threads(1)
torch.set_grad_enabled(False)
native = a.role in ['native','gpu-profile']
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text()) == native
config = OmegaConf.merge(OmegaConf.load(a.tree/'configs/default_config.yaml'),
                         OmegaConf.load(a.tree/'configs/causal_forcing_dmd_framewise_1step.yaml'))
initial_free_gib = get_cuda_free_memory_gb(gpu)
low_memory = initial_free_gib < 40 if a.low_memory is None else a.low_memory
report = dict(role=a.role, frames=a.frames, pixel_frames=4*a.frames-3, height=480, width=832,
              torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
              warmup=a.warmup, repetitions=a.repetitions, low_memory=low_memory,
              initial_free_gib=initial_free_gib, low_memory_override=a.low_memory, seed=42,
              prompt='A cat walks on the grass, realistic style',
              config=OmegaConf.to_container(config,resolve=True),
              source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for d in ['pipeline','utils','wan','demo_utils'] for f in sorted((a.tree/d).rglob('*.py'))},
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              warmup_wall_seconds=[],generation_wall_seconds=[],gx_control_adaptations=[])
def save():
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
start = time.perf_counter()
print('CAUSAL_FORCING_LOADING',flush=True)
text_encoder = None
if a.memory_efficient_load:
    from utils.wan_wrapper import WanTextEncoder
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer
    text_encoder = WanTextEncoder.__new__(WanTextEncoder)
    torch.nn.Module.__init__(text_encoder)
    text_encoder.text_encoder = umt5_xxl(encoder_only=True,return_tokenizer=False,
                                       dtype=torch.bfloat16,device='meta').eval()
    text_state = torch.load(a.base/'models_t5_umt5-xxl-enc-bf16.pth',map_location='cpu',weights_only=True,mmap=True)
    assert all(v.dtype == torch.bfloat16 for v in text_state.values())
    text_encoder.text_encoder.load_state_dict(text_state,strict=True,assign=True)
    text_encoder.text_encoder.requires_grad_(False)
    assert not any(v.is_meta for v in text_encoder.text_encoder.state_dict().values())
    report['loader_setup'] = {'t5_tensors':len(text_state),'dtype':'bfloat16',
        'method':'Meta construction, strict assignment from mmap BF16 checkpoint; identical to public BF16-to-FP32-to-BF16 loading. Outside measured inference.'}
    del text_state
    text_encoder.tokenizer = HuggingfaceTokenizer(name=str(a.base/'google/umt5-xxl'),seq_len=512,clean='whitespace')
pipe = CausalInferencePipeline(config,device=torch.device('cuda'),text_encoder=text_encoder)
state = torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)['generator_ema']
state = {k.replace('model._fsdp_wrapped_module.','model.',1) if k.startswith('model._fsdp_wrapped_module.') else k:v for k,v in state.items()}
pipe.generator.load_state_dict(state,strict=True)
del state
pipe = pipe.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipe.text_encoder,device=gpu)
else:
    pipe.text_encoder.to(device=gpu)
pipe.generator.to(device=gpu)
pipe.vae.to(device=gpu)
pipe.eval()
torch.cuda.synchronize()
ids,mask = pipe.text_encoder.tokenizer([report['prompt']],return_mask=True,add_special_tokens=True)
report['prompt_token_ids'] = ids[0,:int(mask[0].sum())].tolist()
del ids,mask
report['load_wall_seconds'] = time.perf_counter()-start
save()
print('CAUSAL_FORCING_LOADED',report['load_wall_seconds'],flush=True)
if a.audit_controls:
    from collections import Counter
    import functools
    from wan.modules import causal_model
    from utils.wan_wrapper import WanTextEncoder
    scalar_counts = Counter()
    attention_shapes = Counter()
    attention_state = {}
    class CacheScalar:
        def __init__(self,tensor):
            self.tensor = tensor
            self.value = 0
        def item(self):
            observed = self.tensor.item()
            scalar_counts['reads'] += 1
            if native or self.tensor.device.type == 'cpu':
                assert observed == self.value, (observed,self.value)
            elif observed != self.value:
                scalar_counts['emulated_read_mismatches'] += 1
            return self.value
        def fill_(self,value):
            self.tensor.fill_(value)
            self.value = int(value)
            scalar_counts['writes'] += 1
            return self
    original_self_attention = causal_model.CausalWanSelfAttention.forward
    @functools.wraps(original_self_attention)
    def checked_self_attention(self,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache=None,current_start=0,cache_start=None):
        if kv_cache is not None:
            for name in ['global_end_index','local_end_index']:
                if not isinstance(kv_cache[name],CacheScalar):
                    kv_cache[name] = CacheScalar(kv_cache[name])
        attention_state['end'] = current_start+x.shape[1]
        attention_state['local_attn_size'] = self.local_attn_size
        attention_state['max_attention_size'] = self.max_attention_size
        return original_self_attention(self,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache,current_start,cache_start)
    causal_model.CausalWanSelfAttention.forward = checked_self_attention
    original_attention = causal_model.attention
    def checked_attention(q,k,v,*args,**kwargs):
        if attention_state['local_attn_size'] == -1:
            assert k.shape[1] == min(attention_state['end'],attention_state['max_attention_size'])
        assert k.shape[1] == v.shape[1]
        attention_shapes[(q.shape[1],k.shape[1])] += 1
        return original_attention(q,k,v,*args,**kwargs)
    causal_model.attention = checked_attention
    original_text = WanTextEncoder.forward
    @functools.wraps(original_text)
    def checked_text(self,text_prompts):
        ids,mask = self.tokenizer(text_prompts,return_mask=True,add_special_tokens=True)
        lengths = mask.gt(0).sum(dim=1).long().tolist()
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids,mask)
        for u,v,length in zip(context,seq_lens,lengths):
            observed = v.__index__()
            if native: assert observed == length, (observed,length)
            u[length:] = 0.0
        report['text_lengths'] = lengths
        return {'prompt_embeds':context}
    WanTextEncoder.forward = checked_text
    report['control_audit'] = 'CPU-known cache positions and tokenizer lengths; all original GPU reads/writes and text GPU reductions retained. Native runs assert equality; GX substitutes the CPU-known result. Python audit overhead included.'
    if not native:
        report['gx_control_adaptations'] = ['cache_scalar_read_values','text_padding_length']
def generate():
    torch.manual_seed(report['seed']);torch.cuda.manual_seed_all(report['seed'])
    noise = torch.randn([1,a.frames,16,60,104],device='cuda',dtype=torch.bfloat16)
    video,latents = pipe.inference(noise=noise,text_prompts=[report['prompt']],return_latents=True,
                                 initial_latent=None,report_timing=False)
    video = video.permute(0,1,3,4,2).cpu()
    latents = latents[0].cpu()
    pipe.vae.model.clear_cache()
    torch.cuda.synchronize()
    return video,latents
with torch.inference_mode():
    for i in range(a.warmup):
        start = time.perf_counter(); video,latents = generate()
        report['warmup_wall_seconds'].append(time.perf_counter()-start)
        del video,latents
        save();print('CAUSAL_FORCING_WARMUP',i,report['warmup_wall_seconds'][-1],flush=True)
    if a.role == 'drperf':
        report['drperf_regions'] = install_drperf(pipe)
    runtime = ctypes.CDLL(None)
    if a.role == 'gpu-profile':
        Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
    elif a.role == 'cpu-profile':
        runtime.gxvm_ipc_profile_start()
    elif a.role in ['partial_sync','gpu_only_partial_sync']:
        (runtime.gxvm_adopt_now if a.role == 'partial_sync' else runtime.gxvm_start)()
        runtime.gxvm_timeline_mark.argtypes = [ctypes.c_uint,ctypes.c_int]
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.repetitions):
        if a.role in ['partial_sync','gpu_only_partial_sync']: runtime.gxvm_timeline_mark(i+1,0)
        start = time.perf_counter();video,latents = generate()
        report['generation_wall_seconds'].append(time.perf_counter()-start)
        if a.role in ['partial_sync','gpu_only_partial_sync']: runtime.gxvm_timeline_mark(i+1,1)
        save();print('CAUSAL_FORCING_GENERATED',i,report['generation_wall_seconds'][-1],flush=True)
    if a.role == 'gpu-profile':
        Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
    elif a.role == 'cpu-profile':
        assert runtime.gxvm_ipc_profile_stop() == 0
report.update(output_shape=list(video.shape),latent_shape=list(latents.shape),
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
if a.audit_controls:
    report['control_scalar_counts'] = dict(scalar_counts)
    report['attention_shapes'] = [{'query_tokens':q,'key_tokens':k,'calls':n} for (q,k),n in sorted(attention_shapes.items())]
if native:
    report['finite'] = bool(torch.isfinite(video).all() and torch.isfinite(latents).all())
    assert report['finite']
    report['output_float32_sha256'] = hashlib.sha256(video.float().contiguous().numpy().tobytes()).hexdigest()
    report['latent_float32_sha256'] = hashlib.sha256(latents.float().contiguous().numpy().tobytes()).hexdigest()
save()
print('CAUSAL_FORCING_PASS',json.dumps(report),flush=True)
