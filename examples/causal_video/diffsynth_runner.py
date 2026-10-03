"""Actual DiffSynth Wan pipeline: GX discovery/timing and optional hardware oracle."""
import argparse
import cProfile
import ctypes
import hashlib
import json
import os
from pathlib import Path
import pstats
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--tree', required=True)
p.add_argument('--base', required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--role', choices=['emu', 'cpu-profile', 'partial_sync', 'gpu_only_partial_sync', 'gpu-profile', 'native'], default='emu')
p.add_argument('--frames', type=int, default=17)
p.add_argument('--steps', type=int, default=4)
p.add_argument('--height', type=int, default=256)
p.add_argument('--width', type=int, default=448)
p.add_argument('--vram-limit', type=float, default=6)
p.add_argument('--warmup', type=int, default=0)
p.add_argument('--repetitions', type=int, default=1)
p.add_argument('--cprofile', action='store_true')
p.add_argument('--profile-load', action='store_true')
p.add_argument('--load-only', action='store_true')
p.add_argument('--hash-models', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
os.chdir(a.output)
sys.path.insert(0, a.tree)
import numpy as np
import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, WanVideoUnit_PromptEmbedder
from diffsynth.core.vram.layers import AutoWrappedModule

physical = a.role in ['native', 'gpu-profile']
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text()) == physical
torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Keep CPU-known mask lengths before moving onto GX's noncomputing GPU.
# Shared with hardware runs; preserves the original (batch-one) slicing.
def encode_prompt(self, pipe, prompt):
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    seq_lens = mask.gt(0).sum(dim=1).long().tolist()
    prompt_emb = pipe.text_encoder(ids.to(pipe.device), mask.to(pipe.device))
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb
WanVideoUnit_PromptEmbedder.encode_prompt = encode_prompt
base = Path(a.base)
vram = dict(offload_dtype=torch.bfloat16, offload_device='cpu', onload_dtype=torch.bfloat16, onload_device='cpu', preparing_dtype=torch.bfloat16, preparing_device='cuda', computation_dtype=torch.bfloat16, computation_device='cuda')
report = {'config': vars(a) | {'output': str(a.output)}, 'source_sha256': {str(f.relative_to(a.tree)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(a.tree, 'diffsynth').rglob('*.py'))}, 'metadata_adaptation': 'Tokenizer mask lengths computed on CPU before transfer, identical slicing; shared by native/GX. GX skips GPU arithmetic.', 'phase': 'loading', 'async_weight_uploads': os.environ.get('DIFFSYNTH_ASYNC_WEIGHT_UPLOADS') == '1', 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch': torch.__version__, 'device_total_bytes': torch.cuda.get_device_properties(0).total_memory}
def save():
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
save()
load_prof = cProfile.Profile()
if a.profile_load:
    load_prof.enable()
t = time.perf_counter()
pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device='cuda', model_configs=[ModelConfig(path=str(base / n), **vram) for n in ['diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth']], tokenizer_config=ModelConfig(path=str(base / 'google/umt5-xxl')), vram_limit=a.vram_limit, redirect_common_files=False)
torch.cuda.synchronize()
report.update(load_seconds=time.perf_counter() - t, phase='loaded')
if a.profile_load:
    load_prof.disable()
    load_prof.dump_stats(str(a.output / 'load.pstats'))
    with (a.output / 'load.txt').open('w') as f:
        stats = pstats.Stats(load_prof, stream=f)
        stats.strip_dirs().sort_stats('tottime').print_stats(60)
        stats.sort_stats('cumulative').print_stats(70)
save()
if a.hash_models:
    digest = hashlib.sha256()
    count = total_bytes = 0
    for name, tensor in sorted(pipe.state_dict().items()):
        assert tensor.device.type == 'cpu', (name, tensor.device)
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        raw = memoryview(tensor.detach().contiguous().view(torch.uint8).numpy()).cast('B')
        digest.update(raw)
        total_bytes += len(raw)
        count += 1
    report['loaded_state'] = dict(sha256=digest.hexdigest(), tensors=count, bytes=total_bytes)
    save()
print('DIFFSYNTH_LOADED', flush=True)
if a.load_only:
    raise SystemExit(0)
original = AutoWrappedModule.cast_to
calls, active = {}, False
def cast_to(self, module, dtype, device):
    if active:
        row = calls.setdefault(type(module).__name__, {'calls': 0, 'parameter_bytes': 0, 'host_seconds': 0})
        row['calls'] += 1
        row['parameter_bytes'] += sum(x.numel() * x.element_size() for x in module.parameters())
        t = time.perf_counter()
        result = original(self, module, dtype, device)
        row['host_seconds'] += time.perf_counter() - t
        return result
    return original(self, module, dtype, device)
if a.cprofile:
    AutoWrappedModule.cast_to = cast_to
kwargs = dict(prompt='A cat walks on the grass, realistic style', negative_prompt='', seed=42, height=a.height, width=a.width, num_frames=a.frames, num_inference_steps=a.steps, cfg_scale=5.0, tiled=False, progress_bar_cmd=lambda x: x)
for _ in range(a.warmup):
    pipe(**kwargs)
torch.cuda.synchronize()
print('DIFFSYNTH_WARMUP_DONE', flush=True)
runtime = ctypes.CDLL(None)
if a.role == 'cpu-profile':
    runtime.gxvm_ipc_profile_start()
elif a.role == 'partial_sync':
    runtime.gxvm_adopt_now()
elif a.role == 'gpu_only_partial_sync':
    runtime.gxvm_start()
elif a.role == 'gpu-profile':
    Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
if a.role in ['partial_sync', 'gpu_only_partial_sync']:
    runtime.gxvm_timeline_mark.argtypes = [ctypes.c_uint, ctypes.c_int]
report['iterations'] = []
prof = cProfile.Profile()
active = True
if a.cprofile:
    prof.enable()
for iteration in range(a.repetitions):
    if a.role in ['partial_sync', 'gpu_only_partial_sync']:
        runtime.gxvm_timeline_mark(iteration + 1, 0)
    t = time.perf_counter()
    video = pipe(**kwargs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t
    if a.role in ['partial_sync', 'gpu_only_partial_sync']:
        runtime.gxvm_timeline_mark(iteration + 1, 1)
    row = {'iteration': iteration + 1, 'application_clock_seconds': seconds, 'frames_returned': len(video)}
    if physical:
        pixels = np.stack([np.asarray(frame) for frame in video])
        row['output_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
        row['output_shape'] = list(pixels.shape)
    report['iterations'].append(row)
if a.cprofile:
    prof.disable()
active = False
if a.role == 'cpu-profile':
    assert runtime.gxvm_ipc_profile_stop() == 0
elif a.role == 'gpu-profile':
    Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
report.update(phase='completed', cast_to=calls, peak_gpu_bytes=torch.cuda.max_memory_allocated())
assert report['peak_gpu_bytes'] <= report['device_total_bytes'], 'Exceeds modeled GPU capacity'
save()
if a.cprofile:
    prof.dump_stats(str(a.output / 'cpu.pstats'))
    with (a.output / 'cpu.txt').open('w') as f:
        stats = pstats.Stats(prof, stream=f)
        stats.strip_dirs().sort_stats('tottime').print_stats(80)
        stats.sort_stats('cumulative').print_stats(100)
print('DIFFSYNTH_RESULT', json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}), flush=True)
