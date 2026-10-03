"""Exercise the public FlashDreams Self-Forcing API without UI/video encoding."""
import argparse
import copy
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
    import math
    import perfmark
    from transformers import core_model_loading, T5Tokenizer
    from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoder
    names = []
    def wrap(owner, method, name, features):
        original = getattr(owner, method)
        signature = inspect.signature(original)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            with perfmark.region(name, **features(bound.arguments)):
                return original(*args, **kwargs)
        setattr(owner, method, measured)
        names.append(name)
    wrap(pipe, 'initialize_cache', 'fd_initialize', lambda a: {'prompts':len(a['text'])})
    wrap(pipe, 'generate', 'fd_chunk', lambda a: {'chunk':int(a['autoregressive_index']),
         'filling_chunks':min(int(a['autoregressive_index']),7),
         'steady':int(a['autoregressive_index']>=7)})
    wrap(pipe, 'finalize', 'fd_finalize', lambda a: {'chunk':int(a['autoregressive_index'])})
    wrap(UMT5TextEncoder, '__init__', 'fd_text_encoder_load', lambda a: {})
    wrap(T5Tokenizer, 'from_pretrained', 'fd_tokenizer_load', lambda a: {
        'tokenizer_json_bytes': (Path(a['pretrained_model_name_or_path'])/'tokenizer/tokenizer.json').stat().st_size})
    wrap(T5Tokenizer, '__init__', 'fd_tokenizer_construct', lambda a: {
        'vocab_entries': len(a['vocab']) if isinstance(a['vocab'], list) else 0})
    def materialize_features(a):
        tensor = a['tensor']
        shape = tensor.get_shape() if hasattr(tensor, 'get_shape') else tensor.shape
        return {'elements':int(math.prod(shape)), 'dtype':str(a['dtype'])}
    # These calls run in Transformers loader workers. Explicit per-call marks
    # retain their cost even when unmarked thread following is disabled.
    wrap(core_model_loading, '_materialize_copy', 'fd_weight_materialize', materialize_features)
    return names


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tree', type=Path, required=True)
p.add_argument('--base', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--text-model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--role', choices=['emu', 'native', 'gpu-profile', 'cpu-profile', 'partial_sync', 'gpu_only_partial_sync', 'drperf'], default='emu')
p.add_argument('--blocks', type=int, default=9)
p.add_argument('--warmup', type=int, default=2)
p.add_argument('--repetitions', type=int, default=1)
p.add_argument('--no-instantiate', action='store_true')
p.add_argument('--load-in-requested-dtype', action='store_true')
p.add_argument('--retain-cpu-tokenizer', action='store_true')
p.add_argument('--calibration-without-graphs', action='store_true')
a = p.parse_args()
assert a.blocks > 0 and a.warmup >= 0 and a.repetitions > 0
assert not a.calibration_without_graphs or a.role == 'gpu-profile'
sys.path[:0] = [str(a.tree/'flashdreams'), str(a.tree/'integrations_v2')]
import torch
from self_forcing.config import PIPELINE_WAN21_T2V_1PT3B
config = copy.deepcopy(PIPELINE_WAN21_T2V_1PT3B)
config.diffusion_model.transformer.checkpoint_path = str(a.checkpoint)
config.decoder.checkpoint_path = str(a.base/'Wan2.1_VAE.pth')
config.text_encoder.model_id_or_local_path = str(a.text_model)
if a.load_in_requested_dtype:
    assert hasattr(config.text_encoder, "load_in_requested_dtype")
    config.text_encoder.load_in_requested_dtype = True
if a.retain_cpu_tokenizer:
    config.retain_cpu_tokenizer = True
if a.calibration_without_graphs:
    config.diffusion_model.transformer.use_cuda_graph = False
    config.decoder.use_cuda_graph = False
report = dict(role=a.role, torch=torch.__version__, blocks=a.blocks,
              warmup=a.warmup, repetitions=a.repetitions, seed=42,
              prompt='A cat walks on the grass, realistic style',
              height=480, width=832, resolved_config=str(config),
              calibration_without_graphs=a.calibration_without_graphs,
              load_in_requested_dtype=a.load_in_requested_dtype,
              retain_cpu_tokenizer=a.retain_cpu_tokenizer,
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for d in ['flashdreams/flashdreams', 'integrations_v2/self_forcing']
                             for f in sorted((a.tree/d).rglob('*.py'))},
              warmup_wall_seconds=[], generation_wall_seconds=[])
a.output.mkdir(parents=True, exist_ok=False)
def save():
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
save()
if a.no_instantiate:
    print('FLASHDREAMS_CONFIG_OK', str(config), flush=True)
    sys.exit(0)
native = a.role in ['native', 'gpu-profile']
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text()) == native
torch.set_num_threads(1)
report.update(pid=os.getpid(), device=torch.cuda.get_device_name())
start = time.perf_counter()
pipe = config.setup().to('cuda').eval()
torch.cuda.synchronize()
report['load_wall_seconds'] = time.perf_counter()-start
save()
print('FLASHDREAMS_LOADED', report['load_wall_seconds'], flush=True)
def generate():
    cache = pipe.initialize_cache(text=[report['prompt']], height=60, width=104)
    chunks = []
    for i in range(a.blocks):
        video = pipe.generate(autoregressive_index=i, cache=cache)
        pipe.finalize(autoregressive_index=i, cache=cache)
        chunks.append(video.cpu())
        print('FLASHDREAMS_CHUNK', i, tuple(video.shape), flush=True)
    torch.cuda.synchronize()
    return chunks
with torch.inference_mode():
    for _ in range(a.warmup):
        start = time.perf_counter()
        chunks = generate()
        report['warmup_wall_seconds'].append(time.perf_counter()-start)
        del chunks
        save()
    if a.role == 'drperf':
        report['drperf_regions'] = install_drperf(pipe)
    runtime = ctypes.CDLL(None)
    if a.role == 'gpu-profile':
        Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
    elif a.role == 'cpu-profile':
        runtime.gxvm_ipc_profile_start()
    elif a.role in ['partial_sync', 'gpu_only_partial_sync']:
        (runtime.gxvm_adopt_now if a.role == 'partial_sync' else runtime.gxvm_start)()
        runtime.gxvm_timeline_mark.argtypes = [ctypes.c_uint, ctypes.c_int]
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.repetitions):
        if a.role in ['partial_sync', 'gpu_only_partial_sync']:
            runtime.gxvm_timeline_mark(i+1, 0)
        start = time.perf_counter()
        chunks = generate()
        report['generation_wall_seconds'].append(time.perf_counter()-start)
        if a.role in ['partial_sync', 'gpu_only_partial_sync']:
            runtime.gxvm_timeline_mark(i+1, 1)
        save()
    if a.role == 'gpu-profile':
        Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
    elif a.role == 'cpu-profile':
        assert runtime.gxvm_ipc_profile_stop() == 0
transformer = pipe.diffusion_model.transformer
network_wrapper = transformer._cuda_graph_dispatch.cond_call
decoder_wrapper = pipe.decoder.vae._decoder_wrapper
report['graph_state'] = {
    'network_capture_threshold': transformer._cuda_graph_capture_ar_idx,
    'network_graph_present': network_wrapper is not None and network_wrapper._graph is not None,
    'decoder_graph_present': decoder_wrapper is not None and decoder_wrapper._graph is not None,
}
report['output_shapes'] = [list(t.shape) for t in chunks]
report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
if native:
    report['finite'] = all(bool(torch.isfinite(t).all()) for t in chunks)
    assert report['finite']
    digest = hashlib.sha256()
    for t in chunks:
        digest.update(t.float().contiguous().numpy().tobytes())
    report['output_float32_sha256'] = digest.hexdigest()
save()
print('FLASHDREAMS_RESULT', json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','resolved_config']}), flush=True)
