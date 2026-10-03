"""Run the real LightX2V Self-Forcing pipeline for native/GX screening."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def install_drperf(pipe):
    """Mark host dispatch after warmup; GX CUDA exports are excluded externally."""
    import functools
    import inspect
    import perfmark
    from lightx2v.common.kvcache.rolling import RollingKVCachePool
    transformer = pipe.runner.model.transformer_infer
    regions = []

    def wrap(owner, method, name, features):
        original = getattr(owner, method)
        signature = inspect.signature(original)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            arguments = signature.bind(*args, **kwargs).arguments
            with perfmark.region(name, **features(arguments)):
                return original(*args, **kwargs)

        setattr(owner, method, measured)
        regions.append(name)

    wrap(pipe, 'generate', 'lightx2v_generate', lambda a: {})
    wrap(pipe.runner.model, 'infer', 'lightx2v_model',
         lambda a: {'chunk': int(pipe.runner.model.scheduler.seg_index),
                    'context_refresh': str(bool(pipe.runner.model.scheduler.is_rerun))})
    wrap(transformer, '_calculate_q_k_len', 'lightx2v_attention_lengths',
         lambda a: {'query_tokens': int(a['q'].shape[0])})
    wrap(transformer, 'infer_self_attn_with_kvcache', 'lightx2v_self_attention',
         lambda a: {'query_tokens': int(a['x'].shape[0]),
                    'chunk': int(transformer.scheduler.seg_index),
                    'layer': str(transformer.block_idx)})
    wrap(RollingKVCachePool, '_copy_layer_to_gpu', 'lightx2v_kv_load',
         lambda a: {'valid_tokens': a['self'].get_local_end(a['layer_id']),
                    'capacity': a['self'].cache_size_for_layer(a['layer_id']),
                    'ring_active': str(a['self']._is_ring_active(a['layer_id']))})
    wrap(RollingKVCachePool, 'store_kv', 'lightx2v_kv_store',
         lambda a: {'tokens': int(a['end_idx']) - int(a['start_idx']),
                    'start': int(a['start_idx']),
                    'ring_active': str(a['self']._is_ring_active(a['layer_id']))})
    wrap(RollingKVCachePool, 'reset', 'lightx2v_kv_reset',
         lambda a: {'cache_bytes': int(a['self']._k_cpu.nbytes + a['self']._v_cpu.nbytes)
                    if a['self']._kv_offload else 0})
    return regions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tree', required=True, type=Path)
    p.add_argument('--base', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--role', choices=['native', 'gpu-profile', 'emu', 'cpu-profile', 'partial_sync', 'gpu_only_partial_sync', 'drperf'], default='native')
    p.add_argument('--frames', type=int, default=21)
    p.add_argument('--repetitions', type=int, default=1)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--weight-offload', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--kv-offload', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--control-replay', type=Path, help='Native scheduler decisions for computation-free GX; GPU operations still execute')
    p.add_argument('--python-profile', action='store_true', help='Separate native Python call-cost diagnostic')
    a = p.parse_args()
    assert a.repetitions > 0 and a.warmup >= 0
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.tree))
    os.environ.setdefault('PROFILING_DEBUG_LEVEL', '0')
    os.environ.setdefault('DTYPE', 'BF16')
    import torch
    from lightx2v import LightX2VPipeline
    from lightx2v.models.schedulers.wan.self_forcing.scheduler import WanSFScheduler
    torch.set_num_threads(1)
    native = a.role in ('native', 'gpu-profile')
    assert not a.python_profile or a.role == 'native'
    assert native or a.control_replay, 'GX needs native scheduler decisions'
    assert not (native and a.control_replay)
    replay = json.loads(a.control_replay.read_text()) if a.control_replay else None
    controls = {'selected_timesteps': None, 'indices': []}
    cursor = 0
    original_prepare = WanSFScheduler._prepare_index_schedule
    def prepare_schedule(scheduler):
        original_prepare(scheduler)
        if replay:
            scheduler.selected_timesteps = replay['selected_timesteps']
        controls['selected_timesteps'] = list(scheduler.selected_timesteps)
    def index_for_timestep(scheduler, timestep, schedule_timesteps=None):
        nonlocal cursor
        if schedule_timesteps is None:
            schedule_timesteps = scheduler.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        if replay:
            expected = replay['indices'][cursor]
            assert expected['timestep'] == timestep
            # Retain equality, nonzero, scalar indexing and D2H. Only restore
            # native control shape/value; GX does not compute the predicates.
            indices.resize_(expected['shape'])
        pos = 1 if len(indices) > 1 else 0
        result = indices[pos].item()
        if replay:
            result = expected['index']
        controls['indices'].append({'timestep': timestep, 'shape': list(indices.shape), 'index': result})
        cursor += 1
        return result
    WanSFScheduler._prepare_index_schedule = prepare_schedule
    WanSFScheduler.index_for_timestep = index_for_timestep
    assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text()) == native
    config = json.loads((a.tree/'configs/self_forcing/wan_t2v_sf.json').read_text())
    config.update(num_frames=a.frames, dit_original_ckpt=a.checkpoint,
                  cpu_offload=a.weight_offload, offload_granularity='block',
                  t5_cpu_offload=True, vae_cpu_offload=False)
    config['ar_config']['kv_offload'] = a.kv_offload
    for key in ['self_attn_1_type', 'cross_attn_1_type', 'cross_attn_2_type']:
        config[key] = 'flash_attn2'
    config_path = a.output/'config.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    report = {'role': a.role, 'pid': os.getpid(), 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'requested_pixel_frames': a.frames,
              'config': config, 'prompt': 'A cat walks on the grass, realistic style',
              'source_revision': '69018c92b0a42d9b0cf962a248fadbfe0cbc03de',
              'source_sha256': {str(f.relative_to(a.tree)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for directory in ['lightx2v', 'lightx2v_platform']
                                for f in sorted((a.tree/directory).rglob('*.py'))},
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'control_replay_sha256': hashlib.sha256(a.control_replay.read_bytes()).hexdigest() if a.control_replay else None,
              'warmup_wall_seconds': [], 'generation_wall_seconds': []}
    def save():
        (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    start = time.perf_counter()
    pipe = LightX2VPipeline(task='t2v', model_path=a.base, model_cls='wan2.1_sf', dit_original_ckpt=a.checkpoint)
    pipe.create_generator(config_json=str(config_path))
    # T5 slices its output using a GPU reduction read back to Python. Preserve
    # that execution, then restore the native view shapes in computation-free GX.
    encoder = pipe.runner.text_encoders[0]
    original_infer = encoder.infer
    def text_infer(texts):
        full = []
        hook = encoder.model.register_forward_hook(lambda module, args, output: full.append(output))
        try:
            result = original_infer(texts)
        finally:
            hook.remove()
        if replay:
            assert len(full) == 1 and len(result) == len(replay['text_lengths'])
            result = [u[:length] for u,length in zip(full[0], replay['text_lengths'])]
        controls['text_lengths'] = [int(u.shape[0]) for u in result]
        return result
    encoder.infer = text_infer
    torch.cuda.synchronize()
    report['load_wall_seconds'] = time.perf_counter()-start
    report['resolved_config'] = dict(pipe.runner.config)
    save()
    def generate():
        nonlocal cursor
        cursor = 0
        controls['indices'] = []
        result = pipe.generate(seed=42, prompt=report['prompt'], return_result_tensor=True)
        if replay:
            assert cursor == len(replay['indices'])
        return result
    for _ in range(a.warmup):
        start = time.perf_counter()
        output = generate()
        torch.cuda.synchronize()
        report['warmup_wall_seconds'].append(time.perf_counter()-start)
        del output
    save()
    (a.output/'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    print('LIGHTX2V_WARMUP_DONE', flush=True)
    if a.role == 'drperf':
        report['drperf_regions'] = install_drperf(pipe)
    runtime = ctypes.CDLL(None)
    if a.role == 'gpu-profile':
        Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
    elif a.role == 'cpu-profile':
        runtime.gxvm_ipc_profile_start()
    elif a.role in ('partial_sync', 'gpu_only_partial_sync'):
        (runtime.gxvm_adopt_now if a.role == 'partial_sync' else runtime.gxvm_start)()
        runtime.gxvm_timeline_mark.argtypes = [ctypes.c_uint, ctypes.c_int]
    torch.cuda.reset_peak_memory_stats()
    if a.python_profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    for i in range(a.repetitions):
        if a.role in ('partial_sync', 'gpu_only_partial_sync'):
            runtime.gxvm_timeline_mark(i+1, 0)
        start = time.perf_counter()
        output = generate()
        torch.cuda.synchronize()
        report['generation_wall_seconds'].append(time.perf_counter()-start)
        if a.role in ('partial_sync', 'gpu_only_partial_sync'):
            runtime.gxvm_timeline_mark(i+1, 1)
    if a.role == 'gpu-profile':
        Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
    elif a.role == 'cpu-profile':
        assert runtime.gxvm_ipc_profile_stop() == 0
    if a.python_profile:
        profiler.disable()
        import pstats
        stats = pstats.Stats(profiler).stats
        report['python_profile'] = [dict(file=k[0], line=k[1], name=k[2], primitive_calls=v[0], calls=v[1], self_seconds=v[2], cumulative_seconds=v[3])
                                    for k,v in sorted(stats.items(), key=lambda kv: kv[1][2], reverse=True)[:150]]
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    video = output['video']
    report.update(output_shape=list(video.shape), output_dtype=str(video.dtype), finite_checked=native)
    if native:
        report['finite'] = bool(torch.isfinite(video).all())
        assert report['finite']
        values = video.detach().float().cpu().contiguous()
        report['output_float32_sha256'] = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        report['output_min'] = values.min().item()
        report['output_max'] = values.max().item()
    save()
    (a.output/'controls.json').write_text(json.dumps(controls, indent=2)+'\n')
    print('LIGHTX2V_RESULT', json.dumps({k:v for k,v in report.items() if k not in ['source_sha256', 'resolved_config', 'config']}), flush=True)


if __name__ == '__main__':
    main()
