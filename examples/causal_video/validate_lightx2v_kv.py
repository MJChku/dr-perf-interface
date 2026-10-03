"""CPU value oracle for the actual LightX2V valid-entry load/store methods.

CUDA allocation/events are replaced with CPU backing; this checks ring values,
step/layer isolation and head-shard shapes, not asynchronous stream correctness.
"""
import argparse
import ast
import json
from pathlib import Path
import torch


def validate(tree, defer_writeback=False):
    scope = {'torch': torch}
    folder = tree / 'lightx2v/common/kvcache'
    for name in ('base.py', 'rolling.py'):
        module = ast.parse((folder / name).read_text())
        module.body = [node for node in module.body if isinstance(node, ast.ClassDef)]
        exec(compile(module, str(folder / name), 'exec'), scope)
    checks = 0
    torch.manual_seed(42)
    for class_name in ('RollingKVCachePool', 'StepRollingKVCachePool', 'HybridStepRollingKVCachePool'):
        cls = scope[class_name]
        for heads in (1, 2, 4):
            for sink in (0, 2):
                for prefetch_before_roll in (False, True):
                    args = dict(num_layers=3, num_heads=heads, head_dim=2,
                                dtype=torch.float32, device=torch.device('cpu'))
                    if class_name == 'HybridStepRollingKVCachePool':
                        args.update(num_steps=2, layer_cache_sizes=[8, 10, 12])
                    else:
                        args['cache_size'] = 10
                        if class_name == 'StepRollingKVCachePool':
                            args['num_steps'] = 2
                    cache = cls(**args)
                    cache._init_kv_buffer()
                    if hasattr(cache, '_k_buckets'):
                        cache._k_cpu_buckets, cache._v_cpu_buckets = cache._k_buckets, cache._v_buckets
                    else:
                        cache._k_cpu, cache._v_cpu = cache._k_buffer, cache._v_buffer
                    cache._k_gpu_buf = torch.empty(cache._cache_size, heads, 2)
                    cache._v_gpu_buf = torch.empty_like(cache._k_gpu_buf)
                    cache._kv_offload = True
                    cache._record_cpu_update = lambda layer: None
                    cache.sync_all = lambda: None
                    cache._reset_offload_state = lambda: setattr(cache, '_loaded_layer', -1)
                    steps = getattr(cache, 'num_steps', 1)
                    oracle = {}
                    global_ends = {}
                    for step in range(steps):
                        for layer in range(3):
                            oracle[step, layer] = (torch.empty(0, heads, 2), torch.empty(0, heads, 2))
                            global_ends[step, layer] = 0
                    # Multiple requests, many wraps, repeated overwrites, uneven capacities.
                    for request in range(2):
                        if request:
                            cache.reset()
                            for key in oracle:
                                oracle[key] = (torch.empty(0, heads, 2), torch.empty(0, heads, 2))
                                global_ends[key] = 0
                        for chunk in range(18):
                            count = (2, 3, 2, 4)[chunk % 4]
                            for denoise in range(3):
                                for step in range(steps):
                                    cache._current_step = step
                                    for layer in range(3):
                                        key = step, layer
                                        prior_k, prior_v = oracle[key]
                                        capacity = cache.cache_size_for_layer(layer)
                                        advancing = denoise == 0
                                        evicted = max(0, len(prior_k) + count - capacity) if advancing else 0
                                        cache._loaded_layer = layer
                                        # Poison staging so omitted needed slots fail loudly.
                                        cache._k_gpu_buf.fill_(float('nan'))
                                        cache._v_gpu_buf.fill_(float('nan'))
                                        if prefetch_before_roll:
                                            cache._copy_layer_to_gpu(layer)
                                        if evicted:
                                            cache.roll_window(layer, sink, evicted)
                                            prior_k = torch.cat((prior_k[:sink], prior_k[sink + evicted:]))
                                            prior_v = torch.cat((prior_v[:sink], prior_v[sink + evicted:]))
                                        if not prefetch_before_roll:
                                            cache._copy_layer_to_gpu(layer)
                                        start = len(prior_k) if advancing else len(prior_k) - count
                                        k, v = torch.randn(count, heads, 2), torch.randn(count, heads, 2)
                                        expected_k = torch.cat((prior_k[:start], k))
                                        expected_v = torch.cat((prior_v[:start], v))
                                        options = {}
                                        if defer_writeback and class_name == 'RollingKVCachePool':
                                            options['writeback'] = denoise == 2
                                        before_cpu = (cache._k_cpu_layer(layer).clone(), cache._v_cpu_layer(layer).clone())
                                        cache.store_kv(k, v, start, start + count, layer, **options)
                                        if options.get('writeback', True):
                                            for logical, physical, length in cache._logical_chunks(layer, start, start + count):
                                                offset = logical - start
                                                torch.testing.assert_close(cache._k_cpu_layer(layer)[physical:physical + length], k[offset:offset + length], rtol=0, atol=0)
                                                torch.testing.assert_close(cache._v_cpu_layer(layer)[physical:physical + length], v[offset:offset + length], rtol=0, atol=0)
                                        else:
                                            torch.testing.assert_close(cache._k_cpu_layer(layer), before_cpu[0], rtol=0, atol=0, equal_nan=True)
                                            torch.testing.assert_close(cache._v_cpu_layer(layer), before_cpu[1], rtol=0, atol=0, equal_nan=True)
                                        global_ends[key] += count if advancing else 0
                                        cache.set_ends(layer, global_ends[key], len(expected_k))
                                        for attn_start in (0, min(sink, len(expected_k)), max(0, len(expected_k) - 4)):
                                            torch.testing.assert_close(cache.k_cache(layer, attn_start, len(expected_k)), expected_k[attn_start:], rtol=0, atol=0)
                                            torch.testing.assert_close(cache.v_cache(layer, attn_start, len(expected_v)), expected_v[attn_start:], rtol=0, atol=0)
                                            checks += 2
                                        oracle[key] = expected_k, expected_v
    return {'passed': True, 'attention_tensor_comparisons': checks,
            'defer_writeback': defer_writeback,
            'scope': 'Actual CPU-executed cache methods; dense logical oracle; ring wraps, sinks, overwrites, request resets, steps, layer capacities and head shards. CUDA ordering and distributed collectives require separate validation.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tree', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--defer-writeback', action='store_true')
    a = p.parse_args()
    result = validate(a.tree, a.defer_writeback)
    text = json.dumps(result, indent=2) + '\n'
    if a.output:
        a.output.write_text(text)
    print(text, end='')
