"""CPU regression for Inferix Self-Forcing KV transfer bounds.

Executes AST-extracted source forwards with dense CPU attention and CPU cache backing.
"""
import json
from pathlib import Path
import sys


def validate_kv_transfer_cpu(reference, candidate_tree, report_path):
    """Exercise the source forward methods without importing CUDA dependencies."""
    import ast
    import copy
    import math
    from types import SimpleNamespace
    import torch

    trees = [reference, candidate_tree]
    for tree in trees:
        if not tree.is_dir():
            raise SystemExit(f'Missing tree: {tree}')

    def source_method(tree, relative, class_name, method_name, namespace):
        path = tree / relative
        module = ast.parse(path.read_text())
        cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                    and n.name == method_name))
        # The functions are otherwise unmodified. Their annotations refer to
        # imports intentionally absent from this CPU-only harness.
        method.returns = None
        for arg in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
            arg.annotation = None
        ast.fix_missing_locations(method)
        code = compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec')
        scope = dict(namespace)
        exec(code, scope)
        return scope[method_name]

    def dense_attention(q, k, v):
        scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(q.shape[-1])
        weights = scores.softmax(dim=-1)
        return torch.einsum('bhqk,bkhd->bqhd', weights, v)

    model_file = 'inferix/models/self_forcing/causal_model.py'
    cache_file = 'inferix/kvcache_manager/model/self_forcing_kv_cache_manager.py'
    namespace = dict(torch=torch, math=math, causal_rope_apply=lambda x, *a, **kw: x,
                     causal_rope_apply_chunked=lambda x, *a, **kw: x)

    class Backing:
        def __init__(self, capacity, batch, dim):
            self.device = torch.device('cpu')
            self.data = {str(i): torch.full((2, capacity, 1, 1, dim), float('nan'))
                         for i in range(batch)}
            self.reads = []
            self.writes = []

        def get_raw(self, req, layer):
            return self.data[req.request_id]

        def get(self, req, layer):
            value = self.get_raw(req, layer)
            return value.clone()

        def get_range(self, req, layer, start, length):
            return self.get_raw(req, layer)[:, start:start + length].clone()

        def set(self, req, layer, start, size, data):
            assert 0 <= start <= start + size <= self.get_raw(req, layer).shape[1]
            self.writes.append((req.request_id, start, size))
            self.get_raw(req, layer)[:, start:start + size] = data

    def make_run(tree, sink, offload, parallel, local_size, capacity,
                 batch=2, dim=2):
        attn_forward = source_method(tree, model_file, 'CausalWanSelfAttention',
                                     'forward', namespace)
        block_forward = source_method(tree, model_file, 'CausalWanAttentionBlock',
                                      'forward', namespace)
        get_cache = source_method(tree, cache_file, 'SelfForcingKVCacheManager',
                                  'get_kv_cache', namespace)
        set_cache = source_method(tree, cache_file, 'SelfForcingKVCacheManager',
                                  'set_kv_cache', namespace)
        backing = Backing(capacity, batch, dim)
        requests = [SimpleNamespace(request_id=str(i)) for i in range(batch)]
        wrapper = SimpleNamespace(layer_number=0, enable_kv_offload=offload)
        def tracked_get(**kw):
            read_length = kw.get('read_length')
            backing.reads.append((kw['kv_cache_request'].request_id, 0,
                                  capacity if read_length is None else read_length))
            return get_cache(wrapper, **kw)
        wrapper.get_kv_cache = tracked_get
        wrapper.set_kv_cache = lambda **kw: set_cache(wrapper, **kw)
        identity = lambda x: x
        config = None if parallel == 'none' else SimpleNamespace(world_size=1, rank=0)
        attn = SimpleNamespace(num_heads=1, head_dim=dim, norm_q=identity,
                               norm_k=identity, q=identity, k=identity,
                               v=identity, o=identity, attention=dense_attention,
                               parallel_config=config, local_attn_size=local_size,
                               sink_size=sink)

        class AttentionAdapter:
            def __call__(self, *args, **kwargs):
                output = attn_forward(attn, *args, **kwargs)
                self.last_output = output[0].detach().clone()
                return output

        adapter = AttentionAdapter()
        block = SimpleNamespace(modulation=torch.zeros(1, 6, dim), norm1=identity,
                                norm2=identity, norm3=identity, self_attn=adapter,
                                cross_attn=lambda x, *a, **kw: torch.zeros_like(x),
                                ffn=lambda x: torch.zeros_like(x),
                                kv_cache_manager=wrapper, enable_kv_offload=offload,
                                parallel_config=config, local_attn_size=local_size)
        meta = dict(global_end_index=torch.tensor([0]), local_end_index=torch.tensor([0]))

        poisoned_allocations = [0]

        def step(start, length, seed):
            x = torch.arange(batch * length * dim, dtype=torch.float32).reshape(batch, length, dim)
            x = (x + seed) / 13
            e = torch.zeros(batch, length, 6, dim)
            e[:, :, 2] = 1
            original_empty = torch.empty
            def poisoned_empty(*args, **kwargs):
                value = original_empty(*args, **kwargs)
                if tuple(value.shape) == (2, capacity, 1, 1, dim):
                    value.fill_(float('nan'))
                    poisoned_allocations[0] += 1
                return value
            # If attention touches uncopied scratch, NaNs propagate to output.
            if tree == trees[1] and offload:
                torch.empty = poisoned_empty
            try:
                y = block_forward(block, x, e, torch.tensor([length] * batch),
                                  torch.tensor([[length, 1, 1]] * batch), None, None,
                                  None, None, kv_cache_meta=meta,
                                  current_start=start, kv_cache_manager=backing,
                                  kv_cache_requests=requests)
            finally:
                torch.empty = original_empty
            end = meta['local_end_index'].item()
            assert torch.isfinite(adapter.last_output).all(), (start, seed, 'attention')
            assert torch.isfinite(y).all(), (start, seed, 'block')
            for req in requests:
                assert torch.isfinite(backing.get_raw(req, 'layer_0')[:, :end]).all(), (start, seed, 'cache')
            return y.detach().clone(), adapter.last_output, end

        return SimpleNamespace(step=step, meta=meta, backing=backing,
                               poisoned_allocations=poisoned_allocations)

    window_sequence = [(0, 2, 1), (0, 2, 7), (0, 2, 11),
                       (2, 2, 19), (4, 2, 29), (6, 2, 37),
                       (6, 2, 41), (8, 2, 47)]
    # One-token initial context, then seven two-token blocks. Every block is
    # revisited, as in denoising followed by a clean-context cache update.
    global_sequence = [(0, 1, 1), (0, 1, 7), (0, 1, 11)]
    for block in range(7):
        start = 1 + 2 * block
        global_sequence += [(start, 2, 19 + 8 * block),
                            (start, 2, 23 + 8 * block)]
    scenarios = [('window', 6, 6, sink, window_sequence) for sink in (0, 1)]
    scenarios += [('global', 16, -1, 0, global_sequence)]
    cases = 0
    summary = []
    for scenario, capacity, local_size, sink, sequence in scenarios:
        for parallel in ('none', 'world_size_1'):
            for offload in (True, False):
                original, candidate = [make_run(tree, sink, offload, parallel,
                                                local_size, capacity) for tree in trees]
                for start, length, seed in sequence:
                    old_end = original.meta['local_end_index'].item()
                    old_global = original.meta['global_end_index'].item()
                    previous_reads = len(candidate.backing.reads)
                    previous_writes = len(candidate.backing.writes)
                    old_prefix = {key: value[:, :old_end].clone()
                                  for key, value in candidate.backing.data.items()}
                    left = original.step(start, length, seed)
                    right = candidate.step(start, length, seed)
                    assert left[2] == right[2]
                    torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
                    torch.testing.assert_close(left[1], right[1], rtol=0, atol=0)
                    for key in original.backing.data:
                        torch.testing.assert_close(original.backing.data[key][:, :left[2]],
                                                   candidate.backing.data[key][:, :right[2]],
                                                   rtol=0, atol=0)
                    step_reads = candidate.backing.reads[previous_reads:]
                    step_writes = candidate.backing.writes[previous_writes:]
                    assert len(step_reads) == len(step_writes) == 2
                    evicts = local_size != -1 and start + length > old_global and length + old_end > capacity
                    expected_read = old_end if evicts else old_end + start - old_global
                    expected_write_start = min(sink, right[2]-length) if evicts else right[2]-length
                    for _, read_start, read_size in step_reads:
                        assert read_start == 0
                        assert read_size == (expected_read if offload else capacity)
                    for _, write_start, write_size in step_writes:
                        assert write_start == (expected_write_start if offload else 0)
                        assert write_size == right[2] - write_start
                    # A repeated block should leave earlier history exactly intact.
                    if start + length == old_global and old_end >= length:
                        for key in old_prefix:
                            torch.testing.assert_close(candidate.backing.data[key][:, :old_end-length],
                                                       old_prefix[key][:, :old_end-length],
                                                       rtol=0, atol=0)
                    if offload:
                        assert sum(n for _, _, n in candidate.backing.reads) <= sum(
                            n for _, _, n in original.backing.reads)
                        assert sum(n for _, _, n in candidate.backing.writes) <= sum(
                            n for _, _, n in original.backing.writes)
                    cases += 1
                # Pipeline reset clears indices while reusing allocated backing.
                for run in (original, candidate):
                    run.meta['global_end_index'].fill_(0)
                    run.meta['local_end_index'].fill_(0)
                left, right = original.step(0, 2, 59), candidate.step(0, 2, 59)
                torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
                for key in original.backing.data:
                    torch.testing.assert_close(original.backing.data[key][:, :2],
                                               candidate.backing.data[key][:, :2], rtol=0, atol=0)
                cases += 1
                if offload:
                    assert candidate.poisoned_allocations[0] == 2 * (len(sequence) + 1)
                summary.append(dict(scenario=scenario, capacity_tokens=capacity,
                                    local_attn_size=local_size, sink_tokens=sink,
                                    parallel_config=parallel, offload=offload,
                                    paired_steps=len(sequence) + 1,
                                    request_calls=2 * (len(sequence) + 1),
                                    original_read_tokens=sum(x[2] for x in original.backing.reads),
                                    candidate_read_tokens=sum(x[2] for x in candidate.backing.reads),
                                    original_write_tokens=sum(x[2] for x in original.backing.writes),
                                    candidate_write_tokens=sum(x[2] for x in candidate.backing.writes),
                                    poisoned_scratch_allocations=candidate.poisoned_allocations[0]))
    report = dict(result='pass', torch_version=torch.__version__, device='cpu',
                  reference=str(reference), candidate=str(candidate_tree),
                  paired_steps=cases, request_calls=sum(x['request_calls'] for x in summary),
                  total_original_read_tokens=sum(x['original_read_tokens'] for x in summary),
                  total_candidate_read_tokens=sum(x['candidate_read_tokens'] for x in summary),
                  total_original_write_tokens=sum(x['original_write_tokens'] for x in summary),
                  total_candidate_write_tokens=sum(x['candidate_write_tokens'] for x in summary),
                  scenarios=summary,
                  scope='AST-extracted attention/block and KV wrapper methods; mocked projections and RoPE; dense CPU attention and CPU backing; exact output and active-cache comparison',
                  limitations='No CUDA transfer timing, GPU kernels, or distributed world_size>1 execution')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'KV_TRANSFER_CPU_PASS {cases} paired steps; {report["request_calls"]} request calls; report {report_path}')


if __name__ == '__main__':
    import argparse
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=root / 'out/causal-video/Inferix-metadata')
    parser.add_argument('--candidate', type=Path,
                        default=root / 'out/causal-video/Inferix-kvtransfer')
    parser.add_argument('--report', type=Path,
                        default=root / 'out/causal-video/inferix-kv-transfer/cpu-validation.json')
    args = parser.parse_args()
    validate_kv_transfer_cpu(args.reference.resolve(), args.candidate.resolve(),
                             args.report.resolve())
