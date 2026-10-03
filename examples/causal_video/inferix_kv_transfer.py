"""Apply bounded KV transfers to an isolated, pinned Inferix checkout.

Persistent KV remains on CPU. Full-capacity transient device buffers and the
attention operation are retained; only transferred ranges change. Distributed
and GPU-resident paths retain their existing behavior.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

MODEL = "inferix/models/self_forcing/causal_model.py"
MANAGER = "inferix/kvcache_manager/model/self_forcing_kv_cache_manager.py"
HASHES = {
    MODEL: "6ea994efae297f6d2fb1494e5b8a7597a1167c461e550388ab6ec9d9c44ee7fa",
    MANAGER: "07df3f8f2aeb8d9e76d0afeb16debcf198402a2701ebbdd64080c4db96af49bf",
}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one source anchor: {old[:100]!r}")
    return text.replace(old, new)


def adapt(relative, source):
    if relative == MANAGER:
        source = replace_once(source,
            "def get_kv_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest) -> torch.Tensor:",
            "def get_kv_cache(self, kv_cache_manager: KVCacheManager, kv_cache_request: KVCacheRequest, read_length: int | None = None) -> torch.Tensor:")
        return replace_once(source,
            '''        return kv_cache_manager.get(
            kv_cache_request, f"layer_{self.layer_number}",
        ).squeeze(2)''',
            '''        if read_length is None:
            return kv_cache_manager.get(
                kv_cache_request, f"layer_{self.layer_number}",
            ).squeeze(2)
        source = kv_cache_manager.get_raw(kv_cache_request, f"layer_{self.layer_number}")
        if not 0 <= read_length <= source.shape[1]:
            raise ValueError(f"invalid KV history length {read_length}")
        # Keep the allocated capacity: attention's eviction decisions use it.
        # The current block overwrites the uncopied range before attention.
        cache = torch.empty(source.shape, dtype=source.dtype, device=kv_cache_manager.device)
        if read_length:
            cache[0, :read_length].copy_(source[0, :read_length])
            cache[1, :read_length].copy_(source[1, :read_length])
        return cache.squeeze(2)''')

    source = replace_once(source,
        '''                local_start_index = local_end_index - num_new_tokens
            else:
                # Assign new keys/values directly up to current_end''',
        '''                local_start_index = local_end_index - num_new_tokens
                # Rolling changes historical positions as well as new tokens.
                kv_cache_meta["_writeback_start"] = min(sink_tokens, local_start_index)
            else:
                # Assign new keys/values directly up to current_end''')
    source = replace_once(source,
        '''                local_start_index = local_end_index - num_new_tokens
            
            # Update KV cache with new keys/values''',
        '''                local_start_index = local_end_index - num_new_tokens
                kv_cache_meta["_writeback_start"] = local_start_index
            
            # Update KV cache with new keys/values''')
    source = replace_once(source,
        '''        # Fetch self-attention kv cache    
        if kv_cache_meta is not None:''',
        '''        # Bound transfers only for CPU-backed, single-GPU KV caches.
        bounded_kv = self.enable_kv_offload and (
            self.parallel_config is None or self.parallel_config.world_size <= 1)
        # Fetch self-attention kv cache
        if kv_cache_meta is not None:''')
    source = replace_once(source,
        '''            for kv_cache_request in kv_cache_requests:
                kv_cache = self.kv_cache_manager.get_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request)
                all_k_cache.append(kv_cache[0])''',
        '''            for kv_cache_request in kv_cache_requests:
                read_length = None
                if bounded_kv:
                    capacity = kv_cache_manager.get_raw(
                        kv_cache_request, f"layer_{self.kv_cache_manager.layer_number}").shape[1]
                    old_local_end = kv_cache_meta["local_end_index"].item()
                    old_global_end = kv_cache_meta["global_end_index"].item()
                    evicts = self.local_attn_size != -1 and (
                        current_start + x.shape[1] > old_global_end) and (
                        x.shape[1] + old_local_end > capacity)
                    # Eviction needs the old history to perform the existing
                    # roll. Otherwise the current block replaces its own KV.
                    read_length = (old_local_end if evicts else
                                   old_local_end + current_start - old_global_end)
                kv_cache = self.kv_cache_manager.get_kv_cache(
                    kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request,
                    read_length=read_length)
                all_k_cache.append(kv_cache[0])''')
    return replace_once(source,
        '''                cur_k_cache, cur_v_cache = new_k[request_idx], new_v[request_idx]
                self.kv_cache_manager.set_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, start_index=0, k_data=cur_k_cache, v_data=cur_v_cache)''',
        '''                write_start = kv_cache_meta["_writeback_start"] if bounded_kv else 0
                if not 0 <= write_start <= new_k.shape[1]:
                    raise ValueError(f"invalid KV writeback offset {write_start}")
                cur_k_cache = new_k[request_idx, write_start:]
                cur_v_cache = new_v[request_idx, write_start:]
                self.kv_cache_manager.set_kv_cache(kv_cache_manager=kv_cache_manager, kv_cache_request=kv_cache_request, start_index=write_start, k_data=cur_k_cache, v_data=cur_v_cache)''')


def prepare(root):
    changes = {}
    for relative, expected in HASHES.items():
        before = (root / relative).read_bytes()
        if hashlib.sha256(before).hexdigest() != expected:
            raise ValueError(f"unexpected source: {root / relative}")
        after = adapt(relative, before.decode()).encode()
        compile(after, str(root / relative), "exec")
        changes[relative] = (before, after)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[2] / "out/causal-video/Inferix"
    if args.apply and args.root.resolve() == reference.resolve():
        parser.error("copy the reference checkout before applying")
    changes = prepare(args.root)
    patch = "".join("".join(difflib.unified_diff(
        before.decode().splitlines(True), after.decode().splitlines(True),
        fromfile="a/" + relative, tofile="b/" + relative))
        for relative, (before, after) in changes.items())
    if args.patch:
        args.patch.write_text(patch)
    if args.apply:
        for relative, (_, after) in changes.items():
            (args.root / relative).write_bytes(after)
        manifest = {"candidate": "bounded KV transfers with CPU offload retained",
                    "files": {relative: {"before_sha256": hashlib.sha256(before).hexdigest(),
                                         "after_sha256": hashlib.sha256(after).hexdigest()}
                              for relative, (before, after) in changes.items()}}
        (args.root / "inferix-kv-transfer.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(args.root / "inferix-kv-transfer.json")
    elif not args.patch:
        print(patch, end="")


if __name__ == "__main__":
    main()
