"""Independent runtime check; run with the pinned vLLM CPU environment."""
import json
from vllm import SamplingParams
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.utils.hashing import sha256

init_none_hash(sha256)
rows = []
for block_size in (16, 64, 128):
    for n in sorted({1, block_size-1, block_size, block_size+1, 2*block_size-1, 2*block_size, 2*block_size+1, 5*block_size+3}):
        calls = []
        def counted_hash(value):
            calls.append(len(value[1]))
            return sha256(value)
        hasher = get_request_block_hasher(block_size, counted_hash)
        r = Request(request_id=f"verify-{block_size}-{n}", prompt_token_ids=list(range(n)), sampling_params=SamplingParams(max_tokens=1), pooling_params=None, block_hasher=hasher)
        expected = n // block_size
        assert len(calls) == len(r.block_hashes) == expected
        assert sum(calls) == expected * block_size
        r.update_block_hashes()
        assert len(calls) == expected, "unchanged request rehashed blocks"
        rows.append(dict(block_size=block_size, tokens=n, hash_calls=len(calls), hashed_tokens=sum(calls), tail=n%block_size))
print(json.dumps({"passed": True, "cases": rows}, indent=2))
