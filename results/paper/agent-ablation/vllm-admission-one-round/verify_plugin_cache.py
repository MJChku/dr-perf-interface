"""Independent count of actual plugin-discovery API calls across cache states."""
import json
import importlib.metadata as metadata
from unittest.mock import patch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import cached_load_custom_logitsprocs, validate_logits_processors_parameters

params = SamplingParams(max_tokens=1)
original = metadata.entry_points
calls = []
rows = []
def observed_entry_points(*args, **kwargs):
    calls.append(dict(kwargs))
    return original(*args, **kwargs)

cached_load_custom_logitsprocs.cache_clear()
with patch.object(metadata, 'entry_points', observed_entry_points):
    for label, configured in [('cold_none', None), ('warm_none', None), ('cold_empty_tuple', ()), ('warm_empty_tuple', ()), ('warm_none_again', None)]:
        before = len(calls)
        info_before = cached_load_custom_logitsprocs.cache_info()._asdict()
        validate_logits_processors_parameters(configured, params)
        rows.append(dict(label=label, scans=len(calls)-before, cache_before=info_before, cache_after=cached_load_custom_logitsprocs.cache_info()._asdict()))
    cached_load_custom_logitsprocs.cache_clear()
    before = len(calls)
    validate_logits_processors_parameters(None, params)
    rows.append(dict(label='after_cache_clear',scans=len(calls)-before))
assert [r['scans'] for r in rows] == [1,0,1,0,0,1]
assert all(c['group']=='vllm.logits_processors' for c in calls)
print(json.dumps(dict(passed=True, rows=rows, entry_point_queries=calls),indent=2))
