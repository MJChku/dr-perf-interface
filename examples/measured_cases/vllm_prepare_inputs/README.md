# vLLM case A: the model runner's per-step fixed tax, and the per-step rebuild

Markers: `mark_step.py` on top of `examples/vllm_cpu/mark.py`'s seven regions
(tree `vllm-cpu-rb`).  Driver: `driver.py reqs=10`, ten concurrent requests
with different prompt and output lengths.

Study ("vLLM case A"):

    prepare_inputs        758.2*num_reqs + 106.9*num_tokens + 464,927.7   ->  constant 465k -> 205k (-56%)
    cached_request_data   4,916.2*num_running + 13,027.5                  ->  1,671*num_running + 15,247.4

* the ~580k fixed part of `_prepare_inputs` is about 35 tiny tensor ops per
  step; `CPUModelRunner._postprocess_tensors` sets `buffer.gpu = buffer.cpu`,
  so every `copy_to_gpu` copied a tensor onto itself (`prepare_inputs.patch`:
  return when the two alias, keep numpy views of the per-step buffers);
* fifteen decode steps in sixteen a request gets no new blocks, and the
  scheduler rebuilt its cached request data anyway
  (`cached_request_data.patch`: append `None` without calling `get_block_ids`,
  hoist the bound methods, store the token tail the consumer reads).

Equivalence: `verify.py` runs both workloads natively and dumps text, token
ids, cumulative logprob and finish reason per request; `run.sh` compares the
dumps.

## Rerun (2026-10-02, `run.sh`)

```
prepare_inputs       base  777*num_reqs + 94*num_tokens + 486,995   22% irregular
                     fix   769*num_reqs + 45*num_tokens + 223,838   27%
cached_request_data  base  4,930*num_running + 14,360               6%
                     fix   1,660*num_running + 17,589               1%
update_states        unchanged (3,384 -> 3,350 per request)
```
Constant 487k -> 224k (study 465k -> 205k); 4,930 -> 1,660 per running
request (study 4,916 -> 1,671).  Both workloads' outputs identical.
