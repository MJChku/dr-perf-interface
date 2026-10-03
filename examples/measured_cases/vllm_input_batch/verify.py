"""Dump generated text + token ids for the vllm_run_batch workload, natively.

    python examples/vllm_verify_batch.py OUT.json [PLEN]

Same prompts / sampling params / engine settings as examples/vllm_run_batch.py,
so the outputs can be compared before and after an InputBatch change.
"""
import json
import os
import sys

_HF = "/home/ubuntu/drperf/third_party/hf"
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

out_path = sys.argv[1]
PLEN = int(sys.argv[2]) if len(sys.argv) > 2 else 1
REQS, SEQS, MODEL = 24, 8, "facebook/opt-125m"

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

_SRC = os.path.join(os.environ.get("CASE_TREE", "/home/ubuntu/drperf-cases/vllm-cpu-batch"), "vllm")
assert os.path.abspath(os.path.dirname(vllm.__file__)) == _SRC, vllm.__file__

llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_model_len=512,
          max_num_seqs=SEQS, max_num_batched_tokens=256, seed=0, disable_log_stats=True)

base = "The quick brown fox jumps over the lazy dog. "
prompts = [base * (PLEN * (1 + (i % 4))) + "Then, number %d:" % i for i in range(REQS)]
params = [SamplingParams(temperature=0.0, max_tokens=4 + 6 * (i % 7), ignore_eos=True)
          for i in range(REQS)]

llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
             use_tqdm=False)
outs = llm.generate(prompts, params, use_tqdm=False)

rec = [{"i": i, "prompt_ids": list(o.prompt_token_ids),
        "text": o.outputs[0].text, "token_ids": list(o.outputs[0].token_ids)}
       for i, o in enumerate(outs)]
with open(out_path, "w") as f:
    json.dump(rec, f, indent=1, sort_keys=True)
print("wrote", out_path, "plen", PLEN, "reqs", len(rec),
      "tokens", sum(len(r["token_ids"]) for r in rec))
