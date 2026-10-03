"""Equivalence harness for the vllm-cpu-rb third beats (cases A "the rebuild"
and F "a per-step scan of the shared prefix").

    python examples/vllm_verify_rb.py <out.json>

Runs BOTH drivers' workloads in one process against
/home/ubuntu/drperf-cases/vllm-cpu-rb and dumps, as JSON:
  * run1: reqs=10 opt-125m greedy -- per request text, token ids, cumulative
    logprob, finish reason;
  * run2: the shared-prefix / block_size=16 workload -- the same, plus the
    value returned by FullAttentionManager.get_num_common_prefix_blocks at
    every scheduler step (monkeypatched wrapper, so no source edit is needed
    on either side of the comparison).
Run it before and after the patches and `diff` the two files.
"""
import json
import os
import sys

os.environ.setdefault("HF_HOME", "/home/ubuntu/drperf/third_party/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "0")

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

_SRC = os.path.join(os.environ.get("CASE_TREE", "/home/ubuntu/drperf-cases/vllm-cpu-rb"), "vllm")
assert os.path.abspath(os.path.dirname(vllm.__file__)) == _SRC, vllm.__file__

from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager  # noqa: E402

PREFIX_LOG = []
_orig = FullAttentionManager.get_num_common_prefix_blocks


def _logged(self, running_request_id):
    v = _orig(self, running_request_id)
    PREFIX_LOG.append([len(self.req_to_blocks),
                       len(self.req_to_blocks.get(running_request_id, ())), v])
    return v


FullAttentionManager.get_num_common_prefix_blocks = _logged


def dump(outs):
    return [{"text": o.outputs[0].text,
             "token_ids": list(o.outputs[0].token_ids),
             "clogprob": o.outputs[0].cumulative_logprob,
             "finish": o.outputs[0].finish_reason,
             "prompt_ids": list(o.prompt_token_ids)} for o in outs]


res = {}

# ------------------------------------------------------------------ run 1 ---
llm = LLM(model="facebook/opt-125m", dtype="bfloat16", enforce_eager=True,
          max_model_len=256, max_num_seqs=12, max_num_batched_tokens=128, seed=0,
          disable_log_stats=True)
base = "The quick brown fox jumps over the lazy dog. "
prompts = [base * (1 + (i % 4)) + "Then, number %d:" % i for i in range(10)]
params = [SamplingParams(temperature=0.0, max_tokens=24 - 4 * (i % 3), ignore_eos=True)
          for i in range(10)]
llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
             use_tqdm=False)
res["run1"] = dump(llm.generate(prompts, params, use_tqdm=False))
del llm

# ------------------------------------------------------------------ run 2 ---
PREFIX_LOG.clear()
llm = LLM(model="facebook/opt-125m", dtype="bfloat16", enforce_eager=True,
          max_model_len=1024, max_num_seqs=16, max_num_batched_tokens=1024, seed=0,
          enable_prefix_caching=True, block_size=16, disable_log_stats=True)
SYSTEM = (
    "You are a careful assistant running on a CPU backend. "
    "Answer briefly, in plain words, and never invent facts. "
    "Keep the tone neutral and the sentences short. "
    "If a question is ambiguous, state the assumption you make. "
    "Always finish with a single concluding sentence. "
)
BODY = ("The quick brown fox jumps over the lazy dog near the quiet river bank. "
        "A slow grey heron watches the water and waits for the evening light. ")
REPEATS = [0, 2, 5, 8, 12, 18]
prompts = [SYSTEM + BODY * REPEATS[i % len(REPEATS)] + "Question %d: what happened next?" % i
           for i in range(12)]
params = [SamplingParams(temperature=0.0, seed=0, ignore_eos=True,
                         max_tokens=8 + (i * 32) // 11) for i in range(12)]
llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
             use_tqdm=False)
PREFIX_LOG.clear()
res["run2"] = dump(llm.generate(prompts, params, use_tqdm=False))
res["prefix_log"] = list(PREFIX_LOG)

with open(sys.argv[1], "w") as f:
    json.dump(res, f, indent=1, sort_keys=True)
print("wrote %s: run1=%d run2=%d prefix_log=%d steps, sum=%d"
      % (sys.argv[1], len(res["run1"]), len(res["run2"]), len(PREFIX_LOG),
         sum(r[2] for r in PREFIX_LOG)))
