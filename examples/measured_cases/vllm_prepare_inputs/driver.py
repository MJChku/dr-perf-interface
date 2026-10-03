"""Drive vLLM's CPU backend with a tiny model so its marked internals trigger
many times with varying state (batch of requests with different prompt and
output lengths).  Adapted from /home/ubuntu/drperf/examples/vllm_cpu/run.py;
the markers live in the marked COPY of the vLLM package at
/home/ubuntu/drperf-cases/vllm-cpu-rb/vllm, which must be first on PYTHONPATH:

    export PYTHONPATH=/home/ubuntu/drperf-cases/vllm-cpu-rb:/home/ubuntu/drperf/perfmark/python:/home/ubuntu/drperf/build
    export OMP_NUM_THREADS=8 VLLM_CPU_OMP_THREADS_BIND=all
    /home/ubuntu/drperf/third_party/vllm-cpu/.venv/bin/python examples/vllm_run_rb.py [reqs=6] [max_tokens=24]

max_num_seqs=12 so that reqs=10 (out/vllm_marked) all run concurrently; prompts are
16..46 tokens and outputs 16..24 tokens, well inside max_model_len=256.

Under drperf (from /home/ubuntu/drperf-cases, same env exported):

    /home/ubuntu/drperf/bin/drperf-dev run --blocks -q --threads 8 --repeat 1 -o out/vllm_base \\
        -- /home/ubuntu/drperf/third_party/vllm-cpu/.venv/bin/python examples/vllm_run_rb.py reqs=6
    /home/ubuntu/drperf/bin/drperf-dev derive out/vllm_base ; /home/ubuntu/drperf/bin/drperf-dev learn out/vllm_base
"""
import os
import sys
import time

# drperf's own model cache holds facebook/opt-125m (absolute path: this file
# no longer lives inside the drperf tree)
_HF = "/home/ubuntu/drperf/third_party/hf"
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", _HF)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # keep the engine in this process
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")             # GB
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import perfmark  # noqa: E402

st = perfmark.states(reqs=6, max_tokens=24, model="facebook/opt-125m")
REQS, MAX_TOKENS, MODEL = int(st["reqs"]), int(st["max_tokens"]), str(st["model"])

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

_SRC = os.path.join(os.environ.get("CASE_TREE", "/home/ubuntu/drperf-cases/vllm-cpu-rb"), "vllm")
if not os.path.abspath(os.path.dirname(vllm.__file__)) == _SRC:
    sys.exit("vllm_run_rb.py: vllm imported from %s, not the marked copy %s (put %s first on PYTHONPATH)"
             % (vllm.__file__, _SRC, os.path.dirname(_SRC)))

llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_model_len=256,
          max_num_seqs=12, max_num_batched_tokens=128, seed=0, disable_log_stats=True)

# Late attach (`drperf run --late`): DynamoRIO attaches at the first marker, so
# the model load above runs natively.  Stateless on purpose: the first trigger
# of a late-attached run is recorded without its states.
with perfmark.region("vllm_attach"):
    pass

base = "The quick brown fox jumps over the lazy dog. "
prompts = [base * (1 + (i % 4)) + "Then, number %d:" % i for i in range(REQS)]
params = [SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS - 4 * (i % 3), ignore_eos=True)
          for i in range(REQS)]

# warm-up outside any measured intent (markers inside vLLM still fire; they are
# just earlier triggers in the trace)
llm.generate([prompts[0]], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), use_tqdm=False)

t0 = time.time()
outs = llm.generate(prompts, params, use_tqdm=False)
dt = time.time() - t0
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
print("vllm cpu: %d requests, %d generated tokens in %.1fs (%.1f tok/s)" % (REQS, ntok, dt, ntok / max(dt, 1e-9)))
print("sample:", repr(outs[0].outputs[0].text[:60]))
