"""Drive vLLM's CPU backend with a tiny model so its marked internals trigger
many times with varying state (batch of requests with different prompt and
output lengths).  Markers live in the vLLM source (see markers.patch).

    third_party/vllm-cpu/.venv/bin/python examples/vllm_cpu/run.py [reqs=6] [max_tokens=24]

Under drperf:

    bin/drperf run -o out/vllm --threads 4 --repeat 1 --no-native \\
        -- third_party/vllm-cpu/.venv/bin/python examples/vllm_cpu/run.py reqs=6
    bin/drperf fit out/vllm ; bin/drperf learn out/vllm ; bin/drperf trace out/vllm
"""
import os
import time

_HF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "hf")
if os.path.isdir(_HF):
    os.environ.setdefault("HF_HOME", os.path.abspath(_HF))   # drperf's own model cache
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # keep the engine in this process
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")             # GB
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import perfmark  # noqa: E402

st = perfmark.states(reqs=6, max_tokens=24, model="facebook/opt-125m")
REQS, MAX_TOKENS, MODEL = int(st["reqs"]), int(st["max_tokens"]), str(st["model"])

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_model_len=256,
          max_num_seqs=8, max_num_batched_tokens=128, seed=0, disable_log_stats=True)

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
