"""Equivalence check for the output-path fix: run the SAME three phases as
examples/vllm_run_out.py and dump every RequestOutput that comes out, in order,
so `diff` can prove the fix is behaviour-preserving.

Run natively (no drperf):

    export PYTHONPATH=/home/ubuntu/drperf-cases/vllm-cpu-out:/home/ubuntu/drperf/perfmark/python:/home/ubuntu/drperf/build
    /home/ubuntu/drperf/third_party/vllm-cpu/.venv/bin/python examples/vllm_verify_out.py
"""
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

import perfmark  # noqa: E402

st = perfmark.states(reqs=12, model="facebook/opt-125m", phases="ABC")
REQS, MODEL, PHASES = int(st["reqs"]), str(st["model"]), str(st["phases"])

import vllm  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

_SRC = os.path.join(os.environ.get("CASE_TREE", "/home/ubuntu/drperf-cases/vllm-cpu-out"), "vllm")
if os.path.abspath(os.path.dirname(vllm.__file__)) != _SRC:
    sys.exit("vllm_verify_out.py: vllm imported from %s, not the copy %s"
             % (vllm.__file__, _SRC))

llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True, max_model_len=512,
          max_num_seqs=16, max_num_batched_tokens=256, seed=0,
          disable_log_stats=True)

BASE = "The quick brown fox jumps over the lazy dog. "
PROMPTS = [BASE * (1 + (i % 4)) + "Then, number %d:" % i for i in range(REQS)]
TOKS = [64, 8, 32, 4, 48, 16, 56, 12, 40, 24, 60, 6][:REQS]

OUT = []


def sp(max_tokens, kind):
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True,
                          seed=0, output_kind=kind)


def dump(phase, seq, ro):
    """Everything an external consumer of a RequestOutput can observe."""
    co = ro.outputs[0]
    OUT.append(
        "%s #%04d rid=%s fin=%s ncach=%s ncreat=%s kv=%r ec=%r plog=%r "
        "| idx=%d text=%r tok=%s logp=%r cumlp=%r fr=%r sr=%r rexp=%r smask=%r "
        "| prompt=%r ptok=%s"
        % (phase, seq, ro.request_id, ro.finished, ro.num_cached_tokens,
           ro.num_cache_creation_tokens, ro.kv_transfer_params,
           ro.ec_transfer_params, ro.prompt_logprobs,
           co.index, co.text, list(co.token_ids), co.logprobs,
           co.cumulative_logprob, co.finish_reason, co.stop_reason,
           None if co.routed_experts is None else co.routed_experts.tolist(),
           None if co.sampling_mask is None else co.sampling_mask.token_ids,
           ro.prompt, list(ro.prompt_token_ids)))


# warm-up, exactly as in the measured driver (affects prefix cache state)
llm.generate([PROMPTS[0]], sp(2, RequestOutputKind.FINAL_ONLY), use_tqdm=False)

if "A" in PHASES:
    outs = llm.generate(PROMPTS, [sp(t, RequestOutputKind.FINAL_ONLY) for t in TOKS],
                        use_tqdm=False)
    for i, o in enumerate(outs):
        dump("A", i, o)


def drive(kind, tag, base_id):
    engine = llm.llm_engine
    for i, (prompt, mt) in enumerate(zip(PROMPTS, TOKS)):
        engine.add_request(str(base_id + i), prompt, sp(mt, kind))
    seq = 0
    while engine.has_unfinished_requests():
        for out in engine.step():
            dump(tag, seq, out)
            seq += 1


if "B" in PHASES:
    drive(RequestOutputKind.CUMULATIVE, "B", 1000)
if "C" in PHASES:
    drive(RequestOutputKind.DELTA, "C", 2000)

sys.stdout.write("\n".join(OUT) + "\n")
sys.stdout.write("TOTAL %d records\n" % len(OUT))
