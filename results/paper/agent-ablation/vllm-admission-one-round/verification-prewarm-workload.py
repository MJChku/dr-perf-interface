import os
import sys
import time
_HF = os.environ['HF_HOME']
if os.path.isdir(_HF):
    os.environ.setdefault('HF_HOME', _HF)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
os.environ.setdefault('VLLM_CPU_KVCACHE_SPACE', '1')
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import bench_support
st = bench_support.states(reqs=16, max_tokens=4, passes='16,16', model='facebook/opt-125m')
REQS, MAX_TOKENS, MODEL = (int(st['reqs']), int(st['max_tokens']), str(st['model']))
PASSES = [int(x) for x in str(st['passes']).split(',') if x.strip()]
import vllm
from vllm import LLM, SamplingParams
_SRC = os.environ['BENCH_SOURCE']
if os.path.abspath(os.path.dirname(vllm.__file__)) != _SRC:
    sys.exit('vllm_run_req.py: vllm imported from %s, not the prepared source %s (put %s first on PYTHONPATH)' % (vllm.__file__, _SRC, os.path.dirname(_SRC)))
llm = LLM(model=MODEL, dtype='bfloat16', enforce_eager=True, max_model_len=768, max_num_seqs=16, max_num_batched_tokens=1024, seed=0, disable_log_stats=True)
pass
from vllm.v1.sample.logits_processor import cached_load_custom_logitsprocs
cached_load_custom_logitsprocs(None)
SENT = 'The quick brown fox jumps over the lazy dog. '
REPS = [1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 48, 54, 60, 66]
prompts = [SENT * REPS[i % len(REPS)] for i in range(REQS)]
params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True, seed=0)
outs = []
times = []
for n in PASSES:
    t = time.time()
    sub = [prompts[i * REQS // n] for i in range(n)]
    outs.append(llm.generate(sub, params, use_tqdm=False))
    times.append(time.time() - t)
ptok = [len(o.prompt_token_ids) for o in outs[0]]
ntok = sum((len(o.outputs[0].token_ids) for o in outs[0]))
print('vllm cpu req: %d requests, prompt tokens %d..%d (total %d), %d generated tokens' % (len(outs[0]), min(ptok), max(ptok), sum(ptok), ntok))
print('passes ' + ', '.join(('%d prompts %.2fs' % (n, t) for n, t in zip(PASSES, times))) + '   total %.2fs' % sum(times))
print('sample:', repr(outs[0][0].outputs[0].text[:60]))
