#!/bin/bash
# vLLM case H2: one request's logprobs=5 bills every row of the batch with a vocabulary-wide top-k.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=vllm_logprobs
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_vllm
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH PYTHONHASHSEED=0
    dr_run $case-$v --late --state reqs=3,4,5,6,8 -- "$VLLM_PY" "$HERE/driver.py" mode=stop max_tokens=128 lp=1
done
compare $case-base $case-fix gather_logprobs sample
perrun $case-base $case-fix gather_logprobs sample    # the fix makes num_reqs the asking rows: pair runs, not states
echo "== equivalence: token ids, text, stop reasons and the full top-5 logprob dicts, natively"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH PYTHONHASHSEED=0
    SPEC_DUMP=$OUT/$case-$v.json "$VLLM_PY" "$HERE/driver.py" mode=stop max_tokens=128 lp=1 > "$OUT/$case-$v.native.log" 2>&1
done
dump_equal "$OUT/$case-base.json" "$OUT/$case-fix.json"
