#!/bin/bash
# vLLM case A: the model runner's per-step fixed tax (copy_to_gpu copies tensors onto themselves on the CPU backend) and the per-step rebuild of cached request data.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=vllm_prepare_inputs
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_vllm
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    dr_run $case-$v -- "$VLLM_PY" "$HERE/driver.py" reqs=10
done
compare $case-base $case-fix prepare_inputs cached_request_data update_states
echo "== equivalence: text, token ids, cumulative logprob, finish reason of both workloads, natively"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    "$VLLM_PY" "$HERE/verify.py" "$OUT/$case-$v.json" > "$OUT/$case-$v.native.log" 2>&1
done
dump_equal "$OUT/$case-base.json" "$OUT/$case-fix.json"
