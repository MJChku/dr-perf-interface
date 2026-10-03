#!/bin/bash
# vLLM case C: the persistent input batch under churn -- an arrival converts the prompt element by element, a departure costs per row moved.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=vllm_input_batch
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_vllm
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    dr_run $case-$v --late -- "$VLLM_PY" "$HERE/driver.py"
done
compare $case-base $case-fix add_request refresh_metadata condense update_states
echo "== equivalence: generated text and token ids of all 24 requests, natively"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    "$VLLM_PY" "$HERE/verify.py" "$OUT/$case-$v.json" > "$OUT/$case-$v.native.log" 2>&1
done
dump_equal "$OUT/$case-base.json" "$OUT/$case-fix.json"
echo "== equivalence: randomized differential test of the modified InputBatch against the pristine one"
export CASE_TREE=$WORK/$case-fix PYTHONPATH=$WORK/$case-fix:$PERFMARK_PATH
"$VLLM_PY" "$HERE/check_equiv.py" 2>&1 | tail -3
