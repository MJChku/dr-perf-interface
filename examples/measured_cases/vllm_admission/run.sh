#!/bin/bash
# vLLM case B: admitting a request.  process_inputs' per-token slope is a Python bounds check; the detokenizer primes its stream with the whole prompt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=vllm_admission
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_vllm
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    dr_run $case-$v --late -- "$VLLM_PY" "$HERE/driver.py"
done
compare $case-base $case-fix process_inputs engine_add_request request_init
echo "== equivalence: every generated text and token-id list of both passes, natively"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    DUMP=$OUT/$case-$v.json "$VLLM_PY" "$HERE/driver.py" > "$OUT/$case-$v.native.log" 2>&1
done
dump_equal "$OUT/$case-base.json" "$OUT/$case-fix.json"
