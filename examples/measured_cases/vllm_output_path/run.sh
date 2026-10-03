#!/bin/bash
# vLLM case E: the output path -- the surprise was mostly the markers; the real saving is skipping make_request_output's early-return call.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=vllm_output_path
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_vllm
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    dr_run $case-$v --late -- "$VLLM_PY" "$HERE/driver.py" phases=A     # FINAL_ONLY, the path LLM.generate takes
done
compare $case-base $case-fix process_outputs
percall $case-base $case-fix process_outputs    # per call = num_outputs request-steps
echo "== equivalence: every RequestOutput of the three phases, natively"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$WORK/$case-$v:$PERFMARK_PATH
    "$VLLM_PY" "$HERE/verify.py" > "$OUT/$case-$v.dump" 2> "$OUT/$case-$v.native.log"
done
for v in base fix; do grep -v "^\(WARNING\|INFO\|DEBUG\|Loading\)" "$OUT/$case-$v.dump" > "$OUT/$case-$v.records"; done
echo "   $(grep -c . "$OUT/$case-base.records") records"
dump_equal "$OUT/$case-base.records" "$OUT/$case-fix.records"
