#!/bin/bash
# run_all.sh [CASE...]  -- run every case (or the named ones); QUICK=1 shrinks the Wan grids
set -uo pipefail
MC=$(cd "$(dirname "$0")" && pwd)
ALL="vllm_admission vllm_prepare_inputs vllm_input_batch vllm_output_path vllm_logprobs wan_host wan_more wan_xattn wan_t5"
[ $# -gt 0 ] || set -- $ALL
rc=0
for c in "$@"; do
    echo "################ $c"
    "$MC/$c/run.sh" || { echo "!! $c failed"; rc=1; }
done
exit $rc
