#!/bin/bash
# Wan 2.1, the per-step host constant: cross-attention re-projects the fixed prompt in every block of every forward (3,000x per video instead of 60x).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=wan_xattn
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_wan
GRID="${QUICK:+--state frames=5,9 --state steps=4,6 --state hw=128,192}"; GRID="${GRID:---state frames=5,9,17 --state steps=4,6,8 --state hw=128,192,256}"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$PERFMARK_PATH
    dr_run $case-$v --late $GRID -- "$WAN_PY" "$HERE/driver.py"
done
compare $case-base $case-fix attn block transformer_forward denoise_step
percall $case-base $case-fix attn block
echo "== equivalence: sha256 of latents and decoded video at several grid points, natively"
for v in base fix; do PYTHONPATH=$PERFMARK_PATH "$WAN_PY" "$HERE/equiv.py" "$WORK/$case-$v" > "$OUT/$case-$v.equiv" 2> "$OUT/$case-$v.native.log"; done
dump_equal "$OUT/$case-base.equiv" "$OUT/$case-fix.equiv"
