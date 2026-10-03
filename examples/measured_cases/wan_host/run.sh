#!/bin/bash
# Wan 2.1 host side: the rotary table is rebuilt on every forward, the prompt re-projected every step and CFG branch, the VAE decode re-concatenated per frame.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=wan_host
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_wan
GRID="${QUICK:+--state frames=5,9 --state steps=4,6 --state hw=128,192}"; GRID="${GRID:---state frames=5,9,17 --state steps=4,6,8 --state hw=128,192,256}"
BT="${QUICK:+--state batch=1,2,3 --state text=32,64}"; BT="${BT:---state batch=1,2,3,4 --state text=32,64,96}"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$PERFMARK_PATH
    dr_run $case-$v --late $GRID -- "$WAN_PY" "$HERE/driver.py"          # shape grid: rope
    dr_run $case-$v-bt --late $BT -- "$WAN_PY" "$HERE/driver.py"         # batch/text grid: cond_embed
done
compare $case-base $case-fix rope
compare $case-base-bt $case-fix-bt cond_embed
percall $case-base $case-fix rope cond_embed
echo "== equivalence: sha256 of latents and decoded video at several grid points, natively"
for v in base fix; do PYTHONPATH=$PERFMARK_PATH "$WAN_PY" "$HERE/equiv.py" "$WORK/$case-$v" > "$OUT/$case-$v.equiv" 2> "$OUT/$case-$v.native.log"; done
dump_equal "$OUT/$case-base.equiv" "$OUT/$case-fix.equiv"
