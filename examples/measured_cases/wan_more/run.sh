#!/bin/bash
# Wan 2.1, second round: the VAE posterior computes exp over the latent for values never read, callbacks call locals() per name, the rotary application allocates temporaries.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=wan_more
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_wan
G="${QUICK:+--state hw=128,192 --state frames=5,9}"; G="${G:---state hw=128,192,256 --state frames=5,9,17}"
CBG="--state cb=1,2,3"                 # callback_on_step_end_tensor_inputs[:cb] -> ninputs
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$PERFMARK_PATH
    dr_run $case-$v-t2v --late $G $CBG -- "$WAN_PY" "$HERE/driver.py" mode=0
    dr_run $case-$v-enc --late $G -- "$WAN_PY" "$HERE/driver.py" mode=1
done
compare $case-base-t2v $case-fix-t2v attn_rope callback_step
compare $case-base-enc $case-fix-enc dgd_init
echo "== equivalence: rotary application, VAE encode statistics, tiled decode/encode, post-processing, natively"
for v in base fix; do PYTHONPATH=$PERFMARK_PATH "$WAN_PY" "$HERE/equiv.py" "$WORK/$case-$v" > "$OUT/$case-$v.equiv" 2> "$OUT/$case-$v.native.log"; done
dump_equal "$OUT/$case-base.equiv" "$OUT/$case-fix.equiv"
