#!/bin/bash
# Wan 2.1 text encoder: the prompt is padded to max_sequence_length and the encoder's cost follows the padding, quadratically; the real prompt is free.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib.sh"
case=wan_t5
[ -d "$WORK/$case-base" ] && [ -d "$WORK/$case-fix" ] || "$MC/prepare.sh" $case
need_wan
G="${QUICK:+--state mode=6 --state rtok=8,128 --state maxlen=64,512 --state batch=1}"
G="${G:---state mode=6 --state rtok=8,32,128,480 --state maxlen=64,128,256,512 --state batch=1,2}"
for v in base fix; do
    export CASE_TREE=$WORK/$case-$v PYTHONPATH=$PERFMARK_PATH
    dr_run $case-$v --late $G -- "$WAN_PY" "$HERE/driver.py"
done
compare $case-base $case-fix t5_enc
percall_incl $case-base $case-fix t5_enc
echo "== equivalence: prompt embeddings, latents and video over a spread of (batch, max_len, real tokens), natively"
for v in base fix; do PYTHONPATH=$PERFMARK_PATH "$WAN_PY" "$HERE/equiv.py" "$WORK/$case-$v" "$OUT/$case-$v.npz" > "$OUT/$case-$v.equiv" 2> "$OUT/$case-$v.native.log"; done
PYTHONPATH=$PERFMARK_PATH "$WAN_PY" "$HERE/cmp_t5.py" "$OUT/$case-base.npz" "$OUT/$case-fix.npz"
