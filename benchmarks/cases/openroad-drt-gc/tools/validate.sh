#!/bin/sh
# Evaluator tool: run drperf on the reference annotation, original and fixed,
# small and large, and keep what it reports.
#
#   tools/validate.sh [drt-gc-001 ...]        default: all four cases
#
# Writes, per case, reference/assets/drperf/:
#   {original,fixed}-{small,large}.txt          bin/drperf CMD (the formula as printed)
#   {original,fixed}-{small,large}.derive.txt   bin/drperf-dev derive (block counts, irregular share)
#   {original,fixed}-predict.txt                small formula evaluated at the large run's states
# Raw runs go to out/ (not kept). "original" is the reference build with its gate
# unset, "fixed" the fixed build with the gate set.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
DRPERF=$ROOT/bin/drperf
DEV=$ROOT/bin/drperf-dev
cases=${*:-"drt-gc-001 drt-gc-002 drt-gc-003 drt-gc-004"}

for c in $cases; do
  case $c in
    drt-gc-001) scenario=pgvias gate=GC_CUTCORNERCACHE ;;
    drt-gc-002) scenario=pins gate=GC_FIXEDPOLYS ;;
    drt-gc-003) scenario=mix gate=GC_EMPTYLAYERS ;;
    drt-gc-004) scenario=drnets gate=GC_NETCTOR ;;
  esac
  [ -x "$HERE/build/$c/fixed/gcbench" ] || "$HERE/build.sh" $c fixed > /dev/null
  [ -x "$HERE/build/$c/reference/gcbench" ] || "$HERE/build.sh" $c reference > /dev/null
  keep=$HERE/$c/reference/assets/drperf
  mkdir -p "$keep"
  for arm in original fixed; do
    if [ $arm = original ]; then
      bin=$HERE/build/$c/reference/gcbench on=0
    else
      bin=$HERE/build/$c/fixed/gcbench on=1
    fi
    for size in small large; do
      cmd="$bin scenario=$scenario size=$size"
      echo "$c $arm $size: $gate=$on $DRPERF $(basename "$(dirname "$bin")")/gcbench scenario=$scenario size=$size"
      {
        echo "\$ $gate=$on bin/drperf build/$c/$(basename "$(dirname "$bin")")/gcbench scenario=$scenario size=$size"
        env $gate=$on "$DRPERF" $cmd
      } > "$keep/$arm-$size.txt"
      raw=$HERE/out/$c/$arm-$size
      mkdir -p "$HERE/out/$c"
      # shellcheck disable=SC2086
      env $gate=$on "$DEV" run --late --blocks -q --force --repeat 1 --no-native \
        -o "$raw" -- $cmd > "$raw.log" 2>&1 || { cat "$raw.log"; exit 1; }
      "$DEV" derive "$raw" --top 4 | sed "s|$HERE/||g" > "$keep/$arm-$size.derive.txt"
      grep -m1 "cost(" "$keep/$arm-$size.derive.txt" | sed 's/^ */    /'
    done
    "$DEV" derive "$HERE/out/$c/$arm-small" --top 4 --predict "$HERE/out/$c/$arm-large" | sed "s|$HERE/||g" \
      > "$keep/$arm-predict.txt"
  done
done

"$HERE/tools/provenance.py"
