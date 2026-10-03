#!/bin/sh
# Evaluator tool (uses the reference fixes): check that each fix leaves the
# result of worker setup unchanged.
#
#   tools/equivalence.sh            (runs ./build.sh all first if needed)
#
# For each case, the `check` build (src + region.patch + fix.patch) runs every
# scenario with its gate off and on; stdout (totals and the digest of all pins,
# edges, corners with fixed flags, maximal rectangles with fixed and tapered
# flags, special spacing rectangles and the per-layer shapes of every net) must
# be identical, and identical to the neutral build. For the small sizes the
# full dumps are compared byte for byte as well. The random scenario draws
# overlapping, touching, repeated and ring-shaped shapes, fixed and routed, on
# 11, 21 and 70-layer stacks.
#
# Also checks that the corner-vertex cache of src/ gives the same result as
# upstream's per-corner isPolygonCorner() (-DGC_UPSTREAM_CORNER_LOOKUP).
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
B=$HERE/build
T=$B/equivalence
mkdir -p "$T"
[ -x "$B/drt-gc-004/check/gcbench" ] || "$HERE/build.sh" all > /dev/null
fail=0
runs=0

gate_of() {
  case $1 in
    drt-gc-001) echo GC_CUTCORNERCACHE ;;
    drt-gc-002) echo GC_FIXEDPOLYS ;;
    drt-gc-003) echo GC_EMPTYLAYERS ;;
    drt-gc-004) echo GC_NETCTOR ;;
  esac
}

compare() {  # compare LABEL FILE_A FILE_B
  runs=$((runs + 1))
  if cmp -s "$2" "$3"; then
    echo "  same $1"
  else
    echo "  DIFF $1"
    fail=1
  fi
}

for c in drt-gc-001 drt-gc-002 drt-gc-003 drt-gc-004; do
  gate=$(gate_of $c)
  echo "$c ($gate)"
  for s in random pgvias pins mix drnets; do
    for z in tiny small large; do
      [ $s != random ] && [ $z = tiny ] && continue
      dump_off="" dump_on=""
      if [ $z = small ]; then
        dump_off="dump=$T/$c-$s-off.txt" dump_on="dump=$T/$c-$s-on.txt"
      fi
      env $gate=0 "$B/$c/check/gcbench" scenario=$s size=$z $dump_off > "$T/$c-$s-$z-off.out"
      env $gate=1 "$B/$c/check/gcbench" scenario=$s size=$z $dump_on > "$T/$c-$s-$z-on.out"
      "$B/$c/neutral/gcbench" scenario=$s size=$z > "$T/$c-$s-$z-neutral.out"
      compare "$s $z: gate on vs off ($(sed -n 's/^digest //p' "$T/$c-$s-$z-on.out"))" \
        "$T/$c-$s-$z-off.out" "$T/$c-$s-$z-on.out"
      compare "$s $z: gate off vs neutral build" "$T/$c-$s-$z-off.out" "$T/$c-$s-$z-neutral.out"
      if [ $z = small ]; then
        compare "$s $z: full dump ($(wc -c < "$T/$c-$s-on.txt") bytes)" \
          "$T/$c-$s-off.txt" "$T/$c-$s-on.txt"
      fi
    done
  done
done

echo "baseline: corner-vertex cache vs upstream per-corner lookup"
mkdir -p "$T/upstream"
# shellcheck disable=SC2086
${CXX:-g++} -std=c++17 -O3 -DNDEBUG -Wp,-D_GLIBCXX_ASSERTIONS -flto=auto \
  -DGC_UPSTREAM_CORNER_LOOKUP -I"$HERE/src" -o "$T/upstream/gcbench" "$HERE"/src/*.cpp
for s in random pgvias pins mix drnets; do
  "$T/upstream/gcbench" scenario=$s size=small > "$T/upstream-$s.out"
  "$B/drt-gc-001/neutral/gcbench" scenario=$s size=small > "$T/cache-$s.out"
  compare "$s small: upstream lookup vs cache" "$T/upstream-$s.out" "$T/cache-$s.out"
done

if [ $fail = 0 ]; then
  echo "equivalent: $runs comparisons identical"
fi
exit $fail
