#!/bin/sh
# Correctness and reachability checks for the neutral cases (no PCVs, no fix).
#
#   tests/run_tests.sh [drt-gc-001 ...]      default: all four cases
#
# For each case: build src/ + region.patch with the recording marker in
# tests/probe, run the bounded (tiny) workloads, and check
#  - the totals of pins, edges, corners and maximal rectangles, and how many
#    of them are fixed, against counts derived by hand from the generator's
#    geometry (see the comments below; the random scenario's are recorded),
#  - the digest of everything setup built against the recorded value,
#  - that the case's marker was entered once per net or once per worker.
# Native run; the marker probe forwards to perfmark, so the binary also runs
# under drperf.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:-"-std=c++17 -O3 -DNDEBUG -Wall -Wp,-D_GLIBCXX_ASSERTIONS -flto=auto"}
cases=${*:-"drt-gc-001 drt-gc-002 drt-gc-003 drt-gc-004"}
fail=0

check() {  # check LABEL GOT WANT
  if [ "$2" = "$3" ]; then
    echo "  ok   $1"
  else
    echo "  FAIL $1: got '$2', want '$3'"
    fail=1
  fi
}

# expected "nets pins edges fixed-edges corners fixed-corners maxrects fixed-maxrects"
# for scenario=S size=tiny repeat=4 (each grid point runs 4 times):
#  pgvias  k fixed cuts x r routed cuts in {1,3}x{0,2}, 3 power nets + 2 floating
#          supply nets per worker: pins 3 rails + 2(k+r) cuts, all 4-vertex;
#          fixed: rails and fixed cuts (routed via enclosures merge into rails)
#  pins    p cell pins x o macro rects in {1,2}x{1,2}, 2 nets + 2 per worker:
#          pins 3p+o, edges 14p+4o (fixed 6p+4o), maxrects 4p+o (fixed 2p+o)
#  mix     s in {1,2} routed nets (1 wire, 1 pin rect) + 1 empty + 2 per worker:
#          4 pins per routed net, 4 fixed edges and 1 fixed maxrect per net
#  drnets  s in {1,2} routed nets (1 wire, no metal3 wire), 1 design net with
#          2 pins, 1 empty + 2 per worker: pins 6 + 3s, fixed edges 12 per worker
expected_totals() {
  case $1 in
    pgvias) echo "80 432 1728 1344 1728 1344 432 336" ;;
    pins) echo "64 192 864 480 864 480 240 144" ;;
    mix) echo "36 48 192 48 192 48 48 12" ;;
    drnets) echo "44 84 368 96 368 96 100 32" ;;
    random) echo "153 769 3900 2763 3900 2797 1289 915" ;;
  esac
}
expected_digest() {
  case $1 in
    pgvias) echo bd529281d5885b29 ;;
    pins) echo 9eab96c8ec2dc4f3 ;;
    mix) echo 3d63a6048ee99e0d ;;
    drnets) echo a48046826df2de6d ;;
    random) echo 3244772730ba6880 ;;
  esac
}

for c in $cases; do
  echo "$c"
  dir=$HERE/build/tests/$c
  rm -rf "$dir"
  mkdir -p "$dir"
  cp -r "$HERE/src" "$dir/src"
  patch -s -d "$dir" -p1 -F0 < "$HERE/$c/region.patch"
  # shellcheck disable=SC2086
  $CXX $CXXFLAGS -I"$HERE/tests/probe" -I"$dir/src" -I"$ROOT/perfmark" \
    -o "$dir/gcbench" "$dir"/src/*.cpp -L"$ROOT/build" -lperfmark -Wl,-rpath,"$ROOT/build"
  for s in pgvias pins mix drnets random; do
    out=$("$dir/gcbench" scenario=$s size=tiny 2> "$dir/$s.err")
    got=$(echo "$out" | sed -n 's/^nets \([0-9]*\) pins \([0-9]*\) edges \([0-9]*\) (fixed \([0-9]*\)) corners \([0-9]*\) (fixed \([0-9]*\)) maxrects \([0-9]*\) (fixed \([0-9]*\))$/\1 \2 \3 \4 \5 \6 \7 \8/p')
    check "$s totals" "$got" "$(expected_totals $s)"
    check "$s digest" "$(echo "$out" | sed -n 's/^digest //p')" "$(expected_digest $s)"
    workers=$(echo "$out" | sed -n 's/.*: \([0-9]*\) workers$/\1/p')
    nets=$(echo "$got" | cut -d' ' -f1)
    hits=$(sed -n "s/^marker_hits $c \([0-9]*\)$/\1/p" "$dir/$s.err")
    case $c in
      drt-gc-001 | drt-gc-002) want=$nets ;;   # once per net
      drt-gc-003 | drt-gc-004) want=$workers ;; # once per worker
    esac
    check "$s marker $c entered" "${hits:-0}" "$want"
  done
done
[ $fail = 0 ] && echo "all checks passed"
exit $fail
