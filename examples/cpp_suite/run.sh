#!/bin/sh
# Run the C++ suite under drperf and derive/learn/diff every case.
#   examples/cpp_suite/run.sh [OUTDIR]     (default: examples/cpp_suite/out)
# Expected results are stated at the top of each c*.cpp; see the report that
# accompanied the suite for the observed ones.
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
D=${1:-$PWD/out}
B=$PWD/build
[ -x "$B/c1_exact" ] || ./build.sh
rm -rf "$D"; mkdir -p "$D"
R="$ROOT/bin/drperf run --blocks -q --force"
$R --jobs 6 --repeat 3 -o "$D/c1" --state n=100,200,300,500,800,1000 -- "$B/c1_exact"
$R --jobs 6 --repeat 1 -o "$D/c2" --state n=1000,2000,3000,5000,8000,12000 -- "$B/c2_stl"
$R --jobs 6 --repeat 1 -o "$D/c3" --state n=200,400,600,1000,1400,2000 -- "$B/c3_virtual"
$R --jobs 6 --repeat 1 -o "$D/c4" --state n=137,251,389,523,677,911 -- "$B/c4_undeclared"
$R --jobs 8 --repeat 1 -o "$D/c5" --state n=200,400,600,800,1200,1600,2000,2400 -- "$B/c5_regimes"
$R --jobs 3 --repeat 1 --threads 4 -o "$D/c6_t4" --state n=200,400,800,1200,1600,2000 -- "$B/c6_openmp"
$R --jobs 3 --repeat 1 --threads 8 -o "$D/c6_t8" --state n=200,400,800,1200,1600,2000 -- "$B/c6_openmp"
$R --repeat 1 -o "$D/c7" --state rounds=12 -- "$B/c7_queue"
$R --repeat 1 -o "$D/c7b" --state rounds=20 -- "$B/c7_queue"
$R --jobs 4 --repeat 1 -o "$D/c8_base" --state n=100,200,300,500 --label base -- "$B/c8_base"
$R --jobs 4 --repeat 1 -o "$D/c8_head" --state n=100,200,300,500 --label head -- "$B/c8_head"
$R --jobs 4 --repeat 1 -o "$D/c9_O0" --state n=100,200,300,500 --label O0 -- "$B/c9_O0"
$R --jobs 4 --repeat 1 -o "$D/c9_O2" --state n=100,200,300,500 --label O2 -- "$B/c9_O2"
$R --jobs 4 --repeat 1 -o "$D/c10_train" --state n=100,200,300,500 -- "$B/c1_exact"
$R --jobs 3 --repeat 1 -o "$D/c10_test" --state n=50,1000,4000 -- "$B/c1_exact"
DV="$ROOT/bin/drperf derive"
for c in c1 c2 c3 c4 c5 c8_base c8_head c9_O0 c9_O2; do echo "=== derive $c"; $DV "$D/$c" --top 3; done
echo "=== derive c5 --no-split"; $DV "$D/c5" --no-split --top 2
echo "=== derive c6_t4 --predict c6_t8"; $DV "$D/c6_t4" --top 3 --predict "$D/c6_t8"
echo "=== derive c10_train --predict c10_test"; $DV "$D/c10_train" --top 2 --predict "$D/c10_test"
echo "=== learn c7 / c7b"; "$ROOT/bin/drperf" learn "$D/c7"; "$ROOT/bin/drperf" learn "$D/c7b"
echo "=== check c7b"; "$ROOT/bin/drperf" trace "$D/c7b" --check "consume.q == cum(produce.m) - cum(consume.q)" --limit 1
echo "=== diff c8"; "$ROOT/bin/drperf" diff "$D/c8_base" "$D/c8_head" | head -14
echo "=== diff c9"; "$ROOT/bin/drperf" diff "$D/c9_O0" "$D/c9_O2" | head -14
echo "=== determinism c1 (distinct .blocks digests per grid point; 1 = identical over repeats)"
for p in 0 1 2 3 4 5; do printf "point %d: " $p; md5sum "$D"/c1/run00${p}_r*.json.blocks | awk '{print $1}' | sort -u | wc -l; done
