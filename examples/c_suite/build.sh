#!/bin/sh
# Build the drperf C suite into examples/c_suite/bin (needs ../../build/libperfmark.so).
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
mkdir -p bin
CF="-O2 -pthread -Wall -Wl,-rpath,$ROOT/build -L$ROOT/build -lperfmark"
for c in c1_affine c2_rep c3_quadratic c4_undeclared c5_regimes c6_threads c6_queue c8_predict; do
  gcc -O2 -pthread -Wall -o bin/$c $c.c -L"$ROOT/build" -lperfmark -Wl,-rpath,"$ROOT/build"
done
gcc -O2 -Wall -o bin/c7_base c7_delta.c -L"$ROOT/build" -lperfmark -Wl,-rpath,"$ROOT/build"
gcc -O2 -Wall -DHEAD -o bin/c7_head c7_delta.c -L"$ROOT/build" -lperfmark -Wl,-rpath,"$ROOT/build"
ls bin
