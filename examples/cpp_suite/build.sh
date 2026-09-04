#!/bin/sh
# Build the drperf C++ suite into build/ (relative to this directory).
set -eu
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
mkdir -p build
CXX=${CXX:-g++}
LIBS="-L$ROOT/build -lperfmark -Wl,-rpath,$ROOT/build"
$CXX -O2 -o build/c1_exact c1_exact.cpp $LIBS
$CXX -O2 -o build/c2_stl c2_stl.cpp $LIBS
$CXX -O2 -o build/c3_virtual c3_virtual.cpp $LIBS
$CXX -O2 -o build/c4_undeclared c4_undeclared.cpp $LIBS
$CXX -O2 -o build/c5_regimes c5_regimes.cpp $LIBS
$CXX -O2 -fopenmp -o build/c6_openmp c6_openmp.cpp $LIBS
$CXX -O2 -pthread -o build/c7_queue c7_queue.cpp $LIBS
$CXX -O2 -o build/c8_base c8_delta.cpp $LIBS
$CXX -O2 -DHEAD -o build/c8_head c8_delta.cpp $LIBS
$CXX -O0 -o build/c9_O0 c9_optlevel.cpp $LIBS
$CXX -O2 -o build/c9_O2 c9_optlevel.cpp $LIBS
ls build
