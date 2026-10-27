#!/bin/sh
# Build the GX collector without a second PTX emulator or CUDA toolkit.
set -eu
cd "$(dirname "$0")/../.."
mkdir -p gpu/build build
cxx=${CXX:-g++}
build_tmp=$(mktemp -d gpu/build/.gx-build-XXXXXX)
trap 'rm -rf "$build_tmp"' EXIT HUP INT TERM
"$cxx" -std=c++17 -O2 -fPIC -shared -pthread -Wall -Wextra -Werror \
    -Igpu/include gpu/gx/collector.cc -ldl -o "$build_tmp/libdrperf_gpu_gx.so"
mv "$build_tmp/libdrperf_gpu_gx.so" gpu/build/libdrperf_gpu_gx.so
"$cxx" -O3 -std=c++17 -ffp-contract=off -fPIC -shared \
    analysis/affine.cpp -o "$build_tmp/libdrperf_fit.so"
mv "$build_tmp/libdrperf_fit.so" build/libdrperf_fit.so
# Existing perfmark annotations work without building the CPU profiler.
"${CC:-gcc}" -O2 -fPIC -shared perfmark/perfmark.c -ldl -o "$build_tmp/libperfmark.so"
mv "$build_tmp/libperfmark.so" build/libperfmark.so
