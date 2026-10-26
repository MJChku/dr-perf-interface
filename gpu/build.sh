#!/bin/sh
# Independent GPU build: no DynamoRIO, CUDA toolkit, or GX checkout required.
set -eu
cd "$(dirname "$0")/.."
mkdir -p gpu/build build
gpu_root=$(pwd)/gpu
cxx=${CXX:-g++}
# An unrelated /usr/local Boost can shadow distro headers while the linker
# selects distro libraries. Keep the header/library pair explicit.
if [ -n "${DRPERF_GPU_BOOST_INCLUDE:-}" ]; then
  boost_include=$DRPERF_GPU_BOOST_INCLUDE
else
  test -f /usr/include/boost/context/fiber.hpp || {
    echo 'Install libboost-context-dev and libboost-fiber-dev (or set DRPERF_GPU_BOOST_INCLUDE)' >&2
    exit 1
  }
  mkdir -p gpu/build/system-include
  ln -sfn /usr/include/boost gpu/build/system-include/boost
  boost_include=$gpu_root/build/system-include
fi
gpu_tmp=$(mktemp -d "$gpu_root/build/.build-XXXXXX")
trap 'rm -rf "$gpu_tmp"' EXIT HUP INT TERM
"$cxx" -std=c++17 -O2 -fPIC -shared -fno-strict-aliasing -fwrapv \
  -I"$boost_include" -Igpu/include -Igpu/vendor/gx/include \
  gpu/runtime/runtime.cc gpu/vendor/gx/mma.cc gpu/vendor/gx/float_arith.cc \
  gpu/vendor/gx/tcgen05.cc \
  gpu/vendor/gx/wgmma.cc gpu/vendor/gx/tensor_map.cc gpu/vendor/gx/bulk_copy.cc \
  gpu/vendor/gx/host_calls.cc \
  -lboost_context -lboost_fiber -pthread -ldl -Wl,-z,defs \
  -o "$gpu_tmp/libdrperf_gpu.so"
mv "$gpu_tmp/libdrperf_gpu.so" gpu/build/libdrperf_gpu.so
"$cxx" -O3 -std=c++17 -ffp-contract=off -fPIC -shared -Wall -Wextra -Werror \
  analysis/affine.cpp -o "$gpu_tmp/libdrperf_fit.so"
mv "$gpu_tmp/libdrperf_fit.so" build/libdrperf_fit.so
cd gpu/vendor/gxptx
LEAN_NUM_THREADS=${LEAN_NUM_THREADS:-4} lake build gxptx
printf '%s\n' 'drperf GPU build complete (separate collector; no DynamoRIO build)'
