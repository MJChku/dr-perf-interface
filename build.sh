#!/bin/sh
# Build everything: perfmark library, drperf DynamoRIO client, C test, Rust example.
set -eu
cd "$(dirname "$0")"
[ -x third_party/dynamorio/bin64/drrun ] || third_party/get_dynamorio.sh
mkdir -p build
gcc -O2 -fPIC -shared -fvisibility=hidden -o build/libperfmark.so perfmark/perfmark.c -ldl
cp build/libperfmark.so perfmark/libperfmark.so
# preloaded so DynamoRIO can be started at the first marked region instead of
# at process start (see README, "Late attach")
gcc -O2 -shared -fPIC -o build/libdrperf_attach.so perfmark/attach.c \
    -Lthird_party/dynamorio/lib64/release -ldynamorio \
    -Wl,-rpath,"$PWD/third_party/dynamorio/lib64/release"
cmake -S client -B client/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build client/build -j8 | grep -v "^\[" || true
gcc -O2 -o build/ctest tests/ctest.c -Lbuild -lperfmark -Wl,-rpath,"$PWD/build"
gcc -O2 -pthread -o build/cthreads tests/cthreads.c -Lbuild -lperfmark -Wl,-rpath,"$PWD/build"
gcc -O2 -pthread -o build/cmarkers tests/cmarkers.c -Lbuild -lperfmark -Wl,-rpath,"$PWD/build"
PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" 2>/dev/null || echo /usr/include/python3.12)
if [ -f "$PYINC/Python.h" ]; then
  gcc -O2 -shared -fPIC -I"$PYINC" -o build/_perfmark.so perfmark/python/_perfmark.c -Lbuild -lperfmark -Wl,-rpath,"$PWD/build"
else
  echo "build.sh: no Python.h under $PYINC; Python markers fall back to ctypes"
fi
if command -v cargo >/dev/null 2>&1 && [ -f examples/rust_delta/Cargo.toml ]; then
  (cd examples/rust_delta && PERFMARK_DIR="$PWD/../../build" cargo build --release --quiet && \
   PERFMARK_DIR="$PWD/../../build" cargo build --release --quiet --features delta --target-dir target-delta)
fi
ls -la build/
