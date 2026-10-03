#!/bin/sh
# Build gcbench for one case and variant into build/<case>/<variant>/gcbench.
#
#   ./build.sh drt-gc-001 [neutral|reference|fixed|check]
#   ./build.sh all                 every case and variant
#
#   neutral    src/ + region.patch                         (empty marker)
#   reference  src/ + region.patch + reference/assets/pcvs.patch
#   fixed      src/ + region.patch + pcvs.patch + reference/assets/fix.patch
#   check      src/ + region.patch + fix.patch             (no states; for tests)
#
# The fix is gated at run time, as upstream: GC_CUTCORNERCACHE (001),
# GC_FIXEDPOLYS (002), GC_EMPTYLAYERS (003), GC_NETCTOR (004); unset = original.
# Needs g++ (C++17), Boost.Polygon headers and the drperf build (libperfmark).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:-"-std=c++17 -O3 -DNDEBUG -Wall -Wp,-D_GLIBCXX_ASSERTIONS -flto=auto"}
[ -f "$ROOT/build/libperfmark.so" ] || { echo "build.sh: build drperf first ($ROOT/build.sh)" >&2; exit 1; }

build() {
  case=$1 variant=$2
  dir=$HERE/build/$case/$variant
  rm -rf "$dir"
  mkdir -p "$dir"
  cp -r "$HERE/src" "$dir/src"
  patches="$HERE/$case/region.patch"
  ref=$HERE/$case/reference/assets
  case $variant in
    neutral) ;;
    reference) patches="$patches $ref/pcvs.patch" ;;
    fixed) patches="$patches $ref/pcvs.patch $ref/fix.patch" ;;
    check) patches="$patches $ref/fix.patch" ;;
    *) echo "build.sh: unknown variant $variant" >&2; exit 2 ;;
  esac
  for p in $patches; do
    patch -s -d "$dir" -p1 -F0 < "$p"
  done
  # shellcheck disable=SC2086
  $CXX $CXXFLAGS ${EXTRA_INCLUDE:+-I"$EXTRA_INCLUDE"} -I"$dir/src" \
    -I"$ROOT/benchmarks/regions/support" -I"$ROOT/perfmark" \
    -o "$dir/gcbench" "$dir"/src/*.cpp \
    -L"$ROOT/build" -lperfmark -Wl,-rpath,"$ROOT/build"
  echo "$dir/gcbench"
}

if [ "${1:-all}" = all ]; then
  fail=0
  for c in drt-gc-001 drt-gc-002 drt-gc-003 drt-gc-004; do
    pids=""
    for v in neutral reference fixed check; do
      build $c $v &
      pids="$pids $!"
    done
    for pid in $pids; do
      wait "$pid" || fail=1
    done
  done
  exit $fail
else
  build "$1" "${2:-neutral}"
fi
