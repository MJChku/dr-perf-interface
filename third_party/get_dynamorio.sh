#!/bin/sh
# Build the pinned GX DynamoRIO fork. No local GX checkout or root required.
set -eu
cd "$(dirname "$0")"
ROOT=$PWD
REV=7c0717f412f8aae9199b49aff95557ada2cf3413
SHORT=7c0717f412f8
REPO=https://github.com/MJChku/dynamorio.git
SRC=$ROOT/DynamoRIO-src-$SHORT
BUILD=$ROOT/DynamoRIO-build-$SHORT
INSTALL=$ROOT/DynamoRIO-gx-$SHORT
JOBS=${DRPERF_BUILD_JOBS:-8}
if [ ! -f "$INSTALL/.drperf-revision" ] ||
   [ "$(cat "$INSTALL/.drperf-revision")" != "$REV" ] ||
   [ ! -x "$INSTALL/bin64/drrun" ]; then
  if [ ! -d "$SRC/.git" ]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin "$REPO"
  fi
  git -C "$SRC" fetch --depth 1 origin "$REV"
  git -C "$SRC" checkout --detach "$REV"
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$REV" ]
  # Never build silently from edits left in this dependency checkout.
  git -C "$SRC" diff --exit-code
  git -C "$SRC" diff --cached --exit-code
  git -C "$SRC" submodule update --init --recursive --depth 1
  cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_CLIENTS=OFF -DBUILD_DOCS=OFF -DBUILD_SAMPLES=OFF -DBUILD_TESTS=OFF
  cmake --build "$BUILD" -j "$JOBS"
  cmake --install "$BUILD" --prefix "$INSTALL"
  printf '%s\n' "$REV" > "$INSTALL/.drperf-revision"
fi
# Switch only after the replacement runtime has built successfully.
ln -sfn "DynamoRIO-gx-$SHORT" dynamorio
"$INSTALL/bin64/drrun" -version
