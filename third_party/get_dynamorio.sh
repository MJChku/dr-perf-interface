#!/bin/sh
# Fetch the pinned DynamoRIO release into third_party/ (no root needed).
set -eu
cd "$(dirname "$0")"
VER=11.3.0
TAG=release_11.3.0-1
SHA=ae02049df5c4daeb82d4087aaa0d38bf58cdd26b38fb6042dccf27cbc32ceed9
TARBALL=DynamoRIO-Linux-$VER.tar.gz
if [ ! -f "$TARBALL" ]; then
  curl -sSL -o "$TARBALL" "https://github.com/DynamoRIO/dynamorio/releases/download/$TAG/$TARBALL"
fi
echo "$SHA  $TARBALL" | sha256sum -c -
[ -d "DynamoRIO-Linux-$VER-1" ] || tar xzf "$TARBALL"
ln -sfn "DynamoRIO-Linux-$VER-1" dynamorio
dynamorio/bin64/drrun -version
