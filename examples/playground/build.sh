#!/bin/sh
# Builds the playground system twice: bin/base and bin/head (head = the same
# program with -DDELTA, a sanitising pass added to parse).  Needs ../../build
# (run ../../build.sh once).
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
mkdir -p bin
CFLAGS="-O2 -g -Wall -I$ROOT/perfmark -L$ROOT/build -lperfmark -lpthread -Wl,-rpath,$ROOT/build"
gcc -o bin/base system.c $CFLAGS
gcc -DDELTA -o bin/head system.c $CFLAGS
echo "built examples/playground/bin/base and bin/head"
