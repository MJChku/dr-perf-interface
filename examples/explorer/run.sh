#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
OUT=${DRPERF_EXPLORER_OUT:-out/explorer-demo}
mkdir -p "$OUT"
# Each invocation measures a fresh run; previous reports stay intact until the
# replacement has completed. Do not silently mix old/new binaries in one fit.
RAW=$(mktemp -d "$OUT/.raw.XXXXXX")
trap 'rm -rf "$RAW"' EXIT HUP INT TERM
gcc -O2 -pthread examples/explorer/pipeline.c -Lbuild -lperfmark \
  -Wl,-rpath,"$PWD/build" -o "$OUT/pipeline"
PYTHONPATH="$PWD/lib" python3 - "$OUT" "$RAW" <<'PY'
import os,sys
import runner
out=sys.argv[1]
os.environ['DRPERF_FOLLOW_THREADS']='0'
rc,log,files=runner.run([os.path.abspath(out+'/pipeline'), os.environ.get('DRPERF_EXPLORER_SCALE','1'),
                       os.environ.get('DRPERF_EXPLORER_REFINED','0'),
                       os.environ.get('DRPERF_EXPLORER_TOKEN_MULTIPLIER','2')],sys.argv[2],timeout=60)
print(log)
if rc or not files: raise SystemExit(rc or 1)
PY
bin/drperf-export "$RAW" --source-root . --source examples/explorer/pipeline.c \
  -o "$OUT/pipeline.drperf.json"
