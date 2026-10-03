#!/bin/bash
# Collect only a completed capture; requires the original remote environment.
set -euo pipefail
arm=${1:?usage: collect.sh cpu|remote|pressure}
case "$arm" in
  cpu) run=drperf-lmc-refined4-cpu ;;
  remote) run=drperf-lmc-refined4-remote ;;
  pressure) run=drperf-lmc-refined3-pressure ;;
  *) echo "Unknown capture: $arm" >&2; exit 2 ;;
esac
base=$(cd "$(dirname "$0")" && pwd)
host=icdslab2.epfl.ch
root=/home/ubuntu/ditto_kv_gx
# Names above are fixed, not arbitrary user-supplied shell fragments.
state=$(ssh "$host" "docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' '$run'")
if [ "$state" != 'exited 0' ]; then
  echo "Capture is not successfully complete: $run ($state)" >&2
  exit 1
fi
ssh "$host" "docker run --rm -v '$root/runs/$run:/capture' --entrypoint chmod gx-kimi-k3:latest -R a+rX /capture"
mkdir -p "$base/architecture-$arm"
ssh "$host" "docker inspect --format '{{json .Config.Env}}' '$run'" | python3 -c '
import json,sys
allowed={"DRPERF_DR_OPTS","TOKENIZERS_PARALLELISM","HF_HUB_OFFLINE","DRPERF_MODE","DRPERF_FOLLOW_THREADS","DRPERF_NATIVE_GX","LMC_CPU_GB","SESSIONS","TURNS"}
print(json.dumps({k:v for item in json.load(sys.stdin) for k,_,v in [item.partition("=")] if k in allowed},indent=2))
' > "$base/architecture-$arm/runtime.json"
rsync -az --exclude lmcache_pkg --exclude raw "$host:$root/runs/$run/" "$base/architecture-$arm/"
# Raw files form one capture. A different PID must not leave old runs pooled
# into a new report; only this generated raw directory is mirrored exactly.
mkdir -p "$base/architecture-$arm/raw"
rsync -az --delete "$host:$root/runs/$run/raw/" "$base/architecture-$arm/raw/"
python3 "$base/architecture.py" "$arm"
