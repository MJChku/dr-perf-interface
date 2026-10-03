#!/bin/bash
set -eu
arm=${1:-remote_naive}
name=${2:-lmc-waits-baseline}
root=/home/ubuntu/ditto_kv_gx
mkdir -p "$root/runs/$name"
extra_mount=()
if [ -n "${LMC_HELPER_OVERRIDE:-}" ]; then
  extra_mount=(-v "$LMC_HELPER_OVERRIDE:/w/lmc-waits/overlay/lmcache/_drperf.py:ro")
fi
docker run "${extra_mount[@]}" -d --name "$name" --cpus 12 --memory 36g --memory-swap 36g --ipc host \
 --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN \
 -v "$root:/w" -v "${DRPERF_STAGE:-$root/drperf-waits}:/home/ubuntu/drperf:ro" \
 -v "$root/drperf/third_party:/home/ubuntu/drperf/third_party:ro" \
 -v "$root/gxvm_runtime/gx_cuda_mixedfake.so:/home/jma/GX/src/sims/gpu/gx_cuda.so:ro" \
 -e ARM="$arm" -e RUN="/w/runs/$name" -e DRPERF_ROOT=/home/ubuntu/drperf \
 -e DRPERF_MODE=late -e DRPERF_DRRUN=/home/ubuntu/drperf/third_party/dynamorio/bin64/drrun \
 -e DRPERF_CLIENT=/home/ubuntu/drperf/build/libdrperf.so -e DRPERF_ATTACH=/home/ubuntu/drperf/build/libdrperf_attach.so \
 -e DRPERF_ATTACH_UNMASK_SIGNAL=1 -e DRPERF_NATIVE_GX=1 -e DRPERF_EXCLUDE_CUDA_MODULE=gx_cuda.so \
 -e DRPERF_DR_OPTS="${DRPERF_DR_OPTS:--no_follow_children}" -e TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}" -e DRPERF_MAX_STATES_PER_REGION=4096 -e DRPERF_TIMEOUT=1800 \
 -e LMC_CPU_GB="${LMC_CPU_GB:-2}" \
 -e SESSIONS="${SESSIONS:-24}" -e TURNS="${TURNS:-6}" \
 -e HF_HUB_OFFLINE=1 -e LMCACHE_TRACK_USAGE=false -e DO_NOT_TRACK=1 \
 -e DRIVER_PY="${DRIVER_PY:-/w/exp/gx_kimi/drperf_offline.py}" -e LMC_OVERLAY="${LMC_OVERLAY:-}" -e DRPERF_WAIT_DELAY_KIND=event \
 -e DRPERF_WAIT_DELAY_REGION="${DRPERF_WAIT_DELAY_REGION:-}" -e DRPERF_WAIT_DELAY_MS="${DRPERF_WAIT_DELAY_MS:-0}" \
 -e DRPERF_WAITS=1 -e DRPERF_MAX_WAIT_RECORDS=1000000 -e DRPERF_FOLLOW_THREADS=0 \
 --entrypoint /bin/bash gx-kimi-k3:latest -lc \
 'bash /w/lmc-waits/drperf_lmcache.sh > "$RUN/driver.log" 2>&1'
