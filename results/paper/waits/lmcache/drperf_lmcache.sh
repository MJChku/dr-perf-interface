#!/bin/bash
# drperf capture of LMCache's store and retrieve paths under GX, on the same
# workload as the ditto capture (exp/gx_kimi/drperf_offline.sh, qwen preset):
# Qwen2.5-0.5B with dummy weights, a 40 MB GPU KV budget so every turn reloads
# from the cache layer, prompts of 100 to 1,500 words (1 to 8 LMCache chunks).
#
# Runs INSIDE the gx-kimi-k3 container, like drperf_offline.sh. Arms:
#   ARM=cpu              LMCache local CPU tier, no remote, no compression
#   ARM=remote_naive     local tier off, lmcache_server in-container, naive serde
#   ARM=remote_cachegen  the same with remote_serde=cachegen
# DRPERF_MODE=native runs without DynamoRIO (functional smoke test);
# DRPERF_MODE=late is the measured capture.
#
# LMCache (built from source, below) lives in $DITTO_ROOT/pydeps_lmc; its
# compiled extension needs the CUDA 13 runtime, which the ditto capture already
# puts on LD_LIBRARY_PATH. Its regions come from lmcache_overlay/.
set -uo pipefail
DITTO_ROOT=${DITTO_ROOT:-/w}
GX_HOME=${GX_HOME:-/home/jma/GX}
PY=${PY:-/home/jma/bridge-venv2/bin/python}
DRP=${DRPERF_ROOT:-/home/ubuntu/drperf}
ARM=${ARM:-remote_naive}
case "$ARM" in cpu|remote_naive|remote_cachegen) ;; *) echo "ARM=$ARM?" >&2; exit 2 ;; esac
RUN=${RUN:-$DITTO_ROOT/runs/drperf-lmcache-$ARM-$(date +%Y%m%d-%H%M%S)}

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SESSIONS=${SESSIONS:-24}; TURNS=${TURNS:-6}; NEW=${NEW:-40}
PREFIXES=${PREFIXES:-100,200,300,400,500,600,750,900,1050,1200,1350,1500}
MAXLEN=${MAXLEN:-2048}; KV_MEM=${KV_MEM:-40000000}
WARMUP=${WARMUP:-3}
# This vLLM build stores KV as NL x [NB, NH, BS, 2, HS] whatever the attention
# backend. LMCache 0.4.7's copy kernel does not implement that layout
# ("Unsupported GPUKVFormat"), so pydeps_lmc holds LMCache built from source
# (upstream 8e93a34, 2026-09-28) against this torch; its kernel does. With the
# layout handled, the capture uses the ditto capture's backend, so the two
# differ only in the cache layer. FlashAttention does not load under GX.
ATTN=${ATTN:-TRITON_ATTN}
LMC_CPU_GB=${LMC_CPU_GB:-2}
PORT=${LMC_PORT:-65432}
export GX_DEVICE_MODEL=${GX_DEVICE_MODEL:-A100} GX_NUM_LOCAL_GPUS=1

mkdir -p "$RUN/raw" "$DITTO_ROOT/hf" "$DITTO_ROOT/cache/triton"
# LMCache with its drperf regions written into its own source
# (lmcache_overlay/): a symlink copy of the installed package with the
# annotated modules copied over it, so drperf maps each region to LMCache's line.
OVERLAY=${LMC_OVERLAY:-$DITTO_ROOT/example/qwen2.5B/lmcache/lmcache_overlay}
LMC=$RUN/lmcache_pkg
mkdir -p "$LMC"
cp -rs "$DITTO_ROOT/pydeps_lmc/lmcache" "$LMC/lmcache"
(cd "$OVERLAY" && find lmcache -name '*.py') | while read -r f; do
  cp --remove-destination "$OVERLAY/$f" "$LMC/$f"
done
echo "LMCACHE_OVERLAY $(cd "$OVERLAY" && find lmcache -name '*.py' | wc -l) modules -> $LMC"
export GX_HOME HF_HOME=$DITTO_ROOT/hf
export PYTHONPATH=$DITTO_ROOT:$DITTO_ROOT/shims:$LMC:$DITTO_ROOT/pydeps_lmc:$DITTO_ROOT/pydeps:/home/jma
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_DISABLE_PYNCCL=1 VLLM_ALLREDUCE_USE_SYMM_MEM=0 VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GX_PTX_CACHE_ROOT=${GX_PTX_CACHE_ROOT:-$DITTO_ROOT/cache/gxptx} TRITON_CACHE_DIR=$DITTO_ROOT/cache/triton
CU13=/home/jma/bridge-venv2/lib/python3.12/site-packages/nvidia/cu13/lib
export LD_LIBRARY_PATH="$CU13${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GX_CUDA_CONTEXT_LIMITS=${GX_CUDA_CONTEXT_LIMITS:-$DITTO_ROOT/gxvm_runtime/context-limits.tsv}
export DRPERF_ROOT=$DRP DRPERF_MODE=${DRPERF_MODE:-native}
export DRPERF_ATTACH_UNMASK_SIGNAL=${DRPERF_ATTACH_UNMASK_SIGNAL:-1}
export DRPERF_DR_OPTS=${DRPERF_DR_OPTS--no_follow_children} DRPERF_MAX_SLOTS=${DRPERF_MAX_SLOTS:-2097152}
# Not followed: LMCache's event-loop thread (remote sends). drperf gives every
# followed thread its own max_slots counter array per region key; with this
# process's ~190 threads the 96 GB counter budget runs out after ~50 keys.
export DRPERF_FOLLOW_THREADS=${DRPERF_FOLLOW_THREADS:-0} DRPERF_EXCLUDE_CUDA_MODULE=${DRPERF_EXCLUDE_CUDA_MODULE:-gx_cuda.so}
# LMCache hashes chunk keys; a remote endpoint needs a stable seed
export PYTHONHASHSEED=0

CFG=$RUN/lmcache.yaml
{
  echo "chunk_size: 256"
  if [ "$ARM" = cpu ]; then echo "local_cpu: True"; else echo "local_cpu: False"; fi
  echo "max_local_cpu_size: $LMC_CPU_GB"
  if [ "$ARM" != cpu ]; then
    echo "remote_url: \"lm://localhost:$PORT\""
    case "$ARM" in remote_cachegen) echo 'remote_serde: "cachegen"' ;; *) echo 'remote_serde: "naive"' ;; esac
  fi
} > "$CFG"
export LMCACHE_CONFIG_FILE=$CFG
echo "LMCACHE arm=$ARM config:"; sed 's/^/  /' "$CFG"

if [ "$ARM" != cpu ]; then
  "$PY" -m lmcache.v1.server localhost "$PORT" > "$RUN/lmcache_server.log" 2>&1 &
  echo $! > "$RUN/lmcache_server.pid"
  for _ in $(seq 1 60); do (echo > /dev/tcp/localhost/$PORT) >/dev/null 2>&1 && break; sleep 1; done
  (echo > /dev/tcp/localhost/$PORT) >/dev/null 2>&1 \
    || { echo "lmcache_server did not come up"; tail -20 "$RUN/lmcache_server.log"; exit 1; }
fi

KV_CFG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
# the tier capture's workload driver, unchanged; the regions are in LMCache
DRIVER_PY=${DRIVER_PY:-$DITTO_ROOT/exp/gx_kimi/drperf_offline.py}
echo "DRPERF_LMCACHE run=$RUN arm=$ARM mode=$DRPERF_MODE model=$MODEL sessions=$SESSIONS turns=$TURNS $(date -u +%FT%TZ)"
cd "$GX_HOME"
bash "$GX_HOME/tools/gx_run.sh" "$PY" "$DITTO_ROOT/exp/gx_kimi/drperf_serve.py" "$RUN/raw" "$RUN/server.pid" "$RUN/server.log" -- \
  "$PY" "$DRIVER_PY" --sessions "$SESSIONS" --turns "$TURNS" --prefix-words "$PREFIXES" \
  --new-words "$NEW" --max-tokens 4 --warmup-sessions "$WARMUP" \
  --warmup-prefix-words "${WARMUP_PREFIXES:-650,850,1150}" --settle-sessions "${SETTLE:-2}" \
  --settle-prefix-words "${SETTLE_PREFIXES:-700,950}" --out "$RUN/client.json" -- \
  --model "$MODEL" --load-format dummy --enforce-eager --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" --max-num-seqs 1 --kv-cache-memory "$KV_MEM" --enable-prefix-caching \
  --attention-backend "$ATTN" --gpu-memory-utilization 0.3 \
  --disable-hybrid-kv-cache-manager --kv-transfer-config "$KV_CFG" > "$RUN/serve.log" 2>&1
rc=$?
echo "DRPERF_LMCACHE serve rc=$rc $(date -u +%FT%TZ)"
grep -a "^CLIENT_SUMMARY\|OFFLINE engine ready\|OFFLINE_DONE" "$RUN/server.log" | cut -c1-300
echo "LMCache events: stored=$(grep -a -c 'Stored [0-9]* out of' "$RUN/server.log") retrieved=$(grep -a -c 'Retrieved [0-9]* out of' "$RUN/server.log") key_errors=$(grep -a -c 'exceeds maximum' "$RUN/server.log") errors=$(grep -a -c 'LMCache ERROR' "$RUN/server.log")"
[ -f "$RUN/lmcache_server.pid" ] && kill "$(cat "$RUN/lmcache_server.pid")" 2>/dev/null
echo "DRPERF_LMCACHE_DONE $(date -u +%FT%TZ)"
exit "$rc"
