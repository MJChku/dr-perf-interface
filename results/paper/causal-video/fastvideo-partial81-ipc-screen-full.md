# GX partial_sync screening evidence

Model: `optimistic-dependency-timeline`. Complete run: `True`. Marked-interval trace evidence: `True`.

| Record | Iteration | Virtual duration | CPU work summed across threads | GPU work summed across streams |
| --- | ---: | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl` | 1 | 12009.344 ms | 2461.266 ms | 11738.946 ms |

Per modeled device within each marked interval (trace-derived):

| Record | Iteration | Device | Busy | Idle | Busy fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl` | 1 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:gpu:0` | 11738.946 ms | 270.398 ms | 97.75% |

Whole capture: virtual makespan 12310.526 ms; CPU work summed across threads 2762.448 ms.
Compute prediction: 106249 hits, 84 misses, 0 errors, 0 disabled; hit coverage 99.92%.
CPU attribution: 0/155594 sampled segments; sampled modeled CPU cost 0.00%; 0 unresolved-symbol samples; 0 sampled segments with fewer than five samples.

Whole-capture modeled GPU device time (denominator: capture makespan):

| Device | Busy | Idle | Busy fraction |
| --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:gpu:0` | 11738.946 ms | 571.580 ms | 95.36% |

Largest CPU segments ending at GPU submissions, iteration 1 in `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl`:

| Modeled CPU within interval | Following interaction | Samples | Lane |
| ---: | --- | ---: | --- |
| 10.597 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.597 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.597 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.597 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.597 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.597 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.596 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.596 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.596 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 10.596 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |

Interpretation limits:

- Modeled interval CPU work sums overlapping threads; it is not critical-path CPU time.
- Modeled GPU idle is not evidence by itself of a CPU bottleneck.
- Partial synchronization omits arbitrary CPU mutex, shared-memory and thread-creation dependencies.
- Prediction misses and unmodeled copies, memsets and callbacks have zero modeled service cost.
- NCCL rendezvous is launch-granular; independent communication contention is omitted.
- CPU segment labels name the following interaction, not the exclusive cost of that API.
- Prediction and CPU sampling counts are for the whole capture, not individual iterations.

Sources: `/home/ubuntu/drperf/out/causal-video/fastvideo-partial81-ipc-report-full/summary.json`, `/home/ubuntu/drperf/out/causal-video/fastvideo-partial81-ipc-report-full/cpu-attribution.json`, `/home/ubuntu/drperf/out/causal-video/fastvideo-partial81-ipc-report-full/trace.json.gz`, `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial81-ipc-1789610969706374649/result.json`
