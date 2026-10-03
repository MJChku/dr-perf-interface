# GX partial_sync screening evidence

Model: `optimistic-dependency-timeline`. Complete run: `True`. Marked-interval trace evidence: `True`.

| Record | Iteration | Virtual duration | CPU work summed across threads | GPU work summed across streams |
| --- | ---: | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl` | 1 | 12066.437 ms | 2609.091 ms | 11752.095 ms |

Per modeled device within each marked interval (trace-derived):

| Record | Iteration | Device | Busy | Idle | Busy fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl` | 1 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:gpu:0` | 11752.095 ms | 314.342 ms | 97.39% |

Whole capture: virtual makespan 12364.709 ms; CPU work summed across threads 2907.364 ms.
Compute prediction: 106249 hits, 84 misses, 0 errors, 0 disabled; hit coverage 99.92%.
CPU attribution: 0/155594 sampled segments; sampled modeled CPU cost 0.00%; 0 unresolved-symbol samples; 0 sampled segments with fewer than five samples.

Whole-capture modeled GPU device time (denominator: capture makespan):

| Device | Busy | Idle | Busy fraction |
| --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:gpu:0` | 11752.095 ms | 612.615 ms | 95.05% |

Largest CPU segments ending at GPU submissions, iteration 1 in `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl`:

| Modeled CPU within interval | Following interaction | Samples | Lane |
| ---: | --- | ---: | --- |
| 11.992 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.992 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.992 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |
| 11.991 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/host-0/d0/timeline/rank-0-pid-382.jsonl:cpu:382` |

Interpretation limits:

- Modeled interval CPU work sums overlapping threads; it is not critical-path CPU time.
- Modeled GPU idle is not evidence by itself of a CPU bottleneck.
- Partial synchronization omits arbitrary CPU mutex, shared-memory and thread-creation dependencies.
- Prediction misses and unmodeled copies, memsets and callbacks have zero modeled service cost.
- NCCL rendezvous is launch-granular; independent communication contention is omitted.
- CPU segment labels name the following interaction, not the exclusive cost of that API.
- Prediction and CPU sampling counts are for the whole capture, not individual iterations.

Sources: `/home/ubuntu/drperf/out/causal-video/fastvideo-partial-final-report-full/summary.json`, `/home/ubuntu/drperf/out/causal-video/fastvideo-partial-final-report-full/cpu-attribution.json`, `/home/ubuntu/drperf/out/causal-video/fastvideo-partial-final-report-full/trace.json.gz`, `/home/ubuntu/GX/NEX/build/experiments/fastvideo-partial-final-1789611575199136097/result.json`
