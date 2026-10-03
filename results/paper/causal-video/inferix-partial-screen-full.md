# GX partial_sync screening evidence

Model: `optimistic-dependency-timeline`. Complete run: `True`. Marked-interval trace evidence: `True`.

| Record | Iteration | Virtual duration | CPU work summed across threads | GPU work summed across streams |
| --- | ---: | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl` | 1 | 51814.447 ms | 45396.148 ms | 11915.228 ms |

Per modeled device within each marked interval (trace-derived):

| Record | Iteration | Device | Busy | Idle | Busy fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl` | 1 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:gpu:0` | 11915.228 ms | 39899.220 ms | 23.00% |

Whole capture: virtual makespan 51814.464 ms; CPU work summed across threads 45396.165 ms.
Compute prediction: 106449 hits, 168 misses, 0 errors, 0 disabled; hit coverage 99.84%.
CPU attribution: 0/114262 sampled segments; sampled modeled CPU cost 0.00%; 0 unresolved-symbol samples; 0 sampled segments with fewer than five samples.

Whole-capture modeled GPU device time (denominator: capture makespan):

| Device | Busy | Idle | Busy fraction |
| --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:gpu:0` | 11915.228 ms | 39899.236 ms | 23.00% |

Largest CPU segments ending at GPU submissions, iteration 1 in `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl`:

| Modeled CPU within interval | Following interaction | Samples | Lane |
| ---: | --- | ---: | --- |
| 6.738 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.112 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.111 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |
| 6.111 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:cpu:340` |

Interpretation limits:

- Modeled interval CPU work sums overlapping threads; it is not critical-path CPU time.
- Modeled GPU idle is not evidence by itself of a CPU bottleneck.
- Partial synchronization omits arbitrary CPU mutex, shared-memory and thread-creation dependencies.
- Prediction misses and unmodeled copies, memsets and callbacks have zero modeled service cost.
- NCCL rendezvous is launch-granular; independent communication contention is omitted.
- CPU segment labels name the following interaction, not the exclusive cost of that API.
- Prediction and CPU sampling counts are for the whole capture, not individual iterations.

Sources: `/home/ubuntu/drperf/out/causal-video/inferix-partial-report-full/summary.json`, `/home/ubuntu/drperf/out/causal-video/inferix-partial-report-full/cpu-attribution.json`, `/home/ubuntu/drperf/out/causal-video/inferix-partial-report-full/trace.json.gz`, `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/result.json`
