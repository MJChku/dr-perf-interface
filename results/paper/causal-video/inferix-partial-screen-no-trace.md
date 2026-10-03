# GX partial_sync screening evidence

Model: `optimistic-dependency-timeline`. Complete run: `True`. Marked-interval trace evidence: `False`.

| Record | Iteration | Virtual duration | CPU work summed across threads | GPU work summed across streams |
| --- | ---: | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl` | 1 | 51814.447 ms | unavailable | unavailable |

Per modeled device within each marked interval (trace-derived):

| Record | Iteration | Device | Busy | Idle | Busy fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| — | — | unavailable | — | — | — |

Whole capture: virtual makespan 51814.464 ms; CPU work summed across threads 45396.165 ms.
Compute prediction: 106449 hits, 168 misses, 0 errors, 0 disabled; hit coverage 99.84%.
CPU attribution: 0/114262 sampled segments; sampled modeled CPU cost 0.00%; 0 unresolved-symbol samples; 0 sampled segments with fewer than five samples.

Whole-capture modeled GPU device time (denominator: capture makespan):

| Device | Busy | Idle | Busy fraction |
| --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/host-0/d0/timeline/rank-unknown-pid-340.jsonl:gpu:0` | 11915.228 ms | 39899.236 ms | 23.00% |

Interpretation limits:

- Modeled interval CPU work sums overlapping threads; it is not critical-path CPU time.
- Modeled GPU idle is not evidence by itself of a CPU bottleneck.
- Partial synchronization omits arbitrary CPU mutex, shared-memory and thread-creation dependencies.
- Prediction misses and unmodeled copies, memsets and callbacks have zero modeled service cost.
- NCCL rendezvous is launch-granular; independent communication contention is omitted.
- CPU segment labels name the following interaction, not the exclusive cost of that API.
- Prediction and CPU sampling counts are for the whole capture, not individual iterations.

Sources: `/home/ubuntu/drperf/out/causal-video/inferix-partial-report-no-trace/summary.json`, `/home/ubuntu/drperf/out/causal-video/inferix-partial-report-no-trace/cpu-attribution.json`, `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-1789608115131050087/result.json`
