# GX partial_sync screening evidence

Model: `optimistic-dependency-timeline`. Complete run: `True`. Marked-interval trace evidence: `True`.

| Record | Iteration | Virtual duration | CPU work summed across threads | GPU work summed across streams |
| --- | ---: | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl` | 1 | 12267.723 ms | 4728.259 ms | 11915.589 ms |

Per modeled device within each marked interval (trace-derived):

| Record | Iteration | Device | Busy | Idle | Busy fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl` | 1 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:gpu:0` | 11915.589 ms | 352.133 ms | 97.13% |

Whole capture: virtual makespan 12267.739 ms; CPU work summed across threads 4728.275 ms.
Compute prediction: 106449 hits, 168 misses, 0 errors, 0 disabled; hit coverage 99.84%.
CPU attribution: 0/109627 sampled segments; sampled modeled CPU cost 0.00%; 0 unresolved-symbol samples; 0 sampled segments with fewer than five samples.

Whole-capture modeled GPU device time (denominator: capture makespan):

| Device | Busy | Idle | Busy fraction |
| --- | ---: | ---: | ---: |
| `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:gpu:0` | 11915.589 ms | 352.150 ms | 97.13% |

Largest CPU segments ending at GPU submissions, iteration 1 in `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl`:

| Modeled CPU within interval | Following interaction | Samples | Lane |
| ---: | --- | ---: | --- |
| 6.677 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 3.077 ms | `MEMCPY` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.703 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.676 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.674 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.674 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.673 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.671 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.671 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |
| 1.668 ms | `COMPUTE` | 0 | `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/host-0/d0/timeline/rank-unknown-pid-298.jsonl:cpu:298` |

Interpretation limits:

- Modeled interval CPU work sums overlapping threads; it is not critical-path CPU time.
- Modeled GPU idle is not evidence by itself of a CPU bottleneck.
- Partial synchronization omits arbitrary CPU mutex, shared-memory and thread-creation dependencies.
- Prediction misses and unmodeled copies, memsets and callbacks have zero modeled service cost.
- NCCL rendezvous is launch-granular; independent communication contention is omitted.
- CPU segment labels name the following interaction, not the exclusive cost of that API.
- Prediction and CPU sampling counts are for the whole capture, not individual iterations.

Sources: `/home/ubuntu/drperf/out/causal-video/inferix-partial-fixed-report-full/summary.json`, `/home/ubuntu/drperf/out/causal-video/inferix-partial-fixed-report-full/cpu-attribution.json`, `/home/ubuntu/drperf/out/causal-video/inferix-partial-fixed-report-full/trace.json.gz`, `/home/ubuntu/GX/NEX/build/experiments/inferix-partial-fixed-1789609490786545754/result.json`
