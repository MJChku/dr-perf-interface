# llama.cpp GroupNorm: fewer requests, shape-dependent hardware results

Source: [llama.cpp `group_norm_f32`](https://github.com/ggml-org/llama.cpp/blob/623c16393c7d53a37df8bb69c6ea8ed6e5171f62/ggml/src/ggml-cuda/norm.cu),
revision `623c16393c7d53a37df8bb69c6ea8ed6e5171f62`.
This is a retained public CUDA implementation, not an injected inefficiency.
Its three passes compute the mean, write centered values while computing
variance, then reread and scale those values. The source's CPU GroupNorm
implementation provides the correctness reference; the standalone harness
uses an independent double-precision two-pass mean/variance oracle.
No published benchmark shape is associated with this function; the hardware
shapes below are representative standalone tests, not an end-to-end llama.cpp
or diffusion workload measurement.

## Extraction and CPU-only investigation

[before.cu](before.cu) preserves the selected device function unchanged.
[common_extracted.cuh](common_extracted.cuh) contains the unchanged required
reduction and PDL helper excerpts, plus CUDA-only includes/constants. Explicit
template instantiations expose the original 32/1,024-thread variants without
the ggml host framework. The [runner](run.py) verifies the selected source and
helper excerpts against [upstream_norm.cu](upstream_norm.cu) and
[upstream_common.cuh](upstream_common.cuh). [provenance.json](provenance.json)
pins paths/hashes; [the CPU reference excerpt](upstream_cpu_reference.cc)
is preserved but never imported or executed.

All feedback-guided development and correctness gates used CPU PTX execution.
The original dispatch uses 32 threads below width 1,024 and 1,024 threads
otherwise. Separate `.warp` and `.cta` regions reflect these actual compiled
specializations; mixing them in one region initially left their distinct
instruction sites unexplained. PCVs are `elements`, `rows`, `launched_lanes`
and the bounded prefix `cached_elements`, not input-case IDs.

DrPerf's original read-site breakdown identifies two input reads and one
centered-output read, each contributing `4*elements`. Two output stores also
contribute `4*elements` each. Within each dispatch, training interfaces fit:

```
E = rows * group_width
L = rows * threads_per_block
C = rows * min(group_width, 8 * threads_per_block)

                           before           after
requested global reads     12*E             12*E - 8*C
requested global writes     8*E              8*E - 4*C
```

[after.cu](after.cu) retains eight values/thread, then retains their centered
values, skipping their temporary output stores/loads. Remaining elements use
the original staging path. Global requests decrease **60%** when all elements
fit, **30%** at width 16,384 and **7.5%** at width 65,536. Unique-read footprint
also falls because cached centered values are never loaded from output.
These are requested bytes and address footprints, not HBM transactions.

The feedback also exposes a cost: NVCC computes some arithmetic for inactive
unrolled cache slots. Across every measured state the candidate has
`3*(8*L-C)` additional FP32 FLOPs; special-function counts stay unchanged.
This is zero when all eight slots/thread are occupied, but positive on narrow
rows. The extra arithmetic and register pressure are retained, not hidden.

Training widths: 64, 256, 768, 1,024, 4,096, 16,384. Held-out widths: 33, 255,
767, 1,025, 4,097, 16,380, 65,536, using separate random seeds. Each uses one/two
groups and three patterns: finite uniform random values, a constant, and a
single peak among zeros. Both variants pass **78 numerical cases**, immutable
input checks and output guards. Tolerance is 4e-5 absolute/relative; maximum absolute error is below 4e-6.
The runner checks every invocation against the global-byte rules above.
Held-out fits retain unexplained portions: individual unrolled sites have
activation thresholds that aggregate `C` alone does not describe. The aggregate
byte rules were independently verified rather than claiming every site fitted.

Scope is contiguous, complete groups. Ragged final groups, arbitrary strides,
NaNs/infinities and framework integration are not established by these tests.

## Native RCP validation

Frozen variants subsequently ran on one **NVIDIA B300 SXM6 AC**, node
`gpu405.rcp.epfl.ch`, CUDA 13, `-O3 -arch=native` (SM103), without fast math.
Inputs, buffers, launch dimensions, shared-memory policy and stream match
within every pair. All native outputs passed the double-precision oracle and
guards; maximum absolute error is below 3.3e-7.

| Groups × width | Baseline, µs | Candidate, µs | Change |
|---|---:|---:|---:|
| 4,096 × 256 | 6.222 | 6.226 | 0.07% slower |
| 4,096 × 4,096 | 47.244 | 64.428 | 36.37% slower |
| 4,096 × 16,384 | 125.904 | 145.892 | 15.88% slower |
| 1,024 × 65,536 | 236.528 | 181.946 | **23.08% lower** |

The candidate is not universally faster. Registers increase 31→40 on SM103,
with no spills. SM80 separately reports 32→40; SM90 reports 32→48 for the
1,024-thread kernel and 32→39 for the warp kernel, also without spills.
Increased register pressure and the additional arithmetic are plausible costs,
not profiled explanations of the regressions. No tuning followed native timing.

Timing uses CUDA events after 30 warmups/variant, then 11 alternating paired
rounds of 30 launches. The table gives medians of per-round average times;
allocation, transfer and numerical validation are excluded. This is repeated
launch timing on reused buffers, not cold-cache or application latency.

Three finalized experiments shared one **31-second, one-B300 RCP allocation**.
The exact owned job was deleted and verified absent. No H100 or rs3lab benchmark
was run. [run_native_rcp.py](run_native_rcp.py) contains bounded lifetimes,
per-benchmark timeouts, source staging and cleanup in `finally`.

## Evidence and reproduction

```sh
python3 gpu/cases/llamacpp-groupnorm-debug/run.py
# Separately, requests one RCP GPU for the shared frozen batch:
python3 gpu/cases/llamacpp-groupnorm-debug/run_native_rcp.py
```

The GPU translator/runtime must already be built. `CUDA_HOME` defaults to
`/home/ubuntu/GX/toolchains/cuda13`. Temporary PTX, modules, cubins and raw
captures are removed; compact reports retain actual fits/rejections.

- [CPU checks, commands and source/tool hashes](results/verification.json)
- [Baseline report](results/before-report.txt), [candidate report](results/after-report.txt)
- [Baseline training](results/before-train.json), [candidate training](results/after-train.json)
- [Baseline held-out](results/before-heldout.json), [candidate held-out](results/after-heldout.json)
- [Optimization diff](results/before-to-after.patch), [driver/oracle](driver.cc)
- [Paired counts](results/paired-resource-checks.json)
- [Offline SM80/SM90/SM103 compiler resources](results/offline-native-resources.json)
- [Native harness](native_benchmark.cu), [all timing samples](results/drperf-public-next-1026-175153/native/groupnorm/native.json)
- [Native SM103 diagnostics](results/drperf-public-next-1026-175153/native/groupnorm/compile.log)
- [Allocation, frozen-source hashes and confirmed deletion](results/drperf-public-next-1026-175153/allocation.json)
