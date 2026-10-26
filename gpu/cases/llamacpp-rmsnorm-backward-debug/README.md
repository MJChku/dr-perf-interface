# llama.cpp RMSNorm backward: retain input and gradient

Source: [llama.cpp `rms_norm_back_f32`](https://github.com/ggml-org/llama.cpp/blob/623c16393c7d53a37df8bb69c6ea8ed6e5171f62/ggml/src/ggml-cuda/norm.cu),
revision `623c16393c7d53a37df8bb69c6ea8ed6e5171f62`.
This public backward kernel computes reductions of `x*x` and `x*gradient`,
then rereads both arrays to produce the input gradient. No inefficiency or
defect was injected. The public ggml CPU derivative implementation is retained
as a reference; the standalone numerical oracle computes the analytical
RMSNorm derivative independently in double precision.

The reference function is
`gradient / sqrt(mean(x²)+eps) - x * sum(x*gradient) / (N*(mean(x²)+eps)^(3/2))`.
There is no associated published benchmark shape; native shapes below are
representative tests, not an end-to-end llama.cpp training measurement.

## CPU-only feedback and change

[before.cu](before.cu) preserves the selected kernel unchanged, adding only
explicit template instantiations for the original 32/1,024-thread dispatch.
[common_extracted.cuh](common_extracted.cuh) preserves required public helpers;
only framework includes/host wrappers are excluded. [provenance.json](provenance.json)
pins paths and hashes. The [runner](run.py) verifies the extraction and helper
excerpts against [upstream_norm.cu](upstream_norm.cu) and
[upstream_common.cuh](upstream_common.cuh). The
[CPU reference excerpt](upstream_cpu_reference.cc) is saved and never executed.

Development, DrPerf feedback and correctness gates used CPU PTX execution.
Regions separate the actual `.warp` and `.cta` compiled variants, avoiding
mixing their different sites in one fit. The original per-site read breakdown
shows four contributions of `4*elements`: input and gradient in the reduction,
then both again in the output pass. The fitted training interfaces are:

```
E = rows * width
C = rows * min(width, 8 * threads_per_block)

                           before         after
requested global reads     16*E           16*E - 8*C
requested global writes     4*E            4*E
```

[after.cu](after.cu) caches eight input values and eight gradient values/thread,
using the original path for remaining elements. Total global requests fall
**40%** when all values fit, **20%** at width 16,384 and **5%** at width 65,536.
FP32 FLOPs, special-function counts and unique-read footprint remain unchanged
on every CPU test. This is request reduction, not a prediction of HBM traffic
or GPU latency.

Training widths are 64, 256, 768, 1,024, 4,096 and 16,384; held-out widths are
33, 255, 767, 1,025, 4,097, 16,380 and 65,536 with different random seeds.
Each uses one/two rows and three patterns: finite uniform random values,
constant values/gradient, and a single input peak with random gradient.
Both variants pass **78 independent numerical cases**, immutable-input and
output-guard checks, with a 4e-5 absolute/relative tolerance and maximum absolute error below 5e-6. Each invocation
also passes the aggregate requested-byte checks above. The PCVs are elements,
rows, launched lanes and the bounded cached prefix, not case IDs.

Training memory fits are clean within each compiled variant. Held-out per-site
fits can remain unexplained: unrolled cached iterations activate at different
thresholds, while the declared prefix PCV describes their aggregate. The
receipts retain those rejected fits; the aggregate rule was checked directly
against observations, not claimed to be inferred across every site/shape.

Scope is contiguous rows, finite FP32 values and the original dispatch policy;
this does not validate framework integration, other layouts or NaNs/infinities.

## Native RCP validation

The frozen variants subsequently ran on one **NVIDIA B300 SXM6 AC**, node
`gpu405.rcp.epfl.ch`, CUDA 13, `-O3 -arch=native` (SM103), without fast math.
Every paired comparison uses the same inputs, buffers, stream and launch
settings. All outputs and guards passed the double-precision derivative oracle,
with maximum error below 3e-7.

| Rows × width | Baseline, µs | Candidate, µs | Change |
|---|---:|---:|---:|
| 4,096 × 256 | 6.234 | 6.227 | 0.10% lower |
| 4,096 × 4,096 | 51.435 | 65.566 | 27.47% slower |
| 4,096 × 16,384 | 152.499 | 185.402 | 21.58% slower |
| 1,024 × 65,536 | 257.971 | 239.825 | **7.03% lower** |

The near-identical smallest result is not a meaningful gain. The two middle
shapes regress despite fewer requests and unchanged arithmetic. SM103 register
use increases 32→56 per thread, without spills. Separately SM80 is 32→48,
SM90 is 32→56 for the CTA kernel and 32→54 for the warp kernel; all have zero
spills. Higher register pressure is a plausible cost, not a hardware-profiled
explanation. Native results were validation, not feedback used to retune.

CUDA events follow 30 warmups/variant and 11 alternating paired rounds of
30 launches. Medians of per-round average times are reported; allocation,
transfers and numerical validation are outside the timed interval. Inputs
are seeded finite uniform FP32 values. Timing describes repeated launches on
reused buffers, not cold-cache, whole-application or training-step latency.

This kernel shared one **31-second, one-B300 RCP allocation** with GroupNorm
and the second agent's frozen fused convolution case. The exact job was deleted
and verified absent. No H100/rs3lab allocation was used.

## Evidence and reproduction

```sh
python3 gpu/cases/llamacpp-rmsnorm-backward-debug/run.py
# Separately, requests one RCP GPU for the shared frozen batch:
python3 gpu/cases/llamacpp-groupnorm-debug/run_native_rcp.py
```

The GPU runtime/translator must already be built. `CUDA_HOME` defaults to
`/home/ubuntu/GX/toolchains/cuda13`. Temporary PTX, cubins, modules and raw
captures are removed on exit.

- [CPU checks, commands and hashes](results/verification.json)
- [Baseline report](results/before-report.txt), [candidate report](results/after-report.txt)
- [Baseline training](results/before-train.json), [candidate training](results/after-train.json)
- [Baseline held-out](results/before-heldout.json), [candidate held-out](results/after-heldout.json)
- [Change](results/before-to-after.patch), [independent driver/oracle](driver.cc)
- [Paired resource counts](results/paired-resource-checks.json)
- [Offline SM80/SM90/SM103 compiler resources](results/offline-native-resources.json)
- [Native harness](native_benchmark.cu)
- [All native samples and checks](../llamacpp-groupnorm-debug/results/drperf-public-next-1026-175153/native/rmsnorm-backward/native.json)
- [Native SM103 diagnostics](../llamacpp-groupnorm-debug/results/drperf-public-next-1026-175153/native/rmsnorm-backward/compile.log)
- [Shared allocation, hashes and confirmed deletion](../llamacpp-groupnorm-debug/results/drperf-public-next-1026-175153/allocation.json)
