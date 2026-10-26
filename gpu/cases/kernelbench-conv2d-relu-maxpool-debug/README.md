# Public fused convolution, ReLU and max-pool: skip zero atomic updates

This is a CPU-developed experiment on the published
[good-kernels/Conv2D_ReLU_MaxPool](https://github.com/ScalingIntelligence/good-kernels/tree/5c94620c59ab9839a4ad4ccad309f3d91752d042/Conv2D_ReLU_MaxPool),
revision `5c94620c59ab9839a4ad4ccad309f3d91752d042`. The original PyTorch workload
is 100×3×224×224, with 96 convolution channels, kernel 11, stride 4, padding 2,
followed by ReLU and 3×3 max-pooling with stride 2. No inefficiency was injected.
This is a standalone kernel experiment, not an official KernelBench submission.

The proposed change initializes the pooled output to **zero** and skips
`atomicMax` when the ReLU result is zero. Pooling a nonempty window of ReLU
values always produces a nonnegative result, so zero is a valid identity.
Changing the initialization is necessary: skipping all-zero contributions with
the original negative sentinel would leave incorrect outputs. The host change
is retained in [host-initialization.patch](results/host-initialization.patch).
The existing sentinel-cleanup wrapper then becomes redundant.

## What DrPerf contributed

Source inspection identifies the atomic pooling implementation. CPU-only
DrPerf feedback quantifies its resource consequences and shows the same atomic
requests on positive, mixed and all-negative convolution data. Its
`atom.global.max.s32` sites each contribute both a four-byte read request and a
four-byte write request. There are no other global stores inside the kernel.
Across training observations, the baseline composed interface simplifies to:

```
global write bytes = 4 * pool_updates
```

Here `pool_updates` is the number of convolution-value/window memberships,
including overlap. The change makes the aggregate rule:

```
global write bytes = 4 * positive_updates
atomic read + write bytes saved = 8 * (pool_updates - positive_updates)
tensor FLOPs = 524288 * tile_products                 # both variants
```

`tile_products = ceil(pixels/128) * ceil(channels/128) * ceil(K/16)`.
`positive_updates` is computed before the measured region from the independent
CPU convolution/pooling oracle. It is an explanatory data-dependent PCV, not
an efficient runtime prepass proposed for production. Finite binary-fraction
operands in the CPU sweep avoid FP16 input-conversion ambiguity when counting
positive contributions; bias remains FP32.

The runner checks the write and tensor rules exactly in every observation.
Random training states remove 27–52% of atomic updates; held-out random states
remove 49–55%. The all-negative and all-positive cases remove 100% and 0%.
These are PTX requests and arithmetic counts, not measured HBM traffic or
latency.

There are two material qualifications:

* The new positive branch changes NVCC's movement of bias loads and ReLU work
  across pooling-bound checks. At the 1×3×6×8 training input, the candidate
  performs 1,024 more non-atomic read bytes and 512 more FP32 operations than
  the baseline, for every input pattern. At all-positive data this increases
  total reads despite unchanged atomic requests. The matched counts are in
  [matched-resource-checks.json](results/matched-resource-checks.json).
* The aggregate write rule does **not** mean DrPerf fitted every candidate
  instruction site. NVCC unrolls multiple atomic sites, and the total number
  of positive memberships does not explain how positives distribute across
  those sites. Candidate composed write interfaces retain unexplained means
  of 1,882.33 bytes in training and 10,197.78 bytes in held-out states. The
  aggregate rule is independently asserted against measured counts; the
  unexplained site fits are preserved. Several other fits have dependent
  PCVs and nonunique coefficients.

## Numerical validation and compatibility

Both variants passed **21 CPU cases**: 12 training and nine held out. Each
shape uses seeded mixed-sign data, all-negative convolution results and
all-positive results. The oracle independently computes original-FP32
convolution in double precision, then bias, ReLU and max-pooling. Every output,
16 trailing guards and input/weight/bias immutability are checked. Maximum
errors and command receipts are in [verification.json](results/verification.json).
Tolerance is `2e-5*(1+abs(reference))`; no NaN/infinity support is claimed.

Training varies channels (16,32), input geometry, convolution kernel (1,3),
padding, reduction depth and pooling (2/2,3/1,3/2). Held-out states include
17 output channels, overlapping 2/1 pooling, batch two, and the original
11/4/2 convolution geometry on a bounded 32×32 input with 96 channels.
[driver.cc](driver.cc) specifies the exact states and data.

[upstream_src.py](upstream_src.py) and [upstream_ref.py](upstream_ref.py) are
untouched downloads. [before.cu](before.cu) extracts the device code unchanged,
removing the torch/ATen includes and host wrapper only. Initial translation
failed because generated code referenced unresolved `WARP_SZ`; the
[failure receipt](results/original-translation.json) is preserved.
[compatible.cu](compatible.cu) replaces `warpSize` with literal 32. That
case-local ABI-compatible edit is shared by both performance variants.
No DrPerf or translator code was changed for this case.

Earlier harness-symbol and resource-assertion mistakes are retained in the
`initial-*.json` receipts. In particular, a proposed `2*conv_outputs` arithmetic
rule was rejected when NVCC moved work around pooling borders; it is not a
claimed resource law. Zero-valued metrics may be absent from DrPerf JSON.

## Native validation

**Rejected as a speed optimization on the tested B300.** The frozen variants
and [native_benchmark.cu](native_benchmark.cu) ran on one NVIDIA B300 SXM6 AC,
CUDA 13 / SM103, in a shared RCP allocation with GroupNorm/RMSNorm. All five
native cases passed the full-output oracle, with maximum absolute error below
9.36e-4. The job was deleted and its absence verified. No rs3lab or separate
SSH GPU host was used.

| Input / data | Baseline → candidate, µs | Latency change |
|---|---:|---:|
| 1×3×32×32, mixed | 65.671 → 67.725 | 3.13% slower |
| 8×3×224×224, mixed | 98.421 → 104.484 | 6.16% slower |
| **100×3×224×224, mixed (original shape)** | **763.957 → 810.436** | **6.08% slower** |
| 1×3×32×32, all negative | 65.643 → 66.111 | 0.71% slower |
| 1×3×32×32, all positive | 65.645 → 66.322 | 1.03% slower |

The negative hardware result is retained; request savings did not imply a
latency gain. Even the all-negative bounded case, which skips every atomic,
did not improve. No hardware-driven retuning was performed. Native and offline
SM103 compilation both use 123 registers and 35,840 shared bytes per block,
with no stack or spills for either variant. SM80 diagnostics are separate in
verification. Timing alone does not isolate the cause of regression; branch
behavior and changed compiler scheduling are possible contributors.

* [Native samples, errors and FP32-oracle positive membership counts](../llamacpp-groupnorm-debug/results/drperf-public-next-1026-175153/native/fused-conv/native.json)
* [Native compiler diagnostics](../llamacpp-groupnorm-debug/results/drperf-public-next-1026-175153/native/fused-conv/compile.log)
* [Shared allocation, frozen source hashes and confirmed deletion](../llamacpp-groupnorm-debug/results/drperf-public-next-1026-175153/allocation.json)
* [Shared bounded RCP controller](../llamacpp-groupnorm-debug/run_native_rcp.py)

The native harness uses CUDA events, 30 warmups and 11 alternating paired
rounds of 30 launches. It reports median per-launch times and all round means.
Both variants use 256 threads, identical input buffers and launch geometry.
Each output is checked against a full double-precision CPU convolution/pooling
oracle, with guards and input immutability. Native FP32 data are not quantized
to half; tolerance `3e-3+3e-3*abs(reference)` accounts for source FP16 conversion.
There are three random shapes (batch 1 at 32², batch 8 and 100 at 224²) plus
all-negative and all-positive bounded shapes. Output initialization, transfers
and validation are outside timing. Repeated launches reuse the same inputs
and already populated output; atomics are idempotent for that workload.

## Reproduction and retained evidence

```
python3 gpu/cases/kernelbench-conv2d-relu-maxpool-debug/run.py
python3 gpu/cases/kernelbench-conv2d-relu-maxpool-debug/compare_resources.py
```

This uses no GPU. The default toolkit is
`/home/ubuntu/GX/toolchains/cuda13`, overridable with `CUDA_HOME`. Generated
PTX, cubins, translated modules and raw JSONL captures live in temporary
directories and are removed. Compact original reports, metric/site extracts,
code diffs, numerical results, source/tool hashes and command receipts live in
[results](results). Upstream Python files are preserved but never executed.
