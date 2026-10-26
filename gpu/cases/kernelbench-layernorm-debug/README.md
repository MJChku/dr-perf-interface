# Public LayerNorm: redundant statistics requests

Source: [ScalingIntelligence/good-kernels, LayerNorm](https://github.com/ScalingIntelligence/good-kernels/tree/5c94620c59ab9839a4ad4ccad309f3d91752d042/LayerNorm),
revision `5c94620c59ab9839a4ad4ccad309f3d91752d042`.
The published PyTorch reference normalizes 16 samples of 4,194,304 FP32
features each, with affine parameters. No inefficiency was injected.
[before.cu](before.cu) is the unchanged published device source, with only
torch/ATen includes and the host wrapper removed. The runner verifies this
extraction against the saved [source](upstream_src.py); the
[reference](upstream_ref.py) is preserved and never imported.

## CPU-only investigation

The initial investigation, change and correctness gates used no GPU.
DrPerf's per-site feedback identifies the apply kernel's mean and inverse
standard deviation loads: every thread requests the same two values.
With `E = rows * columns` and `RB = rows * blocks_per_sample`, the training
interfaces are, ignoring a numerical fit coefficient below 3e-12:

```
                     before                 after
requested reads      16*E + 2056*RB          16*E + 16*RB
requested writes      4*E + 8*RB + 8*rows      unchanged
FP32 FLOPs             7*E + 2882*RB + 5*rows  unchanged
```

[after.cu](after.cu) lets thread zero load the statistics and broadcasts them
through eight bytes of shared memory plus a block barrier. The optimization
changes only this part of the apply kernel. It saves **2,040 requested global
bytes per row block**, exchanging global requests for shared-memory requests
and a barrier. At one row of 256 columns, total requested global bytes fall
from 7,192 to 5,152 (28.4%). At the original width, the same saving is less
than 0.1% of the aggregate requested global bytes. This is not an HBM traffic
or latency prediction. The unique global address footprint stays unchanged.

Training widths are 256, 1,024 and 16,384; held-out widths are 128, 768,
4,100 and 8,192; each uses one/two rows and three patterns (seeded uniform
finite values, constant values, and a piecewise shifted distribution).
The large training width separates `elements` from the block count, which
saturates at 32 blocks per sample. Both variants pass all **42 numerical
cases**, input immutability and output guards, with maximum absolute error
below 6e-7 against an independent double-precision two-pass variance oracle.
The runner also checks each invocation against the aggregate byte and FLOP
rules above. These tests cover widths divisible by four and non-null affine
parameters; they do not establish correctness of tails, high-offset inputs,
NaNs/infinities or the full original shape in emulation.

DrPerf's memory request and FLOP interfaces fit across these training states.
Unique-read footprints remain unexplained because the PCVs omit the feature
vector length shared by affine parameters across rows. The compact extracts
retain rejected fits and original site breakdowns; no case ID is used as a PCV.

## Evidence and reproduction

```sh
python3 gpu/cases/kernelbench-layernorm-debug/run.py
```

The runtime/translator must already be built. `CUDA_HOME` defaults to
`/home/ubuntu/GX/toolchains/cuda13`. The runner compiles PTX for SM80, translates
it, executes it on CPU and removes temporary PTX, modules, cubins and raw
captures. [verification.json](results/verification.json) retains exact
commands, outputs, source/tool hashes, numerical checks and elapsed time.

- [Baseline report](results/before-report.txt), [after report](results/after-report.txt)
- [Training baseline](results/before-train.json), [training after](results/after-train.json)
- [Held-out baseline](results/before-heldout.json), [held-out after](results/after-heldout.json)
- [Change](results/before-to-after.patch), [driver/oracle](driver.cc)
- [Offline SM80 and SM103 native compiler resources](results/offline-native-resources.json)
- [Native all-output validation and alternating timing harness](native_benchmark.cu)

Offline native compilation reports apply-kernel registers 25→27 on SM80 and
32→32 on SM103. Both targets have zero spills. The after kernel adds eight
shared bytes and one barrier. The two unchanged reduction kernels have their
own register counts in the receipt; no GPU was needed for these diagnostics.

## Native RCP validation: no useful gain

The variants were frozen after CPU correctness/resource gates, then measured
on one **NVIDIA B300 SXM6 AC**, RCP node `gpu405.rcp.epfl.ch`, CUDA 13/SM103.
They use identical inputs, buffers, 256-thread blocks, block-count policy and
stream; each timed operation includes all three LayerNorm launches. Input
values are seeded finite uniform FP32 values with nontrivial affine parameters.
Every output passed the double-precision oracle and guards at all three shapes,
with maximum absolute error below 4.5e-7.

| Rows × columns | Baseline, µs | After, µs | Change |
|---|---:|---:|---:|
| 16 × 256 | 10.547 | 11.345 | 7.56% slower |
| 16 × 16,384 | 12.406 | 12.370 | 0.29% lower |
| **16 × 4,194,304 (original)** | **190.445** | **190.642** | **0.10% slower** |

This is a useful counterexample to treating fewer requested bytes as a GPU
speedup. The original-shape result shows no useful improvement; the smallest
case regresses. The resource change moves requests into shared memory and
adds a barrier, while repeated global statistics reads may already hit cache.
Those mechanisms are plausible explanations, not hardware-profiled attribution.
No tuning followed the native results.

CUDA events follow 30 warmup operations per variant, then 11 alternating paired
rounds of 30 operations each; the table reports median per-round average time.
Allocations, transfers and numerical validation are outside the timed interval.
These are repeated launches on reused buffers, not cold-cache/service latency.
SM103 compilation retained 32 apply-kernel registers and zero spills in both
variants, with eight additional shared bytes and a barrier after the change.

Both public kernels shared one 59-second RCP allocation. Its exact job was
deleted and verified absent; no benchmark ran on rs3lab.

- [All native samples and numerical checks](../kernelbench-conv2d-debug/results/drperf-public-1026-164544/native/layernorm/native.json)
- [Native SM103 diagnostics](../kernelbench-conv2d-debug/results/drperf-public-1026-164544/native/layernorm/compile.log)
- [GPU receipt](../kernelbench-conv2d-debug/results/drperf-public-1026-164544/native/gpu-before.txt)
- [Allocation, frozen source hashes, commands and deletion](../kernelbench-conv2d-debug/results/drperf-public-1026-164544/allocation.json)
- [Paired CPU resource comparisons](results/paired-resource-checks.json)

To repeat native validation, run
`python3 gpu/cases/kernelbench-conv2d-debug/run_native_rcp.py`; this explicitly
requests one GPU for the two frozen experiments.
