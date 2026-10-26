# Public Conv2D: remove padded tensor work

Source: [ScalingIntelligence/good-kernels, Conv2D](https://github.com/ScalingIntelligence/good-kernels/tree/5c94620c59ab9839a4ad4ccad309f3d91752d042/Conv2D),
revision `5c94620c59ab9839a4ad4ccad309f3d91752d042`.
The published reference is PyTorch Conv2d: input 100×3×224×224, 96 output
channels, 11×11 kernel, stride four, padding two. No inefficiency was injected.
The implementation converts FP32 operands to FP16, then computes with WMMA
and accumulates in FP32; this numerical approximation belongs to the original.
This is a standalone experiment, not an official KernelBench score.

## Compatibility boundary

[before.cu](before.cu) extracts the device source unchanged, removing only
framework includes and the host wrapper. The runner verifies this against the
saved [published source](upstream_src.py); the [reference](upstream_ref.py)
is preserved and never imported. Translation of that extraction fails because
NVCC emits symbolic `WARP_SZ` for `warpSize`, which the current translator
renders as an undeclared C++ variable. The
[original failure receipt](results/initial-translation-failure.json) is retained.

[compatible.cu](compatible.cu) spells `warpSize` as the literal `32`, the CUDA
warp size. This changes no launch dimensions or algorithm, and allows CPU
execution. [The compatibility diff](results/before-to-compatible.patch) is
separate from the optimization. No DrPerf/GX implementation was changed.

## CPU-only feedback and change

DrPerf counts **524,288 tensor FLOPs per 128×128×16 tile product**, including
work for output channels beyond the useful channel count. At the original
96 output channels, two of eight warps compute channels 96–127, whose output
is discarded. This resource feedback directed reducing the channel tile from
128 to 96 and the block from eight warps/256 threads to six warps/192 threads.
The spatial tile, reduction tile, FP16 arithmetic, pipeline and memory layout
algorithm are retained. The shared allocation adjusts with the channel tile.

[after.cu](after.cu) reduces tensor work to **393,216 FLOPs per tile product**,
a **25% reduction**. The runner independently checks every invocation against
`2 * tile_channels * 128 * 16 * tile_products`. Requested global reads and
writes remain unchanged at the tested shapes; output writes are exactly
`4 * outputs`. Requested shared bytes decrease. These resource counts do not
establish a timing improvement. The tile is chosen for the published 96-channel
workload, not claimed to be optimal for every output-channel count.

PCVs describe outputs, channels, spatial pixels, reduction length, spatial
tiles, reduction tiles, and their product. Training includes one/two spatial
tiles and reduction lengths 1, 9 and 27. Held-out cases include padding,
stride two, multiple batches, 17 output channels, and the original reduction
length 363 with stride four. Each shape has seeded random exact-FP16 binary
fractions, constants and nonrepresentable finite FP32 random operands.
Both variants pass **33 numerical cases**, with maximum absolute error below
3.6e-4. All outputs are checked against an independent double-precision convolution
of the original FP32 operands. Exact binary-fraction tests use tight
2e-5 absolute/relative tolerance; general FP32 tests allow 2e-3 to accommodate
the source's FP16 conversion. Inputs remain unchanged and output guards pass.

The training dimensions are correlated, so the accepted tensor fit is a
linear combination of several dimensions rather than uniquely identifying
the simple tile-product rule. The receipts retain those actual coefficients;
the rule above is independently checked on held-out observations.

Memory-site and scalar-FLOP fits include unexplained portions: NVCC unrolls
spatial output sites with different bounds, and the PCVs do not separately
encode every tile-edge distribution. These rejected fits are retained. The
tensor-work formula and total output-write rule are separately checked on
all observations, rather than claiming every resource interface fit cleanly.

## Evidence and reproduction

```sh
python3 gpu/cases/kernelbench-conv2d-debug/run.py
```

The runtime/translator must already be built. `CUDA_HOME` defaults to
`/home/ubuntu/GX/toolchains/cuda13`. SM80 PTX uses the source's fast-math flag.
Temporary modules, cubins, PTX and raw captures are removed on exit.
[verification.json](results/verification.json) retains source/tool hashes,
commands, compiler diagnostics and numerical/resource gates.

- [Baseline report](results/compatible-report.txt), [after report](results/after-report.txt)
- [Training baseline](results/compatible-train.json), [training after](results/after-train.json)
- [Held-out baseline](results/compatible-heldout.json), [held-out after](results/after-heldout.json)
- [Optimization diff](results/compatible-to-after.patch), [driver/oracle](driver.cc)
- [Offline SM80/SM103 native compiler resources](results/offline-native-resources.json)
- [Native harness](native_benchmark.cu), [bounded RCP controller](run_native_rcp.py)

Offline native compilation reports registers 122→121 on SM80 and 94→94 on
SM103; both targets have zero spills. Shared memory decreases 35,840→30,720
bytes per block. These diagnostics were obtained without physical GPU access.

## Native RCP validation

After freezing the variants and CPU gates, one **NVIDIA B300 SXM6 AC** on RCP
node `gpu405.rcp.epfl.ch` measured the kernels with CUDA 13, `-O3`,
`--use_fast_math` and `-arch=native` (SM103). All shapes use 96 output channels,
3 input channels, 11×11 kernels, stride four and padding two.

| Input shape (N×C×H×W) | Baseline, µs | After, µs | Change |
|---|---:|---:|---:|
| 1 × 3 × 32 × 32 | 62.865 | 71.391 | 13.56% slower |
| 8 × 3 × 224 × 224 | 81.066 | 90.207 | 11.28% slower |
| **100 × 3 × 224 × 224 (original)** | **722.587** | **632.577** | **12.46% lower** |

The exact original-shape medians are 722.586730957 and 632.577087402 µs,
a 1.142× speedup. Smaller workloads regress despite the same 25% reduction
in counted tensor work. The change was not tuned again after hardware timing.

The baseline/candidate share the same seeded finite uniform FP32 input,
weights, biases, buffers, stream, 128-pixel spatial tiles and 16-element
reduction tiles. The controlled change is the channel tile/block size:
128 channels/256 threads versus 96 channels/192 threads. Every output at all
three shapes was checked against an independent CPU convolution accumulated
in double precision and rounded to FP32. Both variants passed the 3e-3
absolute/relative tolerance and output guards, with maximum absolute error
below 9.84e-4. Their matching error reflects the original FP16-operand
approximation, which this optimization preserves. This is neither bitwise
FP32 Conv2d equivalence nor an official KernelBench correctness score.

CUDA events follow 30 warmups per variant and 11 paired rounds of 30 launches,
alternating variant order. The table uses median per-round average times.
Transfers, allocation and CPU validation are excluded. Timings describe
repeated launches on reused buffers. SM103 reports 94 registers and zero
spills for both variants, with shared memory 35,840→30,720 bytes per block.
Resource fits are CPU-only PTX measurements; hardware timing is a separate
validation and is not inferred from the 25% resource reduction.

- [Exact samples and all-output checks](results/drperf-public-1026-164544/native/conv2d/native.json)
- [SM103 compiler diagnostics](results/drperf-public-1026-164544/native/conv2d/compile.log)
- [GPU receipt](results/drperf-public-1026-164544/native/gpu-before.txt)
- [Allocation, source hashes, commands and confirmed deletion](results/drperf-public-1026-164544/allocation.json)
- [Paired CPU resource comparisons](results/paired-resource-checks.json)

The 59-second allocation also validated LayerNorm. The exact owned RCP job
was deleted and its pod verified absent. No benchmark ran on rs3lab.
To repeat both frozen native experiments in one short one-GPU allocation:

```sh
python3 gpu/cases/kernelbench-conv2d-debug/run_native_rcp.py
```
