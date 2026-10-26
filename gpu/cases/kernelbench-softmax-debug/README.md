# GPU-free debugging of a public softmax kernel

This experiment uses the AI-generated CUDA softmax published in
[ScalingIntelligence/good-kernels](https://github.com/ScalingIntelligence/good-kernels/tree/5c94620c59ab9839a4ad4ccad309f3d91752d042/Softmax),
revision `5c94620c59ab9839a4ad4ccad309f3d91752d042`. No defect was injected.
The upstream reference is PyTorch row-wise softmax. The published workload has
4,096 rows of 65,536 FP32 elements, with one 1,024-thread block per row.

We extracted the CUDA device code unchanged, removing only the unused torch
include and host wrapper. We tested smaller batches and several row widths,
including 65,536, using DrPerf's vendored GX PTX execution on the CPU.
**The initial debugging and optimization stage used no GPU.** A subsequent
user-requested RCP hardware measurement is recorded below. This is not an
official KernelBench timing run.

## What the tools contributed

1. **Numerical execution exposed a synchronization defect.** The unchanged
   kernel failed the double-precision CPU softmax oracle at widths 256 and
   65,536. It reads the row maximum from `sdata[0]`, then reuses that shared
   buffer for per-thread exponential sums without an intervening barrier.
   Another thread can overwrite the maximum before its peers read it.
   Adding `__syncthreads()` after the read made all tested inputs pass.
   This was exposed by emulation plus the numerical oracle, **not by DrPerf's
   resource fitting**, and has not been reproduced on physical hardware.

2. **DrPerf exposed redundant global requests.** On six small PCV states,
   the synchronized kernel's fitted interfaces were:

   ```text
   global bytes read    = 12 * elements
   global bytes written =  8 * elements
   FP32 arithmetic     =  5 * elements + 2047 * rows
   special functions   =  1 * elements
   ```

   The read-site breakdown contains three contributions of `4 * elements`:
   maximum computation, exponentiation, and normalization. The two write
   contributions are temporary exponentials and final output. This directed
   the change toward retaining inputs and exponentials per thread.

3. **An initial cache was too large.** Caching 16 `float4`s per thread
   compiled to 100 registers per thread, exceeding A100's register budget
   for the original 1,024-thread block. That version was discarded. The
   final version caches eight `float4`s per thread and retains the original
   staging path for remaining vectors. Offline `ptxas` compilation with
   `__launch_bounds__(1024)` reports 64 registers and zero spills, versus
   40 registers for the synchronized baseline. Compilation requires no GPU.

4. **DrPerf confirmed the request reduction, and the oracle checked outputs.**
   Both synchronized and optimized versions passed 18 training tests and
   36 held-out tests each. Maximum absolute numerical error was below
   `0.000000125`. FP32 arithmetic and special-function counts were unchanged.

| Scope | Synchronized baseline | Optimized | Reduction |
|---|---:|---:|---:|
| Requested global bytes, small training states | `20 * elements` | `8 * elements` | 60% |
| Requested global bytes, one 65,536-element row | 1,310,720 | 917,504 | 30% |
| Global reads, one 65,536-element row | 786,432 | 524,288 | 33% |
| Global writes, one 65,536-element row | 524,288 | 393,216 | 25% |

These are **requested bytes, not HBM traffic or measured latency**. Cache
behavior, register pressure and execution scheduling can change the GPU
performance outcome. These counts alone establish no GPU speedup or advantage
over another agent; the subsequent hardware measurement establishes the timing
comparison below.

## Real-GPU measurement on RCP

At the user's request, the frozen synchronized and optimized kernels were
measured on **one NVIDIA B300 SXM6 AC**, on RCP node `gpu402.rcp.epfl.ch`.
No benchmark was run on rs3lab. The RCP job was deleted after collecting results.

| Shape (rows × columns) | Baseline, µs | Optimized, µs | Latency change |
|---|---:|---:|---:|
| 4,096 × 256 | 82 | 83 | 1% slower |
| 4,096 × 4,096 | 123 | 105 | 15% lower |
| 4,096 × 32,768 | 338 | 243 | 28% lower |
| **4,096 × 65,536 (original workload shape)** | **587** | **512** | **13% lower** |

Displayed values are rounded; the exact original-shape medians are 587.047
and 511.621 µs, a 12.848% latency reduction (1.147× speedup).
The small-case regression is retained: this is not a universally faster kernel.

Both kernels were compiled together with CUDA 13, `-O3 -arch=native`, targeting
SM103, and without fast-math flags. They use identical seeded normal FP32
inputs, 1,024-thread blocks, 4,096 bytes of dynamic shared memory, and the same
device buffers and stream. Every output element at all four shapes was checked
against a CPU softmax reference computed in double precision and rounded to
FP32. Both kernels passed; output guard values also remained intact.

Timing uses CUDA events after 30 warmup launches per variant, with 11 paired
rounds of 30 launches each, alternating variant order. The table reports the
median per-launch time across rounds. Allocation, host/device transfers and
numerical validation are outside the timed interval. This measures repeated
kernel launches on reused buffers, not serving latency or cold-cache latency.
For the original shape, all round means were within 1 µs of their respective
medians. Registers remained 40 versus 64. Unlike the earlier SM80 compilation,
the B300 SM103 compiler inserted 20 bytes of spill stores and 20 bytes of spill
loads in the optimized kernel, with a 24-byte stack frame. The baseline had no
spills. These are static compiler diagnostics, not measured dynamic byte totals;
the hardware timings include their effect. Hardware timing was validation after the
GPU-free optimization, not feedback used to tune the kernel further.

- [Exact timing samples and correctness results](results/drperf-softmax-1026-151904/native/native.json)
- [Allocation, commands, source hashes and confirmed deletion](results/drperf-softmax-1026-151904/allocation.json)
- [Native timing and numerical harness](native_benchmark.cu)
- [RCP controller with bounded allocation and automatic cleanup](run_native_rcp.py)

To repeat on RCP using the existing authenticated `vca-jma` account:

```sh
python3 gpu/cases/kernelbench-softmax-debug/run_native_rcp.py
```

This command requests one B300 for the measurement; it is separate from the
CPU-only reproduction command below.

## What did not fit

The small training states fit cleanly before and after the optimization.
Refitting on the wider held-out states reports unexplained contributions
for both versions. NVCC distributes iterations over different unrolled PTX
sites, whose individual counts are piecewise functions of row width. The
two declared PCVs (`elements`, `rows`) do not explain those sites separately.
The optimized version also introduces a cache-capacity threshold.

For comparison, source inspection gives this aggregate resource rule, which
the runner checks exactly against **every invocation's measured counts**:

```text
N = rows * columns
C = rows * min(columns, 32768)  # cached elements; tested widths divisible by 4
reads  = 12*N - 8*C
writes =  8*N - 4*C
```

This is an independently checked aggregate rule, **not a formula DrPerf
successfully inferred across the whole sweep**. Adding `C` alone would not
necessarily explain the distribution across all unrolled sites. The
unexplained results are retained in `results/*-heldout.json`. This experiment
shows usefulness for directing a change, while also exposing a limitation of
fitting each compiled instruction site independently.

## Inputs and evidence

- Training widths: 256, 1,024, 4,096; batches: one and two rows.
- Held-out widths: 4, 252, 4,100, 8,192, 65,536, 65,540; same batches.
- Each shape uses seeded random finite values, constant values, and a sharply
  peaked row. Validation uses a double-precision stable softmax oracle,
  normalization checks, immutable-input checks and output guards. It does not
  claim exhaustive correctness or test NaNs/infinities.
- The original 4,096-row batch was not executed. The original row width and
  block size were tested. Scalar-tail and alternative-block-size fallback
  paths are retained but not validated by this experiment.
- [Driver and oracle](driver.cc), [baseline](before.cu),
  [barrier fix](synchronized.cu), [optimized kernel](after.cu).
- [Original DrPerf baseline report](results/synchronized-report.txt) and
  [optimized report](results/after-report.txt) describe the training states.
- [Verification](results/verification.json) records commands, tool hashes,
  compilation diagnostics, numerical failures/passes and elapsed times.
- `results/*-train.json` and `results/*-heldout.json` retain selected original
  fits, state statistics and memory-site breakdowns. These are compact evidence
  extracts, not complete DrPerf profiles for loading into the extension.
- `results/*.patch` gives both code changes.

The complete final reproduction took 111 seconds on this CPU host, including
compilation, translation, emulation and reporting. This is not a GPU slowdown
measurement. Artifacts occupy less than 1 MB; raw captures and generated
modules are deleted after collecting the compact evidence.

## Reproduce

From the DrPerf repository, with its GPU runtime/translator already built:

```sh
python3 gpu/cases/kernelbench-softmax-debug/run.py
```

`CUDA_HOME` selects a CUDA toolkit containing `nvcc` and `ptxas`; the default
is `/home/ubuntu/GX/toolchains/cuda13`. The runner uses a temporary directory,
checks the baseline extraction against the pinned upstream source, and
removes generated PTX, modules, cubins and JSONL traces on exit. Upstream
Python files are preserved as evidence and are never imported or executed.
