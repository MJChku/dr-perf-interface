# GPU resource interfaces

`drperf --gpu` is a separate execution and fitting path. GPU markers use
`drperf_gpu.h`; CPU markers continue to use `perfmark`. The GPU command does
not import the CPU collector, start DynamoRIO, or count instructions executed
by the C++ emulator. Only `analysis/affine.cpp` / `lib/native_fit.py`, the
numerical affine checker, are shared.

The GPU path translates PTX to instrumented C++, executes that code on CPU
using its own copied GX lane/fiber, barrier, warp-exchange and matrix runtime,
and fits work counts against entry PCVs. It needs no physical GPU, CUDA
toolkit, running GX process, installed `gx_cuda.so`, or GX checkout.

## Build and run

Dependencies: a C++17 compiler, Boost.Context and Boost.Fiber development
libraries, and the Lean version in `vendor/gxptx/lean-toolchain` through
`elan`/`lake`. On Debian/Ubuntu the Boost packages are
`libboost-context-dev libboost-fiber-dev`. The build explicitly uses distro
Boost headers to avoid incompatible `/usr/local` installations. A different
header root can be selected with `DRPERF_GPU_BOOST_INCLUDE`.

From the drperf repository:

```sh
bin/drperf --gpu build
bin/drperf --gpu translate gpu/examples/saxpy.ptx -o gpu/build/saxpy.so
g++ -std=c++17 -O2 -Igpu/include gpu/examples/saxpy.cpp \
    -Lgpu/build -ldrperf_gpu -Wl,-rpath,"$PWD/gpu/build" -o gpu/build/saxpy
bin/drperf --gpu --output gpu/build/saxpy-report -- \
    gpu/build/saxpy gpu/build/saxpy.so
bin/drperf --gpu --report gpu/build/saxpy-report
python3 gpu/tests/test_gpu.py
```

The example checks the numerical output and guards for eleven sizes, including
empty work and partially populated blocks. Its interfaces include:

```text
fp32_flops        = 2 * elements
global_read_bytes = 8 * elements
global_write_bytes = 4 * elements
launched_lanes    = 128 * blocks
warps             = 4 * blocks
sum_block_max_lane_flops = 2 * active_blocks
```

`blocks` includes the empty launch used to test `n=0`; `active_blocks` counts
tiles containing input elements. This extra PCV explains the parallel-work
maxima across both the empty and nonempty cases.

The generated `.cpp`, `.so` and `.gpu.json` translation manifest are retained.
The manifest pins the PTX, translator, prelude, runtime and resulting module.
PTX can be produced separately with `nvcc -ptx`; a CUDA toolkit is not needed
when the input is already PTX. This is a standalone translated-kernel runner;
it does not intercept CUDA/BLAS calls in arbitrary existing application binaries.

## Markers and launch API

```cpp
#include "drperf_gpu.h"

drperf::gpu::Module module("my-kernels.so");
{
    drperf::gpu::Region r("attention", {
        {"tokens", tokens}, {"attended_pairs", attended_pairs},
        {"blocks", blocks}
    });
    // Each pointer in args addresses one host parameter value, in PTX order.
    // Device pointer parameters contain real host-buffer addresses.
    void* args[] = {&input_pointer, &output_pointer, &tokens};
    module.launch("attention", dim3(blocks), dim3(128), args, dynamic_shared_bytes);
}
```

The host marker captures region name and signed-integer PCVs. Launches execute
synchronously and belong to the innermost GPU region on that host thread.
Nested parent/child counts are exclusive. Separate host threads can use their
own GPU regions and independent launches. CPU `perfmark` markers do not open a
GPU region. GPU markers do not open a CPU region.

The runner provides logical CUDA `threadIdx`, `blockIdx`, `blockDim` and
`gridDim`, protected per-block shared memory, cooperative lane contexts,
named block barriers, masked warp shuffles/votes/reductions, and the copied
GX MMA/WMMA arithmetic. It executes every lane; no representative-lane
shortcut or instruction-to-CPU-time conversion is used. Blocks run
sequentially on each calling host thread. Cross-block cooperative execution,
clusters, TMA, tcgen05, WGMMA and asynchronous CUDA stream scheduling are not
implemented in this standalone runtime. Missing runtime helpers fail module
loading, and unsupported executed instructions fail the command.

## What is counted

Counters are inserted at original PTX statement boundaries, inside predicates
and after labels. Returns and branches are counted before taking effect.
Each statement has its own generated-C++ line/function/opcode attribution.
Parameter, local, constant, shared and global spaces remain separate; generic
memory addresses inside the current shared arena are recognized as shared;
other unqualified accesses retain a separate `generic_*_bytes` category.

- Scalar FLOPs are separated into FP16, BF16, FP32 and FP64. Add/subtract,
  multiply, divide, square root, floating min/max count as one; FMA/mad count
  as two; packed operations multiply by their element count. Transcendentals,
  reciprocal and reciprocal square root have a separate special-function count.
- Tensor floating work and tensor integer work are separate. An `m*n*k` MMA
  contributes `2*m*n*k` across the participating warp, including padded tile
  work. The runtime's scalar implementation of the matrix product contributes
  no additional work counts. Per-site opcodes retain precision/type details.
- Global/shared/local/constant/parameter read and write bytes are **requested
  PTX bytes**. Atomics contribute both read and write bytes. Matrix transfers
  count the logical operand tile once per collective. Counts do not describe
  cache-line transactions, bank conflicts, cache hits or actual HBM traffic.
- Parallelism features include blocks, launched lanes, warps, per-launch
  grid/block shape, shared-memory declarations, per-block maximum lane FLOPs
  and bytes, and a logical lane-sequence span with block-barrier joins.
  `sum_block_*` sums per-block maxima. The region's
  `sum_launch_max_block_lane_sequence_span` sums per-launch maximum block
  spans. Span uses unit PTX statement steps and serializes each lane's
  instruction order; it has no instruction-level parallelism, warp-exchange
  dependency joins or cross-block edges. It is not a hardware critical path.
- PTX block-barrier and other collective lane arrivals are recorded separately.
  `emulator_*` metrics describe implementation rendezvous and are diagnostic;
  internal matrix-helper barriers must not be interpreted as GPU barriers.

Work whose counter semantics are not implemented, such as `cp.async`, records
`unmodeled_work_events`. A runnable module can therefore produce an incomplete
report and exit status 2. Incomplete work is never presented as a complete
resource measurement. Both translation and launch reject missing counters;
module instrumentation ABI mismatches are rejected.

The copied translator's functional operation models remain tested/trusted at
their existing boundaries. The GPU instrumentation is an empirical collector,
without a new Lean proof of cost counts or native GPU numerical equivalence.

## Fitting and reports

`--gpu` defaults to `drperf-gpu-report/`, separately from CPU `drperf-report/`.
It writes `report.txt`, `profile.gpu.drperf.json`, and retained per-process raw
JSONL captures. CPU and GPU report schemas and query routes are separate:

```sh
bin/drperf --gpu --report drperf-gpu-report --json
```

GPU reports currently use the CLI/text and JSON views; the CPU architecture
graph viewer is not used by this path.

The C++ affine checker fits each resource metric and each statement site's
metrics against all declared PCVs plus a constant. **Every invocation** is a
fit observation, including repeated calls with the same PCV state. Variation
at one state can reject a fit. The report exposes coefficients, dependent
PCVs, maximum reconstruction error, rejected site fits and raw observations.
Unexplained work is feedback, so a complete capture with rejected fits returns
0. Missing lifecycle completion, open markers, failed launches, unmarked
launches or unmodeled work return nonzero. Input counts/PCVs beyond the exact
double-integer range are rejected before fitting.

PCVs supply nonlinear structure: `pairs=n*(n-1)/2`, ceiling divisions for tile
counts, or mode-dependent terms. The tests show a triangular-work fit rejected
with only `n`, then accepted after adding `pairs`. Single-state and correlated
PCV observations are disclosed; acceptance applies only to observed inputs.

These are work/parallelism interfaces. There is no fitted GPU latency model,
occupancy inference, hardware bandwidth calibration, or stream-overlap model.

## Vendored code

`vendor/provenance.json` records the GX source revision and original source
digests; `vendor/GX-LICENSE` preserves its MIT license. `vendor/gxptx` contains
the translator source, with local statement instrumentation and a counter ABI
declaration in its prelude. `vendor/gx` contains the lane/fiber, barrier,
warp-exchange, stack, shared-arena and floating/tensor arithmetic machinery.
The standalone host/runtime adapter lives in `runtime/`. Build artifacts and
experiments stay under `gpu/build/` and are ignored by Git.
