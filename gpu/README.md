# GPU resource interfaces

`drperf --gpu` is a separate execution and fitting path. GPU markers use
`drperf_gpu.h`; CPU markers continue to use `perfmark`. The GPU command does
not import the CPU collector, start DynamoRIO, or count instructions executed
by the C++ emulator. Only `analysis/affine.cpp` / `lib/native_fit.py`, the
numerical affine checker, are shared.

The GPU path translates PTX to instrumented C++, executes that code on CPU
using its own copied GX lane/fiber, barrier, warp-exchange and matrix runtime,
and fits work counts against entry PCVs. WGMMA, TMA, bulk operations and
`tcgen05` use GX's canonical implementations, with descriptor-derived resource
counts and cluster/grid coordination. It needs no physical GPU, CUDA
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
python3 gpu/tests/test_runtime_parity.py
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

### Regions inside CUDA kernels

Agents can choose launch-level host regions, regions inside a kernel, or both.
Include `drperf_gpu_device.cuh` in CUDA sources and put a region around any
lexically scoped section, including an inner loop or a device function:

```cpp
#include "drperf_gpu_device.cuh"
using drperf::gpu::DeviceRegion;
using drperf::gpu::DeviceScope;

__global__ void example(float* out, int history, int selected) {
    const int lane = threadIdx.x;
    {
        DeviceRegion clear("dense_clear", DeviceScope::block, {{"history", history}});
        for (int i = lane; i < history; i += blockDim.x) out[i] = 0.f;
    }
    // Other work can have its own region, PCVs, and nesting.
}
```

The constructor's PCV list registers device-computed signed-integer values,
so PCVs can depend on local indices, loads, and intermediate state. Expressions
must have integer types of at most 64 bits and fit the signed range; report
fitting also requires exact-double range (absolute value at most `2^53`).
All expressions are evaluated before opening the region, and the constructor
seals the PCV list before returning. There is no `.pcv()` mutator: body results
cannot be registered later as entry PCVs. Duplicate names and low-level PCV
registration after entry are errors. C-style paired markers are available as
`DRPERF_GPU_DEVICE_BEGIN(name, DeviceScope::lane, {"n", n}, {"pairs", n*(n-1)/2})`
and `DRPERF_GPU_DEVICE_END()`.
Prefer RAII so early returns close regions correctly; paired markers must be
balanced on every participating lane.

| Scope | Observation and participation |
|---|---|
| `DeviceScope::lane` (default) | One invocation per entering lane; permits divergent entry and different lane PCVs. |
| `DeviceScope::warp` | One invocation per physical warp, including a partial final warp; every physical member must enter/exit with identical PCVs and nesting. |
| `DeviceScope::block` | One invocation per block; every launched lane must enter/exit with identical PCVs and nesting. |

Cooperative regions introduce no CUDA barriers. Use a lane region for divergent
or elected-lane code. Place warp/block markers outside conditional work when
the intended observation is the whole group. Cooperative scopes must have the
same named occurrence order on every member; repeated loop occurrences remain
separate observations. A cooperative child needs the same parent invocation on
all members: a warp under a block is valid; a block under separate lane parents
is not. Missing participation or mismatched PCVs invalidates the capture.

Reports name these `device.lane.NAME`, `device.warp.NAME`, or
`device.block.NAME` and retain scope, kernel, linear block, group and parent
token for every invocation. Work belongs exclusively to the innermost region
on its executing lane. Host parents retain unannotated work, geometry and
launch-wide span metrics; `launches[].metrics` retain **inclusive** totals.
Device scopes report `region_lane_participants`, not an inferred scoped span.
When a device marker executes without a host GPU region, the collector creates
`kernel.KERNEL_NAME` for the remaining work. A launch with no executed markers
still needs a host region, including when conditional annotations are skipped
by every lane. CPU `perfmark` markers remain separate.

Compile the annotated sources to PTX, then use the normal translator and
standalone launch harness:

```sh
nvcc -std=c++17 -O2 -Igpu/include -arch=compute_103a -ptx kernel.cu -o kernel.ptx
bin/drperf --gpu translate kernel.ptx -o kernel.so
```

The header deliberately declares external device markers. Defining their
no-op bodies in the header lets NVCC specialize away arguments and destroys
the instrumentation ABI. For native CUDA, either define
`DRPERF_GPU_DISABLE_DEVICE_REGIONS` to erase the markers, or compile the kernel
and `gpu/runtime/device_marker_stubs.cu` separately with `nvcc -dc`, then device
link them. Do not supply marker definitions during the annotated `-ptx` build
or device-LTO them into it. Marker calls, bodies, and the constructor's
registration phase are excluded from resource counts. PCV expressions evaluated
before entry and caller-side parameter preparation may still be counted in an
enclosing region; annotations can affect optimization. These are resource explanations
of the compiled annotated PTX, not zero-overhead hardware measurements.

See [the runnable inner-region example](examples/device_regions/README.md),
which first rejects an incomplete PCV explanation and then refines it.

The runner provides logical CUDA `threadIdx`, `blockIdx`, `blockDim` and
`gridDim`, protected per-block shared memory, cooperative lane contexts,
named block barriers, masked warp shuffles/votes/reductions, and the copied
GX MMA/WMMA arithmetic. It executes every lane; no representative-lane
shortcut or instruction-to-CPU-time conversion is used. Independent blocks run
sequentially on each calling host thread; cluster and cooperative-grid launches
run resident CTA fibers. CTA-group-1/2 `tcgen05` supports tensor
memory allocation/load/store, copy, shift, reduction, commit and MMA execution;
CTA-local mbarriers retain phase/arrival dependencies. BF16 and block-scaled
NVFP4 MMA numerical fixtures and resource-count oracles are included.
WGMMA warpgroup arithmetic, TMA tiled/gather/scatter copies, descriptor edits,
bulk copies/reductions, cluster shared memory and grid synchronization use the
same functional policies as GX. Asynchronous operations complete eagerly, with
their required collective/barrier dependencies retained; asynchronous CUDA
stream scheduling is outside this standalone runner. Use
`module.launch_cluster(kernel, grid, block, cluster, args, dynamic_bytes)` for
explicit cluster geometry. Encode standalone TMA descriptors through
`drperf_gpu_tensor_map.h`, rather than using opaque hardware driver bytes.
See [the runtime support matrix and tests](RUNTIME_PARITY.md) for boundaries
and counter definitions. Missing runtime helpers fail module loading, and
unsupported executed instructions fail the command.

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
- For `tcgen05.mma`, the instruction descriptor supplies M/N/K and operand
  formats at runtime. One issuing lane is charged the complete tile once;
  these counts are not divided among 32 lanes. `tensor_mma_instructions`
  counts issued MMAs; `tmem_*_bytes` distinguish tensor memory from shared and
  global memory. Implicit MMA operand bytes are logical tile footprints, not
  measured transactions. `tensor_memory_allocated_bytes` sums requested TMEM
  capacity over allocations, not peak live capacity. Packed TMEM accesses count
  logical requested bits, not the host helper's read/modify/write operations.
  Scalar reference/fast-path implementation loops never add resource counts.
  Masked/sparse/weight-stationary/ashift MMA and compressed TC copies currently
  execute where GX supports them but mark accounting incomplete; the tested
  dense unmasked BF16/NVFP4 subset has complete implemented counters.
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

Work whose counter semantics are not complete, such as `multimem` or the
remaining sparse/masked/weight-stationary tcgen05 accounting, records
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

Lean translation assigns each PTX instruction a basic-block ID, carried in its
generated counter name. Function entry and labels start blocks; branches,
returns and calls end them. Predicated instructions stay in their block, and
their counters still respect active lanes and predicates. Marker boundaries
also split blocks. The reporter does not reconstruct control flow.

The C++ affine checker fits each basic block's summed resource counts against
all declared PCVs plus a constant. Repeated invocations with the same PCV values
are averaged first; every distinct state mean must meet the tolerance. States
receive equal weight regardless of invocation frequency. `stateStatistics`
retains the mean, minimum, maximum and invocation count; raw observations remain
available. A single state can yield a constant without establishing scaling.

For additive metrics (FLOPs, actual load instructions, requested bytes), the
region's affine part sums only accepted basic-block fits. Rejected blocks'
entire observed costs remain in `unexplainedByState`, with a block breakdown.
There is one fitting step: no further grouping by function, path, kernel or
region, even if rejected blocks would cancel into an affine total. Geometry
and runtime counters without a statement site form a separate contribution.
Unique-address footprints and maximum-span metrics are fitted as region
aggregates; they are not sums of statement costs or child footprints.

Raw `observations[].sites` retain instruction counts. The report's historical
`sites` fit table now contains block fits, with `basicBlockMembers` mapping each
block to its instructions and `fitSemantics=state-means/basic-block-sum-v3`.
Old modules without block IDs retain separate instruction units and an explicit
legacy notice; rebuild/retranslate and capture again to use basic-block fitting.

The report composes additive resources with `(a*PCVs + d)*F[child]`. It checks
the child-call multiplicity exactly for **every parent invocation**, including
zero calls, rather than averaging structural differences away. Unfitted counts
become `unexplained(F[child])`, retaining the child's entire observed inclusive
cost. An accepted `F` references the child's full interface, including its own
unexplained costs and descendants; it is not multiplied by a global child mean.
Child PCV histograms are retained per parent. `fits` contains own costs;
`interfaces` and the text report include composition. Raw parent identities
are scoped to their capture and process. Missing or cyclic parentage is an error.
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
`global_load_instructions` and `generic_load_instructions` count executed
scalar/vector PTX load statements per lane (one vector instruction is one
load; requested bytes express its width). Assumed `latency_*` values and
`dependent_global_loads_on_critical_path` from the experimental issue model
remain raw diagnostics, never fitted resource interfaces. The latter depends
on latency assumptions and must not be confused with an actual load count.

Existing reports can be regenerated with `--gpu --refit REPORT_DIR`; only
the capture named in the report's provenance is used, even after relocation.
Device annotations using the old incremental `.pcv()` API must be changed to
constructor PCV lists and their PTX regenerated and translated. Legacy device
entry markers fail explicitly. Host GPU region annotations are unchanged.

### Refitting, exact coefficients and readable sites

`bin/drperf --gpu --refit REPORT_DIR [--absolute-tolerance A --relative-tolerance R]`
re-fits the retained raw captures of an existing report without re-running the
emulation. Fits first try all-integer coefficients, then scale-aware snapping,
and keep a candidate only if it still reconstructs every invocation, so least
squares noise such as `2e-07 * pcv` beside an `8912896 * pcv` term disappears;
integers print exactly instead of `%g` rounding (`8912896`, not `8.9129e+06`).
Unexplained-site lines name the demangled kernel without template arguments
(`smxx::decode::flash_fwd_mla_combine_kernel @ cpp3773:ld.global.nc.s32`).

### Allocation alignment

Sector and line footprints depend on where buffers start. `drperf::gpu::Buffer<T>`
(in `drperf_gpu.h`) allocates 256-byte aligned storage like `cudaMalloc`; use it
for kernel arguments instead of `std::vector`, whose 16-byte alignment makes the
same kernel's sector/line counts shift with the host allocator. Line footprints
of records whose size is not a multiple of 128 bytes (e.g. FlashMLA's 656-byte
KV rows) are floor/ceil functions of the selection and need such PCVs to fit.

### Lane stack size

Each simulated lane runs on its own guard-page protected fiber stack. Translated
kernels keep every PTX register and `.local` array as C locals (plus a shadow per
register when the latency model is on), so large kernels such as the FlashMLA
decode core (~6000 registers) overflowed boost's 128 KiB default and crashed with
SIGSEGV inside the translated kernel. The default is now 512 KiB (committed
lazily, so untouched pages cost no memory); `DRPERF_GPU_LANE_STACK_BYTES`
(or `GX_FIBER_STACK_BYTES`) sets another size in [65536, 1073741824]. A SIGSEGV
in a `*_gxptx` frame is the usual sign that the stack needs to grow.

### Uninitialized reads (initcheck)

`drperf::gpu::Buffer<T>::uninitialized(n, "name")` allocates a 256-byte aligned
buffer whose contents are undefined, like fresh `cudaMalloc` memory, and
registers it with the collector (`drperf_gpu_mark_uninitialized`). Host writes
are declared with `buffer.host_written(first, count)`. Device writes (ordinary
stores, atomics, TMA and bulk copies) mark bytes initialized; a device read of a
byte nothing has written is counted as `uninitialized_read_bytes` at the reading
site, fitted like any other metric, and the first 20 occurrences are printed with
site, buffer name and offset. On hardware such reads return whatever an earlier
launch or process left behind, which makes results depend on run order. Only
registered buffers are checked; shared memory and unregistered allocations are
not. `gpu/tests/test_initcheck.py` learns `uninitialized_read_bytes = 4*n - 4*written`
for a consumer that reads a partially written scratch buffer.

## GLM-5.3 scaling example

[The reproducible example](examples/glm53_scaling/README.md) uses two frozen
GLM kernel bodies and a separate compact-output demonstration. It trains on
histories32–512 and checks actual held-out executions at2048 and100000:

- Whole-history materialization writes grow48.828x for a fixed query count.
- Sparse-softmax arithmetic remains unchanged at K2048, while its dense
  output clearing grows requested writes24.914x.
- A compact K-slot output removes the history term from writes, but requires a
  compatible downstream consumer; no serving or latency speedup is claimed.

The initial PCVs are rejected for the affected bytes. Adding history explains
the work; changing the output representation removes that dependency. All
14 held-out byte/SFU predictions have zero count error. Counter/parallelism
residuals outside these accepted interfaces remain visible. The example does
not execute tcgen05; the Blackwell fixtures test that support separately.

## Vendored code

`vendor/provenance.json` records the GX source revision and original source
digests; `vendor/GX-LICENSE` preserves its MIT license. `vendor/gxptx` contains
the translator source, with local statement instrumentation and a counter ABI
declaration in its prelude. `vendor/gx` contains the lane/fiber, barrier,
warp-exchange, stack, shared-arena and floating/tensor arithmetic machinery.
The standalone host/runtime adapter lives in `runtime/`. Build artifacts and
experiments stay under `gpu/build/` and are ignored by Git.

## Global footprint and reuse

Requested bytes count every lane's logical request. To distinguish repeated
metadata requests from an expanding working set, new translations also collect
exact address unions at launch, region-invocation and statement-site scope:

| Metrics (replace `read` with `write` for stores) | Meaning |
|---|---|
| `global_read_bytes` | Existing requested bytes, including repeats |
| `global_unique_read_bytes` | Distinct byte addresses touched |
| `global_unique_read_sectors` | Distinct 32-byte sectors touched |
| `global_unique_read_lines` | Distinct 128-byte lines touched |
| `global_unique_read_sector_bytes` | Sector count × 32 |
| `global_unique_read_line_bytes` | Line count × 128 |

The same metrics with `generic_` retain unresolved generic accesses separately.
Shared, local, parameter, constant and tensor memory are excluded from these
address unions. Generic accesses recognized as shared stay excluded. Read and
write unions are separate, including both sides of an atomic or reduction.
TMA uses in-bounds descriptor-selected addresses, including sub-byte payloads;
OOB fills add no global footprint. Multicast counts the source once. Bulk
stores respect byte masks, `cp.async` respects source tails/ignore predicates,
and WMMA footprints follow the actual row/column stride. Semantic helpers add
logical footprints, never the extra memory operations of their CPU interpreter.
NCCL `annotated_global_*` events remain a separate annotation category.

For 32 lanes loading four-byte words from 128-byte-aligned buffers:

| Access pattern | Requested | Unique bytes | Sectors | Lines | Redundancy |
|---|---:|---:|---:|---:|---:|
| Every lane reads the same word | 128 | 4 | 1 | 1 | 32× |
| Consecutive words | 128 | 128 | 4 | 1 | 1× |
| One word every 32 bytes | 128 | 128 | 32 | 8 | 1× |
| One word every 128 bytes | 128 | 128 | 32 | 32 | 1× |

`redundancyRatio = requestedBytes / uniqueBytes`. View it beside the counts:

```sh
python3 gpu/tools/footprint_report.py gpu/build/my-report --level region
python3 gpu/tools/footprint_report.py gpu/build/my-report --level launch
python3 gpu/tools/footprint_report.py gpu/build/my-report --level site --json
# Also: --level launch-site, --level all (default), --region 'attention*'.
```

The JSON view also gives `requestedPerSectorByte`, `requestedPerLineByte`, and
byte utilization of the covered sectors/lines. Partial-sector coverage can make
requested/sector-bytes less than one; that quantity is distinct from duplicate
byte requests. Ratios with zero denominators are null. Sub-byte TMA requested
payload is rounded per instruction, while unique bytes include every byte
containing a touched bit, so their ratio can also be below one.

All integer footprint metrics flow through the existing raw JSONL `metrics`
and `sites` and the report's region/site affine `fits`. Ratios are derived per
observation by this tool because the existing collector/fitter schema accepts
integer counts only. Rounded sector/line metrics often need ceiling-division
PCVs and consistent buffer alignment; a fit in element count alone can reject.
The tool reads a report directory, JSON file, or compressed `.json.gz` report
without loading the runtime or numerical fitter.

Launch metrics and the added `launches[].sites` are inclusive. Region metrics
and sites are exclusive of nested regions. Unions span all launches in one
region invocation: reading the same address in two launches contributes twice
to requested bytes and once to that region's footprint. Footprints therefore
cannot be added across sites, launches, regions or read/write directions.
Each invocation is still a separate fit observation. Device lane/warp/block
regions keep the existing participation and nesting rules.

Rebuild and retranslate to enable address coverage. Work-counter ABI 3 remains
unchanged; separate prelude and per-kernel translator capability markers both
must be present. New raw observations and launch snapshots carry
`footprintVersion: 1`. Older modules still execute with existing work counters,
but carry version 0 and `footprint_unavailable_*` diagnostics; the viewer shows
unavailable footprints. A region mixing old and new modules also has version 0;
any integer footprint counts retained there cover only its new modules.
Historical reports cannot recover addresses retroactively. The main report's
legacy memory-semantics/header text still describes requested-byte counters;
use the metric names and this viewer to distinguish the new unique counts.

These are address coverage and reuse measurements, not measured L1/L2 hits,
HBM bytes, transactions, GPU time or a cache simulator. A smaller request count
with an unchanged small footprint may leave native GPU time unchanged. No LRU
L2 or per-SM L1 estimate is implemented. Addresses are the runner's host-buffer
proxies for device buffers; use CUDA-like aligned allocations for reproducible
sector/line coverage. Address reuse after free/reallocation within one region
is treated as the same address: buffer lifetimes are not tracked. Interval
unions compress contiguous streams but sparse patterns can consume memory in
proportion to their disjoint ranges. No host-copy or host-launch-cost accounting
is added.

CPU-only regression tests:

```sh
python3 gpu/tests/test_footprint.py
python3 gpu/tests/test_report_tools.py
python3 gpu/tests/test_interfaces.py
python3 gpu/tests/test_basic_blocks.py
```

`gpu/tests/BasicBlocks.lean` also checks the translator's block boundaries
(predication, labels, branches, calls, returns and markers) with Lean `#guard`.

## Comparing fitted interfaces

Compare two retained GPU reports after a kernel change:

```sh
python3 gpu/tools/diff_reports.py gpu/build/before gpu/build/after
python3 gpu/tools/diff_reports.py gpu/build/before gpu/build/after --json
python3 gpu/tools/diff_reports.py before.json.gz after.json.gz \
    --region 'attention*' --metric 'global_*' --fail-on-change
```

The tool compares region totals and each fitting unit's metric coefficients,
constants, fit acceptance and dependent-PCV disclosures. Deltas are **after
minus before**, with relative change where the old coefficient is nonzero.
Added/removed regions, metrics and sites are explicit. The JSON retains both
full fits, including reconstruction errors, and flags differing scope/PCV
schemas as non-comparable. Rejected fits stay visible as diagnostics. Capture
validity is shown for both inputs; neither a rejected fit nor an incomplete
capture is evidence of a verified improvement. Observation counts, measured
values and fit-error-only changes are not coefficient-interface changes.

`--atol` and `--rtol` set nonnegative coefficient/constant comparison tolerances
(default zero). `--include-unchanged` includes identical interfaces. Exit status
is 0 for a successful comparison, 1 for differences with `--fail-on-change`,
and 2 for invalid input. Both tools reject CPU report schemas.

Sites normally match their full `function:cppLINE:opcode` IDs. Generated line
numbers can move after translation changes, producing explicit additions and
removals. Supply `--site-map map.json` with a one-to-one JSON object mapping old
site IDs to new IDs when their correspondence is known; unknown IDs and mapping
collisions fail. The tool never guesses among identical opcodes. Compare both
requested-byte and unique-footprint coefficients when evaluating metadata
broadcast changes; no latency or speedup is inferred from a coefficient delta.

## Register-ready latency model (CPU-only, uncalibrated)

GPU translations also emit a **model**, not a GPU measurement or a prediction
in microseconds. Each lane issues in order, starting at clock zero. Every
instruction uses `issue = max(previous_issue + 1, ready(source registers))`,
then sets each destination's readiness to `issue + latency_class`. Sources
include address registers and predicates. Independent loads can overlap;
repeated load/use pairs serialize. A result never consumed before lane exit
costs only its issuing instruction. Stores and reductions have no completion
wait; an atomic return waits only when consumed. Disabled predicates issue
once, consume the predicate, and leave data destinations unchanged. Existing
executed-work counters retain their original predication rules.

The default latency constants are global/generic loads **400**, shared and
tensor-memory loads **30**, MMA **32**, special functions/division **20**, and
other register operations (including parameter/local/constant loads) **4**.
Control operations and stores cost one issue step without a result latency.
These are illustrative model units, with no clock-frequency conversion.
Runtime settings are `DRPERF_GPU_LATENCY_GLOBAL`, `_SHARED`, `_MMA`, `_SFU`, and
`_ALU` (each spelled with the full `DRPERF_GPU_LATENCY_` prefix). Each accepts
an integer from 0 through 1,000,000,000. `DRPERF_GPU_SM_COUNT` defaults to 148;
`DRPERF_GPU_LANES_PER_SM` defaults to 128, a configurable throughput proxy,
not resident occupancy. The latter two must be positive. Settings are read
per launch and retained as `latency_model_*` launch metrics.

Because the launch command strips inherited `DRPERF_*` variables, apply model
overrides **inside** the profiled command:

```sh
bin/drperf --gpu --output gpu/build/latency -- \
  env DRPERF_GPU_LATENCY_GLOBAL=400 DRPERF_GPU_SM_COUNT=148 \
  path/to/driver path/to/retranslated.so
python3 gpu/tools/latency_report.py gpu/build/latency
python3 gpu/tools/latency_report.py gpu/build/latency --json --sms 148 \
  --lanes-per-sm 128 --threshold 4 --region 'combine.*'
```

`latency_critical_path` is the maximum final issue clock over all lanes and
blocks in a launch, **summed once per launch** in its enclosing host region.
`dependent_global_loads_on_critical_path` travels with the chosen dependency
path: a global/generic load contributes only if its result readiness forces
an issue beyond the next issue slot. Merely issuing a load adds nothing to
this count. Ties prefer the path with more exposed loads; the count belongs
to the longest path, not to an independently selected lane. Front-loading
loads can turn eleven exposed waits into one even with unchanged bytes.
Common parameter/metadata loads still contribute when used. A vector load is
one load instruction, not one exposed event per vector component.

Block/named barriers join participating clocks and path counts before any
participant resumes. Cluster split barriers join the completed arrival
phase at wait; mbarrier completion clocks join successful tests and explicit
modeled waits, including transaction completion. Cooperative grid barriers
join resident lanes. These joins never drain otherwise unused load results.
`griddepcontrol` retains the existing no-op behavior: launches execute
synchronously here and no dependent-grid scheduling model is added.

Statement sites contain the maximum **cumulative path reaching that site**,
paired with its path's exposed-load count, rather than an additive site cost.
Do not sum site paths. Device region timing is an inclusive elapsed envelope
from each participant's begin to end, maximized over participants; its load
count is the nonnegative endpoint-minus-entry path-count difference. Nested
region envelopes overlap. The observation's `latencyMetricScope` explicitly
says `inclusive-envelope`; work metrics remain exclusive. The path-count
difference at a device boundary is approximate when the selected path changes
to a dependency originating before entry. Host envelopes include inner device
work, as do the existing host launch spans. All new numerical metrics use the
ordinary region/site fitter. A maximum or branch boundary can be non-affine;
rejected fits remain visible and must not be interpreted as verified formulas.

### Basic-block schedule (`DRPERF_GPU_LATENCY_SCHEDULE=block`)

The default model issues instructions in PTX order, so a load placed after an
earlier consumer waits behind that consumer's stall. `ptxas` reorders SASS
within a basic block and normally issues independent loads first. With
`DRPERF_GPU_LATENCY_SCHEDULE=block`, an instruction issues at
`max(floor + slots used in the block + 1, source readiness)`; branches,
barriers, returns, calls, fences, mbarrier/griddepcontrol operations and
barrier joins reset the floor to the latest issue, so nothing is hoisted across
control flow. Reports record `latency_model_schedule` (0 = PTX order, 1 =
block). `gpu/tests/test_latency_schedule.py`: four interleaved load/use pairs
expose 4 loads in PTX order and 1 under the block schedule.

Use both when ranking: a chain that survives the block schedule is control- or
data-dependent (e.g. `||` validation of loaded fields, page-table then epoch
lookups) and is worth restructuring; a chain that disappears under it is a PTX
ordering artifact that `ptxas` already removes (GLM-5.3 `attention_iw_prenorm`:
58231 -> 8037 units, 132 -> 23 exposed loads per launch).

### Where the critical path waits (`latency_report.py --stalls N [--warps]`)

Every launch records `criticalLane`: the lane that finished last, with
`path = issue + stalled` exactly. The lane clock advances one unit per issued
instruction; any further advance is a stall charged to the waiting site:
dependency stalls at the consumer, and barrier/mbarrier/async joins at the wait
instruction. A cross-lane wait names what released it,
`wait-site <- releasing-site [thread N]`, where the releasing site is an
`mbarrier.arrive`, an async copy (`[async copy, thread N]`, TMA/cp.async
completion) or a tensor-core MMA (`[tensor pipe, thread N]`). `chain` repeats the
breakdown for the threads named that way, recursively (up to nine lanes), so a
producer/consumer pipeline reads as a chain:

```
"flashmla.core.small" block 1 thread 0: path 12752 = issue 1092 + stalled 11660
   5467  42.9%  K @ try_wait <- tcgen05.mma.ws.kind::f16 [tensor pipe, thread 128]
  waited-on thread 128: ... 3171 24.9%  K @ try_wait <- mbarrier.arrive [thread 256]
  waited-on thread 256: ... (dequant warpgroup) ...
```

`--warps` adds the slowest lane of every warp in the critical CTA. Lane totals
cover the whole launch, not only the interval before the wait.

### Site names: PTX and CUDA source lines

Site keys name the translated C++ line (`KERNEL_gxptx:cppNNN:opcode`). At
translation time `drperf --gpu translate` aligns those sites with the PTX
instruction stream and writes `<module>.so.sites.json`; the capture's finish
record lists the modules each process loaded (`provenance.modules` in the
report). Reports (`report.txt`, `latency_report.py`) then print
`kernel @ helper.cuh:424<caller.cuh:236 ptx1775:opcode`: the CUDA line when the
PTX was compiled with `-lineinfo` (an inlined helper is shown with its nearest
caller in another file) and the PTX line otherwise. The generated `.so.cpp` is
no longer needed to read a report and can be deleted to save disk.

### Tensor pipe (`DRPERF_GPU_TENSOR_FLOPS_PER_UNIT`, default 8192)

`tcgen05.mma` used to be issue-only and `tcgen05.commit` arrived at the issuing
lane's clock, so MMA completion cost nothing. Each CTA now has an in-order
tensor pipe: an MMA starts at max(pipe free, issue) and occupies it for
`FLOPs / rate` units (rate = 16-bit dense FLOPs per SM per unit, x2 for 8-bit
operands, x4 for 4-bit, /2 for tf32, x2 per CTA for `cta_group::2`); the commit
arrives `DRPERF_GPU_LATENCY_MMA` units after the pipe drains. B300 runs ~8192 dense
BF16 FLOP/clock/SM. Small-M shapes (M=64) run below peak on hardware; the model
does not distinguish them.

### mbarrier spin loops

Translated `try_wait` loops really spin on the CPU until another fiber arrives,
and each probe used to count as issued work. A failed probe now snapshots the
lane; when the same barrier phase completes after a tight retry loop (at most
eight instructions per probe), the spin's issue slots and stalls are rolled back
and the lane joins the phase completion instead, as a suspended warp would.
The spun instructions are reported as `mbarrier_spin_lane_instructions` (they
remain in `ptx_lane_instructions`; subtract them for work fits). In the FlashMLA
decode core they were 18% of counted instructions and inflated the modeled
critical path by 29%.

### If-conversion of short forward branches

ptxas predicates short forward hammocks (`@p bra L; ld; mul; L:`, small if/else
or switch diamonds) instead of branching, and schedules across them. The block
schedule resets its issue floor at every branch, so guarded loads of an unrolled,
bounds-checked loop looked serialized: the FlashInfer MoE routing kernel modeled
17 dependent global loads (8867 units) although vLLM runs it in 4.5 us. At
translation, branches of a region that starts at a conditional forward branch and
reaches its furthest target within 16 instrumented instructions (no backward
branch, barrier, fence, call or return, and no jump into it from outside) are
tagged `bra.ifconv` / `bra.uni.ifconv` and are not scheduling boundaries; the
routing kernel then models 2763 units. In the block schedule the exposed-load
count now starts from the last boundary instead of the lane clock, so a load
issued ahead of an earlier stalled consumer no longer counts as behind it.

### Cache-hit latency for repeated lines (`DRPERF_GPU_LATENCY_GLOBAL_HIT`)

When set, a global/generic load whose 128-byte lines were all touched earlier in
the same launch uses this latency instead of `DRPERF_GPU_LATENCY_GLOBAL`
(`latency_model_global_hit` records it; unset means no distinction). It is a
within-launch residency approximation: data reused across launches (per-layer
metadata, weights shared by consecutive kernels) still looks cold.

Calibration observed on B300 (GLM-5.3 decode, CUPTI): removing modeled exposed
loads of hot view metadata saved ~75-110 ns each (`dc_selected_address`
4.70 -> 4.25 us after 11 -> 5 exposed loads), while removing the FlashMLA
combine's serialized loads of freshly written FP32 partials saved ~340 ns each
(9.0 -> 5.6 us after 12 -> 2). Every small graph-launched kernel also kept a
~3-4 us floor the model does not represent. Model units are therefore not
convertible to time without residency information; use them to rank chains and
to compare variants of the same kernel. Further B300 observations: removing 26 modeled
exposed waits from the 256-CTA MoE router (`moe_front_prenorm`, activations re-read by
every CTA and therefore L2-resident, ~8 resident warps per SM) saved only ~0.4 us
(10.4 -> 10.0 us); the per-lane model has no warp-level latency hiding, so chains in
kernels with many resident warps over-predict their gain. Likewise the FlashMLA decode
core: removing a modeled 357-unit start-up chain (index -> scale loads on the producer
warp of a one-block CTA, -5% of the modeled path) changed nothing on B300 (14.46 us/call
before and after); the model runs one CTA on an idle SM and charges each global load
its full latency, while the real core overlaps CTAs and is bandwidth/occupancy bound.
Use the critical-lane chain to find which stage paces a pipeline, then check that the
stage is also on the measured critical path before spending GPU time.

The ranking helper accepts report directories, JSON and JSON.gz. It sorts
regions by the sum of their inclusive launch paths, equivalent to mean path
times launch count, without multiplying an already summed count again. It
shows blocks, launches with fewer blocks than SMs, and a heuristic
`latency-bound` flag when the summed path is at least `--threshold` times
`sum(lane instructions / (SMs * lane proxy))`. The alternative is labeled
`throughput-bound-proxy`, not a hardware diagnosis. Report-only overrides
change this comparison, not captured register readiness. Mixed latency
constants are disclosed. Device-only regions with no owned launches and old
reports without model metrics are reported as unavailable, avoiding duplicate
ranking of nested timing envelopes. Capture validity is always shown.

Rebuild with `sh gpu/build.sh` and **retranslate PTX**. Work-counter ABI 3 is
preserved because its stamp lives outside the GPU tree; a required latency
instrumentation ABI 1 is checked at both module and kernel scope. The loader
rejects missing or wrong latency capabilities. Split translation links a
single weak module marker. Existing address-footprint capability checks are
independent. The older general report prose about work counts remains true
for those counters; the `latency_*` fields are explicitly modeled quantities.

This model omits cache hits/misses, memory queues, bandwidth, warp scheduling,
finite outstanding-load capacity, register pressure, occupancy, spilling,
warp-only rendezvous clocks, cross-lane shuffle/collective **data** dependencies,
interprocedural parameter buffer dependencies, memory alias dependencies outside modeled barriers,
and asynchronous tensor/copy completion latency (the value runtime completes
these eagerly). Tensor-memory loads use the shared-latency proxy. WGMMA
conservatively reads its accumulator shadows even when accumulation is
disabled. Polling loops count their executed PTX iterations, so their modeled
path can reflect CPU scheduling; explicit wait annotations avoid that effect.
Neither a long path nor redundant requested bytes alone establishes a native
GPU speedup. Inspect launch size and footprint alongside the path.

```sh
python3 gpu/tests/test_latency.py
```
