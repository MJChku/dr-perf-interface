# Standalone GPU runtime compatibility with GX

The source of truth for this port is GX branch `gx` at
`cc6dfacbe333c7a5892c1247ad2d7f9c9f464d68` in `/home/ubuntu/GX/NEX`.
`vendor/provenance.json` records upstream and adapted file hashes. This is a
standalone kernel execution/resource collector; it does not run the full GX
CUDA application interceptor, device allocation system, transport or timing
simulator.

Instrumentation ABI 3 adds cross-space async-copy counters. Re-translate
modules built with older instrumentation; the loader rejects them explicitly
rather than silently interpreting the new counter space with old semantics.

The runtime now resolves every C runtime entry declared by its vendored GX PTX
prelude, including host-library dependencies. This removes the missing-runtime
holes for WGMMA, TMA, bulk operations, clusters and cooperative grids. Symbol
closure is checked separately from numerical correctness: it is not an
exhaustive proof of every legal PTX program or instruction combination.

| Family | Execution and resource accounting | Numerical checks |
|---|---|---|
| Scalar/vector PTX, MMA/WMMA, warp operations | Existing GX translator and standalone lane machinery | Existing integration suite |
| Dense WGMMA | Canonical GX arithmetic, all 128 lanes rendezvous, eager completion of groups; tile FLOPs/integer ops and logical shared operand bytes | 33 independent oracle cases: F16/BF16/TF32, four FP8 type pairings, four integer pairings, F16/F32/S32 accumulators, register/shared A, N=8–256, descriptor layouts, transpose, scaling, saturation, chained groups and two warpgroups/3D blocks |
| TMA tiled copies | Canonical tensor-map encoder and execution, rank 1–5 implementation, swizzles, sub-byte layouts, strides and OOB fill; requested bytes and transaction counts | 7 load cases and 1 store case cover rank 3, zero/NaN fill, strides, U4 padding and swizzles |
| TMA multicast / paired completion | Resident CTA shared windows and launch-wide mbarriers; source footprint once, destination bytes per receiving CTA | 6 multicast cases across four CTAs, including cta_group::2 completion routing |
| TMA gather/scatter, replacement and reduction | Canonical gather4/scatter4, tensor-map field edits, integer/floating reductions | 6 extended cases cover rank 2, OOB rows, descriptor publication, null-address placeholders, reductions and masked bulk stores |
| Plain cp.async | Immediate/dynamic source size, ignore-source predicates, zero-filled destination tail; separate global reads and shared writes | Existing full-copy oracle plus 3 tail/ignore-source fixtures with guards |
| Bulk copy / reduction | Canonical g2s/s2s/s2g helpers, multicast, OOB ignored bytes, byte masks and reductions; logical bytes plus operation counts | Bulk reductions and masks in the extended suite; helpers additionally covered by runtime symbol closure |
| Clusters / distributed shared memory | Per-CTA arenas, rank/address mapping, split barriers and remote mbarrier arrivals; explicit cluster launch API | 4 independent cluster geometries, including 3D clusters and multiple clusters per grid |
| Cooperative grid / barrier votes | Resident grid fibers, grid barrier and workspace/envregs; named reductions for OR, AND and POPC | Cross-block read-after-grid-sync and CUDA barrier-vote oracles |
| tcgen05 groups 1 and 2 | Canonical tensor-memory execution and descriptor tile accounting, correct paired-CTA storage | Existing BF16/NVFP4 tests plus 3 paired-CTA BF16 matrix products with independent scalar oracles |
| CLC | Pending-cluster queue, success/failure responses, multicast publication, cancelled-CTA suppression and persistent work stealing | 3 numerical oracles verify every tile is processed exactly once with successful cancellations, for cluster sizes 1/2/4 |
| GX semantic/cohort helpers | Real-buffer execution, declared copy/repeated-load counts, warp ballot columns, polls and payload annotations | Runtime entry coverage; semantic annotations do not reconstruct an absent original expanded PTX body |
| Device printf/assert | Canonical parameter-buffer ABI and host-call adapters | Runtime entry coverage; formatting runs on CPU, not as a GPU latency model |

Rows explicitly distinguish tested cases from a shared generic implementation
or an ABI-only check. The checked-in PTX permits running these tests without
CUDA, a GX checkout, or a GPU:

```sh
bin/drperf --gpu build
python3 gpu/tests/test_gpu.py
python3 gpu/tests/test_runtime_parity.py
```

The suites contain 24 and 8 tests respectively. Set `NVCC=/path/to/nvcc` for
the optional annotation compilation/device-link tests in the first suite; they
compile CUDA but do not run native GPU code. The WGMMA driver generates 33 PTX
fixtures from pinned input/fragment-layout oracles, then the parity suite
translates and executes every fixture on CPU.

## Launch and tensor-map interfaces

Independent launches keep the existing `module.launch` interface. Explicit
clusters use:

```cpp
module.launch_cluster("kernel", dim3(8), dim3(128), dim3(2), args);
```

Grid dimensions must be divisible by the cluster dimensions. Ordinary launches
execute one resident cluster at a time; cancelled pending clusters never start.
GX-supported cooperative-grid metadata selects full-grid resident execution
automatically. Geometry
and static/dynamic shared memory are checked before use; shared offsets occupy
24 bits and the high bits identify a CTA within its cluster. Very large
resident grids still require real CPU memory for their lane contexts.
`blocks` and `launched_lanes` retain the requested grid geometry;
`executed_blocks`/`executed_lanes` and `cancelled_clusters`/`cancelled_blocks`
distinguish actual execution from cancelled work.

`drperf_gpu_tensor_map.h` supplies a namespaced, 128-byte/128-aligned descriptor
type and the GX encoder API without a CUDA toolkit dependency:

```cpp
#include "drperf_gpu_tensor_map.h"
using namespace drperf_gpu_tensor_abi;
CUtensorMap map;
uint64_t dims[] = {128, 32}, strides[] = {512};
uint32_t box[] = {32, 8}, traversal[] = {1, 1};
int result = gx_tensor_map_encode_tiled(
    &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, input,
    dims, strides, box, traversal, CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
// Require result == 0. Pass &map as the by-value PTX parameter's argument slot.
```

Use this encoder for the standalone runner. A hardware-generated opaque CUDA
tensor map has a different internal encoding and cannot be reinterpreted as a
GX descriptor.

## Counter interpretation

- WGMMA tile work is `2*64*N*K` per warpgroup operation. Shared operand tile
  bytes are divided among 128 lanes; register-A forms omit the shared A tile.
  `tensor_mma_instructions` belongs to lane zero of the warpgroup, independent
  of scheduler arrival order. This keeps attribution deterministic for device
  regions; block-level regions aggregate the complete operation naturally.
- TMA global reads count in-bounds source bits rounded to bytes. Shared writes
  include OOB fill. Multicast counts one logical source footprint and one
  destination footprint per receiving CTA; these are not cache or HBM
  transaction measurements. `tma_transaction_bytes` retains destination
  mbarrier transaction bytes, including OOB fill.
- Reductions count their logical source/destination reads and writes and
  `reduction_ops`. Byte-masked stores count selected bytes. Copy helper CPU
  loops and matrix interpreter loops never add PTX work.
- All semantic resource hooks preserve the original PTX site and innermost
  device-region ownership. Launch snapshots remain inclusive, while region
  observations are exclusive.
- These additions supply resource explanations, not hardware latency,
  occupancy, asynchronous overlap, or cycle-accurate instruction scheduling.

## Limits inherited from GX and remaining accounting limits

The port retains GX's accepted forms and numerical policies. For example,
TMA im2col and interleaved maps remain unsupported, and dense WGMMA requires a
full, uniform warpgroup. WGMMA floating accumulation follows GX's documented
double-precision reference policy, not a claim of bit identity with hardware.
Asynchronous operations complete eagerly and still honor their synchronization
contracts.

Some numerically executable tcgen05 variants still flag incomplete resource
accounting (sparse/metadata, masked, weight-stationary, ashift and compressed
copies). `multimem` accounting also remains flagged. Functional support and
complete resource accounting are separate claims: these captures retain
`unmodeled_work_events` and are not reported as complete work measurements.
Original expanded instructions cannot be recovered from a fused GX semantic
annotation alone. Whole-application CUDA stream scheduling and timing remain
GX responsibilities.

## Fixes found while porting

The canonical TMA emitter used a temporary C compound array whose address
`g++` rejects in C++. The adapted emitter uses `std::array::data()` with the
same lifetime through the synchronous helper call. Both TMA suites now compile
with the default `g++` path.

The copied translator handled reductions only under `barrier`, while NVCC
emits `bar.red.or.pred` and `bar.red.and.pred`. Its plain `bar` route performed
synchronization and left the destination predicate unchanged. The adapted
route handles both spellings, optional explicit participant counts and POPC;
the CUDA-intrinsic numerical test covers all three reductions. These changes
are local to DrPerf's vendored translator; the canonical GX checkout has not
been edited.
