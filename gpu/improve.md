# GPU resource-interface experience: DeepSeek V4.1 Flash

2026-10-25. Study workspace: `/home/ubuntu/gen-deepseek41`. DrPerf GPU is the primary analysis tool. It fits counted resources against declared PCVs; it does **not** predict elapsed time. H100 kernel timings, when authorized, are separate measurements. A resource reduction and a timing change answer different questions. None of the requests below relies on treating one as a prediction of the other.

The initial analysis edited only this document; the later user-requested collection also writes under `gpu/cases/`. DrPerf implementation source remains unchanged by this worker. Builds and translations produced generated outputs. A workspace-local toolchain snapshot is described by `eval/drperf-gpu-toolchain.json`; the pinned runtime SHA starts `7fdd0e73a3f7`. Commands below run from the study workspace.

## 1. Compare interfaces across two kernels and decompositions

**Experience.** `python3 tools/run_drperf_gpu_fixtures.py --pinned` executed independent numerical fixtures for 30 distinct custom kernel entries (31 region records; the route-token helper appears in two fixture groups). Comparing `ds41_matvec_tc_partials` + `ds41_matvec_tc_finish` against `ds41_matvec_tc_fused` required our own observation alignment and summation script. Entry names and launch decompositions differ. Some broad routed fits reject span interfaces even when byte interfaces fit: total routes, maximum routes in one block, `K/32`, and `ceil((K/32)/8)` describe different structures. Supplying redundant aliases and products makes accepted coefficient choices difficult to interpret.

**Request.** A comparison view with explicit region mappings (many-to-one), selected PCV bases, units, dependent-PCV warnings, observed domains, held-out checks and rejected-fit counterexamples. Show which coefficients change while keeping constants and nonidentifiable terms visible. Permit named derived PCVs and piecewise regimes without editing drivers for every comparison. Keep rejection visible; do not present an unfitted coefficient as an interface.

**Concrete follow-up.** `python3 tools/run_drperf_route_interfaces.py` varies local routes 0/1/3/6, N 8/32/128 and K 64/512/1280. It compares the original partial/finish decomposition, the serial-route fused kernel, and a new CTA per `(route,N tile)` prototype. Reports live under `eval/drperf-gpu/route-interfaces-{control,serial,parallel}`. This distinguishes work growing *inside a block* from work spread over blocks.

## 2. Distinguish launched work, guard execution and useful parallelism

**Experience.** The routed kernels launch fixed-capacity geometry. Nonlocal routes still execute guards and may clear outputs. `executed_lanes` therefore does not mean lanes doing useful MMA work. A serial-route fused block can repeatedly execute a compute region for six routes, whereas the route-parallel prototype exposes independent blocks. Counts summed over repeated region occurrences alone can hide this difference. Device markers in `eval/drperf-gpu/routed-device-regions` helped separate metadata, route computation and ordered sums.

**Request.** Report work distributions per physical CTA and warp, distinct blocks/warps entering each device region, region occurrence counts, participation masks and productive tensor-work lanes. A repeated device scope inside one CTA should be distinguishable from the same scope executed by several CTAs. Preserve launched/cancelled/executed counts already available.

**Single-row example.** The fused KV projection has N=512, so eight columns per CTA expose 64 CTAs; N=1280 exposes 160. The block interface is directly informative about available parallelism. It is not a latency model.

## 3. Make lane-span semantics and parallelism limits easier to inspect

**Experience.** Routed serial fusion removes the global partial buffer but loops over local routes within a CTA. Its per-block maximum lane work and logical span grow with routes. The route-parallel prototype keeps shared partials and moves routes into grid geometry. These are structural differences independent of the separately measured H100 timing.

**Request.** Expose warp-level dependency/ILP-aware logical spans, operation-class breakdowns and source-site contributions, alongside the present lane-sequence and barrier metrics. Show whether a maximum is taken per region occurrence, CTA, launch or sequence. Explicitly identify missing warp-exchange, asynchronous-operation and cross-launch dependencies. Logical spans should retain their stated units and should not be labeled hardware critical paths.

## 4. Add occupancy-relevant resource interfaces

**Experience.** Shared-partial fusion changes a 32-thread partial CTA into a 256-thread CTA with `8*(K/32)*sizeof(float)` dynamic shared memory. Current dynamic/static shared counts are useful, but interpreting block exposure also requires registers per thread and their allocation granularity. The single-row KV case makes this especially concrete.

**Request.** Register counts, spill/local bytes, static/dynamic shared memory per block, threads/warps per block and optional architecture allocation limits in one view. Fit resource limits against K/N/routes where meaningful. These are occupancy-relevant resource facts, not predicted elapsed time or a claim of achieved occupancy.

## 5. Preserve memory-space provenance and distinguish requests from footprints

**Experience.** Some fixture buffers appear under `generic_*` rather than `global_*`. Our first analysis omitted generic bytes and had to be corrected. The pinned toolchain's exact-address footprint feature (version 1) provides useful unique bytes, 32-byte sector coverage and 128-byte line coverage. Those are address unions, not cache misses, transactions or HBM bytes. Shared partials also change the balance between shared and global requests.

**Request.** Optional allocation/range registration with semantic roles (activation, weight, partial, output), plus combined device-buffer views retaining the underlying PTX space. Add per-warp coalescing/sector-request and shared-bank access patterns where their semantics can be stated precisely. Keep exact footprint coverage separate from transaction accounting and any cache assumptions.

## 6. Unsupported PTX: report the failing instruction and origin early

**Observed failure.** The primary-reference K128 FP8 staging fixture translated successfully, but execution failed with `unsupported GPU PTX operation: dp2a.lo.u32.u32`.

Reproduction: `python3 tools/run_drperf_library_probes.py`; case `k128_fixture_deep_m32_n128`. Exact bounded execution receipt: `eval/runs/20261025T223746.231701Z`; retained partial capture: `eval/drperf-gpu/k128_fixture_deep_m32_n128-library-probe/`. Translation receipt: `eval/runs/20261025T223738.248818Z`. The attention prepare/unpack and grouped-FP4 library adapter probes in the same batch completed successfully. This is a specific unsupported execution form, not evidence that FP8 or all library kernels are unsupported.

**Request.** A translation/validation coverage manifest listing unsupported operations, their PTX source locations and generated call sites before execution, plus a minimal-instruction reproducer export. Implement/test this `dp2a` form with independent numerical cases. Avoid equating successful translation or symbol closure with executable instruction coverage.

## 7. Application integration and resource sequences

**Experience.** We reused independent scalar-oracle fixtures with a workspace-only `Driver` adapter around `drperf::gpu::Module`. Our CPU preflight compiles the actual benchmark-driver source with that adapter, retains case arguments/output names, and runs one iteration of translated PTX. DrPerf deliberately strips GX variables; a fixture guard initially assumed a GX process and had to recognize the standalone backend. Arbitrary CUDA application interception is outside the documented current interface.

**Request.** A documented adapter/backend protocol, scalar argument-to-PCV descriptors, and resource-sequence composition across launches. Explicitly represent stream/graph edges and host/device copy or clear operations when supplied by the application; those operations may have no kernel PTX and must not silently disappear from a sequence-level resource account. Export stable artifact/scalar identities so resource reports can be linked to a separate GX cost database without merging resource counts with timing.

**Limit.** CPU preflight checks argument parsing, fixture execution and resource-result collection. It cannot validate the native CUDA profiler interceptor or the native SQLite signature path; report that boundary explicitly.

## 8. Atomic, versioned toolchain bundles

**Experience.** Live translator/runtime outputs changed during the first batch while another process was rebuilding the shared checkout. We retained those exploratory captures but reran the comparative evidence with a byte-pinned workspace bundle. Runtime and module instrumentation versions matter, including footprint availability.

**Request.** An atomic versioned build/export command containing the executable, runtime, translator, prelude, fitter and a dependency manifest. Stamp every capture with one bundle identity and reject incompatible artifacts. Existing per-module provenance and ABI checks are useful; a portable bundle would make consistent comparisons easier.

## 9. Compact, navigable reports for many kernel states

**Experience.** Thirteen fixture groups plus focused/library/device-region probes produce hundreds of MB of pretty JSON and raw per-site captures. Full evidence is useful, but reviewers mostly need accepted interfaces, rejection examples, resource units and a small per-state comparison. Pointer/format aliases can dominate long coefficient tables.

**Request.** A compact summary/index, transparent lossless compressed report input/output, a metric/region filter, normalized per-real-row/per-route views, and source navigation from coefficients to contributing sites. A two-variant comparison should show observations and fitted terms without hiding the declared PCV domain or single-state status.

## 10. Large context and extrapolation: evidence still incomplete

The present probes cover bounded kernel states, not a full 100K-context resource execution. We have not established DrPerf GPU runtime or memory requirements for that complete workload and do not claim otherwise. The initial grouped-FP4 single-state probe was followed by six real/padded-row states; its fitted tensor work follows padded rows, with zero real-row coefficient at fixed N=128,K=64. This remains a bounded fixture, not a full production-shape survey. Sparse-attention library internals and CUB coverage remain incomplete.

A useful workflow would declare a bounded scaling grid, retain representative large-state validation, report analysis wall time/memory separately from counted GPU resources, and clearly flag extrapolation or single-state evidence. This is a workflow request, not a measured claim that a 100K execution is too slow.

## Follow-up: observed route-interface comparison

The focused grid completed for all three variants (75 independent-oracle cases each, including invalid-ID checks). Concise fits reconstructed buffer bytes and block/lane FLOP maxima with zero residual. Let T=N/8, G=K/32, R=local routes, B=T*ceil(G/4). Control partial+finish buffer requests fit `768B + 512RB + 236RTG + 288T`; both shared-partial variants fit `120T + 16RT + 172RTG`. Main compute blocks fit B / T / 6T for control / serial / route-parallel. The **control also loops over routes within a CTA**, but splits K over many CTAs; the difference is not simply route-parallel control versus serial fusion.

At N=32,K=512 the serial launch-maximum logical span fits `156 + 299.421*R + 11.263*any_route` with <0.527 step residual. Route-parallel fits `65 + 314*any_route`, zero R coefficient and zero residual. Sum of per-block maximum lane FLOPs is unchanged between the shared-partial variants, while their compute-active CTA counts are T and RT. Global/generic/shared byte observations match across those two variants. The prototype increases guard-only blocks and summed block span. This reinforces requests for explicit aggregation levels and productive-CTA counts; it does not establish a timing result.

Evidence: `eval/drperf-gpu/route-interface-summary.json`, `eval/drperf-gpu/route-interfaces.md`, bounded fitting invocation `eval/runs/20261025T230223.975437Z`.

## Follow-up: real/padded-row interface and bounded execution costs

`python3 tools/run_drperf_extra_interfaces.py` completed the grouped-FP4 adapter at six declared states: padded rows0/64/128/192, real rows0/2/53/117/128, active experts0/2/3/4 (four capacity experts, N128,K64). The accepted interfaces have zero residual:

- tensor_float_flops = 16,384 * padded_rows;
- global_read_bytes = 8,192 + 272 * padded_rows;
- shared_read_bytes = 1,024 * padded_rows;
- launched blocks =8.

The real_rows coefficient is zero in these fits. At padded_rows128, real_rows0/53/128 all produce the same counted work. This is a concrete padding interface, not a timing result. Evidence: `eval/drperf-gpu/grouped_fixture-extra-interfaces/profile.gpu.drperf.json` (possibly losslessly archived with `.gz`).

The full benchmark-driver CPU preflight also exposed analysis resource pressure at actual projection shapes, rather than at100K context. Four concurrent instrumented cases (including N2304,K5120,1/3/6 local routes) reached sampled aggregate RSS17,203,392,512 bytes; an independent16GiB guard stopped the batch. The original workspace watchdog missed thread-spawned descendants; that **workspace bug** was fixed and regression-tested. A two-case20GiB bounded resume retains completed hash-matched cases and enforces the32GiB host reserve. Counter/footprint collection is enabled, but allocation sources were not profiled, so the RSS cannot be attributed entirely to DrPerf metadata.

Request: analysis-memory diagnostics broken down by fixture/lane contexts, per-site counters and footprints; explicit collection budgets; compact exact-footprint representations where possible; and a restartable case-grid command that preserves completed states with identities. Keep wall time/RSS of the CPU analysis distinct from GPU resource metrics. This is observed execution overhead, not a claim about GPU elapsed time.

The broad custom HC split-K and attention split fixtures each hit a210s CPU-analysis deadline, retaining completed observations and incomplete lifecycle evidence. They are not numerical failures or complete fitted reports. Smaller explicit slices are being used to finish selected resource states. A partial-capture viewer that clearly distinguishes completed invocations from an incomplete process would help; it must never promote such a capture to a complete-run pass.

## Final bounded result

The exact16case CPU benchmark preflight completed using verified independent case partitions. The full scalar shapes were executed once each, including routed up/down with1/3/6 local routes. The consolidated receipt is `eval/drperf-gpu/native-batch-preflight-final.json`; resource observations and independent H100 timings for all8 pairs are in `native-count-timing-table.md`. Counts and timings exclude the unchanged route-token helper consistently. Native times sum isolated partial/finish medians and are not sequence or request latency. Redundant duplicate CPU cases were stopped after their complete counterparts were hash-verified.

Complete bounded coverage reached40 distinct kernel entries. The HC128split,K20480 slice and the attention selected64/128,8heads,dim512 slice completed with independent numerical oracles. The attention640-key state remains incomplete at its deadline. Toolchain verification still matches all40 pinned files. No new native job was submitted after the preflight requirement; any future native bundle must also prove that its native driver was rebuilt from the CPU-preflight source.

## 2026-10-26: reproducible case collection and additional concrete coverage

The user requested portable examples. The collection is now under [`cases/`](cases/README.md), with one directory per serving-kernel family or issue, local PTX, scalar-oracle drivers and support closure, deterministic PCV grids, byte-pinned toolchain identities, fitted reports, expected-interface checks and exact commands. Only `gpu/cases/` and this document were modified by this worker in the DrPerf repository. The workspace export and accounting tools are `tools/export_drperf_cases.py`, `tools/verify_drperf_cases.py` and `tools/finalize_drperf_cases.py`. Every reproduction is recorded as `drperf_gpu`; no collection command allocates a GPU.

**Unsupported production prologue.** The grouped-FP4 production up/down benchmark preflight also fails on `dp2a.lo.u32.u32`; successful small grouped fixtures do not establish instruction coverage at production N/K. See `cases/ds41-grouped-production-dp2a/`: it includes the full production PTX and an isolated one-instruction probe with an independent unsigned scalar oracle. The native grouped-tail benchmark was not submitted because this exact CPU preflight failed. Please report unsupported reachable forms before a lengthy batch, and make coverage failures structured results carrying the instruction, PTX location and argument shape. Translation success alone is insufficient.

**Unsupported sparse-attention transpose form.** A new direct-library probe, `cases/ds41-sparse-attention-head-padding/`, translates successfully but execution rejects `ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16`. The case preserves the actual PTX, six-state head/selection grid, independent scalar softmax/BF16 oracle and failing command. Its numerical status remains `not_assessed`; the proposed tensor interface proportional to padded heads has not been observed. The 8→16 padding finding came from source/geometry analysis, not this failed DrPerf capture. A coverage manifest should distinguish transpose, shape, multiplicity and element-type forms of `ldmatrix`.

**Existing library coverage that does work.** The bounded CUB pack/block-radix-finish adapters, and selected-only batched decode preparation, completed numerical/resource probes. This does not cover the full host `DeviceRadixSort` sequence. A documented multi-module/host-library adapter and sequence-level resource composition would make that boundary easier to express. Command-level memsets/copies need explicit accounting; kernel-only traces cannot establish their costs or resource work.

**Padding fix with an independent oracle.** The grouped-FP4 full32/tail16 wrapper gives tensor work `16384*padded_rows` at N128/K64. Real53 rows change padded128→96; real26 change128→64, with eight launched CTAs in either case. Full32 and tail16 work share one entry/launch. Production host packing still rounds32 and the candidate is disabled pending production coverage and timing. The useful feature request is a paired PCV-grid view that aligns identical logical expert rows across different packed layouts and reports identifiable padding coefficients; an identical physical offset is not the right alignment key.

**Route parallelism, separate from elapsed time.** At N32/K512, serial logical launch-maximum span has an approximately299.421-step route coefficient (residual<0.527), while the route-parallel interface is `65+314*any_route` with zero positive-route coefficient. Requested buffer interfaces match. A separately authorized, bounded38-case H100 batch measured actual production shapes, all executions completed and deletion was verified: `eval/profiles/gpu/ds41-prof-20261026-002107/`. For up N2304/K5120 R6 the control/serial/parallel timings were84.895/119.84/80.991µs; for down N5120/K2304 R6 they were83.582/78.079/71.279µs. These are separate native measurements, not predictions or corrections of DrPerf. Registers/shared allocation facts, productive-CTA participation and warp/ILP-aware logical dependency metrics would complement the present resource interfaces.

**Fixture/module compatibility.** One collection attempt paired the vector-cast launch geometry with the scalar `data.ptx`, correctly failing its independent oracle. We corrected the case to retain explicit scalar and8-element vector variants with their respective launch geometry. This was our packaging error, not an unsupported instruction or DrPerf numerical defect. An optional entry-point ABI/launch-contract manifest could catch such mistakes before execution; it should not replace numerical oracles.
