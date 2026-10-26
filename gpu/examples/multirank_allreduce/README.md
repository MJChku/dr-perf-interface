# Concurrent-rank Lamport all-reduce on the CPU

**Passed:** drperf GPU mode executes the communicating, original BF16 GLM fused-residual kernel and FlashInfer one-shot all-reduce / residual+RMSNorm kernels with one independent `std::thread`, `Module`, and GPU `Region` per rank. Peer addresses point into real host allocations. Neither peer data nor ready flags are pre-filled. Every sentinel wait, CTA-counter wait, residual operation and RMSNorm reduction remains in the compiled kernel.

The retained run is [results/final/summary.json](results/final/summary.json). It contains **2,088 rank launches in 788 collective calls**, checking **1,154,304 logical elements** across ranks, plus a separate two-thread visibility probe. No GPU, CUDA device execution, network transport, cluster job or git operation was used. NVCC only emits PTX; the translated kernels execute on CPUs.

## Reproduce

From `/home/ubuntu/drperf`:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B gpu/examples/multirank_allreduce/run.py \
  --output gpu/examples/multirank_allreduce/results/reproduction
```

Choose an unused output directory. The runner uses the existing drperf runtime and translator; it does not rebuild or modify them. It invokes:

```text
/home/ubuntu/GX/toolchains/cuda13/bin/nvcc -std=c++17 -O2 -arch=compute_103a -ptx ...
bin/drperf --gpu translate ... -o .../kernels.so
g++ -std=c++17 -O2 -pthread .../driver.cc ... -ldrperf_gpu
bin/drperf --gpu --output ... --timeout 120 -- .../driver .../kernels.so ...
```

`CUDA_VISIBLE_DEVICES` is empty, Python bytecode writes are disabled, and compiler temporaries stay in `gpu/build/ext-work2/tmp`. All products go into this example. Each command has an argv/cwd/time/status record and stdout/stderr logs. Raw JSONL captures, CLI profiles and reports are losslessly compressed after collection. The pinned runtime copy in each run's `tool_snapshot/` is selected with `LD_LIBRARY_PATH` during capture.

To refit the retained evidence without launching anything:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B gpu/examples/multirank_allreduce/analyze.py \
  --run gpu/examples/multirank_allreduce/results/final
```

## Learned interfaces versus the source audit

All formulas below are **per rank per launch**, in requested global plus generic bytes. Let `N = hidden*w`, `R = ranks`, `C = R*hidden*previous_w` **in scalar elements**, `B = w*cluster`, and `L = N/8` launched lanes for BF16. Parameter/local/shared bytes have separate counters. These are resource interfaces, without a GPU timing interpretation.

| Original BF16 kernel | NOTES.md source payload, R=2 | Learned payload, R=2 | Learned payload, R=4 |
|---|---:|---:|---:|
| `allreduce_fusion_kernel_oneshot_lamport<kAllReduce>` | `12N + 2C` | **`12N + 2C`** | **`20N + 2C`** |
| `glm53_ar_fused::ar_residual_oneshot<BF16,true>`, optional AR output null | `18N + 2C` | **`18N + 2C`** | Project kernel fixes R=2 |
| `allreduce_fusion_kernel_oneshot_lamport<kARResidualRMSNorm>` | `18N + 2C` | **`18N + 2C`** | **`26N + 2C`** |

The source lower bounds in the archived `NOTES.md` are now learned from executions. Every payload coefficient reconstructs every training observation and held-out payload count with **zero byte error**. Generic accesses to peer buffers retain their original generic category in raw captures; the table sums categories only because both address global host-backed buffers.

The payload comprises input, pushes, one successful polling round, previous-slot clear, and epilogue buffers. Its learned read/write split is:

| Kernel | Compulsory payload reads | Compulsory payload writes |
|---|---:|---:|
| Plain all-reduce | `(2 + 2R)N` | `(2 + 2R)N + 2C` |
| Fused residual / RMSNorm | `(6 + 2R)N` | `(4 + 2R)N + 2C` |

Scalar workspace/flag/counter traffic is additional and independently verified on every small-state observation:

```text
control reads  = (8R + 28)L + 4B + 4
control writes = 4B + 12
all compulsory global/generic bytes = payload + (8R + 28)L + 8B + 16
```

Thus the two-rank compulsory total is **`17.5N + 2C + 8B + 16`** for plain all-reduce and **`23.5N + 2C + 8B + 16`** for residual/RMSNorm. The extra terms explain the difference between a source payload and a full requested-byte count. Initialization is outside these per-call regions.

FP32 work also fits: plain all-reduce `(R-1)N`, fused residual `2N`, and RMSNorm `(R+6.25)N + 2B` for cluster=1 or `(R+6.25)N + 4B` for cluster=2 in the sampled geometries. RMSNorm has `B` reciprocal-square-root operations. Declaring `block_warps = B*ceil(threads/32)` explains the shared reduction traffic across partial and full warps. Instruction, integer-operation and conversion totals still have rejected fits; they are retained in the compulsory profile, not presented as explained interfaces.

## Shapes and numerical checks

Training uses only hidden **32, 64, 128, 256, 512**, `w=1..4`, and ranks **2 and 4**. The training matrix has 1,560 rank launches in 600 collectives and 773,760 checked logical elements. Each workspace persists for the sequence:

```text
w = 1, 4, 2, 1, 3, 4, 1, 2, 4, 3, 1, 4
```

Held-out hidden **96, 192, 384** exercise non-power-of-two lane counts. Hidden **2048**, excluded entirely from fitting, checks the actual archived launcher's nontrivial cluster selection: two CTAs/cluster, 128 lanes/CTA, at both rank counts. Together these add 528 rank launches in 188 collectives and 380,544 checked logical elements. The 2,112 held-out read/write/payload/total predictions have zero payload error; the maximum total-fit floating-point reconstruction error is `1.048e-9` bytes, i.e. zero after rounding to integer bytes.

The archived launcher chooses cluster=1 for all training widths. Additional training runs explicitly split each token across two CTAs, retaining the original kernel body, index mapping and distributed-shared-memory reduction. These are labelled `.cluster`; `.launcher` uses the original geometry rule. Actual hidden=2048 launcher runs confirm the real cluster path as well.

Inputs vary by rank, element and call, include signed values and real negative zeros, and remain live host buffers. The host oracle accumulates rank inputs in FP32, rounds the collective result to BF16, and implements the original epilogue rounding boundaries. Project residual output is FP32 containing a BF16-rounded result; FlashInfer updates its BF16 residual in place, then applies gamma and RMSNorm. Every checked result matched the host reference, ignoring the sign of exact zero. The norm checker permits one BF16 ULP for reduction/rsqrt differences, but **zero elements used that tolerance** in either training or held-out runs.

Each call also verifies flag rotation, reset block counter, unchanged communication capacity, `flags[4] = R*N`, every cross-wired payload element, the entire previous-size clear extent, allocation/output guards, and unchanged read-only input/gamma/FP32-residual buffers. Host barriers bracket calls and keep oracle inputs stable; the kernels themselves perform the communicating pushes and waits.

## Real workspace and progress

The driver follows `LibAllReduce::init` in the archived `src/lib_allreduce.cuh`. Each rank owns three data slots of `comm_size = R*4*hidden*sizeof(BF16)` bytes and flags `{0,0,0,comm_size,0}`. Every slot starts at BF16 negative zero (`0x8000`). The workspace table has `3R+1` entries: two unused pointer groups, rank-ordered data pointers, then that rank's private flags. Its peer entries point into the other ranks' allocations.

`LamportComm` reads `flags[2]` to choose the current slot and clears `(flag+2)%3`, using `flags[4]` from the previous call. The first call has `C=0`. Later calls exercise triple-slot reuse and decreasing/increasing row counts. A current-shape-only fit rejects payload bytes; adding `C` accepts it. For example, at hidden=512, a w=1 call after w=4 has `C=4096`, versus `C=1024` after w=1: the clear contributes **8,192 versus 2,048 bytes** although the current shape is identical.

Every rank opens its own host-thread GPU region and calls `Module::launch_cluster` concurrently. A second two-rank capture delays one rotating rank by 1 ms before launch. The translated entry metadata is `__gxptx_syncfree = 2`: all CTAs are resident as lane fibers, so block 0's `LamportComm::update` counter wait cannot hide producer CTAs in an unlaunched batch. Backward edges call the runtime's periodic fiber yield. Explicit cluster geometry supplies the original cluster rank mapping, barriers and DSMEM accesses.

## Polling is retained, measured and separated

The kernels are unannotated, original bodies. `analyze.py` discovers the innermost backward branch interval containing each volatile load in the retained generated C++. [poll-site-groups.json](results/final/poll-site-groups.json) lists every vector-poll and CTA-counter-poll statement site and generated line range. Outer payload loops are excluded from this grouping.

For each rank invocation, all R vector-load sites must have the same dynamic round count. One successful round per `N/8` vector accesses is compulsory. One counter read is compulsory. Only excess iterations are removed from the compulsory view; loop-body instruction counts are normalized as well, and retry backedges contribute zero in that view. Assertions check every site's iteration relationship and reconstruct the original bytes exactly:

```text
raw requested bytes = compulsory payload + compulsory controls
                    + 16*R*vector_retry_rounds + 4*counter_retry_polls
```

[spin-observations.json.gz](results/final/spin-observations.json.gz) retains both complete polling work and retry-only work, including statement/integer/read counts for each observation. This run observed **33,012 vector retry rounds**, **15,364 counter retries**, and **1,684,976 retry-read bytes**. There were **374 identical-PCV state groups with different retry costs**; one `N=32,C=0,R=2` state ranged from 0 to 480 retry bytes. These values depend on host scheduling and change on rerun. Raw CLI shape fits correctly reject timing-dependent counts; they are never used as compulsory payload coefficients or latency estimates.

## Memory-ordering boundary and minimal probe

The retained generated C++ re-reads each poll on each iteration: `ld.volatile.global.v4.b32` uses `gx_gld_vec<uint32_t,4>`, and the generic volatile counter uses `gx_gld<uint32_t>`. Both call the external `global_memcpy` function for every visit. The translated shared library is compiled at `-O1` without LTO; these external calls prevent load hoisting. Delayed-peer, triple-buffer-reuse and multi-CTA runs all make progress and pass the data oracles.

There is a precise **qualifier-model limitation**, which did not block these CPU executions: `gpu/vendor/gxptx/Gxptx/Translation/Statement.lean`, `translateStmtScoped`'s `loadE`/guarded vector-load and store paths, lower guarded global/generic accesses through `gx_gld`, `gx_gld_vec` and `gx_gst`/`gx_gst_vec`. Those helpers in `Gxptx/prelude.h` call `global_memcpy`; `gpu/runtime/runtime.cc::global_memcpy` calls `std::memcpy`. The guarded `.b32` vector and scalar paths do **not** preserve volatile/acquire/release as C++ atomic types or a complete PTX synchronization model. A separate `.v4.u32` volatile special case exists, but NVCC emits `.v4.b32` here. Cross-thread memcpy traffic is not a C++ data-race-freedom proof. The passing results are empirical evidence for this Linux/x86 build, not arbitrary-host memory-model equivalence.

[visibility.cu](visibility.cu) and [visibility_driver.cc](visibility_driver.cc) are the minimal executable probe: two host threads exchange 64 fresh 16-byte vectors, using the exact `.b32` volatile load spelling, release/acquire flags, acknowledgements and explicit `membar.sys`. One peer starts 2 ms late. All vector fields and checksums pass. In [visibility.so.cpp](results/final/visibility.so.cpp), system fences lower to `__sync_synchronize()` (three static sites); acquire/release accesses still use the guarded helpers. This validates fresh rereads and the explicit host fences in the tested build, without asserting stronger load-only acquire semantics. No runtime or translator file was fixed or edited.

A minimal operational reproduction using the retained modules is:

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
LD_LIBRARY_PATH=/home/ubuntu/drperf/gpu/examples/multirank_allreduce/results/final/tool_snapshot \
  gpu/examples/multirank_allreduce/results/final/visibility-driver \
  /home/ubuntu/drperf/gpu/examples/multirank_allreduce/results/final/visibility.so
```

## Provenance, evidence and limits

[frozen/](frozen/) contains byte-for-byte copies of the archived FlashInfer header and `lib_ar_residual_fused.cuh`, plus five required FlashInfer dependency headers. [sources.json](sources.json) records original paths and SHA-256 hashes for integration-18 commit `4f5de3aefec13e115d1a9f7ed0e64f5d1a645695`. Dependency headers came from the locally available FlashInfer include tree under `GX/bw-work/corpus/flashinfer/data/include`; their exact hashes are also recorded. `kernels.cu` adds explicit template instantiations and otherwise includes these files unchanged. FP32 project residual PTX is incidentally instantiated by the original host wrapper, but only the BF16 target is launched and reported.

The concurrent external session replaced the translator binary after both modules were generated. Both translation manifests pin translator SHA `8c70ee1d22ad6ce4f3ebb9bdb45d1fa1916d9a779ab576e286d1d1c131405092`, prelude SHA `b2d24f8d0862921d96748cc07bbf711c4e2c55fc0dacb8f2a2335d1fce5ca69b`, and runtime SHA `ccb95fa6e7538851e121e6e73427243b8612b887d236c068022d060646c0ff5c`. All captures load that runtime's retained copy; they never retranslate midway. [manifest.json](results/final/manifest.json) discloses before/after hashes and external drift. Generated C++, PTX and modules are retained. `tool_snapshot/` also retains the exact executed sources/runner; the current runner only removes an unused exploratory binary-reuse option.

Useful evidence:

- [summary.json](results/final/summary.json): learned fits, source comparisons, numerics, spin totals and rejected fits.
- [compulsory-profile.json.gz](results/final/compulsory-profile.json.gz): normalized observations and affine fits, trained on hidden≤512 only.
- [heldout-checks.json.gz](results/final/heldout-checks.json.gz): every held-out prediction and count.
- `results/final/{r2,r2-skew,r2-cluster,r4,r4-cluster}/`: complete raw JSONL, original CLI profiles and reports, compressed without dropping polling.
- `results/final/*.record.json` and `*.stdout.log`: exact commands and per-scenario numerical/protocol results.
- [scratch REPORT](/home/ubuntu/drperf/gpu/build/ext-work2/REPORT.md): concise handoff.

No executed opcode, communication or cluster limitation prevented the requested examples. Initialization and host barriers are not charged as kernel work; PDL predecessor/completion instructions do not model asynchronous CUDA streams. The four-rank workspace extends the same header protocol; the archived GLM owner/project fused kernel supports two ranks only. Cluster sizes above two, production hidden=6144, FP32 collectives, inter-process/device transport and GPU performance were not tested. Requested bytes include duplicate lane requests and are not HBM transactions. These bounded numerical checks and empirical fits are not a proof over all inputs or a native serving benchmark. Only this example and `gpu/build/ext-work2` were used for task files; the shared runtime, vendor, README and tests were left unedited.
