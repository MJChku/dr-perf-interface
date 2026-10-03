# LMCache: instruction counts plus checked waits

This is a record of local experiments. Links into `results/paper/waits/lmcache/` refer to
local captures and source overlays, which are not included in the Git repository.
The portable marker examples and checker tests are in `examples/waits/` and
`tests/`; the commands below that use LMCache also need those local artifacts.

2026-09-29. This extends the [LMCache instruction-count study](../lm_cache_task.md)
with native synchronization observations and passive future-completion checks.
The existing LMCache code and old profiles were left unchanged; checkpoint
annotations live in a separate [overlay](../results/paper/waits/lmcache/overlay/).

## Whole serving workflow: broad region coverage

Historical captures under the previous one-obligation-per-invocation rule
(2026-10-01): these three profiles have **zero `unexplained(Wait[?])`
terms**, with all native invocation obligations covered and all supplied wait
indicators checked. The overlay annotates **553 function sites in 43 modules**,
plus inline transfers and request handlers. Across these captures it exercises
**162 distinct regions**. All **216 requests** completed without failures.
Execution uses Qwen2.5-0.5B/vLLM under GX functional emulation with dummy weights;
these are instruction and dependency observations, with no GPU-output or latency claim.

| Configuration | Requests | Regions | Covered invocations | `Wait[?]` | Declared pairs | Null checkpoints | Profile / graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Local CPU, 0.4 GiB | 144 | 126 | 4031/4031 | 0 | 1 | 9160 | [JSON](../results/paper/waits/lmcache/architecture-cpu.drperf.json) / [archived profile JSON](../results/paper/waits/lmcache/architecture-cpu.drperf.json) |
| Remote TCP | 36 | 149 | 1334/1334 | 0 | 4 | 2659 | [JSON](../results/paper/waits/lmcache/architecture-remote.drperf.json) / [archived profile JSON](../results/paper/waits/lmcache/architecture-remote.drperf.json) |
| Local CPU, 0.12 GiB | 36 | 138 | 1028/1028 | 0 | 1 | 2300 | [JSON](../results/paper/waits/lmcache/architecture-pressure.drperf.json) / [archived profile JSON](../results/paper/waits/lmcache/architecture-pressure.drperf.json) |

The four distinct event-backed pairs remain:

- `lmc.lookup_rpc.request` waits on `lmc.lookup_rpc.handle`.
- `lmc.remote.contains` waits on `lmc.remote.exists_operation`.
- `lmc.remote.get` waits on `lmc.remote.get_operation`.
- `lmc.remote.put_callback` waits on `lmc.remote.put_operation`.

The remaining observations have explicit **`waited(null)` refinements**, whose
reasons are visible in the interface. The [reviewed catalog](../results/paper/waits/lmcache/null-refinements.json)
names each scope and explains its runtime/allocator locks, metadata work,
transport operations, external-server reads, stream fences or lifecycle waits.
The TCP server's internal publisher is outside this capture. An external socket
read remains a visible wait, even though it has no declared region publisher.
Null refinements add no dependency arrows. Reasons require **manual review**;
the checker checks checkpoint occurrence, indicator consistency and the existing
invocation budget. This coverage result does not establish that every dependency
has a known publisher, or that every annotated invocation actually blocked.

The [annotation inventory](../results/paper/waits/lmcache/annotation-inventory.json) records
source sites. Each summary marks observed and unexercised regions separately.
The source overlay and all report source hashes match. Full native evidence is
retained in checksummed sidecars; open the ordinary `.drperf.json` in the viewer.
The broad function annotations initially use `cold`; their instruction-cost PCVs
can be refined independently of these wait annotations.

### Placement and checks

Reviewed scopes opt in through the [overlay helper](../results/paper/waits/lmcache/overlay/lmcache/_drperf.py)
to `region(...).waited_null_on_exit()`. The native binding emits one null
checkpoint immediately before region end, including exception exits. Its
`I[True]` means a scope-end checkpoint occurs on each captured invocation.
It supplies **one** credit: separate descendant invocation obligations remain.
The coroutine/generator wrappers mark active steps, ending before suspension.
Ordinary unreviewed regions keep their original behavior.

The original transfer checkpoints remain conditional on their entry PCVs:

```text
lmc.gpu.to_gpu:         I[True] * waited(null)
lmc.gpu.from_gpu:       I[host_chunks > 0] * chunks * waited(null)
lmc.vllm.start_load_kv:  I[True] * (load_uploads + 1) * waited(null)
lmc.vllm.wait_for_save:  I[True] * (store_uploads + 1) * waited(null)
```

The adapter multiplicity includes its one reviewed scope-exit checkpoint plus
completed nonempty slot uploads. Empty mappings have no upload checkpoint.
The store `chunks` fit describes the observed all-host batches; mixed batches
still require testing. A synchronous child is never used as a CPU event publisher.
The [connector/adapter tests](../results/paper/waits/lmcache/check_copy_steps.py) cover 64
connector and 216 adapter cases, including empty/mixed batches and failures.
The [active-step tests](../results/paper/waits/lmcache/check_architecture_steps.py) cover
suspension, exceptions, cancellation, disabled measurement and excluded PCV work.

The [exporter](../results/paper/waits/lmcache/architecture.py) checks transfer checkpoints
against completed native syncs and rejects contradictory indicators. Removing
null checkpoints in a synthetic control restores uncovered obligations and
invalidates the supplied interfaces; reasons alone provide no credit. The control
keeps its record-count metadata consistent and never modifies the saved capture.
The native shared-helper, ancestor-budget and deliberately invalid child-publisher
cases remain in [examples/waits/coverage.c](../examples/waits/coverage.c).

The [capture harness](../results/paper/waits/lmcache/capture_workload.py) now closes the actual
lookup server and checks that its receive thread finishes before process exit.
The previous forced shutdown left one receive invocation without its checkpoint;
that rejected capture was replaced by a fresh execution, not edited into a pass.
The 144-request CPU capture uses `-no_enable_reset` and
`TOKENIZERS_PARALLELISM=false` after earlier instrumented attempts stalled.
These settings avoided the stall in the accepted run; its root cause is not established.
The runtime settings are recorded in the CPU report's provenance.

Reproduce with [expand_annotations.py](../results/paper/waits/lmcache/expand_annotations.py)
after `annotate.py`, then [launch.sh](../results/paper/waits/lmcache/launch.sh) with
`DRIVER_PY=/w/lmc-waits/capture_workload.py`. The [collector](../results/paper/waits/lmcache/collect.sh)
accepts only successful completed captures and rebuilds each profile separately.
See [waited.md](waited.md) for the marker and declaration contract.

The high-level paths to inspect are:

| Workflow | Region path and architecture question |
| --- | --- |
| Scheduler lookup | vLLM matched-token query → `lmc.lookup_rpc.request` → worker `lmc.lookup_rpc.handle` → `lmc.lookup`. Does foreground scheduling depend on a remote query? |
| Key construction | Token database → chunk/prefix hashing → cache-key construction. How much work is repeated across lookup, retrieve and store? |
| Retrieval | `lmc.vllm.start_load_kv` → `lmc.retrieve` → storage manager → selected backend → GPU restore. What must complete before KV can be used? |
| Store | `lmc.vllm.wait_for_save` → `lmc.store` → host allocation, GPU extraction and storage dispatch. Which work finishes synchronously, and which completes in the background? |
| Lifetime/replacement | Pin/unpin and reference-count changes → allocator/free-list/cache-policy operations. Which buffers stay live, and when can replacement proceed? |
| Remote completion | EXISTS/GET/PUT operation regions → caller completion checkpoints. PUT completion here means local send completion, not remote durable acknowledgment. |

The extra declared lookup relationship is at the **semantic RPC caller**:

```text
lmc.lookup_rpc.request = CPU/child work
    + I[blending == 1 or chunks > 0] * Wait[lmc.lookup_rpc.handle]
    + any remaining unexplained wait terms
```

The entry PCVs come from the original chunking configuration. In the original
code, a non-blending request with no complete chunks returns without RPC; a
successful RPC returns after the handler has computed its result. The handler
publishes before the original response send; the client marks `waited` after
receiving a nonempty response. The generation is a stable 64-bit digest of the
request ID. The checker tests uniqueness, pair ownership, observed publication
order and indicator occurrence. This experiment uses one worker rank in one
process; it does not establish a general multi-rank/cross-process RPC protocol.

The remote backend separately declares these interfaces:

```text
lmc.remote.contains: I[connected == 1 and async_exists == 1] * Wait[exists_operation]
lmc.remote.get:      I[ready == 1 and individual == 1 and chunks > 0] * chunks * Wait[get_operation]
lmc.remote.put_callback: I[True] * Wait[put_operation]
```

Targets above abbreviate `lmc.remote.*`. The indicators are supplied by the
annotation; only multiplicity is fitted. Exceptions/timeouts can invalidate a
normal-completion claim and are not silently converted into successful waits.
Native observations without a covering checkpoint stay in the interfaces as
`unexplained(Wait[?])`. A native lock call alone does not prove contention.

This is broad coverage of the **configured serving paths**, not every LMCache
feature. Separate async prefetch, layerwise transfers, P/D disaggregation, other
remote providers, disk/GDS and distributed controllers need additional workloads.
The TCP server's internal execution is not in the client profile. CacheGen still
needs numerical device-derived outputs that this GX functional-skip setup does
not supply. These boundaries must remain visible rather than being presented as
a claim about all possible paths or waits.

## Real-server optimization: one GET stalls independent lookup work

2026-09-30. Unlike the deliberately restored Ditto reclaim regression, this
experiment starts with the **unmodified installed LMCache connector and real
LMCache CPU TCP server**. No response is withheld and no wait is inserted.
The existing serving capture above pointed to synchronous receives inside
`LMCServerConnector.get`, despite its async interface. `StorageManager` shares
one event loop among its backends. Blocking that loop can delay work that does
not need the GET result.

The [reproduction](../examples/waits/lmcache_real_io.py) queues a bulk GET and
an independent EXISTS lookup on two connections sharing a loop. It transfers
actual bytes through the original server. A CPU allocation adapter supplies
LMCache's `TensorMemoryObj` buffers without requiring a GPU. This is a real
connector/server component benchmark, **not full vLLM serving**, and it does
not establish that the earlier single-connector Qwen workload experiences the
same degree of interference.

Before the change, drperf records the lookup operation starting only after
GET finishes in **6/6 captured pairs**. After the change, it starts before GET
finishes in **6/6 pairs**. Each arm retains **12 ordered completion declarations**,
zero declaration violations/unverified pairs, and zero count/trace validity
errors. [Captured region order](../results/paper/waits/lmcache/real-io/trace-order.json)
records the exact invocation sequence numbers supporting this comparison.

The [candidate patch](../examples/waits/lmcache_nonblocking.patch) sets the
socket nonblocking, awaits complete metadata/payload reads, and retains one
transaction lock across the whole GET response. It also yields between
receive chunks of at most 1 MiB: merely using `sock_recv_into` does not yield
when data is already available. This bounds work per event-loop turn. It is
an isolated experimental patch, not an upstreamed production fix; cancellation
and connection recovery still need broader validation.

Separate **uninstrumented** runs provide the latency numbers below. Each cell
is a median over 11 trials, excluding one warmup per size/arm and alternating
arm order. Both arms use the same local server in a four-CPU container.

| Concurrent GET size | EXISTS before, ms | EXISTS after, ms | GET before, ms | GET after, ms |
| --- | ---: | ---: | ---: | ---: |
| 1 MiB | 0.97 | 0.86 | 0.83 | 0.84 |
| 3 MiB | 2.05 | 0.76 | 1.89 | 1.78 |
| 16 MiB | 9.60 | 1.02 | 9.36 | 8.78 |
| 64 MiB | 79.83 | 2.27 | 79.51 | 70.56 |

For 3, 16 and 64 MiB, the independent lookup finishes before GET in **0/11**
baseline trials and **11/11** patched trials per size. Every retrieved payload
passes SHA-256 comparison. Eight additional protocol checks (both arms × four
sizes) each queue five commands on one connection: two GETs, a hit lookup,
a miss lookup and a missing GET. Both payloads and all hit/miss results match.

**What drperf contributed:** native receive observations and region sequencing
identified serialization worth changing; checked completion edges retained
the caller/result relationships in both implementations. The region-level
wait graph alone has the same semantic edges before and after. There is no
declared `EXISTS waits on GET` dependency: scheduling interference is not a
data dependency, and the checker must not invent that edge. The independent
native timing run establishes the practical effect.

Coverage is not complete: the baseline has 12 uncovered operation invocations,
and the patched capture has 66 (including active coroutine steps). Client
markers do not identify a publisher for remote socket data. Splitting a
coroutine at suspension also increases the invocation-budget count; it is not
evidence that the patch introduced more semantic dependencies or more blocking.
No all-waits-covered or end-to-end speedup claim follows from these profiles.

Evidence: [native measurements, protocol checks and source hashes](../results/paper/waits/lmcache/real-io/native-final.json),
[summary](../results/paper/waits/lmcache/real-io/summary.json),
[before profile](../results/paper/waits/lmcache/real-io/before.drperf.json),
[after profile](../results/paper/waits/lmcache/real-io/after.drperf.json).
The profiles are 108/121 KiB; raw captures and adjacent wait-evidence sidecars
are under the same `real-io/` directory. The installed source snapshot is
[here](../results/paper/waits/lmcache/real-io/lm_connector_before.py).

To reproduce in an environment with this LMCache version and CPU torch:

```sh
# Keep the installed package untouched; patch a copied connector file.
cp /path/to/lmcache/v1/storage_backend/connector/lm_connector.py /tmp/lm_connector_after.py
patch /tmp/lm_connector_after.py < examples/waits/lmcache_nonblocking.patch
python examples/waits/lmcache_real_io.py --patched /tmp/lm_connector_after.py \
  --output /tmp/lmc-native.json --sizes 1,3,16,64 --repeats 11
# Capture each arm separately; capture timings are not latency evidence.
DRPERF_WAITS=1 DRPERF_FOLLOW_THREADS=0 DRPERF_REPORT=/tmp/lmc-before.drperf.json \
  bin/drperf python examples/waits/lmcache_real_io.py \
  --patched /tmp/lm_connector_after.py --output /tmp/lmc-capture.json \
  --capture --mode before --sizes 3,16 --repeats 2
# Repeat with --mode after and a different report/output path.
```

## Semantic regions, rather than synchronization helpers

The current annotations place `waited` in the **caller that consumes the result**.
For example, `lmc.remote.get` waits for a connector GET operation; an implementation
helper such as `Future.result()` does not own the semantic dependency. Marking that
helper separately for instruction accounting must not change the declared edge.
See [waited.md](waited.md) for the API and checking contract.

The connector's original coroutine executes inside an operation region:
`lmc.remote.exists_operation`, `lmc.remote.get_operation`, or
`lmc.remote.put_operation`. These contain actual connector work, rather than a
marker-only `*_ready` region. The [annotation adapter](../results/paper/waits/lmcache/overlay/lmcache/_drperf.py)
closes the region when the coroutine suspends and reopens it when it resumes, so
interleaved tasks cannot become its children. Publication occurs inside the final
step, before the original Task/Future is released. A region invocation therefore
means an active coroutine step, not necessarily one whole asynchronous request.
The [adapter check](../results/paper/waits/lmcache/check_operation_steps.py) exercises
interleaving, exceptional completion, and cancellation.

The consumers are lookup (`lmc.remote.contains`), retrieval (`lmc.remote.get`),
and store completion bookkeeping (`lmc.remote.put_callback`). The last is an
already-ready future consumed by a callback, not evidence of a sleeping thread.
For this connector, PUT completion means its send coroutine returned; it does
**not** establish remote durable storage or acknowledgment.

The fresh baseline completed 144/144 requests with no failures. All **890**
declared dependencies were ordered, with zero violations, zero unverified pairs,
and no instruction-count or trace validity errors:

| Consumer region | Publisher region | Checked occurrences |
|---|---|---:|
| `lmc.remote.contains` | `lmc.remote.exists_operation` | 421 |
| `lmc.remote.get` | `lmc.remote.get_operation` | 381 |
| `lmc.remote.put_callback` | `lmc.remote.put_operation` | 88 |

This adds **one new semantic relationship**, the store-completion callback, and
relocates the two existing lookup/retrieval declarations from primitive or
marker-only regions to application operations. For every observed retrieval
invocation, the number of GET dependencies equals its `chunks` PCV. Each lookup
and store callback has one dependency. The current graph has 38 region names and
four distinct wait relationships: these three declarations plus the previously
captured native `mv.sync → mv.d2h` relationship.

- [Current standalone graph](../results/paper/waits/lmcache/operations.drperf.json)
- [Current baseline evidence](../results/paper/waits/lmcache/operations-analysis.json)
- [Current viewer report](../results/paper/waits/lmcache/operations.drperf.json)
- [Current raw captures and annotation sources](../results/paper/waits/lmcache/operations-evidence.tar.gz)

The viewer report is now **4.7 MB**. Earlier exports embedded the full native
synchronization history plus duplicated operation records, producing a 566 MB
JSON. The exporter now stores full evidence in a checksummed adjacent sidecar
(13.9 MB here). All costs, breakdowns, traces, and checked graph results remain
in the JSON; reloading the evidence reproduces all 890 checks exactly.

The new [publication-delay rerun](../results/paper/waits/lmcache/operations-probe-analysis.json)
also completed 144/144 requests. It injected **117 ms at all 88 PUT publications**;
all 890 declared dependencies remained ordered, with no violations, unverified
pairs, dropped wait records, or count/trace validity errors. Both runs retain four
unfinished background native waits separately. This checks observed publication
order under the perturbation; it does not establish time blocked or all-schedules
correctness. The complete delayed profile is
[available as JSON](../results/paper/waits/lmcache/operations-probe.drperf.json).

Native socket/lock observations remain separate. No annotation claims that a
client operation is the remote server's producer, or that host-side GPU submission
means device completion. This capture uses the synchronous retrieval configuration;
it does not exercise LMCache's separate asynchronous prefetch path.

## Workload and earlier evidence

Qwen2.5-0.5B-Instruct, dummy weights, one request at a time, 24 sessions × 6 turns,
40 MB device KV budget, 256-token chunks, and the `remote_naive` LMCache backend.
The installed LMCache package is pinned to `8e93a34`; the annotated source is the
existing `example/qwen2.5B/lmcache/lmcache_overlay` checkout. The remote cache server
runs in the same container. This is the specific `lm://` connector and synchronous
retrieval path, not a claim about every LMCache backend or configuration.

The real framework runs under **GX functional emulation and drperf**, without a
GPU or GXVM timing. Counts describe observed host instructions and synchronization
operations. They do not measure GPU latency, time actually blocked, or a critical
path. Results from the delay probe must not be used as baseline performance data.

- [Instruction/native-wait baseline evidence](../results/paper/waits/lmcache/baseline-analysis.json)
- [Checkpoint-annotated evidence](../results/paper/waits/lmcache/futures-analysis.json)
- [Complete checkpoint profile, compressed](../results/paper/waits/lmcache/futures.drperf.json.gz)
- [Raw evidence archive](../results/paper/waits/lmcache/evidence.tar.gz), including the original
  baseline, checkpoint run and delay probe with their process logs and sidecars.
- [archived profile JSON](../results/paper/waits/lmcache/operations.drperf.json): updated to the
  current capture above, with expandable hierarchy and cost details.
  The HTML retains graph evidence and summaries; omitted low-level native records
  remain in the raw captures. It is a visualization snapshot, not a fresh checker input.
- [Exact checkpoint patch](../results/paper/waits/lmcache/annotations.patch)
- [Publication-delay plan](../results/paper/waits/lmcache/probe-plan.json)
- [Completed delay-probe evidence](../results/paper/waits/lmcache/probe-analysis.json)
- [Event-loop check source](../results/paper/waits/lmcache/check_event_loop.py) and
  [three-trial output](../results/paper/waits/lmcache/event-loop-check.log)

Both unperturbed runs completed 144/144 requests, exited with `DRPERF_SERVE rc=0`,
and have complete region traces, zero dropped wait records, and no count/trace
validity errors. Four background semaphore waits were still open at process exit;
they remain explicitly unresolved.

The delay run also completed 144/144 requests and exited successfully. It injected
117 ms before each of 381 GET publications. All 802 declared relationships remained
ordered, with zero violations or unverified relationships and no dropped records.

## Observed interfaces

These are **counts of dependencies or synchronization calls**, not durations to
add to a latency formula. `chunks` is the existing region-entry PCV. The relations
below hold for each observed invocation, not just an average across invocations.

| Region | Observed interface | Workload total |
|---|---|---:|
| `lmc.remote.contains` | `1 * Wait[exists_ready]` | 421 dependencies |
| `lmc.remote.get` | `chunks * Wait[get_ready]` | 381 dependencies in 110 batches |
| `lmc.gpu.from_gpu` | `chunks * cudaStreamSynchronize` | 88 syncs in 40 batches |
| `lmc.gpu.to_gpu` | `1 * cudaStreamSynchronize` | 110 syncs in 110 batches |
| slot-mapping uploads inside `mv.h2d` | sync hidden inside a tensor copy | 150 syncs: 110 load, 40 save |

There were 134 lookup-region invocations and 421 remote EXISTS exchanges. GET
also performs one metadata receive per chunk, followed by payload receives.
Payload `recv` counts vary with socket fragmentation, despite equal 3 MB payloads:
1,190 calls in the native-wait baseline and 1,485 in the checkpoint run. A single
coefficient on bytes does not explain that receive-loop work exactly.

The native semaphore waits are not equivalent to semantic dependencies: only
149 `sem_clockwait` calls occurred inside the 381 GET result checks in the annotated
run. Some futures were ready before their result was consumed. The checkpoints
retain all 381 relationships, including already-satisfied ones.

## Design choices worth improving

### 1. The asynchronous connector blocks its event loop

[`exists`](../results/paper/waits/lmcache/overlay/lmcache/v1/storage_backend/connector/lm_connector.py:85)
uses synchronous `sendall` and `recv` inside an `async def` and a shared socket lock.
[`get`](../results/paper/waits/lmcache/overlay/lmcache/v1/storage_backend/connector/lm_connector.py:143)
also reads metadata synchronously, then calls the synchronous payload receive loop.
Submitting multiple GET futures does not make those network transactions concurrent.

This is more than a naming concern. The component test calls the **actual EXISTS
method** over a socket pair and withholds its response for 200 ms. An unrelated
callback queued on the same event loop cannot run until the response is released.
All three trials reproduced that behavior. This test establishes the scheduling
effect; it does not quantify production contention or end-to-end slowdown.

**Candidate change:** nonblocking socket operations with complete-message reads,
plus explicit scheduling/backpressure. Preserve a lock around a complete protocol
transaction; merely making `recv` awaitable does not safely multiplex this protocol.
Consider separate bounded load/store queues or connections if measurements show
that large transfers delay other operations. The existing GET comment intentionally
favors loading over saving; replace that implicit policy with an explicit one.

### 2. Chunk-by-chunk remote metadata gates the caller

[`contains`](../results/paper/waits/lmcache/overlay/lmcache/v1/storage_backend/remote_backend.py:164)
submits EXISTS and immediately calls `future.result()`.
[`batched_contains`](../results/paper/waits/lmcache/overlay/lmcache/v1/storage_backend/remote_backend.py:188)
falls back to individual lookups for this connector. The graph checks 421 caller
dependencies on the corresponding EXISTS completion checkpoints.

**Candidate change:** implement a real prefix/batched EXISTS operation, preserving
first-missing-chunk semantics. This can replace per-chunk round trips with one batch
round trip. Similarly, the GET fallback
[submits individual futures](../results/paper/waits/lmcache/overlay/lmcache/v1/storage_backend/remote_backend.py:453);
batching the wire protocol could reduce its 381 metadata exchanges toward one per
retrieval batch. An API called `batched_get` is insufficient by itself.

The existing `tokens` PCV does not encode remote residency. Prefix hits, misses,
and in-flight puts can change how many keys must be checked. Treat these as missing
state or unexplained behavior rather than claiming a universal formula in tokens.

### 3. Stores synchronize every chunk instead of every batch

[`from_gpu`](../results/paper/waits/lmcache/overlay/lmcache/v1/gpu_connector/gpu_connectors.py:424)
synchronizes after each host-bound copy. Its
[batch wrapper](../results/paper/waits/lmcache/overlay/lmcache/v1/gpu_connector/gpu_connectors.py:445)
calls it repeatedly. The measured multiplicity is exactly `chunks`. The load-side
batch wrapper already synchronizes once after its loop.

**Candidate change:** enqueue each batch's ordered gather/copy operations on the
store stream, then synchronize once before handing host buffers to storage.
For this workload the proposed change is **88 → 40 explicit store syncs**. That is
a structural prediction, not a measured optimization or a latency speedup. Keep
host buffers alive, preserve shared staging-buffer reuse on the same stream, and
retain the standalone `from_gpu` completion contract. Do not just delete the sync.

Native CUDA observations matched 88 `mv.sync` calls to preceding `mv.d2h` transfers.
The 110 load syncs remain unresolved at producer level: the transfer kernel is not
represented by the selected memcpy hooks. No false `mv.h2d` producer was invented.

### 4. Tiny metadata uploads also impose host synchronization

The vLLM adapter's slot-mapping
[load copy](../results/paper/waits/lmcache/overlay/lmcache/integration/vllm/vllm_v1_adapter.py:813)
and [save copy](../results/paper/waits/lmcache/overlay/lmcache/integration/vllm/vllm_v1_adapter.py:1172)
use `.to(self.device)` without `non_blocking=True`. Native interception finds 150
stream synchronizations inside these copies, separate from the 198 explicit
`mv.sync` calls. This is additional overhead that a search for explicit sync calls
would miss.

**Candidate change:** reuse appropriately pinned host/device metadata buffers and
enqueue copies with explicit stream ordering before dependent kernels. Validate
buffer lifetime, reuse and cross-stream dependencies; changing the flag alone is
not the complete design. At the outer adapter regions, `reqs` alone does not explain
the sync counts: only requests whose load/save conditions pass take these paths.

## How the dependency check works

The [separate annotation wrapper](../results/paper/waits/lmcache/overlay/lmcache/_drperf.py:39)
assigns each submitted future a unique event generation. It publishes immediately
before its coroutine returns, and records `waited` immediately after the original
`future.result()` succeeds. It adds no readiness condition and does not replace
the original wait. Regions do not span an `await`, because coroutines can interleave
on one execution thread. Marker implementation instructions are excluded by drperf;
these added Python wrappers still change some caller overhead, so use the original
baseline for instruction-count priorities.

All 802 declared dependencies in the unperturbed checkpoint run are ordered, with
no missing publisher or order violation. The automatically chosen GET publication
delay is 117 ms: the largest observed publication-to-consumption gap, 16,868 us,
rounded up to milliseconds, plus a 100 ms margin. The probe delays publication
before the original future release. The marker binding releases the Python GIL
during the delay. This is an observed-order check under perturbation, not a proof
of dependency necessity for every input or schedule.

Raw socket receives still have unknown remote producers, and native synchronization
is not silently assigned to the nearest declared checkpoint. The graph is deliberately
conservative. In particular, it does not claim that every lock call was contended.

## What this adds to the instruction-count view

The instruction baseline still points to repeated
[KV layout normalization](../results/paper/waits/lmcache/overlay/lmcache/v1/gpu_connector/gpu_connectors.py:138):
469 calls at about 1,169,000 host instructions each, approximately 51% of the captured
LMCache-region instructions in this workload. Cache that result while the relevant
tensor identities/layouts remain unchanged. This is a CPU-work candidate.

Wait analysis adds a different set of questions: why must the caller wait per chunk,
why does one slow socket prevent unrelated loop work, and which ordinary operations
implicitly synchronize? The next useful evaluations are nonblocking connector I/O,
batched metadata requests, and one store barrier per batch. These proposals have
not yet been implemented or validated as end-to-end speedups on a real GPU.

## Capture and reproduction

The [launcher](../results/paper/waits/lmcache/launch.sh) uses the existing remote development
image and mounted workspace on `icdslab2.epfl.ch`, with `DRPERF_WAITS=1`, one million
wait-record slots, and GX excluded from instruction counts. An isolated copy of
drperf is mounted at `/home/ubuntu/drperf` in the container. The
[driver](../results/paper/waits/lmcache/drperf_lmcache.sh) accepts `LMC_OVERLAY`; warmup and
settling invocations are not application-region measurements. Synchronization
capture attaches before engine construction to retain object histories.

```
# Remote host; existing image, package, model config and GX mounts required.
bash ~/ditto_kv_gx/lmc-waits/launch.sh remote_naive lmc-waits-baseline-NEW
LMC_OVERLAY=/w/lmc-waits/overlay \
  bash ~/ditto_kv_gx/lmc-waits/launch.sh remote_naive lmc-waits-futures-NEW
LMC_OVERLAY=/w/lmc-waits/overlay \
DRPERF_WAIT_DELAY_REGION=lmc.remote.put_operation DRPERF_WAIT_DELAY_MS=117 \
  bash ~/ditto_kv_gx/lmc-waits/launch.sh remote_naive lmc-waits-probe-NEW
```

The current captures are `lmc-waits-20260930-operations` and
`lmc-waits-20260930-operations-probe`. Earlier
capture directories are `lmc-waits-20260929-remote-ext`,
`lmc-waits-20260929-futures`, and `lmc-waits-20260929-probe-get` under the remote
workspace's `runs/`. The analysis scripts and local evidence are together in
`/home/ubuntu/drperf/results/paper/waits/lmcache/`. Use the complete raw captures to re-export
or recheck; the HTML is only a compact display snapshot.

To recover the current raw folders and sources, extract `operations-evidence.tar.gz`
in `results/paper/waits/lmcache/`; `evidence.tar.gz` holds the earlier experiments. Open `operations.drperf.json` directly in VS Code. The adjacent
`operations.drperf.json.waits.<hash>.jsonl.gz` contains full evidence for rechecking;
the viewer does not load it. `futures.drperf.json.gz` is the earlier capture. The much smaller standalone HTML opens directly in
a browser. Full native records are large because synchronization capture also
retains object histories and interpreter/allocator lock calls; those counts are
not evidence that the locks were contended.

During this study the exporter was fixed to retain unfinished background waits
without invalidating unrelated completed dependencies. Mappings for every affected
object remain disabled; unfinished application-region calls and dropped records
still invalidate the applicable capture. Wait and event checker tests cover this.
Validation passed: 24 wait tests, 17 event-checker tests, and 12 execution-graph tests.
The standalone HTML was also opened in Chromium: 38 distinct regions represented
by 45 hierarchy boxes, three wait edges, and no JavaScript errors.

## Invocation-budget recheck (2026-09-30)

Rechecking the existing `operations.drperf.json` with the invocation-budget
checker gives 4,372 obligations from 222,063 in-region native calls. Of these,
531 are covered: all 421 synchronization-bearing `lmc.remote.contains`
invocations and all 110 synchronization-bearing `lmc.remote.get` invocations.
The 890 declarations remain ordered with no event-pair conflicts; 359 credits
have no eligible outstanding obligation. Remaining obligations total 3,841.
Many involve internal mutex acquisition; the checker does not treat those as
proof of blocking or silently remove them. Credits cannot cross thread/root
boundaries or cover native calls after their checkpoint.

This is a recheck of the old capture, not a new LMCache run. The newer client's
versioned glibc condition-wait hooks require recapture for additional evidence.
Local detailed results: `results/paper/waits/coverage/lmcache-recheck.json`. The synthetic
coverage suite separately executes all twelve programs with the updated client.

## Conditional prefetch admission: an architectural choice made explicit

2026-10-01. A focused component experiment checks why a one-chunk prefetch must
wait for a separate large prefetch. The installed LMCache storage manager selects
`AsyncSingleSerializer` for async loading: its lock spans the whole prefetch,
including disk I/O. The same module contains `AsyncMultiSerializer`, which admits
concurrent requests within a chunk budget. The experiment compares these existing
implementations using the real `LocalDiskBackend`, separate keys, real 1 MiB files
and returned bytes. A CPU allocation adapter supplies unpinned buffers; this does
not validate a production switch under allocator fragmentation or full serving.

Exact code and evidence:

- Default selection: [storage_manager.py, line 288](../results/paper/waits/lmcache/prefetch-admission/storage_manager.py#L288).
- Exclusive serializer: [line 197](../results/paper/waits/lmcache/prefetch-admission/storage_manager.py#L197);
  budgeted serializer: [line 168](../results/paper/waits/lmcache/prefetch-admission/storage_manager.py#L168);
  budget rule: [line 123](../results/paper/waits/lmcache/prefetch-admission/storage_manager.py#L123).
- Actual backend: [prefetch, line 594](../results/paper/waits/lmcache/prefetch-admission/local_disk_backend.py#L594)
  and [read implementation, line 726](../results/paper/waits/lmcache/prefetch-admission/local_disk_backend.py#L726).
- Annotations: [publisher, line 79](../examples/waits/lmcache_prefetch_admission.py#L79)
  and [consumer, line 141](../examples/waits/lmcache_prefetch_admission.py#L141).
- [Explicit interface declarations](../examples/waits/lmcache_prefetch_admission.interfaces.json),
  [portable results and source hashes](../examples/waits/lmcache_prefetch_admission.results.json),
  [viewer JSON](../results/paper/waits/lmcache/prefetch-admission/baseline/profile.drperf.json),
  [archived profile JSON](../results/paper/waits/lmcache/prefetch-admission/baseline/profile.drperf.json).

The supplied interface is:

```text
small prefetch = CPU work
  + I[True] * Wait[small read]
  + I[peer_active == 1 and (serial == 1 or chunks + 1 > concurrent_budget)]
      * Wait[large read]
```

`peer_active` is the admission-time snapshot retained in the caller's entry
state, not a later estimate of whether the peer is running. `chunks` is the
large request's size; the small request always needs one chunk. The concurrent
budget is half the configured full chunk budget, following LMCache's existing
policy. Each workload sets it to either `chunks + 1` (both fit) or `chunks`
(only the large request fits). The marker follows actual admission state;
the predicate is supplied separately and checked per invocation.

The graph therefore exposes a policy: with exclusive admission, an unrelated
large read remains a prerequisite even with spare capacity. Budgeted admission
removes that prerequisite only when capacity permits. Publication is in the disk
worker before its future is released; the consumer checkpoint follows actual
completion. These are independent invocations, not a parent waiting on a
synchronous child. No region remains open across a coroutine suspension.

Uninstrumented native measurements use 11 trials per state after warmup,
alternating policy order. These are warm filesystem reads, not cold-storage or
end-to-end serving latencies. All 3,828 returned chunks in measured trials pass
SHA-256 checks.

| Competing large request | Exclusive, spare capacity | Budgeted, spare capacity | Exclusive, tight budget | Budgeted, tight budget |
| --- | ---: | ---: | ---: | ---: |
| 4 chunks | 2.840 ms | 2.111 ms | 2.502 ms | 2.521 ms |
| 16 chunks | 7.380 ms | 1.529 ms | 5.888 ms | 5.893 ms |
| 64 chunks | 19.237 ms | 3.660 ms | 19.313 ms | 20.166 ms |

Values are median small-request completion latencies. Under tight budgets, the
small request waits in all 11 trials for both policies. With spare capacity,
budgeted admission avoids the peer wait in all 11 trials at each size. The
16- and 64-chunk cases also complete the small read first in every such trial.

Separate DrPerf captures use 4- and 16-chunk peers: 24 small-request invocations
including warmups, with the peer dependency present in 18 and absent in 6. Both
declared interfaces check. The planner selects a 432 ms publication delay;
rerunning with that delay preserves the valid declarations. A forced peer wait
with an unconditional indicator fails on all six spare-capacity budgeted cases.
A wrong indicator, a wrong publisher, and omitted peer checkpoints with the
declaration retained also fail. Removing just the declaration leaves 18
`unexplained(Wait[large read])` observations.

The experiment also records a coverage limit: removing **both** the peer
checkpoint and its declaration leaves no peer-specific unexplained term. The
own-result checkpoint already covers the caller's single invocation obligation.
This mechanism checks the proposed architecture; coverage does not establish
that it lists every dependency. All four captures retain 48 unexplained
read-region obligations and cover 24 caller obligations. These were not erased
with blanket null refinements.

Reproduce in an environment with the recorded LMCache sources, PyTorch, and a
built DrPerf (no GPU required):

```bash
python3 examples/waits/check_lmcache_prefetch.py \
  --drperf-root "$PWD" --out /tmp/lmcache-prefetch-study
```

The driver runs native timings, captures baseline and planned-delay executions,
checks both interfaces, and asserts the negative controls. Each capture retains
the executed driver, source hashes, raw evidence, and a viewer JSON with its
checksummed wait sidecar. Local [controls](../results/paper/waits/lmcache/prefetch-admission/controls.json)
and [validation log](../results/paper/waits/lmcache/prefetch-admission/validation.log) record
the checked outcomes. This is a concrete use of the wait interface to compare
concurrency policies, not an automated discovery or agent-advantage result.

### Fresh-agent workflow on the prefetch workload

A fresh GPT-5.6 Sol agent received only an unannotated workload, its DrPerf
feedback, generic waited documentation, and access to the installed LMCache
source. It did not receive the preceding experiment, issue, or candidate policy.
This is one preselected component workload, not a controlled comparison against
an agent without DrPerf. Its contemporaneous
[investigation record](../results/paper/waits/lmcache/blind-prefetch/investigation.md)
separates the feedback that directed inspection from conclusions drawn from code.

The initial report has 18 unexplained invocation obligations: six each in the
two disk-read regions and in `lmc.prefetch.consume_b`. Following that feedback,
the agent checked the own-result dependency
`I[need_b == 1] * Wait[lmc.disk.read_b]`, covering the six caller obligations.
It left the twelve read-region obligations unexplained. Tracing the completion
path led it to the serializer's lock across the entire request and the disk
executor's four workers; it independently chose to test the existing budgeted
serializer. Across 160 native observations, all 1,760 returned chunks passed
payload checks. The small request's pooled median latency changed from
2.454 to 1.102 ms with a four-chunk peer and from 5.936 to 1.764 ms with a
16-chunk peer. These are CPU component measurements with an allocation adapter.

The [annotated profile](../results/paper/waits/lmcache/blind-prefetch/annotated/profile.drperf.json)
checks all six declared dependencies. The candidate preserves that edge and
indicator: the own-result interface alone does not describe why the result was
delayed. Thus this trial demonstrates a report-guided investigation; representing
the admission policy itself requires further refinement. Both the
[initial report](../results/paper/waits/lmcache/blind-prefetch/initial/profile.drperf.json)
and [candidate report](../results/paper/waits/lmcache/blind-prefetch/annotated-multi/profile.drperf.json)
are retained, including their unexplained terms. No default LMCache policy was
changed.

In a bounded follow-up, an instrumented copy of the original single-lock path
wrapped each active step of B's real `asyncio.Lock.acquire()` awaitable. The
region closed before each suspension. Six acquisitions each suspended once and
resumed, giving twelve active-step invocations, with **zero native synchronization
operations captured inside those steps**. A supplied `admit_b -> release_a`
dependency checked in all six cases. Omitting its checkpoint failed the supplied
indicator six times, but did not create a native admission obligation. The
[refined profile](../results/paper/waits/lmcache/blind-prefetch/admission-final/profile.drperf.json)
and [omitted-checkpoint profile](../results/paper/waits/lmcache/blind-prefetch/admission-final-omitted/profile.drperf.json)
record this distinction. Detecting an entirely undeclared async admission would
need runtime evidence for a task-scoped logical await's suspension and completion.
The publisher can remain user-declared; automatic producer inference is not
required for missing-annotation coverage. The runtime extension was subsequently implemented and tested below.

## Both granularity gaps fixed: native operations and async admission

The checker now retains multiple obligations within one invocation. Two
successful waits cannot be covered by one later checkpoint just because they
share a region or native object. Consecutive failed attempts at the same
API/object/caller site remain grouped; successful spurious wakes remain separate.
Moving annotations to a parent preserves these obligations. The earlier broad
architecture profiles above used the older invocation rule: their zero-residual
numbers must not be read as results under the stricter accounting.

The [async admission regression](../examples/waits/lmcache_async_admission.py)
uses the installed, original `AsyncSingleSerializer.run` and real
`LocalDiskBackend` I/O. A CPU allocation adapter supplies the data buffers.
Boundary wrappers annotate its original lock methods; the serializer's algorithm
is neither copied nor optimized. The runtime adapter observes the actual
`asyncio.Lock.acquire`, including suspension, and leaves publisher selection to
the supplied `publish`/`waited` declarations. Two peer sizes (4 and 16 one-MiB
chunks), three trials each, check **66 payload hashes per capture**.

| Capture | Admission obligations | Covered | Admission interface |
| --- | ---: | ---: | --- |
| [No admission checkpoint or declaration](../results/paper/waits/lmcache/blind-prefetch/async-omitted/profile.drperf.json) | 6 | 0 | `unexplained(Wait[?])` |
| [Declared admission dependency](../results/paper/waits/lmcache/blind-prefetch/async-annotated/profile.drperf.json) | 6 | 6 | `I[needs_a_release == 1] * Wait[lmc.prefetch.release_a]` |

The second interface checks six logical invocations, not twelve coroutine
steps. Its active region closes before suspension; another task on the event
loop cannot consume its coverage credits. Native operations nested inside the
same scoped runtime primitive remain in the evidence as `runtimeContainer`
links, without charging twice for one operation. Native calls in another task
or outside that primitive retain their own obligations.

Both reports preserve unrelated unexplained synchronization. In these captures
the complete coverage totals are 6/1299 and 12/1302; most remaining observations
are conservative interpreter/library lock candidates in disk I/O. These totals
are schedule-sensitive and are not counts of independent application
prerequisites. The admission result is the bounded regression: a previously
invisible async prerequisite is now reported even when entirely undeclared,
then covered by a checked, explicitly supplied dependency. No latency
improvement or complete whole-system dependency recovery is claimed here.

For a small standalone reproduction, run
`python3 examples/waits/check_async_operations.py`. It tests two actual async
waits in one invocation with zero, one, or two checkpoints, both locally and
relocated to the parent. `examples/waits/check_coverage.py` also tests two
successful native waits at the same object and caller site. None requires a GPU.
