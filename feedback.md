# Feedback from Ditto P/D wait annotations — 2026-09-30

The following comes from applying [waited.md](docs/waited.md) to Ditto's
prefill/decode completion join and shared-memory storage client. These are
documentation/diagnostic suggestions and a feature request, not evidence that
the checker returned an incorrect ordering result.

## Correction: scope of the initial Ditto evidence

The earlier P/D examples covered only notification bookkeeping. The generated
`pd.ready -> pd.poll.*` edges did not identify the work producing KV data.
Ditto has removed those annotations and the extra expected-input registry.
They must not be presented as whole-path coverage or transfer-wait validation.
The feedback below concerns report clarity and cross-process evidence, not a
request to add a dependency-tracking system to the application.

## Identify same-thread delay probes in the report

Our initial P/D publication-delay probes delayed notification handling and then
consumed that notification later on the same thread. They passed, but supplied
little additional evidence about asynchronous transfer synchronization: the
delay also postponed the consumer's execution. These annotations establish CPU
notification ordering, not host DMA/NIXL device completion.

Suggested diagnostic: show publisher/consumer thread identities and label a
same-thread probe as a sequential ordering check. Keep the ordered result;
do not describe it as a failed declaration. Explain that this probe does not
independently hold back a producer while allowing its consumer to run.

Evidence: the earlier [host delay](../compression/ditto_kv/build/pd-qwen3/drperf-pd/delay-host/events.json)
and [NIXL delay](../compression/ditto_kv/build/pd-qwen3/drperf-pd/delay-nixl/events.json)
captures. Ditto's documentation now records their narrower scope.

## Feature request: explicitly scoped cross-process declarations

Ditto's Python client waits on an eventfd-backed mailbox; its Rust server
publishes replies in another process. The current captures cover the client,
and the documented process-local event identities cannot connect these ends.

A future cross-process API would need an explicit capture/session namespace,
connection incarnation, and request generation, with both processes captured.
It should retain unverified status when either endpoint is absent. Equal local
IDs, file descriptor numbers, or timestamps alone must not establish an edge.
This would allow us to declare the storage RPC's actual semantic dependency
without requiring automatic eventfd producer discovery.

Native synchronization coverage should remain distinct from this declared
ordering check, as `waited.md` already specifies.

## Counter allocation feedback from the broader GX worker capture

A capture of actual Ditto shared workers (154 process threads) overflowed the
basic-block table at 262,144 slots. Raising it to 2,097,152 slots with unmarked
thread following enabled then exhausted the 96 GiB virtual counter budget:
`counter_denied=2064980`, despite only 241 exported region/state entries.
The capture was rejected. Its 500,000-record wait history also overflowed.
Evidence: `../compression/ditto_kv/build/pd-qwen3/drperf-storage/shared-worker-gx-r4/raw/run.15.json`.

It would help if the capacity diagnostic reported the number of allocated
thread/key arrays and their per-array reservation, and explained that following
unmarked background threads multiplies that allocation. Increasing the block
table alone can make the counter-budget failure worse. Our next capture keeps
explicitly marked worker threads, disables unmarked-thread following, and
increases wait-history capacity; it must pass validity checks independently.

The next worker run (`shared-worker-gx-r5`) had zero counter/wait/trace drops
in its raw header with 4.45 GiB of reserved counters, but stalled during native
symbol resolution after the workload printed `SHARED_WORKER_PROFILE_PASSED`.
A debugger sample placed its main thread inside `libdrsyms.so` (offset 0x79e0).
After several minutes without report progress we stopped that capture; its
JSON is incomplete and must not be exported as a valid model. A progress
indicator or separately bounded symbolization phase would make this failure
much easier to diagnose. The follow-up capture uses `-no_symbols`, retaining
region instruction counts and module/block identity.


## Workspace capture feedback — 2026-10-01

The instruction attribution was useful again: Ditto's completion-recycle
region exposed a coefficient of about 2.24 instructions per following vector
entry. Replacing repeated removal with stable compaction and caching the
completion query method reduced inclusive reaper instructions by 56.2% in
our repeat-poll replay. Cached dtype lookup reduced array setup by 33.5%.
Evidence: [workspace profiles](../compression/ditto_kv/example/qwen3/workspace/drperf/README.md).

Two reporting improvements would help:

- The same logical PCVs can describe different completion implementations.
  Combining a simple Python completion and threading.Event completion in one
  capture produced a poor reaper fit. Separate cost/wait captures fixed the
  ambiguity. Per-state sample dispersion or an explicit capture-context tag
  would help detect such mixing before interpreting a mean as a formula.
- Wait coverage still flags NumPy/interpreter/CUDA-runtime synchronization
  alongside our declared GPU fence waits. Checked null declarations correctly
  validate the observed drain/backpressure paths, but coverage remains partial.
  A UI grouping by native caller/module and observation kind would help users
  distinguish runtime synchronization from a missing application dependency,
  without suppressing observations or treating a CUDA launch as completion.

An exporter ergonomics issue: a shared wait declaration manifest fails when a
capture omits any declared region. We filtered manifests to observed regions.
An explicit optional mode to label absent regions as unobserved, rather than
verified or invalid, would make a common manifest usable across focused tests.

CLI discoverability: `drperf-check-waits` accepts the older API-oriented claim
schema (`api`, no `event`/`reason`). Passing a semantic waited manifest that
`drperf-export --wait-interfaces` accepts yields "Unsupported claim field;
selectors must not hide observations". The message implies a bad declaration
rather than a different checker/schema. Please identify the supported schema
and direct semantic waited manifests to the exporter's interface checker.
The Ditto reports use the exporter's checked semantic claims.


## Application regions reached during PCV evaluation — 2026-10-01

Broader Ditto worker coverage called decorated placement helpers from
`decode_metrics()`, itself a PCV computation. Excluding the `perf.pcv` region
name alone did not exclude those nested application regions: 22 decoded layer
units appeared as 44 placement-region calls. This was an integration issue in
Ditto's annotation wrapper, not evidence of duplicate production execution.
Ditto now suppresses nested Python annotations during PCV calculation, using a
context-local depth so other threads remain visible, and tests both behaviors.
A report diagnostic for application regions called beneath a declared PCV-only
scope would help catch this mistake before interpreting cost or invocation
counts. Evidence: [worker coverage](../compression/ditto_kv/example/qwen3/workspace/drperf/worker/README.md).
