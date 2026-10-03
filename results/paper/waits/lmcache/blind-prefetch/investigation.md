# Blind LMCache waited investigation

## Scope and evidence order

This trial exercises the real LMCache `LocalDiskBackend` and its real asynchronous
serializer with a CPU-only buffer adapter. It has no GPU, no network, four CPUs,
and an 8 GB container limit. The initial workload and capture scripts are
preserved unchanged under `initial/` before annotation work.

1. **Feedback-guided evidence (before reading LMCache implementation).** The
   completed baseline reported ordered events, `0/18` synchronization-bearing
   invocations covered, and these unexplained obligations: `lmc.disk.read_a`
   (6 observations), `lmc.disk.read_b` (6), and `lmc.prefetch.consume_b` (6).
   There were no event annotations in this capture. This feedback motivates
   inspection of the disk-read path and the small-request future consumption;
   it does not identify a producer or establish an optimization opportunity.
2. **Workload facts visible before implementation inspection.** Each trial
   submits request A first (4 or 16 chunks), then one-chunk request B, both
   through one `AsyncSingleSerializer`. The main thread waits for B first and A
   second. `lmc.disk.read_a` and `lmc.disk.read_b` wrap
   `batched_async_load_bytes_from_disk`; `lmc.prefetch.consume_b` wraps
   `Future.result()` for B. Payload SHA-256 and result counts are checked after
   both futures complete.

## Source investigation

3. **Feedback-guided source trace.** In installed LMCache
   `v1/storage_backend/local_disk_backend.py:594-655`,
   `batched_get_non_blocking` prepares every requested object and then awaits
   one submitted prefetch job. `LocalDiskWorker.submit_task` at lines 42-75
   gives prefetches equal priority and forwards them to an
   `AsyncPQThreadPoolExecutor`. The synchronous batch reader at lines 726-751
   reads the chunks sequentially inside that one job. In
   `v1/storage_backend/job_executor/pq_executor.py:133-181`, the executor starts
   four asynchronous workers and each uses `asyncio.to_thread` for a submitted
   synchronous job.
4. **Source-only observation suggested by that trace.** The workload wraps both
   requests in one `AsyncSingleSerializer`. Its installed implementation at
   `v1/storage_backend/storage_manager.py:197-214` holds one `asyncio.Lock`
   across the entire awaited request, explicitly forcing requests to serialize.
   Therefore request B cannot reach the four-worker disk executor until A's
   whole batch returns. This is a plausible head-of-line delay in this workload,
   not yet a measured optimization and not a general conclusion about LMCache's
   production allocator constraints.

## Annotation and checks

5. The honest cross-thread dependency visible at the workload boundary is that
   `lmc.prefetch.consume_b` completes `Future.result()` only after
   `lmc.disk.read_b` has produced its batch. Event `101` is published after the
   real batch read returns but before the producer region exits, and the same
   generation is marked waited only after `b.result()` succeeds. The consumer
   records entry PCV `need_b=1`; the declaration uses explicit indicator
   `need_b == 1` and names producer `lmc.disk.read_b`.
6. No null markers were added to the disk-read regions. Their detected Python
   mutex/condition calls do not by themselves justify inventing an application
   publisher or treating all internal locking as one reviewed semantic wait.
   Those obligations should remain visible unless stronger evidence identifies
   their meaning.

## Design questions and experiments

7. **What the checked interface says.** The single-serializer annotated capture
   checked all six instances of
   `I[need_b == 1] * Wait[lmc.disk.read_b]`: event order was ordered, with no
   violations or unverified edges. Coverage rose from 0/18 to 6/18. This tells a
   reader that every observed B consumption depended on its B disk-read result.
   It does **not** represent the earlier A request, the serializer lock, or the
   reason B started late. Discovering the A-before-B gate still required source
   inspection.
8. **Candidate change.** I selected LMCache's existing `AsyncMultiSerializer`
   for this isolated workload instead of `AsyncSingleSerializer`. This is a
   workload-level configuration experiment, not a patch to installed LMCache.
   It is warranted here because the CPU adapter reports a 128-chunk budget and
   the tested concurrent totals are only 5 or 17 chunks. Production suitability
   still depends on the real allocator's budget and the race/deadlock constraint
   documented by the serializers.
9. **Native timing and payload correctness.** Two native runs per serializer,
   20 repetitions at each A batch size per run, were executed in opposite mode
   orders. All 160 rows passed the existing SHA-256 check and had exactly
   `chunks + 1` returned payloads. Pooled B-latency medians were 2.454 ms
   (single) versus 1.102 ms (multi) with 4 A chunks, and 5.936 ms versus
   1.764 ms with 16 A chunks: reductions of about 55% and 70%, respectively.
   Individual-run medians agreed in direction: single 2.466/5.069 ms and
   2.163/5.975 ms; multi 1.024/1.924 ms and 1.116/1.750 ms. Files
   `native-{single,multi}.json` and their `-reverse` counterparts retain every
   sample and correctness count.
10. **Instrumented candidate check, kept separate from native timing.** The
    multi-serializer capture retained the same checked declaration and explicit
    condition: six `consume_b -> disk.read_b` edges, all ordered. Thus the
    declared dependency graph and condition did not change; scheduling and
    timing did. Concurrent progress also made `lmc.prefetch.consume_a` carry a
    detected wait in five of six invocations, raising total obligations to 23.
    I did not annotate A merely to improve coverage, so this new observation
    remains `unexplained(Wait[?])` alongside both disk-read regions.

## Conclusion and limitations

The initial `lmc.prefetch.consume_b` feedback usefully directed attention to the
future completion path, while the two disk-read warnings directed inspection of
the backend and executor. The annotation/check phase established only the
result dependency it can directly support. The potential head-of-line issue was
source-derived: `AsyncSingleSerializer` prevents B from being submitted until A
finishes despite a four-worker executor. Native measurements support a material
B-latency benefit from the existing budgeted multi serializer in this CPU-only
test, without payload corruption.

This is not evidence for changing LMCache's default or removing serialization.
The trial uses page-cache-friendly temporary files, one process, a synthetic
CPU allocator, fixed 1 MiB chunks, and no GPU/network. It does not stress staging
memory exhaustion, cancellations, failures, concurrent writes, or production
allocator behavior. The remaining 12 single-mode disk-read obligations, and the
17 multi-mode disk-read/consume-A obligations, are intentionally unresolved.
The ordinary review profile is `annotated/profile.drperf.json` with its adjacent
checksummed waits sidecar; `annotated-multi/` records the candidate scheduling
capture. Re-loading both profiles with wait evidence and rerunning the event
checker reproduced the checked claim and the reported unresolved obligations.

## Focused admission refinement

11. A bounded follow-up tested whether the original single-serializer admission
    can be represented directly. `admission_workload.py` contains an explicitly
    labeled **instrumented copy** of the installed `AsyncSingleSerializer`
    control shape; the untouched source used for comparison is saved at
    `source_snapshot/storage_manager.py`. Request A publishes event `202` in
    caller-supplied region `lmc.prefetch.release_a` immediately before releasing
    the original `asyncio.Lock`. Request B drives the real `lock.acquire()`
    awaitable while wrapping each active `send`/`throw` step in
    `lmc.prefetch.admit_b_step`. Every step region closes before yielding back to
    asyncio, so thread-local regions never span coroutine suspension. After the
    acquire awaitable completes, caller-supplied `lmc.prefetch.admit_b` records
    `event_waited(202, generation)`. Its entry PCVs are
    `needs_a_release`, `peer_chunks`, and the actual number of acquire steps;
    the explicit indicator is
    `needs_a_release == 1 and attempts >= 1`.
12. In `admission-final/`, every B acquisition took two active awaitable steps:
    six `attempt=1` and six `attempt=2` region invocations. Neither step state
    recorded native waiting (`waiting=0`), and the final admitted region also
    had no native wait. The six `admit_b -> release_a` events nevertheless
    checked as ordered, with no mismatches or unverified cases. Native coverage
    remained 0/18; the same disk-A, disk-B, and result-consumer obligations from
    the initial workload remained unexplained.
13. `admission-final-omitted/` repeats the same operation but omits the final
    `event_waited`. The declaration becomes invalid with six
    `unexpectedAbsence` counterexamples, each at `attempts=2`. It does **not**
    create a new unexplained native admission obligation. This means the
    interface checker can catch an omitted declared semantic checkpoint, while
    current native wait capture cannot independently demand that checkpoint for
    an `asyncio.Lock` suspension. The result-consumer wait remains unexplained
    in both refinement captures and cannot be reassigned specifically to lock
    admission without conflating admission with later disk completion.
14. Making admission independently coverage-bearing would require runtime
    evidence beyond intercepted pthread/semaphore calls: task identity, the
    transition where B suspends on this particular lock, the matching A lock
    release/generation that makes B runnable, and B's resumption/completed
    acquire. DrPerf would then need to turn that task-level suspend/resume span
    into one admission obligation per B invocation (and one per actual retry if
    acquisition logic retries), rather than counting each active coroutine step
    or pretending the inactive suspension is a native blocked call. No blocking
    behavior was added to manufacture such evidence.

## Checker-design clarification after review

The final admission profiles were independently reloaded and checked by the
parent: both contain 12 active-step invocations and **zero captured native
synchronization operations** in `lmc.prefetch.admit_b_step`. The omitted variant
has six explicit-indicator failures. These counts support the stated async
coverage gap without relying on a post-acquire-only region.

The minimal coverage extension does not require discovering which region
published readiness. It needs a task-scoped identity for one logical await,
evidence that it suspended, and evidence of its completion, so one obligation
can survive coroutine suspension and retries. The existing user-declared
publish/waited channel supplies the producer and generation for order checking.
Retries within the same logical invocation should remain grouped, just as native
retries are today. Inferring a matching release automatically is a separate,
stronger feature and is not required to raise `unexplained(Wait[?])`. This runtime
extension has not been implemented in this trial.
