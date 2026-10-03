# Declaring and checking waits with drperf

**drperf is a checker. The human or agent declares the semantic dependency;
drperf checks that declaration against observed execution.** It does not discover
the application's intended dependency graph, implement synchronization, or make
an incorrect program wait correctly.

A declaration says: **region A has waited for a particular result published by
region C**. The graph displays `A → C`, from consumer to publisher. The lock,
future, socket, or synchronization helper used to implement that wait need not
be a region.

## One command

```bash
drperf python3 app.py
```

Use `drperf ./myapp` for a native application. This captures instructions and
synchronization, fits interfaces, checks declarations, and automatically writes
`drperf-report/report.txt`, `profile.drperf.json`, and `graph.html`. The text
report includes an architecture graph for agents; the HTML provides the
interactive human graph and region details. There is no separate export, check,
or graph command in the public workflow.

Declare the indicator directly in `wait`; no separate JSON file is needed.
Unexplained work is feedback; contradictory declarations fail the run's checks.
Delay injection is an explicit optional probe. Keep companion evidence files
with the report for rechecking. See [SKILL.md](../SKILL.md) for the agent workflow.

## Inline indicators

```python
from perfmark import region, release, wait

# In the producer, immediately before the application's real release:
with region("cache.fetch"):
    fetch_result()
    release(42, generation)
    ready.set()

# In the consumer (normally executing separately):
with region("request.load", need_fetch=int(need_fetch)):
    if need_fetch:
        ready.wait()
        wait(42, generation, indicator="need_fetch == 1", producer="cache.fetch")
```

`wait(event, generation, indicator="expression", producer="region")` records a
completed dependency. The indicator is a **string expression over entry PCVs**,
not a Boolean computed at the checkpoint. It asserts when the checkpoint should
appear; it never suppresses the checkpoint or implements synchronization.
The third positional argument can also supply the indicator. `producer` is
optional: without it, the event channel names the dependency and the checked
release region supplies its displayed name. Supplying it additionally checks
that specific region name. Use `"True"` for an unconditional dependency.

The interface is `CPU_formula + I[need_fetch == 1] * Wait[cache.fetch]`.
For each captured region invocation, DrPerf compares the declared indicator
with checkpoint occurrence. Indicator 1 without a checkpoint, or 0 with one,
is a failed check. Equal PCV states are checked individually, not averaged.
Expressions support integer arithmetic, comparisons, and Boolean operations;
they must evaluate to 0 or 1. DrPerf does not fit a replacement indicator.
Repeated checkpoints have a separately fitted affine multiplicity; unexplained
multiplicity remains visible.

Inline metadata is captured at runtime. Literal Python `wait` sites inside
literal `region` or `async_region` scopes are also read from the selected source
tree, so an entirely unexecuted marker can still be checked for missing
occurrences. Use literal event IDs or module-level integer constants, literal
indicator strings, and direct `perfmark` imports (aliases are supported).
Dynamic sites that cannot be resolved from source are reported as unverified.
For native code or Python declarations outside these supported source forms,
a declaration must execute at least once to be captured; an entirely unseen
site cannot establish an absence check. Source must match the captured program.

Observed declarations are embedded in `waitDeclarations`; results and
counterexamples are in `eventModel.interfaceChecks`. Native coverage is in
`eventModel.coverage`. The viewer displays indicators and unexplained terms.
Old `event_publish`/`event_waited` markers and optional `drperf.waits.json`
declarations remain supported for compatibility; new annotations should use
`release`/`wait`. Conflicting declarations are rejected.

## Refine a synchronization without naming an event

```python
with perfmark.region("counter.update", need=need):
    if need:
        with counter_lock:
            counter += 1
        perfmark.wait(None, indicator="need == 1",
                      reason="Protect a bookkeeping counter; no publisher is claimed.")
```

This retains `I[need == 1] * waited(null)` in the report. The required reason
is for **manual review**, not a statement the checker proves. There is no event,
generation, publisher, delay probe, or dependency arrow. A shared helper can
contain this refinement. Its indicator and coverage are checked normally:
one completed marker covers at most one eligible completed wait obligation
in itself or descendants, never a future operation and never twice.

A parent event-backed wait on its synchronous child is invalid, including
transitive descendants: publication in the child followed by a checkpoint
after its return only restates call/return order. A publication to its own
invocation is invalid too. These checks use invocation ancestry and thread
identity, not region names; a separately executing producer remains eligible.
A GPU stream fence without a claimed semantic publisher can instead use
`wait(None, indicator=..., reason=...)` in the child or its caller.

Native obligations without covering markers remain `unexplained(Wait[?])`.
Legacy checkpoints without indicators remain `unexplained(Wait[publisher])`;
DrPerf never silently supplies `I[True]`. CPU residual percentages do not
measure wait coverage or latency.

## Place the checkpoints at the semantic level

Use the same `(event_id, generation)` at both ends:

- `perfmark.release(event_id, generation)`: in C, immediately **before the
  real release** that makes the result available to the consumer.
- `perfmark.wait(event_id, generation, indicator="True")`: in A, **after the real wait or
  successful readiness check** has completed.

For example, A calls helper B to wait for C:

```python
import threading
import perfmark

ready = threading.Event()
event_id, generation = 1, 1

# Establish capture before either thread can publish, including with late attach.
with perfmark.region("capture", n=1):
    pass


def C():
    with perfmark.region("C"):
        # Prepare the result here.
        perfmark.release(event_id, generation)
        ready.set()  # The application's real release.


def B():
    ready.wait()  # Ordinary synchronization; no region annotation is required.


producer = threading.Thread(target=C)
producer.start()
with perfmark.region("A"):
    B()
    perfmark.wait(event_id, generation, indicator="True")
producer.join()
```

The declaration is **A waits for C**, not B waits for C. If B is marked for
separate instruction accounting, keep `wait` outside B, after its region
has ended. It still belongs to A. Different callers can use the same helper for
different dependencies and declare their own events at their call sites.

Similarly, put publication in the region that semantically produces the result;
you do not need to create an artificial region just for the marker. Do not leave
a region open across an `await` that can interleave unrelated coroutines: region
nesting follows execution threads, not asynchronous task identities.

## Event identities, branches, and timeouts

The two numbers are annotation identities, not synchronization objects. Use a
consistent event ID for the logical channel and a fresh generation for each
publication. Both sides must agree on the pair. Within a captured process, each
`(event_id, generation)` must have exactly one publication. An event ID belongs
to one ordered `(publisher region, consumer region)` pair across generations.
Reusing it for another region pair is a violation; choose a separate event ID.
Repeated invocations of the same consumer region may wait for the same publication.
Different processes' identical numbers are not matched by the current checker.

Original control flow determines whether a checkpoint executes; the inline
indicator asserts its presence. For example:

```python
with perfmark.region("A", need=int(need_result)):
    if need_result:
        if ready.wait(timeout=1):
            perfmark.wait(event_id, generation, indicator="need == 1")
        else:
            handle_timeout()  # Do not claim successful completion.
```

If a timeout occurs while `need == 1`, this indicator fails: it predicted a
completed dependency that did not occur. Exercise such paths and revise the
interface or region boundary explicitly; do not turn a timeout into a completion.

Already-ready results can also execute `wait`: it declares completed
readiness, not that the thread slept. A skipped wait, timeout, cancellation, or
exception must not accidentally execute a successful-wait checkpoint. Declare
PCVs on A for its explicitly declared indicator and any fitted multiplicity.

## Capture and check

Run `drperf python3 app.py` after building the tool and its Python extension.
Wait checking is enabled by default. `DRPERF_WAITS=0` explicitly disables it.
The text and JSON reports include the check outcomes and any optional
publication-delay suggestions; the HTML contains the region graph. Full
synchronization histories are stored in an adjacent checksummed evidence file.
No second command is required to perform the checks.

Capture must include publishers and consumers. Late attachment can miss earlier
publications. The Python checkpoint functions require the built `_perfmark`
extension; rebuild with `./build.sh` if needed.

The C equivalents in `perfmark.h` are:

```c
perfmark_release(event_id, generation);
perfmark_wait(event_id, generation, "need == 1", "producer");
perfmark_wait_null("need == 1", "Protect a bookkeeping counter.");
```

The marker calls and their instrumented bodies are excluded from instruction
counting. The application's actual wait and surrounding work retain their normal
accounting.

## Check again while delaying publication

A baseline can look ordered simply because the producer happened to finish
first. The report suggests a publication delay from observed checkpoint distances,
with a bounded maximum. It prints rerun settings; it does not launch the rerun.
For the example above:

```bash
DRPERF_REPORT_DIR=drperf-probe \
DRPERF_WAIT_DELAY_KIND=event DRPERF_WAIT_DELAY_REGION=C \
DRPERF_WAIT_DELAY_MS=500 \
  drperf python3 app.py
```

Drperf delays the publication marker before the following real release. The
consumer's existing synchronization should therefore keep it from reaching
`wait` too early. If B does not actually wait, a delayed run can expose
A reaching its checkpoint before C publishes. The markers themselves never
block the consumer on the declared event.

Keep the perturbed run separate from baseline cost measurements. A requested
probe that never reaches its target is reported as unverified, not counted as an
executed check. Delays may also exercise real timeout or cancellation paths.

## What the result establishes

- **Ordered:** each checked consumer checkpoint followed its unique matching
  publication in the observed execution.
- **Violation:** for example, the consumer passed its checkpoint before
  publication completed, or a generation was published more than once.
- **Unverified:** evidence is missing or ambiguous, capture is incomplete, or a
  requested delay was not executed.

The command exits with 1 for contradictory declarations and 2 for incomplete
checks or measurement errors. Unexplained waits remain report feedback. Passing a delayed run is additional
execution evidence; it is not a proof for every possible schedule or input, nor
proof that the declared producer is necessary for the consumer.

## Coverage budgets across nested regions

A region invocation may have **multiple wait obligations**. Each completed
native wait contributes one. Consecutive failed attempts at the same API,
object, and caller site are grouped until success, unless a checkpoint separates
them. Two successful waits stay separate, even at the same site on the same
object. Successful spurious wakes cannot safely be identified as retries from
this evidence, so they also stay separate. Two different operations inside one
region no longer disappear behind a single parent checkpoint.

Each runtime-observed logical await contributes one obligation, including its
internal suspension/resumption steps. This is still conservative observation,
not proof of blocking. The [granularity tests](../tests/test_wait_granularity.py)
and [async tests](../tests/test_async_waits.py) cover both forms.

```python
with region("A"):
    B()              # suppose B performs one logical wait
    B()              # another invocation, another obligation
    wait(e1, generation, indicator="True")
    wait(e2, generation, indicator="True")
```

Each valid, ordered `waited` checkpoint supplies one credit. Children spend
credits locally first; outstanding obligations pass to their enclosing caller,
recursively. A credit covers at most one obligation and cannot escape its
invocation into a sibling, another thread, or another root. It is never banked
for future work. Coverage for a group requires its **last** native call to have
returned before the checkpoint: a marker between retries does not cover later
calls in that group. This is checked after collecting the execution.

B does not warn just because its own body has no checkpoint. A later declaration
in A or a higher ancestor can cover it. At the end of the enclosing invocation
chain, remaining obligations are reported as `unexplained(Wait[?])`. Unfinished
or incomplete captures cannot establish coverage. A valid marker without a synchronization
observation remains an order check; it provides no credit for later operations.

The JSON stores `eventModel.coverage`, per-region totals, and bounded examples
of uncovered obligations. Each edge's `coverageCredit` records the invocation,
obligation number, and operation IDs charged to that credit. This is a deterministic accounting witness (local-first,
then oldest completed group), **not a native-object/publication association**.
The invariant is `obligations = covered + uncovered`. Native producer resolution
remains separate metadata, including for covered groups. A resolved native
publisher does not replace a semantic declaration.

Fresh native captures also record `callerModule` and `callerOffset` (the call
instruction's final byte, relative to its loaded module). Uncovered examples
retain up to three frequent caller sites, with the remaining call count. This
helps distinguish an interpreter or allocator lock from an application's
completion primitive. A caller site identifies who invoked the API, **not who
released it**. It neither proves blocking nor discharges the obligation; an
uncontended lock can still appear among the observations.

Native-GX captures have an explicit implementation boundary: a CPU
synchronization call whose recorded caller is the configured, excluded GX
module is kept in the raw evidence and counted under
`waits.implementationSynchronization`, rather than consuming an application
coverage credit. This requires the recorded `native_gx` configuration and an
exact caller-module match. CUDA synchronization stays in scope, as do Python,
PyTorch, application callbacks, and calls with unknown origin. The same rule
applies when rechecking an older report that contains the required provenance.

An unresolved obligation is not proof that a semantic annotation is missing.
The compiled `uncontended` example has one application thread and a fresh,
private, initially unlocked mutex. Its acquisition still contributes one
conservative API obligation, although there is no publishing peer to name.
Adding an unrelated or repeated `waited` to clear that term would misstate the
interface. This is a coverage-model limitation; no timing/syscall-count heuristic
is used to hide it.

This intentionally groups independent native waits within a single invocation
as well as retries. Use separate subregions if they need separate obligations.
Retries that re-enter a marked child contribute multiple child invocations;
those are not merged. Coverage is only for captured, supported API operations,
not unexecuted branches, custom spinning, or synchronization missed by late attach.
Passing both coverage and publication-order checks is not proof of causal pairing.

The [compiled examples](../examples/waits/coverage.c) exercise deep recursion,
timed retries, repeated calls, branches, shared helpers, concurrent callers,
wrong event reuse, premature markers, and missing credits:

```sh
./build.sh
python3 examples/waits/check_coverage.py
```

Outputs are under `results/paper/waits/coverage/`. `shared.drperf.json` deliberately fails
pair ownership; `relocated.drperf.json` moves declarations to the two callers
with separate event IDs and covers both B invocations. `missing.drperf.json`
leaves one of two calls uncovered. `retries.drperf.json` groups ten native calls
into one covered invocation. The CLI exits 1 for declaration violations, 2 for
unverified or incomplete coverage, and 0 only when both checks pass.

Re-export old captures to apply budget checking. To observe versioned glibc
condition-wait entry points missed by older clients, rebuild and recapture.
Viewer reload alone does not recompute checks stored in an existing report.

For GPU work, a CPU-side kernel or copy submission is **not completion**. Do not
label a submission checkpoint as GPU completion. These passive markers check
CPU checkpoint order; stronger device-completion claims need corresponding
captured device evidence.

The runnable [Python example](../examples/waits/events.py) includes a deliberate
`missing-wait` mode. See [wait_task.md](../wait_task.md) for capture details and
[the LMCache study](LMCache_WAITS.md) for application evidence.

No helper/semantic classification is required. Region-pair reuse is checked
from captured publications and checkpoints. The checker does not move annotations
automatically or reject a helper solely because it has multiple callers.

### Ditto page-reclaim regression

Run `python3 examples/waits/check_ditto_reclaim.py --ditto /path/to/ditto_kv`
after building drperf. This exercises the actual `DittoOffloadingWorker.wait`,
`CacheRuntime.flush_source_lease`, and `CompactionEngine.on_lease_flush` methods
using Ditto's upstream CPU GC harness and its native FTL library. A controlled
native semaphore supplies asynchronous completion; this is **not a full vLLM
inference run or real GPU compression**. The Ditto checkout is left unchanged.
The driver extracts the worker method without importing vLLM, as upstream
control-path tests do, and restores a blocking fence only in an isolated copy
of `on_lease_flush`.

Outputs live in `results/paper/waits/ditto-reclaim/`. `forced-wait.patch` shows the deliberate
regression, `source/` holds source snapshots, and `results.json` records source
hashes and checks. Each variant exports a `.drperf.json` profile:

| Variant | Expected observation over three invocations |
| --- | --- |
| `fixed` | Current reclaim path aborts pending compaction without waiting. |
| `forced` | Reclaim waits for completion; three ordered declarations cover three obligations. |
| `omitted` | Same blocking code without `waited`; three uncovered obligations. |
| `fixed-probe` | Delaying publication by 400 ms still lets reclaim return before completion. |
| `forced-probe` | Delayed completion preserves all three declared dependencies and coverage. |
| `false-claim-probe` | Claiming a dependency in the nonblocking code violates publication order. |

The declared edge is `vllm.reclaim_pages -> ditto.compression.complete` (waiter
to publisher). Names come from annotations; native capture flags missing
declarations without inventing their semantic destination. A forced wait is a
valid dependency and an undesirable design choice, not an annotation error.
This rule applies to the Python graph builder, saved graphs, and the viewer's
expanded hierarchy. Native CUDA stream history, event-record matches, and
semaphore matches remain API evidence; none creates a region wait edge without
a declared event pair. Older automatically inferred edges are discarded when
opening the JSON in the current viewer. A native match does not pay a missing
annotation's coverage obligation.
Both implementations also check that aborted compression output is discarded
and the source lease is eventually released. No GPU latency improvement is
measured by this experiment.

## Async task waits

With `DRPERF_WAITS=1`, the Python binding observes `asyncio.Lock.acquire`,
`Semaphore.acquire`, `Event.wait`, `Condition.wait`, and `Queue.get/put`.
Rebuild with `./build.sh`. `DRPERF_ASYNCIO_WAITS=0` disables these adapters.
They observe existing operations; they do not choose publishers or insert
application waits. Immediate successful calls are conservative candidates too.
Nested implementation calls, such as Condition.wait's lock reacquisition,
remain part of the enclosing logical operation. Native calls inside a captured
primitive are retained as `runtimeContainer`/`nativeOperations` evidence, but
do not create duplicate obligations. This requires the same logical task scope
and enclosing begin/end interval; overlap on the same OS thread is insufficient. Cancellation records an
exceptional completion, not successful readiness. Custom Future implementations
and arbitrary direct `await future` are not covered by this adapter set.

Use a suspension-aware region around coroutine code:

```python
async def load():
    await first_ready.wait()
    await second_ready.wait()
    event_waited(first_event, generation)
    event_waited(second_event, generation)

await perfmark.async_region("load", need_first=1, need_second=1).run(load())
```

Supply both PCV indicators in the ordinary interface declarations. An
`async_region` can also decorate an async function. It counts each active step
in a physical region and closes it before yielding. The checker joins these
steps into one logical invocation for coverage and indicator checking; entry
PCVs must stay fixed. Instruction formulas still describe the counted active
steps, not suspended time or a summed coroutine cost. Do not leave an ordinary
thread-local `with region(...)` open across `await`.

A parent `async_region` may contain the waited checkpoints after the child
returns. The child's obligations persist across suspension and move only along
its logical invocation ancestry. A different task running on the same thread
cannot cover them. An otherwise unmarked supported await appears in a small
`asyncio.<primitive>.<method>` observation region, so omission is visible without
a fabricated semantic edge. The native marker bodies and async observation callbacks are excluded; the
original primitive executes outside that exclusion. Python dispatch and the
coroutine region driver remain counted, as other Python wrapper work does.

Run `python3 examples/waits/check_async_operations.py` for six actual DynamoRIO
captures: zero, one, and two checkpoints, placed locally or in the parent.
Reports are under `out/waits/async-operations/*.drperf.json`.
