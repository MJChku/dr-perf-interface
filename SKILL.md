---
name: drperf
description: Use DrPerf to explain and review marked code's CPU instruction cost and declared wait dependencies, inspect a system's region graph, and guide performance changes using observed-run feedback.
---

# DrPerf: interactive performance-interface discovery

DrPerf checks a proposed cost model against actual execution. Use it when
implementing or reviewing performance-sensitive code, investigating unexpected
library costs, or understanding the structure and dependencies of a system
such as LMCache. It counts CPU instructions with DynamoRIO, including Python
interpreter and native library execution inside marked regions. It does not
measure GPU computation or predict end-to-end latency.

The agent supplies the semantic vocabulary: region boundaries,
performance-critical variables (PCVs), and claimed wait dependencies. DrPerf
supplies affine coefficients, checks observed behavior, and reports what the
annotations do not explain. Discovery is the agent's iterative task; DrPerf is
the checker.

## Run one command

From the application's source tree, use the command that normally runs its
small test or workload, prefixed with `drperf`:

```bash
drperf python3 app.py
```

A native application uses `drperf ./myapp`. The installation's `bin` directory
must be on PATH. All fitting, wait checking, and report generation happen in
this command. Do not ask the user to run separate export, check, or graph tools.
Wait capture is on by default; delay injection is not. Existing input/output
side effects of the application still occur.

Each run writes `drperf-report/` in the working directory:

- **`report.txt`**: the complete text view. It contains checks, coverage, the text architecture
  graph, region formulas, function breakdowns, unexplained work, and observed
  PCV relationships. Search for `REGION "name"`, `Wait[?]`, and `counterexample`.
- **`graph.html`**: a self-contained interactive human view. Open it in a browser;
  it uses the same region viewer and standard ELK layout as the VS Code extension.
  Expand regions, fit the graph, select a node, and inspect its performance details.
- **`profile.drperf.json`**: structured region states, fits, function attribution,
  composition, `eventModel` checks, and `executionGraph`. VS Code can open it too.
- **Companion wait-evidence files**: preserve these with the JSON for full
  evidence. They are not the entry point for reading the report.

Useful JSON entry points: `regions[].regimes` holds fits and function
attribution; `regions[].points` holds observed states and costs; `composition`
holds symbolic child terms; `eventModel.interfaceChecks` holds indicators and
counterexamples; `eventModel.coverage` holds unpaid wait obligations;
`executionGraph` holds the graph; `codeCoverage` holds annotation reachability;
`relations` holds observed cross-region PCV equalities.

A new run replaces these reports. Preserve an important baseline before
rerunning; `DRPERF_REPORT_DIR` can select another directory. Set
`DRPERF_SOURCE_ROOT` only when the working directory is not the source tree.

## Query a saved report

Use the same executable to inspect results without running the application again:

```bash
drperf --report --stats
drperf --report --top unexplained --topk 10
drperf --report --top costly --topk 10
drperf --report --region 'region.name'
drperf --report --region 'region.name' --json
drperf --report --full
```

The default is `drperf-report/profile.drperf.json`. Supply a JSON path or report
directory immediately after `--report` to inspect another capture. `--json`
works for every query. Queries are read-only and do not require the raw wait
sidecars. Start with stats and rankings, then inspect a specific region's
formula, function breakdown, observed-state table, sources and waits.

`costly` ranks invocation-weighted **own** instructions, excluding children.
`unexplained` ranks own residual plus the full estimated cost of unresolved
`F[child]` terms. Residual inside a fitted child term belongs to that child's
interface. Child estimates use observed nesting and state means; overlapping
parent/child totals must not be added. Unavailable estimates are reported, not
treated as zero. Instruction work is not elapsed time.

Stats disclose single-state regions. A zero residual with one observed state
means a constant fits that state's mean; it does not establish growth or
explain individual-call variation. Keep fitting these observations, but do not
mistake a zero residual for evidence across untested states. Function breakdowns
show the retained leading functions; older captures retain at most eight per
coefficient/constant/residual group. Coefficient contributions and equal-state
residual means have different units from invocation-weighted ranking totals.

## Describe CPU work with PCVs

```python
from perfmark import region

with region("lookup", entries=len(table), requests=len(keys)):
    results = [lookup(table, key) for key in keys]
```

PCVs are integer-valued entry-state expressions, not fitted coefficients. Start with sizes,
counts, modes, and other semantically meaningful quantities. DrPerf fits each
observed basic block using all declared PCVs plus a constant. Products or
branch-sensitive quantities belong in the PCV expression, e.g. `n*m`; the
coefficient template remains affine. Avoid reproducing the measured computation
as a PCV merely to make the fit pass.

A region interface has explained affine work, symbolic child terms `F[child]`,
and unexplained work. Parent interfaces retain child references rather than
expanding them. Read the named function breakdown to locate surprises. Inspect
per-state mean costs and invocation variation as well as the unexplained percentage;
a fit from one state or correlated PCVs is evidence only for those observations.
Do not interpret an unused coefficient as proof that a variable never matters.

The first task is to explain: run small cases, inspect residuals, refine PCVs or
boundaries, and rerun. Then investigate unexpected coefficients or growth,
change the implementation, test correctness, and compare equivalent cases.
Symbolic insights can suggest large-input problems without running those cases;
extrapolation remains a hypothesis until validated.

## Read the architecture graph

The text graph lists every observed region once and uses adjacency lists, so
shared helpers and cycles do not expand into an unreadable execution tree.

- `contains -> B`: B is a directly nested region; inspect B's interface separately.
- `next -> B`: observed sibling execution order, with its enclosing context.
  Sequence alone is not a semantic waited dependency.
- `waited -> B`: a declared consumer-to-publisher claim, with event ID, observed
  count, order status, and the supplied indicator's check status.
- `unexplained(Wait[?])`: observed synchronization lacks enough covering annotations.
- `waited(null)`: an explicit event-free refinement with a reason for manual review.

For a complex system, follow its load/store/reload or request paths through
containment, then inspect cross-region waits and their indicators. Ask what
work must finish before the path can proceed and whether that dependence is
intended. Do not infer a bottleneck, critical path, total cost, or latency from
these aggregated arrows: parallelism, unmarked work, and external execution
are not modeled by summing region formulas.

## Declare and check waits

Keep the application's synchronization. Put `release(event, generation)`
immediately before its real release and `wait(event, generation, indicator=...)`
after its existing readiness check. These passive markers do not synchronize.
The indicator is an assertion over entry PCVs, not a condition that skips the
marker. Use distinct event channels for distinct region pairs; generations
identify repeated publications. A parent waiting on its synchronous child is
rejected.

Declare the indicator **inside the marker**, with no separate JSON:

```python
from perfmark import region, release, wait

with region("cache.fetch"):
    fetch_result()
    release(42, generation)
    ready.set()

with region("request.load", need_fetch=int(need_fetch)):
    if need_fetch:
        ready.wait()
        wait(42, generation, indicator="need_fetch == 1", producer="cache.fetch")
```

The producer and consumer normally execute separately. `producer` is optional;
when omitted, the checked release supplies its name. Use literal indicator
strings and event IDs (or module-level constants) in Python region scopes so
DrPerf can also check never-executed sites from source. Unresolvable sites are
reported; a runtime-only declaration must execute once before missing occurrences
can be checked. Native declarations currently have this runtime-only limitation.
`wait(None, indicator="condition", reason="manual explanation")` refines a wait
without claiming a publisher. Old markers and JSON declarations remain readable.

`need_fetch` must be an actual declared entry PCV of `request.load`. The interface
is `I[need_fetch == 1] * Wait[cache.fetch]`. DrPerf checks both unexpected presence
and unexpected absence of the checkpoint. It also checks observed publication
order and rejects event sharing across region pairs. It does not infer the
application's intended publisher or prove necessity from one ordered trace.

Multiple waits in one invocation retain multiple obligations. Each valid
checkpoint covers at most one eligible completed obligation in that invocation
or its descendants. Credits are not reused or banked. Consecutive failed native
attempts at one API/object/site can form a retry chain; distinct successes stay
separate. Moving a waited annotation to a semantic caller must preserve coverage.
Uncovered waits remain report terms, not fabricated dependencies.

For coroutine code use `await perfmark.async_region("name", ...).run(coro)` or
its decorator form. Ordinary thread-local regions must not remain open across
suspension. Supported asyncio primitives contribute logical await observations;
other tasks on the same thread cannot spend those obligations' credits. Custom
atomics, arbitrary futures, and unsupported APIs may require further capture
support. A clean report is not proof that all synchronization was observable.

Read [docs/waited.md](docs/waited.md) for marker examples, null refinements,
explicit delay probes, and capture boundaries. A null refinement is not a way
to dismiss a real prerequisite: its reason remains manually reviewed.

## Check code coverage

`report.txt` includes `CODE COVERAGE`; the JSON field is `codeCoverage`.
**Coverage means source lines lexically inside marked regions / total source
lines**, in `codeCoverage.lines`. It excludes blank/comment-only lines, includes
marker/decorator lines and string literals, and counts nested spans once.
Unexecuted marked regions count as covered. It does not follow callees outside
the marked lexical span. The selected source root defines the denominator;
file exclusions, unsupported wrappers, and scan limits are reported. Per-file
counts and unmarked line ranges help identify where to add annotations.

Separately, the report compares literal region names with executed
regions and lists unobserved annotations with file/line locations. Exercise
those paths before treating their interfaces as checked. Names shared between
multiple annotation sites are explicitly ambiguous, and dynamic names may lack
source locations. An incomplete trace produces unknown coverage, not a pass.

Source coverage describes annotation scope; region reachability describes the
test run. Neither is runtime line/branch coverage. Read the scope before
claiming whole-system coverage.

## Assess the result

Keep three questions separate: did the run exercise the code, do PCVs explain
its instruction cost, and do the declarations explain observed synchronization?
Missing code coverage is not zero cost. Zero instruction residual does not mean
all waits are explained. Incomplete capture is not a successful check.

DrPerf reports missing explanations as feedback. Failed wait claims return a
nonzero exit status; inspect the report before proposing a fix. When reporting
an improvement, cite the region, source location, observed input states,
before/after interface, correctness checks, and remaining unexplained behavior.
Distinguish a cost-model finding from a measured end-to-end speedup.
