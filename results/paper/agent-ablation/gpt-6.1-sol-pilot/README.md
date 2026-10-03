# GPT-6.1 Sol: with and without DrPerf

Six fresh sessions completed on 2026-10-02: one pair each for SQLGlot, LibCST,
and ComfyUI, using GPT-6.1 Sol with medium reasoning. All 102 workload processes
succeeded. These are actual agent sessions, separate from the earlier harness
smoke runs. No optimization phase was run.

Both arms recovered the expected main dependency in all three cases, according
to the [supervisor review](review.json). **This pilot does not show a discovery
accuracy advantage for DrPerf.** Recovery of the expected mechanism is narrower
than scoring every PCV claim or validating predictions on independent inputs.

| Case | Without DrPerf: rounds / launches / process time | With DrPerf: rounds / launches / process time | Review |
| --- | --- | --- | --- |
| sqlglot | 2 / 12 / 25 s | 3 / 18 / 57 s | Both recovered the main dependency |
| libcst | 3 / 18 / 6 s | 3 / 18 / 51 s | Both recovered the main dependency |
| comfyui | 3 / 18 / 43 s | 3 / 18 / 95 s | Both recovered the main dependency |

Process time includes startup and instrumentation, but excludes report fitting
and agent reasoning. It is a cost of running these experiments, not application
latency or a matched-input profiler-slowdown estimate: agents chose different
configurations. The SQLGlot DrPerf agent spent its first round on conventional
profiling; each choice consumed one of its three allowed rounds.

## What the tool added

- **SQLGlot:** both agents found repeated outer-scope collection during CTE
  merging after their first timing round. DrPerf supplied structural instruction
  attribution, but final collector block residual remained 59%. An offline
  total-state fit was much tighter; these are different tests.
- **LibCST:** both recovered linear plus triangular expression-prefix PCVs.
  DrPerf attributed nonlinear instruction work to native clone/allocation/free
  functions. The control inferred cloning from Rust source and scaling; its
  Python profiler could not attribute native internals. DrPerf used at most
  128 terms for addition/call chains, versus 256 in the control, but did not
  save measurement time. Both used 256 for unions. Agent-selected saved-data
  prediction splits are retrospective, not evaluator-reserved held-out tests.
- **ComfyUI:** both distinguished node count from total ancestor-closure and
  input/link volume. DrPerf still reports unexplained work: 15% in recursive
  hashing, 47% in the outer hashing wrapper, and 14% in assembly (rounded).
  The local helpers with zero residual exercise only a few correlated states.

DrPerf agents also exposed annotation overhead: entry preparation outside an
excluded scope can add work to a parent. Their reports acknowledge this.
Changing annotations and inputs between rounds prevents treating residual
changes as a controlled optimization result.

## Evidence and limits

[Manifest](manifest.json) records order, settings, source and tool hashes,
input envelopes, launch results, and archive hashes. [Common prompt](prompt.txt)
and [pre-final-answer rubric](grading-reference.json) are retained. Each session
folder contains its final claims, notes, round-by-round submissions, and a
compressed evidence archive with final source, raw captures and saved profiles:

- sqlglot: [without DrPerf](sqlglot-timing/RESULTS.md), [with DrPerf](sqlglot-drperf/RESULTS.md), [last DrPerf report](sqlglot-drperf/report.txt), [last profile JSON](sqlglot-drperf/profile.drperf.json).
- libcst: [without DrPerf](libcst-timing/RESULTS.md), [with DrPerf](libcst-drperf/RESULTS.md), [last DrPerf report](libcst-drperf/report.txt), [last profile JSON](libcst-drperf/profile.drperf.json).
- comfyui: [without DrPerf](comfyui-timing/RESULTS.md), [with DrPerf](comfyui-drperf/RESULTS.md), [last DrPerf report](comfyui-drperf/report.txt), [last profile JSON](comfyui-drperf/profile.drperf.json).

The last LibCST report is the union-shape round; earlier shape reports remain in
its archive. Archives omit bytecode caches, native binaries (hashes retained),
and reproducible HTML graphs. Reports retain original absolute paths as
provenance. Source can be restored from each archive's `workspace/` directory;
intermediate source versions were not snapshotted, although per-round source
hashes and raw evidence were retained.

Each arm had the same initial task/source/dependencies, a maximum of three
rounds with six configurations per round, and a 15-minute agent wall budget.
Agents ran sequentially in fresh contexts with no inherited conversation.
Arm order was randomized per case with a recorded seed. Filesystem separation
was instructed rather than OS-enforced; the review was not condition-blind.
Token telemetry was unavailable. One trial per case, coupled workload factors,
and no independently reserved inputs make this an exploratory development
pilot, not a statistically powered controlled evaluation. Broader case
selection, hard isolation, repeated trials and held-out grading remain needed.
