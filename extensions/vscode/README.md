# drperf Region Explorer

Inspect what each region's cost depends on, then explore how a change to one or
more performance-critical variables (PCVs) could affect other regions. The
explorer does **not** add region costs into a total or predict latency,
parallelism, GPU overlap, or scheduling.

## Install and open

Install the packaged `.vsix` with VS Code's **Extensions: Install from VSIX**
command, or:

```sh
code --install-extension drperf-explorer-0.1.0.vsix
```

Run **drperf: Open Producer/Consumer Demo** from the Command Palette for a
self-contained example measured by drperf. No profiler or GPU is needed to view
it. The bundled example has concurrent producer/consumer threads, ordinary
affine regions, a branch with two fitted regimes, and a lookup whose cost is
partly unexplained.

To inspect your program, build drperf and save a report during the normal run:

```sh
DRPERF_REPORT=out/my-program.drperf.json \
DRPERF_SOURCE_ROOT=. bin/drperf python app.py
```

Or export measurements you already have:

```sh
bin/drperf-export out/raw --source-root /path/to/project \
  -o out/my-program.drperf.json
```

Then run **drperf: Open Performance Report**. The **drperf Regions** tree appears
in the Explorer. Clicking a region's CodeLens, choosing **Inspect Region at
Cursor**, or moving the cursor into a region while the panel is open selects its
interface. Hovering over the annotation shows a compact formula. Source hashes
flag code that has changed since export.

## Four views

- **Interface:** affine formulas, PCV ranges, observed states, unexplained work,
  per-function attribution, source links, and suggestions for the next small
  experiment. Each formula describes the region's own instructions per call;
  nested regions remain separate.
- **Relationships:** the full list of discovered state equalities and a compact
  dependency diagram. `last`, `cum`, `cumend`, and `count` retain their distinct
  meanings. Search is by region, PCV, or operation. The diagram is a dependency
  view of equations, not a causal or timing graph.
- **What-if:** edit one or more PCVs using multiply/add/set, explicitly choose
  observed equations as assumptions, and inspect the affected regions. Results
  show mean explained instructions per call over the same recorded calls,
  changed PCVs, unknown cases, and extrapolation. Save the scenario and results
  as JSON for review or agent use. **Open Saved Scenario** checks and restores
  it against its original report.
- **Compare runs:** compare observed costs at exactly shared PCV states across
  two reports, with equal weight per shared state within each region. It can
  reveal coefficient changes without confusing
  a different workload mix with a change in the region's cost.

## What a scenario means

A scenario replays the recorded marker sequence. At each region entry, selected
relations compute new state values from preceding calls; direct interventions
are then applied. `cumend` includes only calls that ended before that entry.
Histories reset at each process/run boundary. Call counts, nesting, and marker
order stay fixed. The replay makes no assumption about elapsed time or
concurrency beyond that recorded sequence.

A relationship that held in the measurements is **observed evidence**, not proof
of causality. Choosing it for a scenario is an explicit assumption. Alternative
equations can coincide in the original trace and disagree after an intervention;
choose one per target PCV. Direct interventions override the selected relation
for that target.

The checker only accepted formulas at observed states within its tolerance.
Unobserved points and extrapolations are labelled. A gap between fitted regimes
has no assumed branch boundary. New combinations of correlated PCVs are not
identifiable. Unexplained cost at changed states remains unknown, and partial
predictions show how many calls were modelled. Large int64 PCVs are preserved as
strings in reports rather than rounded; the current JavaScript replay rejects
states outside its exact integer range.

Relationship discovery is bounded and not exhaustive. Reports say when a search
or trace limit prevented exploration. A missing or incomplete trace, counter
validity errors, or invalid trace sequences prevent scenario replay. Viewing
recorded interfaces is still possible.

## Configuration

- `drperf.sourceRoot`: root for the relative source paths in a report. Defaults
  to its workspace folder. Python context managers/decorators and literal C/C++
  and Rust marker calls are supported. Dynamic names need manual navigation.
- `drperf.codeLens`, `drperf.followCursor`: toggle source decorations and cursor
  tracking.
- `drperf.pythonPath`, `drperf.exporterPath`: executables used only by the explicit
  **Export Raw Measurements** command. Exporting executes Python and requires a
  trusted workspace. Opening a report does not execute its measured program.

Reports contain local names, source paths, PCV values, and measurement metadata.
The extension performs no telemetry and sends no report data to a service. It
works with local or remote VS Code extension hosts; the viewer itself does not
need DynamoRIO or Python.

Scenario checks, comparisons, proposals, and experiment searches run in a local
web worker so the panel remains responsive. Only extension assets are loaded;
reports stay local. New edits supersede queued scenario work, and stale results
cannot replace a newer edit or report.

## Build and test

Developer tooling requires Node.js 22 or later. The extension has no runtime npm
dependencies.

```sh
cd extensions/vscode
npm ci
npm test
npx playwright install chromium
npm run test:ui
# Linux extension-host test (requires Xvfb):
xvfb-run -a node test/run-host.cjs
npm run package
```

The tests include actual VS Code activation, CodeLens, hover, cursor mapping,
and stale-source detection; browser checks cover the interactive panel. The
backend tests live in `tests/test_explorer.py`. Regenerate the measured example
with `examples/explorer/run.sh` from the repository root.

Implementation uses VS Code's [Webview API](https://code.visualstudio.com/api/extension-guides/webview)
and [programmatic language features](https://code.visualstudio.com/api/language-extensions/programmatic-language-features).

## Propose, check, and validate relationships

The What-if panel accepts proposed **state relationships**, including products,
integer powers, and conditional expressions. For example, target `lookup.pairs`
with `last("dequeue", "items") ** 2`. The checker reports counterexamples or
confirms agreement at every recorded target call. A separate button adopts a
checked proposal as a scenario assumption. Cost interfaces remain affine;
PCVs themselves are still expressions in the measured program's language.
The small expression editor is only a notation for cross-region relationships.

History references are `last("region", "pcv")`, `cum(...)`, `cumend(...)`, and
`count("region")`; absent prior history is zero. Arithmetic uses exact integers
with `+`, `-`, `*`, `//` (floor division), `%`, and `**` (exponents 0..16).
Comparisons and `condition ? then : else` express piecewise relationships.
Expressions are parsed, never evaluated as JavaScript. Their size and arithmetic
are bounded. Unsupported expressions require a different annotation or an
external checker; the editor is not a general program interpreter.

**Compare against a new measured report** checks a changed execution. It first
requires matching call counts, marker order, thread roles, and nesting. Groups
are paired by their order in the reports, with threads matched by first
appearance; unrelated process sets should not be compared this way. State
matches and cost errors are reported separately. Cost checks use observed
per-state means and omit predictions with unexplained blocks or unsupported
states. Different instrumentation settings are rejected. This is evidence for
a specific intervention, not a universal proof.

The [measured example](../../examples/explorer/README.md) demonstrates both a
failed extrapolation and a successful PCV refinement. **Open Refined PCV Demo**
opens its improved interface. For agents and terminal workflows:

```sh
bin/drperf-explore report.drperf.json --list
bin/drperf-explore report.drperf.json --edit producer items scale 2 \
  --relation RELATION_ID --validate changed.drperf.json -o scenario.json
```

Scenario JSON includes each changed region, coverage and support labels,
representative calls, and the actual history values used by their equations.
It contains no cross-region total or latency estimate.

## Recorded counts and marker estimates

The default view preserves drperf's current calibrated formulas. **View recorded
counts** reverses the marker-overhead subtraction, retaining the marker API's
instructions in each region's own cost. The selected basis also applies to
scenario validation and is saved with the scenario. The terminal equivalent is
`--recorded`. CUDA/native exclusions and OpenMP-wait exclusions are unchanged.

Marker calibration is an estimate, not an exact removal of instrumentation
cost. The existing average nested-marker correction can over-subtract on
exclusive-cost traces; some saved vLLM measurements expose negative calibrated
costs. Such predictions are rejected and the report shows the issue. Recorded
counts remain available for inspection without that correction. The explorer
does not change the underlying checker or silently clamp negative formulas.

If one region name is used with different sets of PCV names, the exporter creates
separate interface variants with stable IDs. Reordering the same named PCVs does
not create a variant. Each variant keeps the original source links; its schema
appears in the display name. This prevents values for one PCV from being fitted
under another PCV's name.

## Find an experiment that distinguishes explanations

Enable **Compare alternative relationships** in a scenario. The checker first
verifies each candidate on the original trace, then compares its prediction to
the selected equation using the scenario's history at each target call. A
disagreement identifies a possible discriminating experiment. It is a one-step
comparison in that history, not a full alternative replay or a measured failure.

For example, the demo supports both `copy.bytes = 8*last(decode.tokens)` and
`copy.bytes = 16*last(dequeue.items)`. Changing decode's expansion factor makes
them disagree. If that change is implemented in the program, a small measured
run can tell which equation survives. This helps distinguish accidental
correlations from relationships useful under an intended change. The CLI flag
is `--audit-alternatives`.

**Find distinguishing experiments** searches from the recorded baseline using
your selected relationship assumptions. It tries `add 1` and `multiply by 2`
on source PCVs of competing equations, prioritizes sources that participate in
more ambiguous targets, and checks up to 12 probes. Suggestions rank by how many
target PCVs they separate, not by aggregate cost. The panel records rejected
probes, including changes that produce non-integer states under an assumption.

These are candidate PCV interventions, not generated program inputs. You still
need to implement a realizable input/configuration change, check that the fixed
call structure is appropriate, and measure it. A missing suggestion is not
proof that the equations are equivalent. The CLI can raise the search budget
to 64 probes:

```sh
bin/drperf-explore report.drperf.json --assume-first \
  --suggest-experiments --max-probes 12 -o experiments.json
```

Every suggested scenario retains its selected equations and checked proposals,
so it can be replayed independently. The measured demo's decode expansion
experiment is found automatically by this search.

## Compare interfaces across runs or versions

**Compare runs** pairs interfaces by original region name and PCV names. It
compares measured per-state means only at exactly shared states, regardless of
call order. States without a counterpart remain unpaired; negative calibrated
costs are excluded. The panel also shows both formulas and identifiable
coefficient differences, and flags changes to captured PCV source expressions.
PCV meanings still need to be consistent across the reports.

```sh
bin/drperf-explore before.drperf.json --compare after.drperf.json -o comparison.json
```

Unlike scenario validation, this comparison does not need a complete trace or
unchanged call counts. It needs valid instruction measurements with matching
measurement settings. Each region remains separate, and the observed
differences are not a statistical significance test. The portable report
[schema](schemas/report.schema.json) also enables validation and completion
when editing `*.drperf.json` in VS Code.
