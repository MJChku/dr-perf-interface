# Retained research evidence

Generated workspaces were removed on 2026-10-02. This directory retains the
measurements, annotations, controls and provenance needed to assess the paper's
current examples. It is an evidence archive, not a claim that every experiment
succeeded or used the current checker.

| Evidence | Entry point | Interpretation |
| --- | --- | --- |
| With/without DrPerf | [GPT-6.1 Sol pilot](agent-ablation/gpt-6.1-sol-pilot/README.md), [protocol](../../benchmarks/PIPELINE_DESIGN.md), [manifest](../../benchmarks/evaluator/ablation.json), [harness checks](agent-ablation/smoke.json) | Six actual agent sessions, 102 workload processes; both arms recover the expected main mechanism in all three development cases. Separate smoke record contains no agent trials. |
| Current LMCache architecture | [Report](waits/lmcache/architecture-report/report.txt), [JSON](waits/lmcache/architecture-report/profile.drperf.json), [graph](waits/lmcache/architecture-report/graph.html) | Saved capture rechecked with the current checker: 149 regions, four declared wait channels, 51 null refinements, and incomplete wait coverage. No application rerun. |
| LMCache source and annotations | [Overlay](waits/lmcache/overlay/), [inventory](waits/lmcache/annotation-inventory.json) | Source hashes are retained. Static line coverage applies to this overlay, not the whole upstream repository. |
| Historical serving captures | [CPU](waits/lmcache/architecture-cpu.drperf.json), [remote](waits/lmcache/architecture-remote.drperf.json), [pressure](waits/lmcache/architecture-pressure.drperf.json) | The paper's older coverage table used different obligation rules. Its all-covered counts must not be presented as current-checker results. |
| Prefetch admission | [Native measurements](waits/lmcache/prefetch-admission/native.json), [controls](waits/lmcache/prefetch-admission/controls.json), [baseline](waits/lmcache/prefetch-admission/baseline/profile.drperf.json) | Component-level policy comparison and deliberately false declarations; not an end-to-end serving speedup. |
| Blind investigation and async checks | [Investigation](waits/lmcache/blind-prefetch/investigation.md), [annotated](waits/lmcache/blind-prefetch/async-annotated/profile.drperf.json), [omitted](waits/lmcache/blind-prefetch/async-omitted/profile.drperf.json) | Keep the successes and remaining uncovered waits together. |
| Real LMCache I/O | [Native validation](waits/lmcache/real-io/native-final.json), [before](waits/lmcache/real-io/before.drperf.json), [after](waits/lmcache/real-io/after.drperf.json) | Payload checks, timing and instruction evidence. |
| Conditional PCVs and CPython dictionaries | [Branch example](../../examples/branch_pcvs/), [dictionary study](../../examples/hash_table_cost/README.md) | Source and compact evidence remain in `examples/`; selected original counts are in the raw archive. |
| Wan and GX experiments | [Wan findings](../../examples/wan_gx/RESULTS.md), [native validation](../../examples/wan_gx/NATIVE_RESULTS.md), [GX case records](../../examples/causal_video/README.md) | Existing patches, compact traces, kernel databases, correctness checks and negative results remain in `examples/`. Additional compact summaries are retained here under `wan-gx/`, `wan-a100/`, `wan-timing/` and `causal-video/`. |

Query the current saved profile from the repository root:

```sh
bin/drperf --report results/paper/waits/lmcache/architecture-report --stats
bin/drperf --report results/paper/waits/lmcache/architecture-report --region lmc.store
bin/drperf --report results/paper/waits/lmcache/architecture-report --top unexplained --topk 10
```

`manifest.json` records original locations, preservation reasons, original and
current hashes, and removed cache paths. Portable report source roots were
updated to the retained source trees. Historical measurement JSON and copied
experiment scripts may still name the original workspace; those paths are
provenance, not evidence that the deleted workspace remains available.

`raw-captures.tar.gz` preserves selected raw counts, traces and synchronization
records with original `out/...` member names. Extract into a separate workspace
when refitting historical results; do not expand it for ordinary report queries.
Adjacent `*.waits.*.jsonl.gz` files belong to their reports and are required for
rechecking wait evidence. Historical `evidence.tar.gz` and
`operations-evidence.tar.gz` retain the earlier published capture/source bundles.

Deleted material consists of downloaded environments, model/cache copies,
intermediate runs, unselected debug traces, screenshots, obsolete binaries and
rebuildable dependency caches. The active DynamoRIO package, its source,
benchmark cases, source edits, and paper-ready evidence already in `examples/`
were retained. The modified vLLM checkout was retained; its environment and build
products can be regenerated with `third_party/vllm-cpu/setup.sh`.
