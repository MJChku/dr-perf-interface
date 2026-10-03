# Initial packaging validation

Checked on 2026-09-14 in the development workspace. These are harness smoke
checks, not agent trials or benchmark accuracy results.

| Pilot | Source revision | Check |
| --- | --- | --- |
| SQLGlot | `2e86ded7d0f6474e9041882058358616fb911464` | Six timing-profile processes at 4, 8, 12, 16, 24, 32 tables completed; expected output column counts were 2 times the table count. |
| Wan | `c5469b7ceb606edd7ba6570dcd17d38590a18db6` | CPU pipeline completed at 32x32, 5 frames, 2 steps, text length 16; output shape `(1, 5, 32, 32, 3)`. |
| vLLM | `2cf0a6915ce544dc493a0990f2ea38d81601128a` | CPU OPT-125M completed two batches of four requests, one generated token each. |

The same six SQLGlot sizes also completed under DynamoRIO after adding test
annotations in the temporary workspace. Every process returned zero, the
client reported no validity warnings, and the supervisor merged the records
and printed an affine cost interface. Those test annotations are not included
in the exported task. This check does not validate the semantic answer key or
held-out predictive accuracy. Wan/vLLM drperf runs have not been smoke-tested
through this new harness yet.

The SQLGlot export logs a missing installed version-metadata warning but loads
the pinned source and computes its result. vLLM reports CPU fallback/library
configuration warnings in this environment; those settings must be identical
between conditions in an actual evaluation. Dependency versions are captured
on export, but dependency lockfiles/container images are still needed for a
portable benchmark release.

Unit/integration checks cover paired scoring, unsupported-claim penalties,
missing/duplicate pair rejection, losing a previously correct answer, immutable
round submissions, failed-run budget consumption, timing-only tool restrictions,
invalid-trace feedback suppression, answer removal from the SQLGlot driver,
and hashes of the imported evidence. Run them with:

```sh
python3 -m unittest discover -s benchmarks/tests -v
```

The small-to-large addition brings this to 11 passing checks, including rejecting
oversized/implicit inputs before execution and preventing comparisons that mix
input-access tracks. A freshly exported small-to-large SQLGlot workspace also
completed its six default timing-profile inputs (4 through 32 tables). This
validates the small-input runner, not transfer to larger inputs.

## Paired harness refresh (2026-10-02)

The primary ablation is with versus without the whole DrPerf tool. The updated
supervisor exports current checked reports instead of calling the removed
`bin/drperf.cost_lines` entry point. Each DrPerf round produces text, JSON and
HTML, including composition and declared wait checking. Merged raw files use
hard links, avoiding a second copy of capture data. Paired preparation rejects
payload differences beyond supervisor configuration. Inherited delay/profiling
settings are cleared before a measurement.

SQLGlot, libcst and ComfyUI each completed six timing-profile and six DrPerf
processes (36 total), with matching workload result lines. Both arms start with
byte-identical exports; smoke-only raw-input markers were added afterwards.
These are infrastructure checks, **not agent ablation results**. Native libcst
libraries are hashed, its Rust source is exported, and the source checkout is
checked against the pinned revision. ComfyUI uses real cache-key/graph code and
CPU Torch, with adapters for unused model-management imports.

Compact evidence: [smoke.json](../results/paper/agent-ablation/smoke.json).
Reproduction automatically removes temporary workspaces and captures:

```sh
python3 benchmarks/tools/smoke_pairs.py --archive /home/ubuntu/drperf-cases \
  --output results/paper/agent-ablation/smoke.json
```

Fourteen benchmark tests pass, including neutral-driver answer removal,
paired-payload equality, rejection of an extra hint in one arm, restricted tool
access, budgets, failed captures and semantic scoring. Historical evidence
hashes now preserve both original archive hashes and local hashes for the
previous developer-command relocation; only command paths changed.

## Actual GPT-6.1 Sol pilot (2026-10-02)

Separate from the harness smoke checks, six fresh agent sessions completed 102 successful workload processes across SQLGlot, LibCST and ComfyUI. Settings, per-round answers, source, raw captures, final reports and limitations are retained in the [pilot record](../results/paper/agent-ablation/gpt-6.1-sol-pilot/README.md). Both arms recovered the expected main mechanism in each case under supervisor review; this is not a blinded or held-out accuracy result.
