# vLLM: a missing admission-cache state, checked after one agent round

A fresh GPT-6.1 Sol agent (medium reasoning, no inherited conversation) inspected
pinned vLLM CPU source, froze its predictions, then received one DrPerf measurement
round with three configurations. All three workload executions succeeded. The
workload runs real OPT-125m CPU inference; the six marked regions cover admission.
The supervisor selected this case from historical evidence, but did not reveal
the historical issue or an expected answer to the agent. Isolation was by
instruction, not an OS filesystem barrier. This is a targeted single-session case
study, not a paired comparison or an estimate of agent failure rates.

## What actually failed

The initial `admit-total` interface proposed prompt tokens, output fanout, maximum
output tokens, and resident requests; it predicted fixed exclusive overhead plus
symbolic child work. It did not declare plugin-loader cache state. It did mention
generic cold/cache uncertainty, so this is an incomplete proposed interface, not
evidence that the agent confidently denied all cache effects.

The first feedback report left 97% of `admission.request` exclusive instruction
work unexplained. At identical declared state (12 prompt tokens, one output,
max_tokens=1, zero resident requests), the first configuration observed 48,446,537
versus 496,115 exclusive instructions. The other two configurations reproduced
the gap: 48,701,744 versus 492,349, and 48,722,202 versus 493,754 (the last uses
max_tokens=4). Same-state variation, rather than residual percentage alone,
establishes that these PCVs do not distinguish the observed executions.

The agent independently traced a candidate to `cached_load_custom_logitsprocs`:
`SamplingParams.verify` validates logits processors through an LRU cache. On a
miss, it queries `importlib.metadata.entry_points` for installed plugins. Model
initialization uses the uncached loader and does not populate this validation
cache. The revised submission proposes
`int(cached_load_custom_logitsprocs.cache_info().currsize == 0)` for the fixed
`None` cache key in this workload, explicitly pending validation.

## Independent verification after the frozen answer

These checks were performed by the supervisor, not fed back for a second agent
round. One agent round was sufficient to propose the correction; establishing
this case also required the following independent validation executions.

1. A direct runtime probe of the real validator counted plugin-discovery API calls:
   first None key: 1 scan; repeat: 0; first empty-tuple key: 1; repeat: 0; None
   again: 0; after cache clear: 1. This confirms cache-key-specific behavior and
   why global cache emptiness is sufficient only in the fixed-key workload.
2. An end-to-end four-request/two-pass capture added the proposed PCV, preserving
   the program logic. The 12-token admission with `plugin_cache_empty=1` cost
   48,497,755 exclusive instructions; the repeat with value 0 cost 497,456.
3. A second end-to-end capture warmed only that cache after model initialization
   and before admissions. The two matching admissions both had PCV value 0 and
   cost between 496,409 and 562,747 instructions. The large spike disappeared.
   This intervention shifts initialization work earlier; it is a causal check,
   not a claimed end-to-end optimization or latency improvement.

The cold-PCV verification fit assigns about 47 million instructions to the cold
state and reports 2% own residual. Do NOT report this as a controlled 97%-to-2%
improvement: verification used four prompt lengths, whereas discovery used 16
across three configurations. Refitting the original four-length capture alone
already gives 3% residual with extreme cancelling coefficients, despite the
same-state cold/warm difference. This is a concrete limit of fitting state means
on a small, correlated sample. The repeated-state evidence and cache intervention
are the grounds for the verified conclusion.

## Suitable paper text

In a fresh study of vLLM request admission, an agent's initial interface described
cost using prompt length, output count, generation limit, and resident requests,
but omitted plugin-loader cache state. Its first DrPerf round left 97% of
exclusive admission work unexplained: two calls with the same declared state
executed about 49 million and 0.5 million instructions. The agent traced this gap
to first-use logits-processor discovery and proposed a cache-state PCV.
Independent execution checks confirmed the dependency: recording that state
separated the two calls, and warming the cache before admission eliminated the
large spike. This case shows how execution feedback turns an incomplete cost
model into a concrete, checkable state dependency.

This supports omitted-state correction, not failure to infer quadratic growth,
universal correctness, or an advantage over conventional profiling. The initial
agent already predicted the originally suspected complete-block hashing behavior;
that negative result is retained in its submissions and RESULTS.md.

## Evidence

- `round-00.json`: frozen source-only claims; `round-01.json`: frozen revised claims.
- `RESULTS.md`: the agent's own final account, including anticipated uncertainties.
- `profile.drperf.json`, `report.txt`, `graph.html`: the complete discovery report.
- `matched-state-evidence.json` and `evidence.csv`: raw per-state count summaries.
- `verification-cold.drperf.json` and `verification-cold-report.txt`: cold-state capture.
- `verification-cold-counts.json`, `verification-prewarm-counts.json`: intervention counts.
- `verify_plugin_cache.py`, `plugin-verification-output.txt`: direct causal API probe.
- `verify_hash_blocks.py`, `verification-output.txt`: 24 independent boundary checks
  confirming the block dependency the agent already predicted.
- `matched-baseline-report.txt`: same four-length baseline refit demonstrating why
  low average-state residual alone does not resolve this omitted state.
- `evidence.tar.gz`: source (native binaries omitted, hashes retained), annotations,
  workload, raw counters/sidecars and source hashes for the discovery round.
- `verification-*-raw.tar.gz`: validation raw captures and marker/workload changes.
- `protocol.json`, `initial-payload-sha256.json`, `checksums.json`: provenance.

The prewarm workload exited successfully and its raw capture has no validity
warnings. Its automatic report export failed on a source symlink resolving
outside the workspace; the original result records that failure. Conclusions
about the intervention use the retained raw counters, not a fabricated report.
The discovery report records 5,654 unresolved synchronization observations; this
CPU-cost study declares no wait dependencies. Its static coverage scanner misses
the imported marker alias and reports zero lexical coverage despite six runtime
regions. Neither limitation is represented as a successful wait or coverage check.

To reproduce, restore `evidence.tar.gz`, obtain the pinned vLLM revision from
`protocol.json` and the native binaries matching `session.json`, then update the
absolute workspace/environment paths in that session file. The validation archives
contain the cold-PCV engine file and both drivers. Run each driver under DrPerf
with the corresponding session environment. Run the two direct probes using the
same Python environment with `perfmark/python` on PYTHONPATH and PERFMARK_LIB set.
