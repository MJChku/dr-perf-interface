# GLM kernel scaling: measured resources, not predicted latency

Two original kernel bodies from gen-glm53 are retained with source hashes. A separate compact-output softmax changes only the output representation and is a demonstration candidate, not installed in serving. Every output/guard passed independent numerical checks.

Train the resource interfaces on histories32–512, varying selected counts16–64. Hold out both2048 and100000 histories, selected count2048. The small-state formulas predict every reported held-out byte/SFU count exactly.

| Resource per one reduced-width row | 2,048 history | 100,000 history | Growth |
|---|---:|---:|---:|
| Original materialization writes |24,576bytes|1,200,000bytes|48.828x|
| Original sparse-softmax FP32 work |31,230FLOPs|31,230FLOPs|1x|
| Original sparse-softmax special ops |2,048|2,048|1x|
| Original sparse-softmax writes |8,192bytes|204,096bytes|24.914x|
| Compact-output demonstration writes |4,096bytes|4,096bytes|1x|

## Discovery and refinement

1. Propose that one-query materialization depends only on query count. DrPerf rejects its byte fits because history length changes behind the same PCV state. Add history: reads=12T and writes=12T for H=N=R=V=1. This pure-copy work grows48.828x from2K to100K.
2. Propose that sparse softmax depends only on selected count K. Its arithmetic and input reads fit, but writes do not: the implementation clears a dense T-sized row. Add history: writes=2T+2K, reads=26K, special_ops=K for these fixtures. Merely counting FLOPs would miss this scaling problem.
3. Change the demonstration's output to K probability slots aligned with the selected IDs, explicitly writing zero to invalid slots. DrPerf now accepts writes=2K without T. All selected probabilities match the original dense row and the independent oracle. A dense PV consumer cannot use this representation unchanged; a compact/sparse consumer and whole-sequence timing are required before claiming an optimization.

The lesson is to challenge an explained dependency: adding T explains the dense clear, while changing the producer-consumer interface can remove it. A good fit alone does not certify efficiency.

## Scope

These are actual standalone PTX executions at2K and100K history with reduced widths, not extrapolated observations or physical GPU timings. The illustrative full-width generalization of materialization is4*H*(N+R+V)*T requested read bytes and the same write bytes; that general expression is source analysis, not a measured full-width run. No full-model numerical run was performed.

These components were concrete bottlenecks in the GLM study. A historical native softmax comparison regressed at2K and improved3.58–3.66x at16K, including dense zeroing; see gen-glm53/docs/moe-helper-profile.md and its archived receipts. That historical comparison uses a different fixture/measurement and is not a native validation of this compact-output prototype or a100K timing claim. Full request benchmarks also change concurrency/output count, so their ratio is not assigned to this kernel alone.

The tcgen05 extension and this example are complementary: Blackwell instruction accounting is tested with separate BF16/NVFP4 matrix fixtures; these two scaling kernels do not execute tcgen05. No claim is made that tcgen05 itself causes the history-scaling issue. Current standalone support is CTA-group1; cluster/TMA-dependent library kernels still require additional support.

See summary.json, training-profile.json.gz and profile.gpu.drperf.json.gz for formulas, residuals, held-out checks, scope and raw provenance. All original emitted captures remain in the run directory.
