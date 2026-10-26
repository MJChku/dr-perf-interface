# ds41-hc-post-merge

**Contribution: QUANTIFIED.** Kernel: Fused HC coefficients + post/merge.

Folds HC post processing and expert-result merges to remove decode launches. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,width,elements,stream count,iterations and matrix scalars. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

After-only resource fit for passing fusion. Original launch histogram, not DrPerf, identified it; no paired resource baseline is manufactured.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Validated but no new isolated serving claim.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/post_merge-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/post_merge-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| fused | `ds41_hc_post_merge` | fp32_flops | `29013.3333 * arg_kind + 1216 * blocks + -14506.6667` | 6.741e+04 | false |
| fused | `ds41_hc_post_merge` | global_read_bytes | `116053.333 * arg_kind + 4352 * blocks + -58026.6667` | 2.697e+05 | false |
| fused | `ds41_hc_post_merge` | global_write_bytes | `256 * blocks` | 0 | true |
| fused | `ds41_hc_post_merge` | blocks | `1 * blocks` | 0 | true |
| fused | `ds41_hc_post_merge` | sum_launch_max_block_lane_sequence_span | `14 * arg_kind + 123` | 0 | true |
