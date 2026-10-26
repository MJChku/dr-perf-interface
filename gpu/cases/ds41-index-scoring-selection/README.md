# ds41-index-scoring-selection

**Contribution: POTENTIAL.** Kernel: Sparse index score + top-k.

Chooses compressed KV history entries at long contexts. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

rows,context keys,heads,dim,selected keys and argument scalars (see support descriptor). `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Actual kernels captured against independent score/selection oracles; look for context-scaling score-buffer traffic and selection passes. No after variant.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Captured bounded states; 100K scaling extrapolation not validated.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/index-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/index-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_index_reduce` | fp32_flops | `3216` | 0 | true |
| baseline | `ds41_index_reduce` | global_read_bytes | `3216` | 0 | true |
| baseline | `ds41_index_reduce` | global_write_bytes | `402` | 0 | true |
| baseline | `ds41_index_reduce` | blocks | `3` | 0 | true |
| baseline | `ds41_index_reduce` | sum_launch_max_block_lane_sequence_span | `207` | 0 | true |
| baseline | `ds41_index_select` | fp32_flops | `772.5939 * arg_candidate_stride + 10.436931 * arg_index_stride + 6.17960283 * arg_keys + -129.65837 * arg_mask_stride + 751.172535 * arg_mode + -1474.34505` | 3537 | false |
| baseline | `ds41_index_select` | global_read_bytes | `-20634.7512 * arg_candidate_stride + -91.9005085 * arg_index_stride + 100.508511 * arg_keys + 3478.9204 * arg_mask_stride + -1231.59684 * arg_mode + 38321.0965` | 3.545e+04 | false |
| baseline | `ds41_index_select` | global_write_bytes | `-190.895357 * arg_candidate_stride + 32.0067631 * arg_index_stride + 1.51381261 * arg_keys + -29.7732308 * arg_mask_stride + -471.218599 * arg_mode + 455.874118` | 2937 | false |
| baseline | `ds41_index_select` | shared_read_bytes | `60251.0586 * arg_candidate_stride + 58.2049926 * arg_index_stride + -236.778769 * arg_keys + 1942.14024 * arg_mask_stride + 22065.7449 * arg_mode + -41346.3489` | 1.623e+05 | false |
| baseline | `ds41_index_select` | shared_write_bytes | `16088.9517 * arg_candidate_stride + 33.2803966 * arg_index_stride + -63.3197136 * arg_keys + 202.833507 * arg_mask_stride + 10775.4408 * arg_mode + 29190.8216` | 7.977e+04 | false |
| baseline | `ds41_index_select` | blocks | `5` | 0 | true |
| baseline | `ds41_index_select` | sum_launch_max_block_lane_sequence_span | `1400.82507 * arg_candidate_stride + 0.999940943 * arg_index_stride + -5.24360167 * arg_keys + 20.6151997 * arg_mask_stride + -151.266784 * arg_mode + -1466.08031` | 3002 | false |
