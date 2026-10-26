# ds41-gather-movement

**Contribution: POTENTIAL.** Kernel: Gather selected KV.

Copies selected compressed/history KV into attention input layout. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

rows,width,selected keys,source/context rows,strides and valid row counts. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Requested bytes and guards are captured. Repeated global reads across queries can be examined with a broader shared-selection PCV grid. No fix here.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Captured baseline; cross-query reuse proposed.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/gather-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/gather-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_gather_bytes` | global_read_bytes | `363.778818 * arg_cache_rows + -1214.45102 * arg_keys + 3416.73327 * arg_queries + 2.2916251 * arg_width + 144.225967 * blocks + -2704.57866` | 1.396e-08 | true |
| baseline | `ds41_gather_bytes` | global_write_bytes | `159.902721 * arg_cache_rows + -475.881746 * arg_keys + 886.29294 * arg_queries + -6.73253826 * arg_width + 136.27121 * blocks + -698.852586` | 1.164e-09 | true |
| baseline | `ds41_gather_bytes` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_gather_bytes` | sum_launch_max_block_lane_sequence_span | `53.5936199 * arg_cache_rows + -180.907219 * arg_keys + 629.90932 * arg_queries + -1.22150504 * arg_width + 1.97195015 * blocks + -402.346166` | 3.071e-10 | true |
