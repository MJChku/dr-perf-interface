# ds41-decode-selected-prepare

**Contribution: QUANTIFIED.** Kernel: Selected-only batched decode preparation.

Uses per-session metadata to prepare only selected KV rows and pad queries to16 heads. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

rows1/3/9,heads3/8,dim32/512,selected_keys4/8,window4,compressed/paged states; exact argument scalars. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Bounded48-state independent fixture quantifies selected-only reads and query padding. Earlier host/shape analysis found the change; no matched before/after resource coefficient is manufactured.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Existing selected-only path; shared-query reuse proposed.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/decode-one-token-launches.json](/home/ubuntu/gen-deepseek41/eval/decode-one-token-launches.json)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_prepare_decode_bf16` | fp32_flops | `-0.977639571 * arg_dim + -9.39511418 * arg_rows + 594.125 * arg_selected + 565.070234 * blocks + -3871.0371` | 3569 | false |
| baseline | `ds41_prepare_decode_bf16` | global_read_bytes | `4.61129049 * arg_dim + -107.998088 * arg_rows + 7960.33333 * arg_selected + 5746.37748 * blocks + -50848.8654` | 4.639e+04 | false |
| baseline | `ds41_prepare_decode_bf16` | global_write_bytes | `6.01465746 * arg_dim + -82.4227297 * arg_rows + 2379 * arg_selected + 5657.42326 * blocks + -17260.3426` | 1.38e+04 | false |
| baseline | `ds41_prepare_decode_bf16` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_prepare_decode_bf16` | sum_launch_max_block_lane_sequence_span | `-0.00230267525 * arg_dim + 1.86464418 * arg_rows + 23.0833333 * arg_selected + 0.0386351567 * blocks + 279.679169` | 11.81 | false |
