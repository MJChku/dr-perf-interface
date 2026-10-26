# ds41-integer-fp8-fp4-matvec

**Contribution: POTENTIAL.** Kernel: SIMT integer-format matvec.

Fallback FP8/FP4 matrix-vector products without TC staging. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

M,N,K,weight_fp4,activation_fp4,minimum_weight_bytes,row_weight_payload_bytes,useful_fmas. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Counts distinguish FP32 scalar arithmetic, format conversion and repeated per-row requests. No matched improvement is claimed.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Baseline captured; compare tensor path separately.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/integer-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/integer-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_scaled_matvec` | fp32_flops | `6.375 * K + -17.8897959 * N + 1.25116467e-08 * activation_fp4 + 8.00408158 * arg_m + -125.964626 * blocks + 45.4544218 * outputs + -148.591837` | 6.727e-08 | true |
| baseline | `ds41_scaled_matvec` | global_read_bytes | `105.1875 * K + 327.757143 * N + 5.62708785e-07 * activation_fp4 + -4.03632058e-08 * arg_b_stride + -2.82928935e-07 * arg_formats + 5605.50952 * arg_m + -8779.69523 * blocks + 2391.00476 * outputs + -12634.2143` | 2.01e-05 | true |
| baseline | `ds41_scaled_matvec` | global_write_bytes | `2 * outputs` | 0 | true |
| baseline | `ds41_scaled_matvec` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_scaled_matvec` | sum_launch_max_block_lane_sequence_span | `2.35811086 * K + 0.235918025 * N + 0.088905074 * activation_fp4 + -0.171645017 * arg_a_stride + -0.160462955 * arg_b_stride + -0.0734847436 * arg_formats + 7.44905826 * arg_m + -9.38071537 * blocks + -0.000162363975 * minimum_activation_bytes + -0.000775726912 * minimum_weight_bytes + 2.32271575 * outputs + 2.91333939e-05 * row_weight_payload_bytes + 121.474227` | 2.004 | false |
