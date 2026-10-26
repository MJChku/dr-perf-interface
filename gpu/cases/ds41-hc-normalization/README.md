# ds41-hc-normalization

**Contribution: POTENTIAL.** Kernel: HC coefficients/pre/post + RMS.

HC stream mixing and normalization around attention and MoE. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

rows,width,elements,streams,iterations,projection size; exact argument-derived vocabulary in support/ds41/drperf_gpu_argument_pcvs.inc. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Actual captures fit FP32 work and requested bytes vs shape. No new DrPerf-found defect or before/after optimization is asserted.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Measured baseline; fusion opportunities remain.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/hc-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/hc-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_dense_bf16` | tensor_float_flops | `4096 * arg_as + 40960` | 0 | true |
| baseline | `ds41_dense_bf16` | global_read_bytes | `11.7588583 * arg_as + 953.163386` | 1.137e-12 | true |
| baseline | `ds41_dense_bf16` | global_write_bytes | `-0.0339566929 * arg_as + 14 * arg_flags + 126.290354` | 23 | false |
| baseline | `ds41_dense_bf16` | shared_read_bytes | `384 * arg_as + 3840` | 0 | true |
| baseline | `ds41_dense_bf16` | shared_write_bytes | `192 * arg_as + 1920` | 0 | true |
| baseline | `ds41_dense_bf16` | blocks | `1` | 0 | true |
| baseline | `ds41_dense_bf16` | sum_launch_max_block_lane_sequence_span | `10.2392963 * arg_as + -7.875 * arg_flags + 382.156742` | 6.75 | false |
| baseline | `ds41_dense_gemm` | fp32_flops | `3570` | 0 | true |
| baseline | `ds41_dense_gemm` | global_read_bytes | `510 * arg_flags + 8925` | 3825 | false |
| baseline | `ds41_dense_gemm` | global_write_bytes | `19.4285714 * arg_flags + 85` | 41.29 | false |
| baseline | `ds41_dense_gemm` | blocks | `2` | 0 | true |
| baseline | `ds41_dense_gemm` | sum_launch_max_block_lane_sequence_span | `-12.0952381 * arg_flags + 389.833333` | 70.55 | false |
| baseline | `ds41_dense_matvec` | fp32_flops | `7.39665354 * arg_as + 11448.9272` | 1.819e-12 | true |
| baseline | `ds41_dense_matvec` | global_read_bytes | `27.6230315 * arg_as + 3182.85714 * arg_flags + -1479.6752` | 4.258e+04 | false |
| baseline | `ds41_dense_matvec` | global_write_bytes | `-0.0339566929 * arg_as + 10.6666667 * arg_flags + 116.957021` | 45.67 | false |
| baseline | `ds41_dense_matvec` | blocks | `-0.0031988189 * arg_as + 15.1215551` | 1.776e-15 | true |
| baseline | `ds41_dense_matvec` | sum_launch_max_block_lane_sequence_span | `0.46875 * arg_as + -20.8809524 * arg_flags + 166.520833` | 263.7 | false |
| baseline | `ds41_dense_tiled` | fp32_flops | `507.590551 * arg_as + 16551.5591` | 1.455e-11 | true |
| baseline | `ds41_dense_tiled` | global_read_bytes | `17.5607776 * arg_as + 2213.7619 * arg_flags + -6000.47621` | 2.57e+04 | false |
| baseline | `ds41_dense_tiled` | global_write_bytes | `-0.0339566929 * arg_as + 10.6666667 * arg_flags + 116.957021` | 45.67 | false |
| baseline | `ds41_dense_tiled` | shared_read_bytes | `2030.3622 * arg_as + 66206.2362` | 5.821e-11 | true |
| baseline | `ds41_dense_tiled` | shared_write_bytes | `125.984252 * arg_as + 11596.5984` | 1.819e-11 | true |
| baseline | `ds41_dense_tiled` | blocks | `-0.000246062992 * arg_as + 2.00935039` | 0 | true |
| baseline | `ds41_dense_tiled` | sum_launch_max_block_lane_sequence_span | `7.99901575 * arg_as + -33.6666667 * arg_flags + 261.370735` | 398.3 | false |
| baseline | `ds41_hc_coefficients` | fp32_flops | `16 * arg_dim + 5.03125 * launched_lanes + 14164` | 2.92e-07 | true |
| baseline | `ds41_hc_coefficients` | global_read_bytes | `16 * arg_dim + 576` | 0 | true |
| baseline | `ds41_hc_coefficients` | global_write_bytes | `192` | 0 | true |
| baseline | `ds41_hc_coefficients` | shared_read_bytes | `4.125 * launched_lanes` | 9.468e-09 | true |
| baseline | `ds41_hc_coefficients` | shared_write_bytes | `0.125 * launched_lanes + 8` | 4.816e-11 | true |
| baseline | `ds41_hc_coefficients` | blocks | `2` | 0 | true |
| baseline | `ds41_hc_coefficients` | sum_launch_max_block_lane_sequence_span | `1.06451613 * arg_dim + -13.6964286 * launched_lanes + 4.96774194 * scale_groups + 1680.57143` | 7.628e-11 | true |
| baseline | `ds41_hc_post` | fp32_flops | `72 * arg_dim` | 0 | true |
| baseline | `ds41_hc_post` | global_read_bytes | `240 * arg_dim + 4.57551273e-08 * blocks + -1.44937076e-08` | 1.463e-05 | true |
| baseline | `ds41_hc_post` | global_write_bytes | `16 * arg_dim` | 0 | true |
| baseline | `ds41_hc_post` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_hc_post` | sum_launch_max_block_lane_sequence_span | `109` | 0 | true |
| baseline | `ds41_hc_pre_norm` | fp32_flops | `24 * arg_dim + 5.03125 * launched_lanes + 4` | 5.515e-09 | true |
| baseline | `ds41_hc_pre_norm` | global_read_bytes | `56 * arg_dim` | 0 | true |
| baseline | `ds41_hc_pre_norm` | global_write_bytes | `8 * arg_dim` | 0 | true |
| baseline | `ds41_hc_pre_norm` | shared_read_bytes | `4.125 * launched_lanes` | 9.468e-09 | true |
| baseline | `ds41_hc_pre_norm` | shared_write_bytes | `0.125 * launched_lanes + 8` | 4.816e-11 | true |
| baseline | `ds41_hc_pre_norm` | blocks | `2` | 0 | true |
| baseline | `ds41_hc_pre_norm` | sum_launch_max_block_lane_sequence_span | `-23.390625 * launched_lanes + 37.5 * scale_groups + 1595` | 3.824e-08 | true |
| baseline | `ds41_rms_norm` | fp32_flops | `12 * arg_dim + 5.03125 * launched_lanes + 6` | 2.532e-08 | true |
| baseline | `ds41_rms_norm` | global_read_bytes | `18 * arg_dim + -6.92198228e-08` | 6.922e-08 | true |
| baseline | `ds41_rms_norm` | global_write_bytes | `6 * arg_dim` | 0 | true |
| baseline | `ds41_rms_norm` | shared_read_bytes | `4.125 * launched_lanes` | 8.549e-09 | true |
| baseline | `ds41_rms_norm` | shared_write_bytes | `0.125 * launched_lanes + 12` | 1.449e-10 | true |
| baseline | `ds41_rms_norm` | blocks | `3` | 0 | true |
| baseline | `ds41_rms_norm` | sum_launch_max_block_lane_sequence_span | `1.15625 * launched_lanes + 56` | 1.413e-09 | true |
| baseline | `ds41_rope` | fp32_flops | `448` | 0 | true |
| baseline | `ds41_rope` | global_read_bytes | `1024` | 0 | true |
| baseline | `ds41_rope` | global_write_bytes | `256` | 0 | true |
| baseline | `ds41_rope` | blocks | `1` | 0 | true |
| baseline | `ds41_rope` | sum_launch_max_block_lane_sequence_span | `112` | 0 | true |
| baseline | `ds41_swiglu` | fp32_flops | `134 * arg_weighted + 2144` | 0 | true |
| baseline | `ds41_swiglu` | global_read_bytes | `536 * arg_weighted + 536` | 0 | true |
| baseline | `ds41_swiglu` | global_write_bytes | `268` | 0 | true |
| baseline | `ds41_swiglu` | blocks | `2` | 0 | true |
| baseline | `ds41_swiglu` | sum_launch_max_block_lane_sequence_span | `11 * arg_weighted + 69` | 0 | true |
