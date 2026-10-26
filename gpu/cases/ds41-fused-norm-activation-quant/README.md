# ds41-fused-norm-activation-quant

**Contribution: QUANTIFIED.** Kernel: HC-pre/RMS/SwiGLU quant fusions.

Combines decode HC preprocessing, RMS or SwiGLU activation with FP8 quantization. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,width,elements,scale_groups,weighted_elements plus argument scalars. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Captured fused resource interfaces for four entries. Launch histogram motivated these fusions before DrPerf; this is after-only quantification, no invented baseline coefficient delta.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Existing passing fusion; serving claims require matched GX pairs.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/fusion-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/fusion-before-pinned)
- [eval/DECODE-FUSION-PLAN.md](/home/ubuntu/gen-deepseek41/eval/DECODE-FUSION-PLAN.md)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| fused | `ds41_hc_coeff_post` | fp32_flops | `132 * arg_dim + 5.03125 * launched_lanes + 21246` | 2.91e-11 | true |
| fused | `ds41_hc_coeff_post` | global_read_bytes | `144 * arg_dim + 864` | 0 | true |
| fused | `ds41_hc_coeff_post` | global_write_bytes | `24 * arg_dim + 288` | 0 | true |
| fused | `ds41_hc_coeff_post` | shared_read_bytes | `240 * arg_dim + 4.125 * launched_lanes + 288` | 1.164e-10 | true |
| fused | `ds41_hc_coeff_post` | shared_write_bytes | `0.125 * launched_lanes + 300` | 0 | true |
| fused | `ds41_hc_coeff_post` | blocks | `3` | 0 | true |
| fused | `ds41_hc_coeff_post` | sum_launch_max_block_lane_sequence_span | `1.11730769 * arg_dim + -1.88031136 * launched_lanes + 2222.74066` | 977 | false |
| fused | `ds41_hc_pre_quant` | fp32_flops | `65.0439744 * arg_dim + 4.9983196 * launched_lanes + 8.79208791` | 2.028e+04 | false |
| fused | `ds41_hc_pre_quant` | global_read_bytes | `90 * arg_dim` | 0 | true |
| fused | `ds41_hc_pre_quant` | global_write_bytes | `15.09375 * arg_dim` | 1.455e-11 | true |
| fused | `ds41_hc_pre_quant` | shared_read_bytes | `4.125 * launched_lanes` | 0 | true |
| fused | `ds41_hc_pre_quant` | shared_write_bytes | `0.125 * launched_lanes + 12` | 0 | true |
| fused | `ds41_hc_pre_quant` | blocks | `3` | 0 | true |
| fused | `ds41_hc_pre_quant` | sum_launch_max_block_lane_sequence_span | `0.625865385 * arg_dim + -1.04211081 * launched_lanes + 890.904176` | 575.9 | false |
| fused | `ds41_rms_quant` | fp32_flops | `39.0759615 * arg_dim + 5.03012821 * launched_lanes + 5.95384615` | 0.2769 | false |
| fused | `ds41_rms_quant` | global_read_bytes | `24 * arg_dim` | 0 | true |
| fused | `ds41_rms_quant` | global_write_bytes | `9.09375 * arg_dim` | 7.276e-12 | true |
| fused | `ds41_rms_quant` | shared_read_bytes | `4.125 * launched_lanes` | 9.095e-13 | true |
| fused | `ds41_rms_quant` | shared_write_bytes | `0.125 * launched_lanes + 12` | 0 | true |
| fused | `ds41_rms_quant` | blocks | `3` | 0 | true |
| fused | `ds41_rms_quant` | sum_launch_max_block_lane_sequence_span | `0.490793269 * arg_dim + -0.804385302 * launched_lanes + 707.694066` | 428.2 | false |
| fused | `ds41_swiglu_quant` | fp32_flops | `79.1272817 * arg_dim + 2.26722193 * arg_weighted + -60.0307695 * blocks + 1.36642759 * weighted_elements + 15.6817737` | 4.481 | false |
| fused | `ds41_swiglu_quant` | global_read_bytes | `12 * arg_dim + 4 * weighted_elements` | 0 | true |
| fused | `ds41_swiglu_quant` | global_write_bytes | `9.09375 * arg_dim + -2.19333183e-08 * blocks` | 5.72e-09 | true |
| fused | `ds41_swiglu_quant` | blocks | `1 * blocks` | 0 | true |
| fused | `ds41_swiglu_quant` | sum_launch_max_block_lane_sequence_span | `4 * arg_weighted + 161` | 0 | true |
