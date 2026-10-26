# ds41-single-row-mma-padding

**Contribution: FOUND.** Kernel: m16n8k32 MMA for one useful row.

Single-token projection uses tensor-core MMA whose row tile contains16 rows. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

M,N,K,tile_groups,useful_fmas. Source grid includes M1 N8 K32, eliminating N padding. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

tensor_flops=8192*tile_groups. At M1 N8 K32:8192 tensor FLOPs vs512 useful FLOPs, exactly16× row expansion. No after kernel removes this expansion.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Open: same tensor row padding remains in fused and control.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

Actual H100 production shapes, separate from small-shape CPU resource fixtures: QA N1280/K5120 M1 23.2945→20.144 µs, M4 41.374→30.5755; KV N512/K5120 M1 19.807→19.904 (fused rejected), M4 26.7355→20.4. Receipt: /home/ubuntu/gen-deepseek41/eval/profiles/gpu/ds41-prof-20261026-002107. Summary: /home/ubuntu/gen-deepseek41/eval/tc-route-parallel-results.json. No new serving claim; identical-DB pair pending disk.

## Full receipts

- [eval/drperf-gpu/matvec-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/matvec-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| control | `ds41_matvec_tc_finish` | fp32_flops | `0.305068239 * K + 1.74599958 * N + 5.70790996 * arg_m + 84.8086554 * blocks + 6.2906991 * outputs + 2.15340817 * tile_groups + -102.725747` | 2.485e-10 | true |
| control | `ds41_matvec_tc_finish` | global_read_bytes | `-0.00120496708 * K + -2.76903314 * N + -39.2045836 * arg_m + 154.093042 * blocks + 2.18518273 * outputs + 22.670951 * tile_groups + -100.850047` | 4.274e-10 | true |
| control | `ds41_matvec_tc_finish` | global_write_bytes | `2 * outputs` | 0 | true |
| control | `ds41_matvec_tc_finish` | blocks | `1 * blocks` | 0 | true |
| control | `ds41_matvec_tc_finish` | sum_launch_max_block_lane_sequence_span | `0.416880503 * K + -0.216637771 * N + -0.735235414 * arg_m + 14.9275226 * blocks + 0.071252543 * outputs + -0.410813216 * tile_groups + 73.0414318` | 1.296e-11 | true |
| control | `ds41_matvec_tc_partials` | tensor_float_flops | `4.05084612e-08 * N + -8.4626129e-07 * activation_fp4 + 1.98699738e-08 * arg_b_stride + 4.2593735e-07 * arg_formats + 1.05774561e-07 * arg_m + -8.6723675e-08 * blocks + 8192 * tile_groups + -1.10873953e-06` | 9.117e-05 | true |
| control | `ds41_matvec_tc_partials` | fp32_flops | `-0.137229355 * K + -5.88664731 * N + -62.315378 * arg_m + 30.9118199 * blocks + 0.798913652 * outputs + 5.19729852 * tile_groups + 87.2994681` | 2.472e-09 | true |
| control | `ds41_matvec_tc_partials` | global_read_bytes | `6.66926843 * K + -110.218327 * N + 273.523669 * activation_fp4 + -13.2119963 * arg_a_stride + -2173.01773 * arg_m + 1413.47535 * blocks + 7.82342284 * minimum_activation_bytes + -18.3726626 * outputs + 1 * row_weight_payload_bytes + -320.289256 * tile_groups + 2029.40286` | 1295 | false |
| control | `ds41_matvec_tc_partials` | global_write_bytes | `-0.274458709 * K + -11.7732946 * N + -124.630756 * arg_m + 61.8236398 * blocks + 1.5978273 * outputs + 10.394597 * tile_groups + 174.598936` | 4.944e-09 | true |
| control | `ds41_matvec_tc_partials` | blocks | `1 * blocks` | 0 | true |
| control | `ds41_matvec_tc_partials` | sum_launch_max_block_lane_sequence_span | `0.619407366 * K + 18.6323175 * N + 1.72548016 * activation_fp4 + -0.188538094 * arg_a_stride + -2.13828427 * arg_b_stride + 12.5993315 * arg_formats + 201.51484 * arg_m + -143.538317 * blocks + -0.0238169351 * minimum_activation_bytes + 0.130722939 * minimum_weight_bytes + 2.97181069 * outputs + -0.00675739584 * row_weight_payload_bytes + 31.7854376 * tile_groups + -63.9492224` | 29.31 | false |
