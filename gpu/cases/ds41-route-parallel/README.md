# ds41-route-parallel

**Contribution: EXPLAINED.** Kernel: Routed TC partial/serial/parallel.

Expert up/down device-indexed launches during decode. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

N=8/32/128,K=64/512/1280,R=0/1/3/6,M1 plus invalid-ID guard case; route_groups and route tile products. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Serial fusion moves route work inside one CTA: maximum block lane span grows with R. Parallel variant puts (route,N-tile) in grid geometry; maximum block span is independent of R>0 while total work/aggregate spans still grow with R.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Route-parallel adopted at measured production M1 shapes; R4/5 native-unmeasured, labeled model only.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

Separate H100: up N2304 K5120 R3 control50.288/serial53.568/parallel49.1835 µs; R6 84.895/119.84/80.991. Down N5120 K2304 R6 83.582/78.079/71.279 µs. Full table: /home/ubuntu/gen-deepseek41/eval/profiles/gpu/ds41-prof-20261026-002107 and /home/ubuntu/gen-deepseek41/eval/TC-ADOPTION.md.

## Full receipts

- [eval/drperf-gpu/route-interface-summary.json](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/route-interface-summary.json)
- [eval/drperf-gpu/route-interfaces-control](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/route-interfaces-control)
- [eval/drperf-gpu/route-interfaces-serial](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/route-interfaces-serial)
- [eval/drperf-gpu/route-interfaces-parallel](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/route-interfaces-parallel)
- [eval/tc-route-parallel-results.json](/home/ubuntu/gen-deepseek41/eval/tc-route-parallel-results.json)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| control | `ds41_route_token_ids` | global_read_bytes | `30 * arg_rows + -6` | 0 | true |
| control | `ds41_route_token_ids` | global_write_bytes | `30 * arg_rows + -6` | 0 | true |
| control | `ds41_route_token_ids` | blocks | `1` | 0 | true |
| control | `ds41_route_token_ids` | sum_launch_max_block_lane_sequence_span | `0.5 * arg_rows + 36.5` | 0 | true |
| control | `ds41_routed_tc_finish` | fp32_flops | `-0.0422744507 * K + -7.21674762 * N + 401.88118 * active_experts + 4.82355036 * active_n + 0.0266860716 * active_n_k + 0.40651795 * any_groups + -3.78505789 * any_iters8 + 2.23414067 * any_route + -1340.35616 * arg_tokens + -1.57272092e-08 * blocks + 6.70982023 * outputs + 8.17855967 * route_groups + -168 * route_iters4 + 0.135180639 * route_tile_visits + 1366.50995` | 1067 | false |
| control | `ds41_routed_tc_finish` | global_read_bytes | `0.125 * active_n_k + 2.33669751e-08 * blocks + 24 * outputs + -1.42572389e-08` | 4.064e-07 | true |
| control | `ds41_routed_tc_finish` | global_write_bytes | `12 * outputs` | 0 | true |
| control | `ds41_routed_tc_finish` | blocks | `1 * blocks` | 0 | true |
| control | `ds41_routed_tc_finish` | sum_launch_max_block_lane_sequence_span | `6 * any_groups + 51 * any_iters8 + 8 * any_route + 56` | 0 | true |
| control | `ds41_routed_tc_partials` | tensor_float_flops | `-5.34647907e-08 * active_experts + 32 * active_n_k + -2.6986033e-08 * any_groups + 2.35573515e-07 * any_iters8 + -1.0475347e-07 * any_route + 1.50437791e-07 * arg_tokens + 2.02736731e-08 * route_iters4 + -1.67172402e-07` | 1.434e-07 | true |
| control | `ds41_routed_tc_partials` | fp32_flops | `0.0625 * active_n_k` | 2.91e-11 | true |
| control | `ds41_routed_tc_partials` | global_read_bytes | `-0.225463737 * K + -38.4893206 * N + 2143.36629 * active_experts + 25.7256019 * active_n + 0.616284049 * active_n_k + 2.16809573 * any_groups + -20.1869754 * any_iters8 + 11.9154169 * any_route + -7148.56619 * arg_tokens + 772.32578 * blocks + 35.7857079 * outputs + 43.6189849 * route_groups + -896 * route_iters4 + 7288.05309` | 5691 | false |
| control | `ds41_routed_tc_partials` | global_write_bytes | `0.125 * active_n_k` | 5.821e-11 | true |
| control | `ds41_routed_tc_partials` | blocks | `1 * blocks` | 0 | true |
| control | `ds41_routed_tc_partials` | sum_launch_max_block_lane_sequence_span | `47.5 * active_experts + -1.5 * all_routes + -0.5 * any_route + -142.5 * arg_tokens + 78.5 * route_iters4 + 206.5` | 3.365e-11 | true |
| serial | `ds41_route_token_ids` | global_read_bytes | `30 * arg_rows + -6` | 0 | true |
| serial | `ds41_route_token_ids` | global_write_bytes | `30 * arg_rows + -6` | 0 | true |
| serial | `ds41_route_token_ids` | blocks | `1` | 0 | true |
| serial | `ds41_route_token_ids` | sum_launch_max_block_lane_sequence_span | `0.5 * arg_rows + 36.5` | 0 | true |
| serial | `ds41_routed_tc_fused` | tensor_float_flops | `-5.34647907e-08 * active_experts + 32 * active_n_k + -2.6986033e-08 * any_groups + 2.35573515e-07 * any_iters8 + -1.0475347e-07 * any_route + 2.50669289e-08 * capacity_rows + 2.02736731e-08 * route_iters4 + -1.67172402e-07` | 1.434e-07 | true |
| serial | `ds41_routed_tc_fused` | fp32_flops | `0.09375 * active_n_k` | 2.037e-10 | true |
| serial | `ds41_routed_tc_fused` | global_read_bytes | `2 * active_n + 0.140625 * active_n_k + 24 * blocks` | 2.328e-10 | true |
| serial | `ds41_routed_tc_fused` | global_write_bytes | `96 * blocks` | 0 | true |
| serial | `ds41_routed_tc_fused` | shared_read_bytes | `1.35278242 * K + 230.935924 * N + -12860.1978 * active_experts + 357.646388 * active_n + 0.271045708 * active_n_k + -13.0085744 * any_groups + 121.121852 * any_iters8 + -71.4925013 * any_route + 4426.28602 * blocks + 7148.56619 * capacity_rows + -261.713909 * route_groups + 5376 * route_iters4 + -4.32578044 * route_tile_visits + -43728.3186` | 3.414e+04 | false |
| serial | `ds41_routed_tc_fused` | shared_write_bytes | `2 * active_n + 0.125 * active_n_k + 24 * blocks` | 2.91e-11 | true |
| serial | `ds41_routed_tc_fused` | blocks | `1 * blocks` | 0 | true |
| serial | `ds41_routed_tc_fused` | sum_launch_max_block_lane_sequence_span | `185.596491 * active_experts + 0.666666667 * all_routes + -0.245614035 * any_groups + 3.96491228 * any_iters8 + 7.50877193 * any_route + -92.7397661 * capacity_rows + 16 * route_groups + -35.5789474 * route_iters4 + 712.438596` | 0.7018 | false |
| parallel | `ds41_route_token_ids` | global_read_bytes | `30 * arg_rows + -6` | 0 | true |
| parallel | `ds41_route_token_ids` | global_write_bytes | `30 * arg_rows + -6` | 0 | true |
| parallel | `ds41_route_token_ids` | blocks | `1` | 0 | true |
| parallel | `ds41_route_token_ids` | sum_launch_max_block_lane_sequence_span | `0.5 * arg_rows + 36.5` | 0 | true |
| parallel | `ds41_routed_tc_parallel` | tensor_float_flops | `-5.40320393e-08 * active_experts + 32 * active_n_k + -2.69741894e-08 * any_groups + 2.3546324e-07 * any_iters8 + -1.0468838e-07 * any_route + 2.57464245e-08 * capacity_rows + 2.02736731e-08 * route_iters4 + -1.6996637e-07` | 1.416e-07 | true |
| parallel | `ds41_routed_tc_parallel` | fp32_flops | `0.09375 * active_n_k` | 2.037e-10 | true |
| parallel | `ds41_routed_tc_parallel` | global_read_bytes | `2 * active_n + 0.140625 * active_n_k + 4 * blocks` | 1.455e-10 | true |
| parallel | `ds41_routed_tc_parallel` | global_write_bytes | `16 * blocks` | 0 | true |
| parallel | `ds41_routed_tc_parallel` | shared_read_bytes | `1.35278242 * K + 230.935924 * N + -12860.1978 * active_experts + 357.646388 * active_n + 0.271045708 * active_n_k + -13.0085744 * any_groups + 121.121852 * any_iters8 + -71.4925013 * any_route + 737.714337 * blocks + 7148.56619 * capacity_rows + -261.713909 * route_groups + 5376 * route_iters4 + -4.32578044 * route_tile_visits + -43728.3186` | 3.414e+04 | false |
| parallel | `ds41_routed_tc_parallel` | shared_write_bytes | `2 * active_n + 0.125 * active_n_k + 4 * blocks` | 2.91e-11 | true |
| parallel | `ds41_routed_tc_parallel` | blocks | `1 * blocks` | 0 | true |
| parallel | `ds41_routed_tc_parallel` | sum_launch_max_block_lane_sequence_span | `3.33333333 * any_groups + 102.333333 * any_iters8 + 56 * any_route + 65` | 1.722e-11 | true |
