# ds41-routed-metadata-cache

**Contribution: FOUND.** Kernel: Routed fused matvec metadata.

Each token selects six experts; local rank owns a subset. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,N,K,active_routes,max_routes,active_experts,route_tile_visits,k_groups. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Uncached v1-equivalent reads route IDs and expert-pointer pairs inside repeated work. Cached v2 stages six IDs and pointer pairs once per CTA. Aligned observations quantify requested global-read coefficient changes; tensor work unchanged.

The uncached input is an explicit reconstruction of the former metadata-read structure from the current kernel; source and transformation provenance are included. It is not claimed byte-identical to a historical binary. Native v1→v2 timing was not isolated.

Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Cache implemented; serial fusion not the final adopted route policy.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/fused_routed-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/fused_routed-before-pinned)
- [eval/drperf-gpu/routed-after](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/routed-after)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| uncached | `ds41_route_token_ids` | global_read_bytes | `24.3529412 * arg_rows + -0.352941176` | 11.29 | false |
| uncached | `ds41_route_token_ids` | global_write_bytes | `24.3529412 * arg_rows + -0.352941176` | 11.29 | false |
| uncached | `ds41_route_token_ids` | blocks | `1` | 0 | true |
| uncached | `ds41_route_token_ids` | sum_launch_max_block_lane_sequence_span | `0.0294117647 * arg_rows + 36.9705882` | 0.9412 | false |
| uncached | `ds41_routed_tc_fused` | tensor_float_flops | `45.1764706 * active_n_k + 2.27142292e-08 * any_groups + -3.43806831e-07 * any_iters8 + 3.22355756e-07 * any_route` | 4.005e-08 | true |
| uncached | `ds41_routed_tc_fused` | fp32_flops | `0.09375 * active_n_k` | 8.913e-11 | true |
| uncached | `ds41_routed_tc_fused` | global_read_bytes | `6553.6 * active_experts + 0.857352941 * active_n_k + 3686.4 * any_groups + -55296 * any_iters8 + 51609.6 * any_route + 6144 * blocks` | 8.709e-09 | true |
| uncached | `ds41_routed_tc_fused` | global_write_bytes | `68 * blocks` | 0 | true |
| uncached | `ds41_routed_tc_fused` | shared_read_bytes | `0.125 * active_n_k` | 4.002e-11 | true |
| uncached | `ds41_routed_tc_fused` | shared_write_bytes | `0.125 * active_n_k` | 4.002e-11 | true |
| uncached | `ds41_routed_tc_fused` | blocks | `1 * blocks` | 0 | true |
| uncached | `ds41_routed_tc_fused` | sum_launch_max_block_lane_sequence_span | `-1 * all_routes + -21.6421053 * any_groups + 324.157895 * any_iters8 + -304.778947 * any_route + 141.189474 * max_routes + 10.3894737 * route_groups + 147` | 0.5263 | false |
| cached | `ds41_route_token_ids` | global_read_bytes | `24.3529412 * arg_rows + -0.352941176` | 11.29 | false |
| cached | `ds41_route_token_ids` | global_write_bytes | `24.3529412 * arg_rows + -0.352941176` | 11.29 | false |
| cached | `ds41_route_token_ids` | blocks | `1` | 0 | true |
| cached | `ds41_route_token_ids` | sum_launch_max_block_lane_sequence_span | `0.0294117647 * arg_rows + 36.9705882` | 0.9412 | false |
| cached | `ds41_routed_tc_fused` | tensor_float_flops | `45.1764706 * active_n_k + 2.27142292e-08 * any_groups + -3.43806831e-07 * any_iters8 + 3.22355756e-07 * any_route` | 4.005e-08 | true |
| cached | `ds41_routed_tc_fused` | fp32_flops | `0.09375 * active_n_k` | 8.913e-11 | true |
| cached | `ds41_routed_tc_fused` | global_read_bytes | `48 * active_experts + 0.198529412 * active_n_k + 24 * blocks` | 5.093e-11 | true |
| cached | `ds41_routed_tc_fused` | global_write_bytes | `68 * blocks` | 0 | true |
| cached | `ds41_routed_tc_fused` | shared_read_bytes | `6553.6 * active_experts + 0.783823529 * active_n_k + 3686.4 * any_groups + -55296 * any_iters8 + 51609.6 * any_route + 6144 * blocks` | 1.135e-08 | true |
| cached | `ds41_routed_tc_fused` | shared_write_bytes | `48 * active_experts + 0.125 * active_n_k + 24 * blocks` | 6.185e-11 | true |
| cached | `ds41_routed_tc_fused` | blocks | `1 * blocks` | 0 | true |
| cached | `ds41_routed_tc_fused` | sum_launch_max_block_lane_sequence_span | `-21.6421053 * any_groups + 325.157895 * any_iters8 + -292.778947 * any_route + 133.189474 * max_routes + 10.3894737 * route_groups + 156` | 0.5263 | false |
