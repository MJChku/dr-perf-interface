# ds41-scatter-merge-clear

**Contribution: POTENTIAL.** Kernel: Expert scatter/merge and zero initialization.

Routes weighted expert results into token accumulators; clears accumulation/scratch. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

Proposed:real_rows,capacity_rows,active_routes,width,context_keys,clear_bytes,scatter_pairs. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Independent17-case fixture exercises duplicate/invalid/padding routes, nonzero poisoned accumulators, shared BF16 merge boundary, casts and HC expansion. Command-level CUDA clears are outside PTX and remain a proposed sequence-accounting case.

Scalar and8-element vector cast entry points use their respective production launch geometry. The first packaging attempt paired the vector launch geometry with scalar data.ptx and failed its oracle; this was a case-driver/module mismatch, corrected explicitly. The oracle sums per-token contributions independently and checks guards and input preservation. Only kernel requests are counted; no unobserved API clear is silently assigned zero work.

Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Bounded scatter/merge/cast/expansion captured; API-clear accounting remains proposed.**

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
| scalar | `ds41_cast_rows` | global_read_bytes | `-40.5 * arg_dim + -12696.7501 * arg_is + 51216.0002 * arg_mode + 12746.2501 * arg_os + -2.358275e-07 * blocks + -25608.0001` | 3.583e+04 | false |
| scalar | `ds41_cast_rows` | global_write_bytes | `-40.5 * arg_dim + 12746.2501 * arg_is + -51216.0002 * arg_mode + -12696.7501 * arg_os + -2.358275e-07 * blocks + 25608.0001` | 3.583e+04 | false |
| scalar | `ds41_cast_rows` | blocks | `1 * blocks` | 0 | true |
| scalar | `ds41_cast_rows` | sum_launch_max_block_lane_sequence_span | `8 * arg_mode + 45` | 0 | true |
| scalar | `ds41_engram_gate` | fp32_flops | `120 * arg_dim + 13440` | 0 | true |
| scalar | `ds41_engram_gate` | global_read_bytes | `144 * arg_dim` | 0 | true |
| scalar | `ds41_engram_gate` | global_write_bytes | `24 * arg_dim` | 0 | true |
| scalar | `ds41_engram_gate` | blocks | `12` | 0 | true |
| scalar | `ds41_engram_gate` | sum_launch_max_block_lane_sequence_span | `1.86644584 * arg_dim + 192.444828` | 29.9 | false |
| scalar | `ds41_expand_hc` | global_read_bytes | `792` | 0 | true |
| scalar | `ds41_expand_hc` | global_write_bytes | `840` | 0 | true |
| scalar | `ds41_expand_hc` | blocks | `4` | 0 | true |
| scalar | `ds41_expand_hc` | sum_launch_max_block_lane_sequence_span | `48` | 0 | true |
| scalar | `ds41_finish_moe` | fp32_flops | `99` | 0 | true |
| scalar | `ds41_finish_moe` | global_read_bytes | `792` | 0 | true |
| scalar | `ds41_finish_moe` | global_write_bytes | `198` | 0 | true |
| scalar | `ds41_finish_moe` | blocks | `1` | 0 | true |
| scalar | `ds41_finish_moe` | sum_launch_max_block_lane_sequence_span | `72` | 0 | true |
| scalar | `ds41_scatter_add` | global_read_bytes | `1518` | 66 | false |
| scalar | `ds41_scatter_add` | global_write_bytes | `594` | 66 | false |
| scalar | `ds41_scatter_add` | blocks | `2` | 0 | true |
| scalar | `ds41_scatter_add` | sum_launch_max_block_lane_sequence_span | `57` | 0 | true |
| scalar | `ds41_table_lookup` | fp32_flops | `192 * arg_mode` | 0 | true |
| scalar | `ds41_table_lookup` | global_read_bytes | `3968` | 0 | true |
| scalar | `ds41_table_lookup` | global_write_bytes | `896` | 0 | true |
| scalar | `ds41_table_lookup` | blocks | `4` | 0 | true |
| scalar | `ds41_table_lookup` | sum_launch_max_block_lane_sequence_span | `22 * arg_mode + 74` | 0 | true |
| vector | `ds41_cast_rows` | global_read_bytes | `-40.5 * arg_dim + -12696.7501 * arg_is + 51216.0002 * arg_mode + 12746.2501 * arg_os + -1.16378037e-07 * blocks + -25608.0001` | 3.583e+04 | false |
| vector | `ds41_cast_rows` | global_write_bytes | `-40.5 * arg_dim + 12746.2501 * arg_is + -51216.0002 * arg_mode + -12696.7501 * arg_os + -1.16378037e-07 * blocks + 25608.0001` | 3.583e+04 | false |
| vector | `ds41_cast_rows` | blocks | `1 * blocks` | 0 | true |
| vector | `ds41_cast_rows` | sum_launch_max_block_lane_sequence_span | `-47.125 * arg_dim + 21.8124992 * arg_is + 71.0000013 * arg_mode + 25.3125008 * arg_os + -1.08808984e-07 * blocks + 62.9999994` | 9.773e-07 | true |
| vector | `ds41_engram_gate` | fp32_flops | `120 * arg_dim + 13440` | 0 | true |
| vector | `ds41_engram_gate` | global_read_bytes | `144 * arg_dim` | 0 | true |
| vector | `ds41_engram_gate` | global_write_bytes | `24 * arg_dim` | 0 | true |
| vector | `ds41_engram_gate` | blocks | `12` | 0 | true |
| vector | `ds41_engram_gate` | sum_launch_max_block_lane_sequence_span | `1.86644584 * arg_dim + 192.444828` | 29.9 | false |
| vector | `ds41_expand_hc` | global_read_bytes | `792` | 0 | true |
| vector | `ds41_expand_hc` | global_write_bytes | `840` | 0 | true |
| vector | `ds41_expand_hc` | blocks | `4` | 0 | true |
| vector | `ds41_expand_hc` | sum_launch_max_block_lane_sequence_span | `48` | 0 | true |
| vector | `ds41_finish_moe` | fp32_flops | `99` | 0 | true |
| vector | `ds41_finish_moe` | global_read_bytes | `792` | 0 | true |
| vector | `ds41_finish_moe` | global_write_bytes | `198` | 0 | true |
| vector | `ds41_finish_moe` | blocks | `1` | 0 | true |
| vector | `ds41_finish_moe` | sum_launch_max_block_lane_sequence_span | `72` | 0 | true |
| vector | `ds41_scatter_add` | global_read_bytes | `1518` | 66 | false |
| vector | `ds41_scatter_add` | global_write_bytes | `594` | 66 | false |
| vector | `ds41_scatter_add` | blocks | `2` | 0 | true |
| vector | `ds41_scatter_add` | sum_launch_max_block_lane_sequence_span | `57` | 0 | true |
| vector | `ds41_table_lookup` | fp32_flops | `192 * arg_mode` | 0 | true |
| vector | `ds41_table_lookup` | global_read_bytes | `3968` | 0 | true |
| vector | `ds41_table_lookup` | global_write_bytes | `896` | 0 | true |
| vector | `ds41_table_lookup` | blocks | `4` | 0 | true |
| vector | `ds41_table_lookup` | sum_launch_max_block_lane_sequence_span | `22 * arg_mode + 74` | 0 | true |
