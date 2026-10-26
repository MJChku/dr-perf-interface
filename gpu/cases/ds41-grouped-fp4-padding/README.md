# ds41-grouped-fp4-padding

**Contribution: QUANTIFIED.** Kernel: Grouped FP4 M32/M16 tail.

Prefill groups real token rows by expert before FP4 GEMMs. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,padded_rows,active_experts,capacity_rows=256,N128,K64,tiles16,tiles32,active_tiles; eight expert histograms in inputs.json and driver. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

tensor_flops=16384*padded_rows, zero real-row term at fixed N/K. M32→M16 tail packing: real53 padded128→96 (−25%); real26 padded128→64 (−50%); full tiles unchanged. Same8 launched CTAs; no added kernel launch.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Custom full32/tail16 wrapper passes oracle; host production packing remains32; production preflight blocked on dp2a, so disabled in serving.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No H100 timing: exact production control fails CPU coverage preflight on dp2a.lo.u32.u32; it was excluded from the native batch.

## Full receipts

- [eval/drperf-gpu/grouped-tight-control32](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/grouped-tight-control32)
- [eval/drperf-gpu/grouped-tight-tight16](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/grouped-tight-tight16)
- [eval/grouped-tight-source.json](/home/ubuntu/gen-deepseek41/eval/grouped-tight-source.json)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| control | `ds41_library_grouped_fp4` | tensor_float_flops | `524288 * active_tiles` | 0 | true |
| control | `ds41_library_grouped_fp4` | fp32_flops | `24576 * active_tiles` | 0 | true |
| control | `ds41_library_grouped_fp4` | global_read_bytes | `8704 * active_tiles + 8192` | 0 | true |
| control | `ds41_library_grouped_fp4` | global_write_bytes | `8192 * active_tiles` | 0 | true |
| control | `ds41_library_grouped_fp4` | shared_read_bytes | `32768 * active_tiles` | 0 | true |
| control | `ds41_library_grouped_fp4` | shared_write_bytes | `22528 * active_tiles` | 0 | true |
| control | `ds41_library_grouped_fp4` | blocks | `8` | 0 | true |
| control | `ds41_library_grouped_fp4` | sum_launch_max_block_lane_sequence_span | `237.238236 * active_experts + 6.42739171 * active_tiles + -1.48018074 * real_rows + 213.541913` | 263.1 | false |
| tight | `ds41_library_grouped_fp4` | tensor_float_flops | `1.41247582e-08 * active_tiles + 16384 * padded_rows` | 9.872e-08 | true |
| tight | `ds41_library_grouped_fp4` | fp32_flops | `768 * padded_rows` | 0 | true |
| tight | `ds41_library_grouped_fp4` | global_read_bytes | `7168 * active_tiles + 64 * padded_rows + 4096` | 0 | true |
| tight | `ds41_library_grouped_fp4` | global_write_bytes | `256 * padded_rows` | 0 | true |
| tight | `ds41_library_grouped_fp4` | shared_read_bytes | `8192 * active_tiles + 768 * padded_rows` | 0 | true |
| tight | `ds41_library_grouped_fp4` | shared_write_bytes | `12288 * active_tiles + 320 * padded_rows` | 0 | true |
| tight | `ds41_library_grouped_fp4` | blocks | `8` | 0 | true |
| tight | `ds41_library_grouped_fp4` | sum_launch_max_block_lane_sequence_span | `526.683766 * active_experts + -823.918141 * active_tiles + 35.6660149 * padded_rows + -16.2750479 * real_rows + 122.244322` | 244 | false |
