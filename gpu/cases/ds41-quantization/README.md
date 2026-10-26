# ds41-quantization

**Contribution: POTENTIAL.** Kernel: Per-row FP8 quantization.

Scales and quantizes activations before projections. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,width,elements,scale_groups,activation format and stride. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Baseline interface exposes standalone conversion/read/write work; folding quantization into a neighboring kernel remains a possible target. No after variant here.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Captured baseline; proposed neighbor fusion.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/quantize-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/quantize-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_quantize_rows` | fp32_flops | `9.57321429 * arg_input_stride + -156.342857 * arg_mode + -0.610714286 * arg_packed_stride + -641.028571 * arg_scale_stride + 4011.2 * blocks + -3398.01607` | 691.2 | false |
| baseline | `ds41_quantize_rows` | global_read_bytes | `1.42857143 * arg_input_stride + 282.571429` | 8.072e-12 | true |
| baseline | `ds41_quantize_rows` | global_write_bytes | `1.01785714 * arg_input_stride + -73.1428571 * arg_mode + 0.714285714 * arg_packed_stride + -28.4285714 * arg_scale_stride + 136 * blocks + 138.589286` | 512 | false |
| baseline | `ds41_quantize_rows` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_quantize_rows` | sum_launch_max_block_lane_sequence_span | `0.0369419643 * arg_input_stride + 8.65714286 * arg_mode + -0.0169642857 * arg_packed_stride + -2.40357143 * arg_scale_stride + 11.95 * blocks + 139.465513` | 20 | false |
