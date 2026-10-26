# ds41-greedy-head

**Contribution: POTENTIAL.** Kernel: Local/global greedy argmax.

Selects next token from sharded vocabulary logits. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,width,world size,elements,pair count,rank offset; NaN/Inf/tie cases. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Actual local/global reduction resource interfaces replace assumptions about full-vocabulary work. No timing prediction; CPU scan removal was identified by host analysis earlier.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **GPU head exists; further scan/launch work remains a possible target.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/head-before-pinned](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/head-before-pinned)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_head_global` | global_read_bytes | `96 * arg_rows + 36 * arg_world + -288` | 0 | true |
| baseline | `ds41_head_global` | global_write_bytes | `4 * arg_rows` | 0 | true |
| baseline | `ds41_head_global` | blocks | `1` | 0 | true |
| baseline | `ds41_head_global` | sum_launch_max_block_lane_sequence_span | `0.0156337242 * arg_rows + 16.5585068 * arg_world + 41.7279253` | 3.243 | false |
| baseline | `ds41_head_local` | global_read_bytes | `4 * elements + 1.21343747e-08 * scale_groups + -1.10967449e-08` | 6.117e-06 | true |
| baseline | `ds41_head_local` | global_write_bytes | `12 * arg_rows` | 0 | true |
| baseline | `ds41_head_local` | shared_read_bytes | `0.000289978987 * arg_first + 1.13217113 * arg_input_stride + 5733.11983 * arg_rows + -7.35321656 * elements + 199.271113 * scale_groups + -132.07509` | 2677 | false |
| baseline | `ds41_head_local` | shared_write_bytes | `-0.000191515721 * arg_first + -1.43895504 * arg_input_stride + 3971.53623 * arg_rows + -9.66545195 * elements + 357.285654 * scale_groups + 165.926648` | 3398 | false |
| baseline | `ds41_head_local` | blocks | `1 * arg_rows` | 0 | true |
| baseline | `ds41_head_local` | sum_launch_max_block_lane_sequence_span | `-2.61475082e-05 * arg_first + 0.0802803818 * arg_input_stride + -0.827590488 * arg_rows + -0.0588206618 * elements + 1.83100354 * scale_groups + 228.011864` | 12.23 | false |
