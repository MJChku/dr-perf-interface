# ds41-attention-prepare-unpack

**Contribution: QUANTIFIED.** Kernel: Attention gather/prepare/unpack.

Expands selected/compressed KV, pads heads and unpacks library outputs. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows,heads,head_dim,selected_keys,context keys,valid pairs; argument-derived composite PCVs. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Captured adapter requests and padding interfaces. This does not yet measure library attention compute, hardware cache transactions, or a proposed multi-query redesign.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Adapter captured; prefill multi-query path open.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/attention_prepare-library-probe](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/attention_prepare-library-probe)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_prepare_sparse_bf16` | fp32_flops | `572.146101 * arg_cr + 40 * arg_cs + 1.1083896e-07 * blocks + -6493.60711` | 8.433e-07 | true |
| baseline | `ds41_prepare_sparse_bf16` | global_read_bytes | `38.2979092 * arg_cr + 2.27173119 * arg_cs + 25.515851 * arg_is + 2153.01313 * blocks + -1948.45375` | 900 | false |
| baseline | `ds41_prepare_sparse_bf16` | global_write_bytes | `1124.41369 * arg_cr + 207.855369 * arg_cs + 7.94156319 * arg_is + 4.38610046 * blocks + -13363.4948` | 10.18 | false |
| baseline | `ds41_prepare_sparse_bf16` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_prepare_sparse_bf16` | sum_launch_max_block_lane_sequence_span | `-0.49770024 * arg_cr + -0.0452657049 * arg_cs + 0.0559532505 * arg_is + 1.68030888 * blocks + 483.552497` | 17.98 | false |
| baseline | `ds41_unpack_sparse_bf16` | global_read_bytes | `8 * arg_dim + 682.666667 * arg_heads + -1365.33333` | 1.819e-12 | true |
| baseline | `ds41_unpack_sparse_bf16` | global_write_bytes | `8 * arg_dim + 682.666667 * arg_heads + -1365.33333` | 1.819e-12 | true |
| baseline | `ds41_unpack_sparse_bf16` | blocks | `0.166666667 * arg_heads + 0.666666667` | 2.22e-16 | true |
| baseline | `ds41_unpack_sparse_bf16` | sum_launch_max_block_lane_sequence_span | `-2.33333333 * arg_heads + 66.6666667` | 7.105e-15 | true |
