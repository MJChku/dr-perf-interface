# ds41-attention-split

**Contribution: POTENTIAL.** Kernel: Split-KV attention partial + finish.

Decode attention partitions selected KV into64-key chunks. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_rows=1,heads8,dim512,selected_keys64/128,valid_pairs,valid_pair_dim,selected_tiles. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Actual two-state resource interfaces for partial/finish. Selected-key terms distinguish scan work from reduction. Grid is bounded, not a100K-context cost estimate.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Captured baseline; no after fix.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/attention_split-extra-interfaces](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/attention_split-extra-interfaces)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_attention_split_finish` | fp32_flops | `128 * selected_keys + 4096` | 0 | true |
| baseline | `ds41_attention_split_finish` | global_read_bytes | `512 * selected_keys + 16384` | 0 | true |
| baseline | `ds41_attention_split_finish` | global_write_bytes | `8192` | 0 | true |
| baseline | `ds41_attention_split_finish` | blocks | `32` | 0 | true |
| baseline | `ds41_attention_split_finish` | sum_launch_max_block_lane_sequence_span | `0.1875 * selected_keys + 76` | 0 | true |
| baseline | `ds41_attention_split_probabilities` | fp32_flops | `192 * selected_keys + 96` | 0 | true |
| baseline | `ds41_attention_split_probabilities` | global_read_bytes | `32 * selected_keys + 32` | 0 | true |
| baseline | `ds41_attention_split_probabilities` | global_write_bytes | `16.5 * selected_keys + 32` | 0 | true |
| baseline | `ds41_attention_split_probabilities` | blocks | `8` | 0 | true |
| baseline | `ds41_attention_split_probabilities` | sum_launch_max_block_lane_sequence_span | `2.359375 * selected_keys + 78` | 0 | true |
| baseline | `ds41_attention_split_scores` | fp32_flops | `6152 * blocks` | 0 | true |
| baseline | `ds41_attention_split_scores` | global_read_bytes | `7936 * blocks` | 0 | true |
| baseline | `ds41_attention_split_scores` | global_write_bytes | `16 * blocks` | 0 | true |
| baseline | `ds41_attention_split_scores` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_attention_split_scores` | sum_launch_max_block_lane_sequence_span | `1033` | 0 | true |
| baseline | `ds41_attention_split_values` | fp32_flops | `23808 * blocks` | 0 | true |
| baseline | `ds41_attention_split_values` | global_read_bytes | `31488 * blocks` | 0 | true |
| baseline | `ds41_attention_split_values` | global_write_bytes | `512 * blocks` | 0 | true |
| baseline | `ds41_attention_split_values` | shared_read_bytes | `32768 * blocks` | 0 | true |
| baseline | `ds41_attention_split_values` | shared_write_bytes | `256 * blocks` | 0 | true |
| baseline | `ds41_attention_split_values` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_attention_split_values` | sum_launch_max_block_lane_sequence_span | `2.65625 * blocks + 3136` | 0 | true |
