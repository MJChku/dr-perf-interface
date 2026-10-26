# ds41-device-routing

**Contribution: POTENTIAL.** Kernel: Full384-way router/top6 and initializer.

Chooses six experts from full logits while retaining route IDs and weights on device. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

rows,expert_count384,selected_routes6,logit/input/output strides; initializer count/format. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Bounded independent fixture covers full384-score routing and deterministic payload initialization. Baseline resource interface only; no DrPerf-found routing fix is asserted.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Captured baseline; router/top-k/scatter fusion remains proposed.**

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
| baseline | `ds41_initialize_matrix` | global_write_bytes | `111` | 0 | true |
| baseline | `ds41_initialize_matrix` | blocks | `1` | 0 | true |
| baseline | `ds41_initialize_matrix` | sum_launch_max_block_lane_sequence_span | `3.675 * arg_kind + 73.5222222` | 16.87 | false |
| baseline | `ds41_initialize_payload` | global_write_bytes | `137` | 0 | true |
| baseline | `ds41_initialize_payload` | blocks | `2` | 0 | true |
| baseline | `ds41_initialize_payload` | sum_launch_max_block_lane_sequence_span | `3.675 * arg_kind + 53.5222222` | 16.87 | false |
| baseline | `ds41_route_from_logits` | fp32_flops | `64182` | 9828 | false |
| baseline | `ds41_route_from_logits` | global_read_bytes | `15360` | 0 | true |
| baseline | `ds41_route_from_logits` | global_write_bytes | `240` | 0 | true |
| baseline | `ds41_route_from_logits` | blocks | `2` | 0 | true |
| baseline | `ds41_route_from_logits` | sum_launch_max_block_lane_sequence_span | `1862` | 3 | false |
