# ds41-cub-selector-adapters

**Contribution: POTENTIAL.** Kernel: CUB pack + block radix finish.

Prepares BF16 score keys and final position-sorted top-k after library radix sorting. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

count,topk,block_size,mode,length,score stride,candidate mask,selected units. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Existing independent selector fixture checks adapters and CUB block sorting. This is not a whole DeviceRadixSort sequence capture; host CUDA library interception would be needed for that.

Library code is included as PTX; the scalar selection oracle is independent. CUB is3.0.1 from the CUDA toolchain. No numerical exemption is used to turn failed execution into a pass.

Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Bounded adapter probe; whole sequence integration remains proposed.**

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
| baseline | `ds41_cub_finish2048` | global_read_bytes | `12836.8` | 8741 | false |
| baseline | `ds41_cub_finish2048` | global_write_bytes | `13078.4` | 4370 | false |
| baseline | `ds41_cub_finish2048` | shared_read_bytes | `294880` | 0 | true |
| baseline | `ds41_cub_finish2048` | shared_write_bytes | `246048` | 0 | true |
| baseline | `ds41_cub_finish2048` | blocks | `1` | 0 | true |
| baseline | `ds41_cub_finish2048` | sum_launch_max_block_lane_sequence_span | `2547.6` | 40.6 | false |
| baseline | `ds41_cub_finish512` | global_read_bytes | `881.006452` | 2191 | false |
| baseline | `ds41_cub_finish512` | global_write_bytes | `872.206452` | 1316 | false |
| baseline | `ds41_cub_finish512` | shared_read_bytes | `196576` | 0 | true |
| baseline | `ds41_cub_finish512` | shared_write_bytes | `172320` | 0 | true |
| baseline | `ds41_cub_finish512` | blocks | `1` | 0 | true |
| baseline | `ds41_cub_finish512` | sum_launch_max_block_lane_sequence_span | `1284.36129` | 31.64 | false |
| baseline | `ds41_cub_pack` | fp32_flops | `-1.01839382 * arg_count + 385.772455 * arg_cs + -83.5264175 * arg_mode + 871.387097 * arg_row + -44.4336424 * blocks + -2768.13681` | 1.66e+04 | false |
| baseline | `ds41_cub_pack` | global_read_bytes | `19.7314938 * arg_count + -4870.69135 * arg_cs + 988.082216 * arg_mode + 2489.66452 * arg_row + 1470.62181 * blocks + 7291.54149` | 3.262e+04 | false |
| baseline | `ds41_cub_pack` | global_write_bytes | `5.98688972 * arg_count + -1536.40013 * arg_cs + 244.12945 * arg_mode + 1.03225806 * arg_row + 1539.39638 * blocks + 3071.04054` | 178.8 | false |
| baseline | `ds41_cub_pack` | blocks | `1 * blocks` | 0 | true |
| baseline | `ds41_cub_pack` | sum_launch_max_block_lane_sequence_span | `-0.0631112285 * arg_count + 16.1119691 * arg_cs + 49.7075488 * arg_mode + 17.0129032 * arg_row + 0.314119256 * blocks + 19.7549245` | 49.5 | false |
