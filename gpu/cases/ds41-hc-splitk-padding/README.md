# ds41-hc-splitk-padding

**Contribution: QUANTIFIED.** Kernel: HC TF32x3 split-K projection.

Projects HC state to mixing coefficients. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

Observed M3,N24,K20480,split128; proposed broader grid M,N,K,split_k,padded_M,padded_N. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Single shape exposes tensor-vs-useful work including TF32x3 decomposition and64×32 tile padding. Observed tensor/useful FLOPs ratio85.33 is a work ratio, not elapsed time. One state cannot identify independent scaling coefficients.



Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Single-state measured; broader PCV grid proposed.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/drperf-gpu/hc_splitk-extra-interfaces](/home/ubuntu/gen-deepseek41/eval/drperf-gpu/hc_splitk-extra-interfaces)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| baseline | `ds41_hc_splitk` | tensor_float_flops | `251658240` | 0 | true |
| baseline | `ds41_hc_splitk` | fp32_flops | `5701632` | 0 | true |
| baseline | `ds41_hc_splitk` | global_read_bytes | `2211840` | 0 | true |
| baseline | `ds41_hc_splitk` | global_write_bytes | `36864` | 0 | true |
| baseline | `ds41_hc_splitk` | shared_read_bytes | `17563648` | 0 | true |
| baseline | `ds41_hc_splitk` | shared_write_bytes | `5521408` | 0 | true |
| baseline | `ds41_hc_splitk` | blocks | `128` | 0 | true |
| baseline | `ds41_hc_splitk` | sum_launch_max_block_lane_sequence_span | `4445` | 0 | true |
| baseline | `ds41_hc_splitk_reduce` | fp32_flops | `9216` | 0 | true |
| baseline | `ds41_hc_splitk_reduce` | global_read_bytes | `36864` | 0 | true |
| baseline | `ds41_hc_splitk_reduce` | global_write_bytes | `288` | 0 | true |
| baseline | `ds41_hc_splitk_reduce` | blocks | `1` | 0 | true |
| baseline | `ds41_hc_splitk_reduce` | sum_launch_max_block_lane_sequence_span | `745` | 0 | true |
