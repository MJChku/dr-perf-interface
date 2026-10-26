# ds41-sparse-attention-head-padding

**Contribution: BLOCKED/POTENTIAL.** Kernel: Library padded16-head sparse attention.

Prefill library attention pads8 useful heads to16 before tensor compute. This case belongs to the DeepSeek-V4.1 serving study in `/home/ubuntu/gen-deepseek41`.

## Declared PCVs and inputs

real_heads4/8/16,padded_heads16,selected_keys64/128,dim512,useful_head_keys. `inputs.json` and `driver.cpp` contain the deterministic grid, patterns and independent numerical oracle. Shapes are bounded fixtures; production shapes are explicitly identified separately.

## Resource interface and contribution

Direct library PTX translates but runtime rejects ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 before its scalar oracle can finish. The8→16 head padding is a source/geometry finding; no DrPerf resource interface or real-head coefficient has been measured here.

The actual kernel and complete deterministic six-state grid/oracle are included. Only the expected coverage failure reproduces; tensor_flops=32768*selected_keys is a proposed shape hypothesis to check after instruction coverage is available, not a fitted result.

Before/after fitted interfaces, nonzero coefficients, observed errors, accepted/rejected flags, dependent-PCV warnings and per-state metrics are retained in `results/<variant>/summary.json` (or `.json.gz`) and `report.txt`. A single-state or after-only case does not establish a before/after coefficient delta. Redundant PCVs may produce equivalent coefficient bases: compare evaluated interfaces on the declared grid, rather than interpreting a nonidentifiable coefficient in isolation.

DrPerf reports dynamic resource interfaces. Requested bytes are not HBM traffic or cache transactions. Logical lane span is not hardware latency or an ILP-aware critical path; launched/active lanes are not achieved occupancy. No serving speed claim follows from these counts.

Fix status: **Blocked direct-library execution;8-head/multi-query redesign and resource interface remain proposed.**

## Reproduce

Run `python3 run.py` from this directory; `DRPERF_HOME` or `--toolchain PATH` selects the byte-pinned toolchain. `toolchain.json` pins every relevant runtime/translator/driver file (commit c8ca439f145c015be904b8a459ed0002b6f76394). PTX and all fixture/support source are local; only that declared toolchain and a C++20 compiler are external dependencies. No physical GPU or CUDA runtime is loaded. See `commands.txt`; the exact expanded commands are saved in `results/verification.json`. `run.py` checks all pinned hashes, compiles the local scalar-oracle driver, executes translated PTX and verifies expected interfaces. Coverage blockers are expected failures with numerical status `not_assessed`; proposed cases do not pretend to execute a missing fixture.

Site-level raw captures and generated C++/CPU modules are regenerable and removed after compact reports are written. Files larger than1MB are gzip-compressed. This collection is intentionally small.

## Measured timing (not from DrPerf)

No isolated native timing for this case.

## Full receipts

- [eval/prefill-efficiency.md](/home/ubuntu/gen-deepseek41/eval/prefill-efficiency.md)

## Reproduced fitted interfaces

Nonzero coefficients below are the DrPerf fits at the observed states. `accepted=false` and nonidentifiable PCVs remain visible in the full summary; they are not validated independent coefficients. Exact structural assertions are in `results/interface-checks.json`.

| Variant | Region | Metric | Fitted expression | Max residual | Accepted |
|---|---|---|---|---:|---|
| padded16 | coverage | none | unsupported GPU PTX operation: ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 | — | blocked |
