# glm53-flashmla-core-interface

**Contribution: QUANTIFIED.** Kernel: FlashMLA SM100 sparse decode core (sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel<V32>).

Decode attention over 2048 selected tokens per request, split into ~11 partials; 14.4 us/call, 78 calls/step.

## Declared PCVs

w 1..4, K 64/128/256, parts 1..8, tasks, split_tasks, blocks64 = w*K/64

## DrPerf finding

First complete official capture of the unmodified kernel after DrPerf fixes (ld.global.nc with eviction + prefetch-size hints, scalar mul.bf16, mov.b128, div.approx.f32, opcode[operand] lexing, b128 store bytes, weight-stationary MMA with all-zero column mask). Exact fits: tensor_float_flops = 8912896*blocks64 (64 heads for 32 useful = 2x), global_read_bytes = 42240*blocks64 + 12288*parts + 75264*tasks (660 B per selected token; query requested once per task but unique once per request), global_write_bytes = 65536*split_tasks + 65792*tasks (FP32 partial per split). At w4/K2048: 11.6 MB partial round trip vs 5.4 MB KV; 44 of 148 SMs carry work.

## Fix

None yet (H32 specialization or 32-head epilogue proposed).

Fix status: **Open.**

## Measured timing (not from DrPerf)

Core 14.4 us/call in the b101 CUPTI trace (unchanged by the combine fix).

## Credit

Interface and padding counts: DrPerf. Tool gaps fixed in gpu/vendor (Statement.lean, prelude.h, tcgen05.cc).

## Reproduce

- see gpu/examples/flashmla_decode (results/grid_aligned)

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53-codex-audit/NOTES2.md

## Copied sources

- `src/flash_core.cu` from /home/ubuntu/drperf/gpu/examples/flashmla_decode/drivers/flash_core.cu
- `src/flash_core_bounded_driver.cc` from /home/ubuntu/drperf/gpu/examples/flashmla_decode/drivers/flash_core_bounded_driver.cc
- `src/flash_params.h` from /home/ubuntu/drperf/gpu/examples/flashmla_decode/drivers/flash_params.h

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
