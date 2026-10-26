# glm53-deepgemm-paged-logits-bq1

**Contribution: FOUND/NEGATIVE.** Kernel: DeepGEMM sm100_paged_mqa_logits (indexer scoring, pinned 8b1392b9).

FP8 index-key scoring over ~100K history per request on 21 full-indexer layers; 12.8 us/call in graph (b101).

## Declared PCVs

padded tokens L, splits S, pages, heads 1..32, dim 8..128, CTAs, in-bounds query rows

## DrPerf finding

Exact: 132 B K+scale reads per padded token, 4 B logits writes; tensor_float_flops = 32768*L, i.e. 4x the useful work, because BLOCK_Q = 128/heads = 4 at H32 while decode has one query per request. Critical-lane chain (latency model with tensor pipe, source lines via -lineinfo): the page loop is an MMA <-> epilogue ping-pong - the epilogue waits at sm100_mqa_logits.cuh:423 on the MMA issued at :334 [tensor pipe], the MMA warp waits for the epilogue to free the accumulator; with BLOCK_Q=1 the MMA wait vanishes (path -14% at 2 pages/request, per-page cost 375 -> 219 units). Every warp role also starts with three dependent scheduler loads (sm100_paged_mqa_logits.cuh:46 ceil_div on context_lens).

## Fix

BLOCK_Q=1 instantiation of the existing core (dg_bq1.cu): 8192 tensor FLOPs per token (exactly 4x less), identical KV/logit bytes; 40/40 drperf reference states pass.

Fix status: **Measured in graph: -2%, not kept (HBM-bound in the step).**

## Measured timing (not from DrPerf)

B300 standalone b112 (bitwise, 400222 logits): batch 4 hot 12.48 -> 10.78 us (1.16x), cold HBM 17.06 -> 16.86 us (1.01x); batch 1 hot 10.14 -> 8.58 us. In graph (b113 harness, GLM53_INDEXER_PAGED_BQ1): 12.83 -> 12.54 us/call (-2%), step unchanged, hot tokens exact - the step runs it in the HBM-bound regime, so the 4x tensor-work cut and the modeled MMA/epilogue chain are hidden behind KV traffic.

## Credit

DrPerf found the 4x tensor work, the ping-pong chain and checked the prototype; GPU confirmed the L2-resident gain.

## Reproduce

- DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu -- build/dg_bq{4,1}_driver build/dg_bq{4,1}.so
- python3 gpu/tools/latency_report.py results/lat_bq4_li --region "deepgemm*" --stalls 6

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b112
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b113

## Copied sources

- `src/dg_bq1.cu` from /home/ubuntu/drperf/gpu/examples/deepgemm_paged_logits/drivers/dg_bq1.cu
- `src/dg_scoring.cu` from /home/ubuntu/drperf/gpu/examples/deepgemm_paged_logits/drivers/dg_scoring.cu
- `src/dg_bq4_driver.cc` from /home/ubuntu/drperf/gpu/examples/deepgemm_paged_logits/drivers/dg_bq4_driver.cc
- `src/dg_bq1_driver.cc` from /home/ubuntu/drperf/gpu/examples/deepgemm_paged_logits/drivers/dg_bq1_driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
