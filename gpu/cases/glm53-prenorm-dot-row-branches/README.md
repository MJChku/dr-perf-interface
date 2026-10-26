# glm53-prenorm-dot-row-branches

**Contribution: FOUND.** Kernel: GLM prenorm_dot (moe_front_prenorm router CTAs, attention_iw_prenorm).

BF16 activation x FP32 weight-row dots: 256 router CTAs per MoE layer (75/step, ~10.8 us/call) and 32 indexer-weight CTAs (21/step).

## Declared PCVs

rows 1..4 at production width 6144 (front_driver; one launch per width)

## DrPerf finding

Latency model (basic-block schedule) with site attribution: the router CTA critical path at w4 is 11945 units with 32 exposed global loads, one per (activation step, row): the activation loop loads one row, branches on `if (t < m)`, consumes it, then loads the next row, so the four rows' loads never overlap. A first attempt (branch-free `t < m ? load : 0` selects) lowered to predicated loads and did not help in the model; a compile-time row count (switch on m once, template body) removes the per-row branches.

## Fix

GLM53_ENABLE_DECODE_PRENORM_STATIC_ROWS (gen-glm53-decode-step glue-eager-checks d55406d8): bitwise outputs (drperf A/B: packed NVFP4, scales, router logits and indexer weights identical for rows 1..4); model w4 6625 / 6 exposed, constant across widths.

Fix status: **Implemented, token-exact; small measured gain.**

## Measured timing (not from DrPerf)

b110 CUPTI: moe_front_prenorm 10.4 -> 10.0 us/call (-0.03 ms/step), attention_iw_prenorm 11.1 -> 10.7-11.0 us; harness step -0.03..-0.18 ms (noisy); hot tokens exact. The model over-predicts: removing 26 exposed waits saved ~0.4 us, because the activations are L2-resident (re-read by 256 CTAs) and ~8 warps per SM hide much of the per-lane chain - the per-lane model has no warp-level latency hiding.

## Credit

DrPerf latency model + site attribution found it and checked the fix before GPU time.

## Reproduce

- DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu -- build/front_driver build/kernels{,_sr}.so (kernels_sr = -DGLM53_ENABLE_DECODE_PRENORM_STATIC_ROWS)

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b110

## Copied sources

- `src/front_driver.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/front_driver.cc
- `src/ab_prenorm.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/ab_prenorm.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
