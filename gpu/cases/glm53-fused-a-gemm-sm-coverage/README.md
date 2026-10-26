# glm53-fused-a-gemm-sm-coverage

**Contribution: NEGATIVE/FOUND.** Kernel: TRT-LLM dsv3FusedAGemm (vLLM csrc copy, src/lib_fused_a_gemm.cu).

Small-M BF16 GEMM for the attention prefix (6144->2624/2752, 78 calls/step) and q_b (2048->8192/12288, 78 calls/step).

## Declared PCVs

M (hd_out) 64..256, K (hd_in) 512..2048, tile_m 16/32, tokens 1..4 (small instantiations; production extrapolated)

## DrPerf finding

Exact: global_unique_read_bytes = 2*M*K + 2*tokens*K, tensor_float_flops = 16*M*K for any tokens, blocks = M/tile_m. Hypothesis from CTA counts: the prefix shapes launch 82/86 CTAs on 148 SMs (one CTA/SM), so they should be 1.72x slower than balanced. A standalone B300 benchmark (HBM-resident weights) refuted it: tile_m 16 (164/172 CTAs) is 0.68-0.97x of the current speed; current 7.3-7.9 us = 4.1-4.4 TB/s. The same sweep found that the hd_in 2048 shapes run faster with 4 pipeline stages instead of 8 (98 KB -> 49 KB shared, 2 -> 4 CTAs/SM, 512/768 CTAs from ~1.7/2.6 waves to one): 2048->8192 9.00 -> 7.65 us, 2048->12288 12.5 -> 11.4 us, bitwise. The new latency_report residency columns (CTAs/SM from shared memory and threads, waves) expose that wave structure directly.

## Fix

GLM53_FUSED_A_K2048_STAGE4 (gen-glm53-decode-step glue-eager-checks e74147cd).

Fix status: **Implemented, token-exact, merge-ready.**

## Measured timing (not from DrPerf)

Standalone B300 (b108): see finding. In-graph (b109 CUPTI): 2048->8192 13.38 -> 12.1 us, 2048->12288 17.4 -> 16.4 us (-0.095 ms/step), harness step -0.05..-0.13 ms uniform, hot tokens exact.

## Credit

DrPerf supplied the exact work interface and the (wrong) coverage hypothesis; the GPU sweep found the stage effect; the residency view in latency_report was added afterwards so DrPerf flags wave structure.

## Reproduce

- see gpu/examples/fused_a_gemm (results/small2); python3 gpu/tools/latency_report.py gpu/examples/fused_a_gemm/results/small2

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b108
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b109

## Copied sources

- `src/kernels.cu` from /home/ubuntu/drperf/gpu/examples/fused_a_gemm/kernels.cu
- `src/driver.cc` from /home/ubuntu/drperf/gpu/examples/fused_a_gemm/driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
