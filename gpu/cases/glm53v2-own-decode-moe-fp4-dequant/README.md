# glm53v2-own-decode-moe-fp4-dequant

**Contribution: QUANTIFIED/NEGATIVE.** Kernel: Own decode NVFP4 MoE (glm_moe_dec_gemm1_k: routed gemm1 + SwiGLU + NVFP4 requant; glm_moe_dec_gemm2_k: gemm2 + weighted finalize).

Replacement attempt for TRT-LLM-gen fused NVFP4 block-scale MoE in decode (T=4 rows, 8 of 256 experts per row); 78 calls per step.

## Declared PCVs

None declared: one fixture state (T=4, C4 routing). The report flags SINGLE STATE, so no scaling was learned.

## DrPerf finding

gx explain gpu (drperf-gpu) on the gx check fixture: gemm1 21.1M PTX lane instructions for 3.17 MB of requested global reads (6.7 lane instructions per weight byte): 5.05M conversion ops + 6.31M integer ops (E2M1 nibble extraction) + 4.28M fp16 flops (block-scale multiplies) vs 58.7M tensor flops on mma.sync. The FP4->f16 dequant path is issue-bound; block scales cannot be factored out of mma.sync k-tiles (4 lanes hold different blocks).

## Fix

None: matching TRT-LLM-gen needs tcgen05 block-scaled FP4 MMA (kind::mxf4nvf4). Kernel kept behind GLM_MOE_DEC=1, default off (commit f9a0743 in gen-glm53-v2).

Fix status: **Shelved (negative result).**

## Measured timing (not from DrPerf)

Native cold A/B on B300 (agent batch 18, C4 routing, T=4): own 57.2 us vs TRT-LLM-gen 29.9 us per call.

## Credit

Instruction mix and the issue-bound diagnosis: DrPerf GPU resource interface (found by the study agent). Timing: native B300 A/B.

## Reproduce

- cd /home/ubuntu/gen-glm53-v2 && gx explain gpu --gpu B300 -q --out RUN -- build/t_moedec

## Receipts

- receipts-v2/glm53v2-own-decode-moe/drperf-gpu-report.txt
- receipts-v2/glm53v2-own-decode-moe/result.json
- /home/ubuntu/gen-glm53-v2/eval/WORKLOG.md (2026-10-28 17:05)
- /home/ubuntu/gen-glm53-v2/eval/GX-FEEDBACK.md

## Copied sources

- `src/moe_decode.cu` from /home/ubuntu/gen-glm53-v2/src/kernels/moe_decode.cu
- `src/t_moedec.cu` from /home/ubuntu/gen-glm53-v2/src/tests/t_moedec.cu

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
