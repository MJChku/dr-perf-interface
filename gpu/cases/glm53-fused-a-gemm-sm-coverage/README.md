# glm53-fused-a-gemm-sm-coverage

**Contribution: FOUND/POTENTIAL.** Kernel: TRT-LLM dsv3FusedAGemm (vLLM csrc copy, src/lib_fused_a_gemm.cu).

Small-M BF16 GEMM for the attention prefix (6144->2624/2752) and o_proj (8192->6144); 156 calls/step, ~1.1 ms exclusive.

## Declared PCVs

M (hd_out) 64..256, K (hd_in) 512..2048, tile_m 16/32, tokens 1..4 (small instantiations; production extrapolated)

## DrPerf finding

Exact: global_unique_read_bytes = 2*M*K + 2*tokens*K (weights and activations once), tensor_float_flops = 16*M*K for any tokens (8 token columns always computed), blocks = M/tile_m with one CTA per SM. At production: prefix 6144->2752 at tile 32 launches 86 CTAs on 148 SMs with 393 KB of weights each, 1.72x the balanced time; o_proj 384 CTAs = 2.6 waves (1.16x).

## Fix

Proposed: split-K or tile choice giving ~148 equal CTAs for the prefix shapes (model: ~8 -> ~4.8 us/call).

Fix status: **Open.**

## Measured timing (not from DrPerf)

12.5 us/call inclusive average over both uses (b101).

## Credit

DrPerf (CTA counts and per-CTA bytes); the latency model does not yet see this kernel's cp.async/mbarrier pipeline (0 exposed loads).

## Reproduce

- see gpu/examples/fused_a_gemm (results/small2)

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101

## Copied sources

- `src/kernels.cu` from /home/ubuntu/drperf/gpu/examples/fused_a_gemm/kernels.cu
- `src/driver.cc` from /home/ubuntu/drperf/gpu/examples/fused_a_gemm/driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
