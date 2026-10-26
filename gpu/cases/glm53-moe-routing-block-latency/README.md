# glm53-moe-routing-block-latency

**Contribution: EXPLAINED.** Kernel: FlashInfer 0.6.18 routingCustom::routingIndicesBlockKernel<float, bf16, 256, 8, SigmoidBias + ScaledSumNormalize>.

DeepSeekV3-style MoE routing for <=4 tokens (top-8 of 256 experts, permutation/histogram outputs); 75 calls per decode step; identical kernel and launch (1x256) in vLLM.

## Declared PCVs

tokens 1..4, local experts 128/256, tile tokens 8/16, TK = tokens*8

## DrPerf finding

One CTA of 256 lanes. Exact: global_read_bytes = 260*TK, unique = 128*TK + 1024, fp32_flops = 649*TK, block barriers = 9 per launch (2304 lane arrivals), collective arrivals 84*TK + 608. Latency model: critical path 10344..10596 units with 17 exposed global loads per launch, essentially constant in tokens: a fixed serial chain of global round trips and block barriers on a single CTA. It is latency-bound by construction, which explains the ~4.5 us floor seen in vLLM for 4 tokens.

## Fix

No kernel change from DrPerf. The CUPTI timeline shows our instance at 8..15 us when it overlaps the shared-expert GEMMs on the side stream (a single-CTA latency chain sharing SMs with GEMM CTAs); candidate fixes: higher priority for the main stream, or folding routing into the preceding MoE-front kernel.

Fix status: **Open (scheduling, not kernel work).**

## Measured timing (not from DrPerf)

vLLM 4.5 us/call; ours 13.8 us exclusive / 18.4 us inclusive median, 8 us when overlap is small (b101).

## Credit

DrPerf explained why the kernel has a fixed ~latency floor; the contention with the side stream comes from the CUPTI timeline. Tool gaps found: drperf_gpu.h dim3 clashed with CUDA headers in nvcc-compiled drivers (fixed); latency metrics are per-launch diagnostics without per-site attribution of exposed loads.

## Reproduce

- see gpu/examples/moe_routing (results/run1); latency: python3 gpu/tools/latency_report.py results/run1

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b54

## Copied sources

- `src/kernel.cu` from /home/ubuntu/drperf/gpu/examples/moe_routing/kernel.cu
- `src/driver.cu` from /home/ubuntu/drperf/gpu/examples/moe_routing/driver.cu
- `src/routing_block_kernel.cuh` from /home/ubuntu/drperf/gpu/examples/moe_routing/routing_block_kernel.cuh

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
