# glm53-moe-routing-block-latency

**Contribution: EXPLAINED (corrected).** Kernel: FlashInfer 0.6.18 routingCustom::routingIndicesBlockKernel<float, bf16, 256, 8, SigmoidBias + ScaledSumNormalize>.

DeepSeekV3-style MoE routing for <=4 tokens (top-8 of 256 experts, permutation/histogram outputs); 75 calls per decode step; identical kernel and launch (1x256) in vLLM.

## Declared PCVs

tokens 1..4, local experts 128/256, tile tokens 8/16, TK = tokens*8

## DrPerf finding

One CTA of 256 lanes. Exact: global_read_bytes = 260*TK, unique = 128*TK + 1024, fp32_flops = 649*TK, block barriers = 9 per launch (2304 lane arrivals), collective arrivals 84*TK + 608. The first latency model reported a fixed chain of 17 dependent global loads (10344..10596 units); with the critical-lane chain and -lineinfo source lines this turned out to be a model artifact: the loads are guarded by short forward branches (RoutingKernel.cuh:127 score loads behind idx<numExperts, RoutingCustomPolicy.cuh:234 bias loads in a dtype switch) that ptxas predicates. With if-conversion modeled the path is 2763 units (one exposed load plus barriers): the kernel is short on its own, consistent with vLLM's 4.5 us, so our 22 us/call in graph (1.57 ms/step) is queuing - the shared-expert GEMMs are enqueued first on the side stream and the 1-CTA routing kernel waits for an SM.

## Fix

Scheduling (gen-glm53-decode-step): GLM53_DECODE_ROUTED_FIRST enqueues the routed path before the shared expert; GLM53_DECODE_ROUTED_PRIORITY gives the main decode streams the greatest priority, the shared-expert stream the least, and instantiates graphs with cudaGraphInstantiateFlagUseNodePriority (1654 high / 300 low priority kernel nodes captured).

Fix status: **Priority variant merge-ready (token-exact).**

## Measured timing (not from DrPerf)

b114 B300 harness: routing 23.0 -> 17.0 us/call (routed first) / 16.6 us (priority); step -0.22..-0.24 ms (routed first) and -0.26..-0.32 ms (priority) uniform, -0.24..-0.32 ms hot; the deprioritized shared down GEMM slows 5.3 -> 24.9 us off the critical path. The rest of the gap to vLLM 4.5 us is likely PDL early launch (the kernel waits on its predecessor inside its CUPTI interval).

## Credit

DrPerf explained why the kernel has a fixed ~latency floor; the contention with the side stream comes from the CUPTI timeline. Tool gaps found: drperf_gpu.h dim3 clashed with CUDA headers in nvcc-compiled drivers (fixed); latency metrics are per-launch diagnostics without per-site attribution of exposed loads.

## Reproduce

- see gpu/examples/moe_routing (results/run1); latency: python3 gpu/tools/latency_report.py results/run1

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b54
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b114

## Copied sources

- `src/kernel.cu` from /home/ubuntu/drperf/gpu/examples/moe_routing/kernel.cu
- `src/driver.cu` from /home/ubuntu/drperf/gpu/examples/moe_routing/driver.cu
- `src/routing_block_kernel.cuh` from /home/ubuntu/drperf/gpu/examples/moe_routing/routing_block_kernel.cuh

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
