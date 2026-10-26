// FlashInfer 0.6.18 routingCustom::routingIndicesBlockKernel (kernel body copied
// verbatim from csrc/fused_moe/trtllm_backend/trtllm_fused_moe_routing_custom.cu
// lines 1-298, host launch code omitted) instantiated with the exact parameters
// the GLM-5.3 lib_moe decode path and vLLM 385dce36 run (b101/b54 traces).
#include "routing_block_kernel.cuh"
using GlmRoutingParams = moe::dev::routing::routingCustom::KernelParams<float, __nv_bfloat16, 256, 8,
    moe::dev::routing::TopKExpertSelect<moe::dev::routing::SigmoidBiasPreprocess,
                                        moe::dev::routing::ScaledSumNormalizePostprocess>>;
template __global__ void moe::dev::routing::routingCustom::routingIndicesBlockKernel<GlmRoutingParams>(GlmRoutingParams);
