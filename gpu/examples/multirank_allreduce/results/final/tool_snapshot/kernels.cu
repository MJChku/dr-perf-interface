// Original archived device bodies; only explicit template instantiations added.
#include "frozen/flashinfer/comm/trtllm_allreduce_fusion.cuh"
#include "frozen/lib_ar_residual_fused.cuh"
using namespace flashinfer::trtllm_allreduce_fusion;
#define INSTANTIATE(PAT, RANKS) \
 template __global__ void flashinfer::trtllm_allreduce_fusion::allreduce_fusion_kernel_oneshot_lamport<AllReduceFusionPattern::PAT, __nv_bfloat16, RANKS, true, true>(AllReduceFusionParams<__nv_bfloat16>);
INSTANTIATE(kAllReduce, 2)
INSTANTIATE(kARResidualRMSNorm, 2)
INSTANTIATE(kAllReduce, 4)
INSTANTIATE(kARResidualRMSNorm, 4)
