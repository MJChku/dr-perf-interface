// Original FlashMLA combine (from flash_aux.cu) and the GLM-5.3 wide variant.
#include "flash_aux.cu"
#include "/home/ubuntu/gen-glm53-decode-step/third_party/flashmla_glm53/combine_glm53.cuh"
template __global__ void glm53_flashmla::combine_wide_kernel<cutlass::bfloat16_t,32,16>(__grid_constant__ const CombineParams);
