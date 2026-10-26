// TRT-LLM dsv3FusedAGemm (vLLM csrc copy used by the GLM-5.3 stack, src/lib_fused_a_gemm.cu
// at integration-18) instantiated at small shapes for drperf: learn scaling in
// K (hd_in), M (hd_out), tile_m and tokens, then extrapolate to production.
#include "/home/ubuntu/gen-glm53-integration/src/lib_fused_a_gemm.cu"
#define STAGES(K, TM, TN) ((K / 256) > (1024 * 192 / ((TM + TN) * 256 * 2)) ? (1024 * 192 / ((TM + TN) * 256 * 2)) : (K / 256))
#define INST(M, K, TM) template __global__ void fused_a_gemm_kernel<1, M, K, TM, 8, 256, STAGES(K, TM, 8)>(bf16_t*, bf16_t const*, bf16_t const*, int, int);
INST(64, 512, 16) INST(64, 1024, 16) INST(64, 2048, 16)
INST(128, 512, 16) INST(128, 1024, 16) INST(128, 2048, 16)
INST(256, 512, 16) INST(256, 1024, 16)
INST(64, 512, 32) INST(128, 1024, 32) INST(256, 2048, 32)
