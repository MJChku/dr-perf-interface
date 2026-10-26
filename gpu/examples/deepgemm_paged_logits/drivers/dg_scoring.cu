#include <deep_gemm/impls/sm100_mqa_logits.cuh>
using E4M3=cutlass::float_e4m3_t;
auto dg_paged_kernel=&deep_gemm::sm100_paged_mqa_logits<1,32,128,128,false,true,false,3,5,256,16,128,256,E4M3,float,float>;
extern "C" __global__ void dg_smem_size(int*out){*out=sizeof(deep_gemm::layout::MQALogitsSharedStorage<32,128,false,4,256,3,5,3,E4M3,float>);}
