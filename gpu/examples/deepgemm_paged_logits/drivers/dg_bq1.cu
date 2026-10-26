#include <deep_gemm/impls/sm100_mqa_logits.cuh>
using E4M3=cutlass::float_e4m3_t;
// Diagnostic specialization, not an exported pinned DeepGEMM paged entry.
// Valid only for kTokensPerRequest=1: the original scheduler's BLOCK_Q=4
// and a BLOCK_Q=1 scheduler have the same single-token spans and tile starts.
extern "C" __global__ __launch_bounds__(384,1)
void dg_paged_bq1(const uint32_t n,const uint32_t stride,const uint32_t btstride,
 const uint32_t*lens,float*out,const uint32_t*bt,const uint32_t*idx,const uint32_t*meta,
 const __grid_constant__ cute::TmaDescriptor tq,const __grid_constant__ cute::TmaDescriptor tsq,
 const __grid_constant__ cute::TmaDescriptor tk,const __grid_constant__ cute::TmaDescriptor tsk,
 const __grid_constant__ cute::TmaDescriptor tw){
 const auto sched=[&](const uint32_t&sm,uint32_t*,uint32_t*){
 return deep_gemm::sched::SM100PagedMQALogitsScheduler<1,true,false,32,256,128,16>(sm,lens,meta,idx,bt,btstride,n);};
 deep_gemm::sm100_mqa_logits_core_impl<32,128,false,false,1,256,3,5,0,128,256,E4M3,float,float,decltype(sched),2>(stride,out,tq,tsq,tk,tsk,tw,sched);
}
extern "C" __global__ void dg_smem_size(int*out){*out=sizeof(deep_gemm::layout::MQALogitsSharedStorage<32,128,false,1,256,3,5,3,E4M3,float>);}
