#include "decode_head64_kernel_glm53.cuh"
namespace sm100::decode::head64 {
template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V32>(const SparseAttnDecodeParams &params);
}
extern "C" __global__ void flash_shared(int* p){*p=sizeof(sm100::decode::head64::KernelTemplate<ModelType::V32>::SharedMemoryPlan);}
using FK=sm100::decode::head64::KernelTemplate<ModelType::V32>;
using FS=decltype(cute::make_shape(int{},int{},int{},int{}));
using FQ=decltype(cute::make_tma_copy(cute::SM90_TMA_LOAD{},cute::make_tensor(cute::make_gmem_ptr((bf16*)nullptr),cute::make_layout(FS{},cute::make_stride(int{},cute::_1{},int{},int{}))),FK::SmemLayoutQ_SW128{}));
using FO=decltype(cute::make_tma_copy(cute::SM90_TMA_STORE{},cute::make_tensor(cute::make_gmem_ptr((bf16*)nullptr),cute::make_layout(FS{},cute::make_stride(int{},cute::_1{},int{},int{}))),FK::SmemLayoutOBuf_TMA{}));
using FT=FK::TmaParams<FS,FQ,FS,FO>;
extern "C" __global__ void flash_tma_init(unsigned char*out,const CUtensorMap*maps,int b,int s_q){
 FT t{};t.shape_Q_SW128=cute::make_shape(64,576,s_q,b);t.shape_O=cute::make_shape(64,512,s_q,b);
 *const_cast<CUtensorMap*>(t.tma_Q_SW128.get_tma_descriptor())=maps[0];
 *const_cast<CUtensorMap*>(t.tma_O.get_tma_descriptor())=maps[1];
 t.tensor_map_q_sw64=maps[2];t.tensor_map_kv_nope=maps[3];t.tensor_map_kv_rope=maps[4];
 for(unsigned i=threadIdx.x;i<sizeof(t);i+=blockDim.x)out[i]=((unsigned char*)&t)[i];
}
