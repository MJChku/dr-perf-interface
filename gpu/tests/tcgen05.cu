// Bounded numerical/resource fixture derived from GX tcgen05_util.cuh (MIT).
// This compiles to PTX only; execution is on the standalone CPU runtime.
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef DRPERF_GPU_TEST_DEVICE_REGIONS
#include "drperf_gpu_device.cuh"
#endif
__device__ uint32_t smem(const void*p){return uint32_t(__cvta_generic_to_shared(p));}
__device__ uint64_t desc(uint32_t p){return uint64_t(p>>4)|(8ull<<16)|(16ull<<32)|(1ull<<46);}
extern "C" __global__ void tcgen_bf16(const uint16_t*a,const uint16_t*b,float*out,int repeats){
#ifdef DRPERF_GPU_TEST_DEVICE_REGIONS
 drperf::gpu::DeviceRegion region("bf16_tile",drperf::gpu::DeviceScope::block, {{"repeats",repeats}});
#endif
 __shared__ alignas(128) uint16_t sa[128*16],sb[64*16];
 __shared__ uint32_t slot;__shared__ alignas(8) uint64_t bar;
 for(int i=threadIdx.x;i<128*16;i+=blockDim.x)sa[i]=a[i];
 for(int i=threadIdx.x;i<64*16;i+=blockDim.x)sb[i]=b[i];
 if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(smem(&bar)));
 if(threadIdx.x<32)asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"::"r"(smem(&slot)));
 __syncthreads();asm volatile("tcgen05.fence::after_thread_sync;");
 uint32_t d=slot;
 if(threadIdx.x==0){
  #pragma unroll 1
  for(int i=0;i<repeats;i++){
   uint32_t accumulate=i!=0;
   asm volatile("{.reg .pred p;setp.ne.u32 p,%4,0;tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;}"::"r"(d),"l"(desc(smem(sa))),"l"(desc(smem(sb))),"r"((1u<<4)|(1u<<7)|(1u<<10)|(8u<<17)|(8u<<24)),"r"(accumulate));
  }
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(smem(&bar)));
 }
 asm volatile("{.reg .pred p;W%=:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra W%=;}"::"r"(smem(&bar)));
 asm volatile("tcgen05.fence::after_thread_sync;");
 for(uint32_t c=0;c<64;c+=8){
  uint32_t r[8];uint32_t addr=d+((threadIdx.x/32*32)<<16)+c;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  for(int j=0;j<8;j++)out[(blockIdx.x*128+threadIdx.x)*64+c+j]=__uint_as_float(r[j]);
 }
 __syncthreads();
 if(threadIdx.x<32){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,64;"::"r"(d));asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");}
}

extern "C" __global__ void tcgen_nvfp4(float*out,int repeats){
#ifdef DRPERF_GPU_TEST_DEVICE_REGIONS
 drperf::gpu::DeviceRegion region("nvfp4_tile",drperf::gpu::DeviceScope::block, {{"repeats",repeats}});
#endif
 __shared__ alignas(128) uint8_t sa[128*32],sb[64*32];
 __shared__ uint32_t slot;__shared__ alignas(8) uint64_t bar;
 for(int i=threadIdx.x;i<128*32;i+=blockDim.x)sa[i]=0x22;
 for(int i=threadIdx.x;i<64*32;i+=blockDim.x)sb[i]=0x22;
 if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(smem(&bar)));
 if(threadIdx.x<32)asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"::"r"(smem(&slot)));
 __syncthreads();asm volatile("tcgen05.fence::after_thread_sync;");
 uint32_t d=slot;
 uint32_t sa_addr=d+((threadIdx.x/32*32)<<16)+64,sb_addr=sa_addr+4;
 asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%1,%1,%1};"::"r"(sa_addr),"r"(0x38383838u));
 asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0],{%1,%1,%1,%1};"::"r"(sb_addr),"r"(0x38383838u));
 asm volatile("tcgen05.wait::st.sync.aligned;");__syncthreads();
 asm volatile("tcgen05.fence::after_thread_sync;");
 if(threadIdx.x==0){
  #pragma unroll 1
  for(int i=0;i<repeats;i++){
   uint32_t accumulate=i!=0;
   asm volatile("{.reg .pred p;setp.ne.u32 p,%4,0;tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0],%1,%2,%3,[%5],[%6],p;}"::"r"(d),"l"(desc(smem(sa))),"l"(desc(smem(sb))),"r"((1u<<7)|(1u<<10)|(8u<<17)|(8u<<24)),"r"(accumulate),"r"(d+64),"r"(d+68));
  }
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(smem(&bar)));
 }
 asm volatile("{.reg .pred p;W%=:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra W%=;}"::"r"(smem(&bar)));
 asm volatile("tcgen05.fence::after_thread_sync;");
 for(uint32_t c=0;c<64;c+=8){
  uint32_t r[8];uint32_t addr=d+((threadIdx.x/32*32)<<16)+c;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  for(int j=0;j<8;j++)out[(blockIdx.x*128+threadIdx.x)*64+c+j]=__uint_as_float(r[j]);
 }
 __syncthreads();
 if(threadIdx.x<32){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,128;"::"r"(d));asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");}
}

// Functional mask form executes, but resource model must stay incomplete.
extern "C" __global__ void tcgen_bf16_mask(const uint16_t*a,const uint16_t*b,float*out,int repeats){
 __shared__ alignas(128) uint16_t sa[128*16],sb[64*16];
 __shared__ uint32_t slot;__shared__ alignas(8) uint64_t bar;
 for(int i=threadIdx.x;i<128*16;i+=blockDim.x)sa[i]=a[i];
 for(int i=threadIdx.x;i<64*16;i+=blockDim.x)sb[i]=b[i];
 if(threadIdx.x==0)asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(smem(&bar)));
 if(threadIdx.x<32)asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;"::"r"(smem(&slot)));
 __syncthreads();asm volatile("tcgen05.fence::after_thread_sync;");
 uint32_t d=slot;
 if(threadIdx.x==0){
  #pragma unroll 1
  for(int i=0;i<repeats;i++){
   uint32_t accumulate=i!=0;
   asm volatile("{.reg .pred p;setp.ne.u32 p,%4,0;tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,{%5,%6,%7,%8},p;}"::"r"(d),"l"(desc(smem(sa))),"l"(desc(smem(sb))),"r"((1u<<4)|(1u<<7)|(1u<<10)|(8u<<17)|(8u<<24)),"r"(accumulate),"r"(0u),"r"(0u),"r"(0u),"r"(0u));
  }
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"::"r"(smem(&bar)));
 }
 asm volatile("{.reg .pred p;W%=:mbarrier.try_wait.parity.shared::cta.b64 p,[%0],0;@!p bra W%=;}"::"r"(smem(&bar)));
 asm volatile("tcgen05.fence::after_thread_sync;");
 for(uint32_t c=0;c<64;c+=8){
  uint32_t r[8];uint32_t addr=d+((threadIdx.x/32*32)<<16)+c;
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7},[%8];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]):"r"(addr));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  for(int j=0;j<8;j++)out[(blockIdx.x*128+threadIdx.x)*64+c+j]=__uint_as_float(r[j]);
 }
 __syncthreads();
 if(threadIdx.x<32){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,64;"::"r"(d));asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");}
}


extern "C" __global__ void tcgen_group2(){
 __shared__ uint32_t slot;
 if(threadIdx.x<32)asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0],64;"::"r"(smem(&slot)));
}
