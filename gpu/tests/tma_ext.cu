// GX extended TMA device cases; C entry names and inline reduction wrappers.
// TMA and bulk-copy forms beyond tiled load/store, under GX strict PTX:
//   cp.reduce.async.bulk.tensor (add.f32, min.s32) with partially
//   out-of-bounds boxes, tile::gather4 loads and tile::scatter4 stores,
//   device-side tensormap.replace + tensormap.cp_fenceproxy, and plain
//   cp.reduce.async.bulk (add.u32, max.s32) and cp.async.bulk .cp_mask stores,
//   and a null-address/zero-stride placeholder map (CUTLASS grouped GEMM)
//   completed on the device.
// Host references follow the PTX ISA / cuda.h definitions. Exit 0 = all pass.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ void bar_init(uint64_t* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(bar)) : "memory");
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ void bar_expect(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ void bar_wait(uint64_t* bar, unsigned parity) {
  asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W%=; }"
               :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ void bulk_done() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

// 1. Tensor reductions: box (32 x 8) of `src` reduced into the tensor at (x, y).
template <bool F32>
__device__ __forceinline__ void reduce_impl(const CUtensorMap& map, const uint32_t* src, int x, int y) {
  __shared__ alignas(128) uint32_t s[32 * 8];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) s[i] = src[i];
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    if (F32)
      asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];"
                   :: "l"(&map), "r"(x), "r"(y), "r"(smem_u32(s)) : "memory");
    else
      asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.bulk_group [%0, {%1, %2}], [%3];"
                   :: "l"(&map), "r"(x), "r"(y), "r"(smem_u32(s)) : "memory");
    bulk_done();
  }
}

// 2. gather4 (load 4 rows of 64 u16 starting at column x) and scatter4 (store them to rows q0..q3).
extern "C" __global__ void k_gather_scatter(const __grid_constant__ CUtensorMap in, const __grid_constant__ CUtensorMap outmap,
                                 int x, int r0, int r1, int r2, int r3, int q0, int q1, int q2, int q3, uint16_t* dump) {
  __shared__ alignas(128) uint16_t s[4 * 64];
  __shared__ alignas(8) uint64_t bar;
  if (threadIdx.x == 0) bar_init(&bar);
  __syncthreads();
  if (threadIdx.x == 0) {
    bar_expect(&bar, sizeof s);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 :: "r"(smem_u32(s)), "l"(&in), "r"(x), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(smem_u32(&bar)) : "memory");
  }
  bar_wait(&bar, 0);
  for (int i = threadIdx.x; i < 256; i += blockDim.x) dump[i] = s[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
                 :: "l"(&outmap), "r"(x), "r"(q0), "r"(q1), "r"(q2), "r"(q3), "r"(smem_u32(s)) : "memory");
    bulk_done();
  }
}

// 3. Device-side tensor-map edit: copy the parameter map to shared memory,
//    point it at another buffer with a smaller height, publish it to a
//    global slot, then TMA-load a box through the slot.
extern "C" __global__ void k_replace(const __grid_constant__ CUtensorMap map, void* other, uint32_t height, CUtensorMap* slot,
                          uint32_t* dump) {
  __shared__ alignas(128) CUtensorMap sm;
  __shared__ alignas(128) uint32_t s[32 * 8];
  __shared__ alignas(8) uint64_t bar;
  if (threadIdx.x < 8) reinterpret_cast<uint4*>(&sm)[threadIdx.x] = reinterpret_cast<const uint4*>(&map)[threadIdx.x];
  if (threadIdx.x == 0) bar_init(&bar);
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint64_t sa = smem_u32(&sm);
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "l"(sa), "l"(other) : "memory");
    asm volatile("tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], 1, %1;" :: "l"(sa), "r"(height) : "memory");
  }
  __syncthreads();
  if (threadIdx.x < 32)
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
                 :: "l"(slot), "r"(smem_u32(&sm)) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(slot) : "memory");
    bar_expect(&bar, sizeof s);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_u32(s)), "l"(slot), "r"(0), "r"(4), "r"(smem_u32(&bar)) : "memory");
  }
  bar_wait(&bar, 0);
  for (int i = threadIdx.x; i < 256; i += blockDim.x) dump[i] = s[i];
}

// 5. CUTLASS grouped-GEMM pattern: the host encodes a placeholder map (null
//    address, zero stride); the kernel sets the real address and row stride
//    before the first load.
extern "C" __global__ void k_placeholder(const __grid_constant__ CUtensorMap map, void* data, uint64_t row_bytes,
                              CUtensorMap* slot, uint32_t* dump) {
  __shared__ alignas(128) CUtensorMap sm;
  __shared__ alignas(128) uint32_t s[32 * 8];
  __shared__ alignas(8) uint64_t bar;
  if (threadIdx.x < 8) reinterpret_cast<uint4*>(&sm)[threadIdx.x] = reinterpret_cast<const uint4*>(&map)[threadIdx.x];
  if (threadIdx.x == 0) bar_init(&bar);
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint64_t sa = smem_u32(&sm);
    asm volatile("tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;" :: "l"(sa), "l"(data) : "memory");
    asm volatile("tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], 0, %1;" :: "l"(sa), "l"(row_bytes) : "memory");
  }
  __syncthreads();
  if (threadIdx.x < 32)
    asm volatile("tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
                 :: "l"(slot), "r"(smem_u32(&sm)) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" :: "l"(slot) : "memory");
    bar_expect(&bar, sizeof s);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_u32(s)), "l"(slot), "r"(0), "r"(2), "r"(smem_u32(&bar)) : "memory");
  }
  bar_wait(&bar, 0);
  for (int i = threadIdx.x; i < 256; i += blockDim.x) dump[i] = s[i];
}

// 4. Plain bulk reductions and a byte-masked bulk store.
extern "C" __global__ void k_bulk(const uint32_t* src, uint32_t* add_dst, int32_t* max_dst, uint8_t* mask_dst, uint32_t byte_mask) {
  __shared__ alignas(128) uint32_t s[64];
  for (int i = threadIdx.x; i < 64; i += blockDim.x) s[i] = src[i];
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], 256;" :: "l"(add_dst), "r"(smem_u32(s)) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [%0], [%1], 256;" :: "l"(max_dst), "r"(smem_u32(s)) : "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group.cp_mask [%0], [%1], 256, %2;"
                 :: "l"(mask_dst), "r"(smem_u32(s)), "h"((uint16_t)byte_mask) : "memory");
    bulk_done();
  }
}


extern "C" __global__ void k_reduce_f32(const __grid_constant__ CUtensorMap map,const uint32_t* src,int x,int y){reduce_impl<true>(map,src,x,y);}
extern "C" __global__ void k_reduce_s32(const __grid_constant__ CUtensorMap map,const uint32_t* src,int x,int y){reduce_impl<false>(map,src,x,y);}

// src-size and ignore-source forms, including complete zero-fill.
extern "C" __global__ void k_async_tail(const uint8_t* input,uint8_t* output,int source_bytes){
 __shared__ alignas(16) uint8_t s[48];
 if(threadIdx.x==0){
  for(int i=0;i<48;++i)s[i]=0xa5;
  auto a=smem_u32(s),b=smem_u32(s+16),c=smem_u32(s+32);
  asm volatile("cp.async.ca.shared.global [%0],[%1],16,4;"::"r"(a),"l"(input):"memory");
  asm volatile("{.reg .pred p;setp.ne.u32 p,%2,0;cp.async.ca.shared.global [%0],[%1],16,p;}"::"r"(b),"l"(input),"r"(source_bytes):"memory");
  asm volatile("cp.async.ca.shared.global [%0],[%1],16,%2;"::"r"(c),"l"(input),"r"(source_bytes):"memory");
  asm volatile("cp.async.commit_group;cp.async.wait_all;":::"memory");
  for(int i=0;i<48;++i)output[i]=s[i];
 }
}
