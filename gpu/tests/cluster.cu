// Thread-block cluster semantics under GX strict PTX: cluster ids/ranks,
// barrier.cluster, distributed shared memory (shared::cluster and generic
// mapa), and remote mbarrier arrivals. Exit status 0 means every check passed.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
namespace cg = cooperative_groups;

struct Record { unsigned rank, nrank, cx, cy, cz, peer_cluster, peer_generic, barrier_ok, address_rank; };

__device__ unsigned ld_cluster_u32(unsigned address) {
  unsigned v;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(address));
  return v;
}

extern "C" __global__ void cluster_kernel(Record* out) {
  __shared__ unsigned word;
  __shared__ alignas(8) unsigned long long bar;
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned rank = cluster.block_rank(), n = cluster.num_blocks();
  const unsigned linear = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  if (threadIdx.x == 0) {
    word = 1000 + linear;
    unsigned addr = (unsigned)__cvta_generic_to_shared(&bar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(n));
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  cluster.sync();
  // Every CTA arrives once on rank 0's mbarrier (remote except for rank 0).
  // Hardware shared addresses carry the cluster rank in bits [24,32); clearing
  // them (as CUTLASS clears the peer bit) addresses rank 0's copy.
  if (threadIdx.x == 0) {
    unsigned local = (unsigned)__cvta_generic_to_shared(&bar);
    unsigned remote = local & 0x00FFFFFFu;
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
  }
  unsigned peer = (rank + 1) % n;
  unsigned local = (unsigned)__cvta_generic_to_shared(&word), remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(peer));
  unsigned via_cluster = ld_cluster_u32(remote);
  unsigned via_generic = *cluster.map_shared_rank(&word, peer);
  unsigned barrier_ok = 1;
  if (rank == 0 && threadIdx.x == 0) {
    unsigned addr = (unsigned)__cvta_generic_to_shared(&bar), done = 0;
    for (int spins = 0; !done && spins < 1000000; ++spins)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(addr) : "memory");
    barrier_ok = done;
  }
  cluster.sync();
  if (threadIdx.x == 0) {
    dim3 cid = cluster.block_index();
    out[linear] = Record{rank, n, cid.x, cid.y, cid.z, via_cluster, via_generic, barrier_ok,
                         (unsigned)__cvta_generic_to_shared(&word) >> 24};
  }
}


extern "C" __device__ void gx_ptx_grid_sync();
extern "C" __global__ void grid_kernel(unsigned* scratch,unsigned* output){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=gridDim.x*blockDim.x;
 scratch[i]=i*3+7;gx_ptx_grid_sync();output[i]=scratch[(i+n/2)%n];gx_ptx_grid_sync();
}
extern "C" __global__ void vote_kernel(unsigned* output){
 int a=__syncthreads_or(threadIdx.x%2),b=__syncthreads_and(threadIdx.x<blockDim.x);
 int c=__syncthreads_count(threadIdx.x%2);
 output[blockIdx.x*blockDim.x+threadIdx.x]=a+2*b+4*c;
}
