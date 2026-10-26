// clusterlaunchcontrol (dynamic persistent scheduling) under GX strict PTX.
// Each launched cluster processes its own tile, then repeatedly cancels a
// pending cluster and processes that cluster's tile instead, until no
// cancellation succeeds. Every tile must be processed exactly once, by every
// CTA of the cluster that owns it. Exit 0 = pass.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

extern "C" __global__ void persistent(unsigned* visits, unsigned* steals) {
  __shared__ alignas(16) uint4 response;
  __shared__ alignas(8) uint64_t bar;
  unsigned rank, nrank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(nrank));
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(smem_u32(&bar)));
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  unsigned x = blockIdx.x - rank % nrank, phase = 0, stolen = 0;  // cluster's first CTA
  for (;;) {
    if (threadIdx.x == 0) atomicAdd(&visits[x + rank], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;" :: "r"(smem_u32(&bar)) : "memory");
      if (rank == 0)
        asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
                     :: "r"(smem_u32(&response)), "r"(smem_u32(&bar)) : "memory");
    }
    asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W; }"
                 :: "r"(smem_u32(&bar)), "r"(phase) : "memory");
    phase ^= 1;
    unsigned ok, cx = 0, cy = 0, cz = 0;
    asm volatile("{ .reg .b128 r; .reg .pred p; ld.shared.b128 r, [%4];"
                 " clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, r; selp.u32 %0, 1, 0, p;"
                 " @p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, r; }"
                 : "=r"(ok), "+r"(cx), "+r"(cy), "+r"(cz) : "r"(smem_u32(&response)) : "memory");
    asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
    if (!ok) break;
    x = cx;
    ++stolen;
  }
  if (threadIdx.x == 0) atomicAdd(steals, stolen);
}

