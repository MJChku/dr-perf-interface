#pragma once
#include "../include/drperf_gpu.h"
#include <light/kernel_fiber.hh>
#include <light/named_barrier.hh>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "latency.h"

struct ThreadContext {
    dim3 threadIdx, blockIdx, blockDim, gridDim;
    uint64_t instructions=0, flops=0, bytes=0;
    DrperfGpuReady latency_clock;
    // Basic-block schedule (DRPERF_GPU_LATENCY_SCHEDULE=block): issue floor at the
    // last control boundary and instructions issued since.
    uint64_t latency_block_floor=0, latency_block_issued=0;
    bool latency_load_hit=false;  // set by the memory hook for the next latency issue
    DrperfGpuReady latency_async;  // completion of this lane's issued cp.async copies
    std::string last_site;
    // Critical-lane breakdown: the lane clock advances by one per issue
    // (latency_issue) or waits; waits beyond the issue slot are charged to the
    // waiting site (dependency stalls and barrier/mbarrier/async joins), so
    // latency_issue + sum(latency_stalls) == latency_clock.time exactly.
    uint64_t latency_issue=0;
    std::unordered_map<std::string,uint64_t> latency_stalls;
    // Failed mbarrier try_wait loop: state after the first failed probe. On the
    // eventual success the spin's issue slots are rolled back (hardware try_wait
    // suspends the warp) and the lane joins the phase's completion instead.
    struct Spin { uint64_t address=0, phase=0, failures=0, instructions=0, issue=0; DrperfGpuReady clock;
                  std::vector<std::pair<std::string,uint64_t>> stalls; } latency_spin;  // stalls: undo log
    DrperfGpuReady* const* latency_last_dst=nullptr; size_t latency_last_ndst=0;  // last issue's results
    uint64_t linear_block=0, cluster_barrier_token=0;
    NamedBarrierManager* barriers=nullptr;
};
inline ThreadContext* gx_thread_context() {
    auto* f = gx::KernelFibers::active();
    if (!f || !f->data) throw std::runtime_error("GPU helper outside a lane");
    return static_cast<ThreadContext*>(f->data);
}
#define current_named_barrier_manager (gx_thread_context()->barriers)
int get_block_id();
int get_warp_id_in_block();
int get_lane_id();
void record_barrier_call();
void gpu_barrier_wait(NamedBarrierManager::BarrierIface*);
#define BARRIER_WAIT(b) gpu_barrier_wait((b).get())
struct GxBarScope { GxBarScope(int,int) {} };
