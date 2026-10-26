#pragma once
#include "../include/drperf_gpu.h"
#include <light/kernel_fiber.hh>
#include <light/named_barrier.hh>
#include <stdexcept>
#include <string>
#include "latency.h"

struct ThreadContext {
    dim3 threadIdx, blockIdx, blockDim, gridDim;
    uint64_t instructions=0, flops=0, bytes=0;
    DrperfGpuReady latency_clock;
    // Basic-block schedule (DRPERF_GPU_LATENCY_SCHEDULE=block): issue floor at the
    // last control boundary and instructions issued since.
    uint64_t latency_block_floor=0, latency_block_issued=0;
    std::string last_site;
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
