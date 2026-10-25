#pragma once
#include "../include/drperf_gpu.h"
#include <light/kernel_fiber.hh>
#include <light/named_barrier.hh>
#include <stdexcept>

struct ThreadContext {
    dim3 threadIdx, blockIdx, blockDim, gridDim;
    uint64_t instructions=0, flops=0, bytes=0;
};
inline ThreadContext* gx_thread_context() {
    auto* f = gx::KernelFibers::active();
    if (!f || !f->data) throw std::runtime_error("GPU helper outside a lane");
    return static_cast<ThreadContext*>(f->data);
}
extern thread_local NamedBarrierManager* current_named_barrier_manager;
int get_block_id();
int get_warp_id_in_block();
int get_lane_id();
void record_barrier_call();
void gpu_barrier_wait(NamedBarrierManager::BarrierIface*);
#define BARRIER_WAIT(b) gpu_barrier_wait((b).get())
struct GxBarScope { GxBarScope(int,int) {} };
