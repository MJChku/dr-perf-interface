// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The GX Project Authors
// Extracted from GX cuda_emulator.hh; see ../provenance.json.
#pragma once
#include <light/fiber_barrier.hh>
#include <light/fiber_exchange.hh>
#include <unordered_map>
#include <memory>
extern thread_local int waiting_revive_count;
class NamedBarrierManager {
public:
    // don't pay the cost of yield
    const static bool _use_simple_barrier = false;
public:
    struct BarrierIface {
        virtual ~BarrierIface() = default;
        virtual void wait() = 0;
    };

    struct FiberBarrier final : BarrierIface {
        explicit FiberBarrier(int nThreads) : barrier(nThreads) {}
        void wait() override { 
             waiting_revive_count += 1;
             barrier.wait(); 
             waiting_revive_count -= 1;
        }
        gx::LocalFiberBarrier barrier;
    };

    struct SimpleBarrier final : BarrierIface {
        explicit SimpleBarrier(int nThreads) : expected_threads(nThreads) {}
        void wait() override {
            const int gen = generation;
            const int c = ++count;
            if (c == expected_threads) {
                count = 0;
                generation = gen + 1;
                return;
            }
            waiting_revive_count += 1;
            while (generation == gen) {
                gx::kernel_fiber_yield();
            }
            waiting_revive_count -= 1;
        }
        const int expected_threads;
        int count = 0;
        int generation = 0;
    };

        struct BarrierData {
        std::unique_ptr<BarrierIface> barrier;
        std::vector<bool> votes;
        std::vector<uint32_t> vals;  // per-lane exchange slots (gx_shfl_sync)
        int vote_count = 0;
        bool result = false; // For vote results
        unsigned barrier_result_uint = 0; // For __ballot_sync results
        std::vector<uint32_t> shuffle_values;
        int expected_threads;
        
        BarrierData(int nThreads) : expected_threads(nThreads) {
            if (_use_simple_barrier) {
                barrier = std::make_unique<SimpleBarrier>(nThreads);
            } else {
                barrier = std::make_unique<FiberBarrier>(nThreads);
            }
        }
    };
    
    // Semantic block scan: all CUDA lanes participate; no application memory
    // is bypassed. Keep scratch per CTA rather than per emulated thread.
    struct BlockScanData {
        struct Arrival {
            gx::FiberRef fiber;
            uint64_t* result;
            uint32_t value;
            bool exclusive;
        };
        std::vector<Arrival> lanes;
        unsigned arrived = 0;
        explicit BlockScanData(int threads) : lanes(threads) {}
        uint64_t scan(unsigned lane, uint32_t value, bool exclusive) {
            uint64_t result;
            auto self = gx::FiberRef::active();
            lanes[lane] = {self, &result, value, exclusive};
            if (++arrived != lanes.size()) {
                self.suspend();
            } else {
                uint32_t total = 0;
                for (const auto& entry : lanes) total += entry.value;
                uint32_t prefix = 0;
                for (auto& entry : lanes) {
                    prefix += entry.value;
                    *entry.result = (uint64_t(total) << 32) |
                        uint32_t(prefix - (entry.exclusive ? entry.value : 0));
                }
                arrived = 0;
                // Publish into caller-owned stacks before any caller resumes.
                // schedule() only queues, so fast reentry cannot clobber results.
                for (unsigned i = 0; i < lanes.size(); ++i)
                    if (i != lane) self.schedule(lanes[i].fiber);
            }
            return result;
        }
    };
private:
    std::unordered_map<int, std::unique_ptr<BarrierData>> barriers_;
    // Consecutive CUDA lanes normally arrive at the same barrier. Keep the
    // lookup local to this launch/pthread, not in every emulated lane.
    int last_barrier_name_ = 0;
    BarrierData* last_barrier_ = nullptr;
    std::unordered_map<int, std::unique_ptr<BlockScanData>> block_scans_;
    std::unordered_map<uint64_t, std::unique_ptr<gx::LocalWarpExchange>> warp_exchanges_;
    std::unordered_map<uint64_t, std::unique_ptr<gx::WarpgroupRendezvous>> warpgroup_rendezvous_;
    // bar.warp.sync rendezvous per (block, warp, membermask) and the lanes of
    // each (block, warp) that have exited the kernel.
    struct WarpSync {
        uint32_t arrived = 0;
        std::array<gx::FiberRef, 32> fibers{};
    };
    std::unordered_map<uint64_t, WarpSync> warp_syncs_;
    std::unordered_map<uint64_t, uint32_t> exited_lanes_;
    uint64_t last_exit_key_ = ~uint64_t(0);
    uint32_t* last_exit_ = nullptr;   // element references survive rehashing
    unsigned pending_warp_syncs_ = 0;
    std::unique_ptr<BarrierData> grid_barrier_;
    // std::mutex manager_mutex_;
    int default_thread_count_;
    
public:
    NamedBarrierManager(int default_threads) : default_thread_count_(default_threads) {}
    
    BarrierData* get_barrier(int name, int nThreads = 0) {
        if (last_barrier_ && last_barrier_name_ == name) return last_barrier_;
        // std::lock_guard<std::mutex> lock(manager_mutex_);
        auto entry = barriers_.find(name);
        if (entry == barriers_.end()) {
            int thread_count = nThreads > 0 ? nThreads : default_thread_count_;
            entry = barriers_.emplace(name, std::make_unique<BarrierData>(thread_count)).first;
        }
        last_barrier_name_ = name;
        return last_barrier_ = entry->second.get();
    }
    
    void clear() {
        last_barrier_ = nullptr;
        // std::lock_guard<std::mutex> lock(manager_mutex_);
        barriers_.clear();
        block_scans_.clear();
        warp_exchanges_.clear();
        warpgroup_rendezvous_.clear();
        warp_syncs_.clear();
        exited_lanes_.clear();
        last_exit_key_ = ~uint64_t(0);
        last_exit_ = nullptr;
        pending_warp_syncs_ = 0;
        grid_barrier_.reset();
    }

    BlockScanData* get_block_scan(int block, int threads) {
        auto& scan = block_scans_[block];
        if (!scan) scan = std::make_unique<BlockScanData>(threads);
        if (scan->lanes.size() != unsigned(threads))
            throw std::runtime_error("block scan participant count changed within launch");
        return scan.get();
    }

    BarrierData* get_grid_barrier(int participants) {
        if (!grid_barrier_) grid_barrier_ = std::make_unique<BarrierData>(participants);
        if (grid_barrier_->expected_threads != participants)
            throw std::runtime_error("grid barrier participant count changed within launch");
        return grid_barrier_.get();
    }

    // bar.warp.sync: wait until every lane of `mask` has arrived or exited.
    // Exited lanes count as arrived, as on sm_70+ hardware, whose warp
    // barrier only waits for lanes that can still execute it (kernels such as
    // DeepGEMM's scheduler let some lanes leave a loop and exit before the
    // others' final __syncwarp). `mask` is already limited to the warp's
    // population and includes the calling lane.
    void warp_sync(long block, unsigned warp, unsigned lane, uint32_t mask) {
        const uint64_t warp_key = (uint64_t(block) << 5) | warp;
        WarpSync& ws = warp_syncs_[(warp_key << 32) | mask];
        auto self = gx::FiberRef::active();
        ws.fibers[lane] = self;
        if (!ws.arrived) ++pending_warp_syncs_;
        ws.arrived |= 1u << lane;
        auto exited = exited_lanes_.find(warp_key);
        const uint32_t required = mask & ~(exited == exited_lanes_.end() ? 0u : exited->second);
        if ((ws.arrived & required) != required) { self.suspend(); return; }
        release(ws, lane);
    }
    void lane_exited(long block, unsigned linear_thread) {
        const uint64_t warp_key = (uint64_t(block) << 5) | (linear_thread / 32);
        if (warp_key != last_exit_key_) {
            last_exit_ = &exited_lanes_[warp_key];
            last_exit_key_ = warp_key;
        }
        const uint32_t exited = *last_exit_ |= 1u << (linear_thread % 32);
        if (!pending_warp_syncs_) return;
        for (auto& [key, ws] : warp_syncs_) {
            const uint32_t mask = uint32_t(key);
            if ((key >> 32) != warp_key || !ws.arrived || (ws.arrived & mask & ~exited) != (mask & ~exited))
                continue;
            release(ws, 32);
        }
    }
private:
    void release(WarpSync& ws, unsigned self_lane) {
        const uint32_t waiting = ws.arrived & ~(self_lane < 32 ? 1u << self_lane : 0u);
        ws.arrived = 0;
        --pending_warp_syncs_;
        auto self = gx::FiberRef::active();
        for (uint32_t bits = waiting; bits; bits &= bits - 1) self.schedule(ws.fibers[__builtin_ctz(bits)]);
    }
public:
    gx::WarpgroupRendezvous* get_warpgroup_rendezvous(long block, unsigned warpgroup) {
        auto& entry = warpgroup_rendezvous_[(uint64_t(block) << 8) | warpgroup];
        if (!entry) entry = std::make_unique<gx::WarpgroupRendezvous>();
        return entry.get();
    }

    gx::LocalWarpExchange* get_warp_exchange(int block, int warp, uint32_t mask) {
        // Distinct masks with the same population must not rendezvous together.
        const uint64_t key = ((uint64_t(block) * 32 + warp) << 32) | mask;
        auto& entry = warp_exchanges_[key];
        if (!entry) entry = std::make_unique<gx::LocalWarpExchange>(mask);
        return entry.get();
    }
};

// CUDA emulator with Boost.Fiber
