// SPDX-License-Identifier: MIT
#pragma once
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <light/kernel_fiber.hh>

// CTA-local PTX mbarriers. Owned by one launch; CUDA fibers on its pthread
// execute these operations serially. Shared addresses distinguish CTAs.
class GxMbarriers {
    struct State {
        uint32_t expected, pending;
        int64_t transactions = 0;
        uint64_t phase = 0;
        std::vector<gx::FiberRef> waiters;
        void complete() {
            if (pending == 0 && transactions == 0) {
                ++phase;
                pending = expected;
                if (!waiters.empty()) {
                    auto self = gx::FiberRef::active();
                    for (auto waiter : waiters) self.schedule(waiter);
                    waiters.clear();
                }
            }
        }
    };
    std::unordered_map<uint64_t, State> states;
    State& get(uint64_t address) {
        auto it = states.find(address);
        if (it == states.end()) throw std::runtime_error("uninitialized PTX mbarrier");
        return it->second;
    }
public:
    GxMbarriers() = default;
    GxMbarriers(const GxMbarriers&) = delete;
    GxMbarriers& operator=(const GxMbarriers&) = delete;
    void init(uint64_t address, uint32_t count) {
        if (!count || count >= (1u << 20) || (address & 7))
            throw std::runtime_error("invalid PTX mbarrier initialization");
        auto old = states.find(address);
        if (old != states.end() && !old->second.waiters.empty())
            throw std::runtime_error("PTX mbarrier reinitialized with waiting fibers");
        states.insert_or_assign(address, State{count, count});
    }
    uint64_t arrive(uint64_t address, uint32_t count, uint32_t expected_bytes) {
        auto& s = get(address);
        if (!count || count > s.pending)
            throw std::runtime_error("PTX mbarrier arrival count underflow");
        const uint64_t token = s.phase;
        s.transactions += expected_bytes;
        s.pending -= count;
        s.complete();
        return token;
    }
    void complete_tx(uint64_t address, uint32_t bytes) {
        auto& s = get(address);
        // Copy completion may precede arrive.expect_tx. Keep its credit until
        // both the pending arrival count and net transaction count reach zero.
        s.transactions -= bytes;
        s.complete();
    }
    // mbarrier.expect_tx: raise the pending transaction count of the
    // current phase without an arrival.
    void expect_tx(uint64_t address, uint32_t bytes) {
        get(address).transactions += bytes;
    }
    bool test(uint64_t address, uint64_t token, bool parity) {
        auto& s = get(address);
        return parity ? (s.phase & 1u) != token : s.phase != token;
    }
    uint64_t phase(uint64_t address) { return get(address).phase; }
    // Only an explicitly annotated empty wait loop may park here. A normal
    // try_wait/test remains nonblocking. Completion is observed when the
    // waiter is queued to resume, even if another phase finishes meanwhile.
    void wait(uint64_t address, uint64_t token, bool parity) {
        auto& s = get(address);
        if (parity ? (s.phase & 1u) != token : s.phase != token) return;
        auto self = gx::FiberRef::active();
        s.waiters.push_back(self);
        self.suspend();
    }
};
