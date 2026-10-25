#pragma once
#include <light/kernel_fiber.hh>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace gx {
// Kernel fibers are owned by one pthread, with cooperative (not preemptive)
// scheduling. This barrier must never be shared across pthreads. A suspended
// fiber remains alive through run_fiber_launch's owning fiber handles.
class LocalFiberBarrier {
    const unsigned expected_;
    unsigned remaining_;
    std::vector<FiberRef> waiters_;
public:
    explicit LocalFiberBarrier(unsigned expected)
        : expected_(expected), remaining_(expected) {
        if (!expected) throw std::invalid_argument("empty fiber barrier");
        waiters_.reserve(expected - 1);
    }
    LocalFiberBarrier(const LocalFiberBarrier&) = delete;
    LocalFiberBarrier& operator=(const LocalFiberBarrier&) = delete;
    ~LocalFiberBarrier() { assert(waiters_.empty()); }
    void wait() {
        auto self = FiberRef::active();
        if (--remaining_) {
            waiters_.push_back(self);
            self.suspend();
        } else {
            remaining_ = expected_;
            // schedule() only queues a fiber; it does not switch to it. Clear
            // this generation's queue before any participant can enter again.
            for (auto other : waiters_) self.schedule(other);
            waiters_.clear();
        }
    }
};
}
