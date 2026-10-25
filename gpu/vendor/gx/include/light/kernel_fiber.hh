#pragma once
#include <boost/context/fiber.hpp>
#include <boost/fiber/context.hpp>
#include <boost/fiber/operations.hpp>
#include <cassert>
#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gx {
// Launch-local, single-pthread cooperative contexts. No host tasks,
// transport fibers or cross-pthread wakeups belong to this scheduler.
class KernelFibers {
public:
    struct Fiber {
        boost::context::fiber body, caller;
        KernelFibers* owner;
        Fiber* next = nullptr;
        void* data = nullptr;
        bool parked = false;
        bool queued = false;
        explicit Fiber(KernelFibers* scheduler) : owner(scheduler) {}
    };
private:
    std::vector<std::unique_ptr<Fiber>> fibers_;
    Fiber* head_ = nullptr;
    Fiber* tail_ = nullptr;
    Fiber* current_ = nullptr;
    std::exception_ptr error_;
    inline static thread_local Fiber* active_ = nullptr;
    void enqueue(Fiber* fiber) {
        if (fiber->owner != this || fiber->queued || fiber->parked)
            throw std::logic_error("invalid kernel fiber enqueue");
        fiber->queued = true;
        fiber->next = nullptr;
        if (tail_) tail_->next = fiber;
        else head_ = fiber;
        tail_ = fiber;
    }
public:
    KernelFibers() = default;
    ~KernelFibers() {
        // Forced unwinding still runs stack-local destructors with that lane
        // as the active context. In successful launches all bodies are empty.
        assert(!active_);
        for (auto& fiber : fibers_) {
            active_ = fiber.get();
            fiber->body = {};
        }
        active_ = nullptr;
    }
    KernelFibers(const KernelFibers&) = delete;
    KernelFibers& operator=(const KernelFibers&) = delete;
    template<class Stack, class Fn> void add(Stack& stack, Fn fn) {
        auto owned = std::make_unique<Fiber>(this);
        Fiber* fiber = owned.get();
        fiber->body = boost::context::fiber(std::allocator_arg, stack,
            [this, fiber, fn = std::move(fn)](boost::context::fiber&& caller) mutable {
                fiber->caller = std::move(caller);
                try { fn(); }
                catch (const boost::context::detail::forced_unwind&) { throw; }
                catch (...) { error_ = std::current_exception(); }
                return std::move(fiber->caller);
            });
        fibers_.push_back(std::move(owned));
        enqueue(fiber);
    }
    Fiber* current() const { return current_; }
    static Fiber* active() { return active_; }
    void yield() {
        assert(current_);
        Fiber* fiber = current_;
        fiber->caller = std::move(fiber->caller).resume();
    }
    void suspend() {
        assert(current_ && !current_->parked);
        current_->parked = true;
        yield();
    }
    void wake(Fiber* fiber) {
        if (fiber->owner != this || !fiber->parked)
            throw std::logic_error("invalid kernel fiber wake");
        fiber->parked = false;
        enqueue(fiber);
    }
    void run() {
        if (current_ || active_) throw std::logic_error("recursive kernel scheduler");
        while (head_) {
            Fiber* fiber = head_;
            head_ = fiber->next;
            if (!head_) tail_ = nullptr;
            fiber->queued = false;
            current_ = fiber;
            active_ = fiber;
            fiber->body = std::move(fiber->body).resume();
            active_ = nullptr;
            current_ = nullptr;
            if (error_) std::rethrow_exception(error_);
            if (fiber->body && !fiber->parked) enqueue(fiber);
        }
        for (auto& fiber : fibers_)
            if (fiber->body) throw std::runtime_error("kernel fibers have no runnable participant");
    }
};

// The existing Boost transport/runtime tests retain their own scheduler.
// A rendezvous never mixes those fibers with launch-local CUDA contexts.
struct FiberRef {
    void* pointer;
    bool kernel;
    static FiberRef active() {
        if (auto* fiber = KernelFibers::active()) return {fiber, true};
        return {boost::fibers::context::active(), false};
    }
    void suspend() const {
        if (kernel) static_cast<KernelFibers::Fiber*>(pointer)->owner->suspend();
        else static_cast<boost::fibers::context*>(pointer)->suspend();
    }
    void schedule(FiberRef other) const {
        if (kernel != other.kernel) throw std::logic_error("mixed kernel barrier schedulers");
        if (kernel) static_cast<KernelFibers::Fiber*>(pointer)->owner->wake(
            static_cast<KernelFibers::Fiber*>(other.pointer));
        else static_cast<boost::fibers::context*>(pointer)->schedule(
            static_cast<boost::fibers::context*>(other.pointer));
    }
};
inline void kernel_fiber_yield() {
    if (auto* fiber = KernelFibers::active()) fiber->owner->yield();
    else boost::this_fiber::yield();
}
}
