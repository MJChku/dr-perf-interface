#pragma once
#include <light/kernel_fiber.hh>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace gx {
// One warp collective on one cooperative kernel pthread. Results live on each
// caller's suspended stack, not in reusable shared slots. The last arrival
// publishes every result before waking anyone, so no departure barrier is
// needed, even when it immediately enters the next collective.
class LocalWarpExchange {
    struct Arrival {
        FiberRef fiber;
        uint32_t* result;
        uint32_t value;
        unsigned source;
    };
    const uint32_t mask_;
    uint32_t arrived_ = 0;
    enum class Operation { gather, sum, ballot, columns, min_u, max_u, min_s, max_s, bit_and, bit_or, bit_xor };
    Operation operation_ = Operation::gather;
    std::array<Arrival, 32> lanes_{};
public:
    explicit LocalWarpExchange(uint32_t mask) : mask_(mask) {
        if (!mask) throw std::invalid_argument("empty warp exchange");
    }
    LocalWarpExchange(const LocalWarpExchange&) = delete;
    LocalWarpExchange& operator=(const LocalWarpExchange&) = delete;
    ~LocalWarpExchange() { assert(!arrived_); }
    uint32_t exchange(unsigned lane, uint32_t value, unsigned source, bool sum) {
        return collect(lane, value, source, sum ? Operation::sum : Operation::gather);
    }
    uint32_t ballot(unsigned lane, bool predicate) {
        return collect(lane, predicate, lane, Operation::ballot);
    }
    // ABI order: add, min.u32, max.u32, min.s32, max.s32, and, or, xor.
    uint32_t reduce(unsigned lane, uint32_t value, unsigned operation) {
        static constexpr Operation ops[] = {Operation::sum, Operation::min_u, Operation::max_u,
            Operation::min_s, Operation::max_s, Operation::bit_and, Operation::bit_or, Operation::bit_xor};
        if (operation >= 8) throw std::invalid_argument("invalid warp reduction");
        return collect(lane, value, lane, ops[operation]);
    }
    // Independent predicate columns share one rendezvous. Results still live
    // on each suspended caller's stack and are published before any wakeup.
    void ballot_columns(unsigned lane, uint32_t predicates, unsigned columns,
                        uint32_t* results) {
        if (!columns || columns > 32 || !results ||
            (arrived_ && lanes_[__builtin_ctz(arrived_)].source != columns))
            throw std::invalid_argument("invalid warp ballot columns");
        collect(lane, predicates, columns, Operation::columns, results);
    }
private:
    uint32_t collect(unsigned lane, uint32_t value, unsigned source, Operation operation,
                     uint32_t* outputs = nullptr) {
        if (lane >= 32 || !(mask_ & (1u << lane)) ||
            (operation == Operation::gather && (source >= 32 || !(mask_ & (1u << source)))) ||
            (arrived_ & (1u << lane)) || (arrived_ && operation_ != operation))
            throw std::invalid_argument("invalid warp exchange participant");
        uint32_t result = 0;
        auto self = FiberRef::active();
        lanes_[lane] = {self, outputs ? outputs : &result, value, source};
        operation_ = operation;
        arrived_ |= 1u << lane;
        if (arrived_ != mask_) {
            self.suspend();
        } else {
            if (operation == Operation::columns) {
                uint32_t votes[32] = {};
                for (uint32_t bits = mask_; bits; bits &= bits - 1) {
                    const unsigned i = __builtin_ctz(bits);
                    for (unsigned col = 0; col < source; ++col)
                        if (lanes_[i].value & (1u << col)) votes[col] |= 1u << i;
                }
                for (uint32_t bits = mask_; bits; bits &= bits - 1) {
                    auto& a = lanes_[__builtin_ctz(bits)];
                    for (unsigned col = 0; col < source; ++col) a.result[col] = votes[col];
                }
            } else {
                uint32_t total = (operation == Operation::gather || operation == Operation::sum ||
                                  operation == Operation::ballot || operation == Operation::bit_xor)
                                    ? 0 : lanes_[__builtin_ctz(mask_)].value;
                if (operation != Operation::gather)
                    for (uint32_t bits = mask_; bits; bits &= bits - 1) {
                        const unsigned i = __builtin_ctz(bits);
                        const uint32_t value = lanes_[i].value;
                        switch (operation) {
                        case Operation::sum: total += value; break;
                        case Operation::ballot: if (value) total |= 1u << i; break;
                        case Operation::min_u: if (value < total) total = value; break;
                        case Operation::max_u: if (value > total) total = value; break;
                        case Operation::min_s: if ((value ^ 0x80000000u) < (total ^ 0x80000000u)) total = value; break;
                        case Operation::max_s: if ((value ^ 0x80000000u) > (total ^ 0x80000000u)) total = value; break;
                        case Operation::bit_and: total &= value; break;
                        case Operation::bit_or: total |= value; break;
                        case Operation::bit_xor: total ^= value; break;
                        default: break;
                        }
                    }
                for (uint32_t bits = mask_; bits; bits &= bits - 1) {
                    auto& a = lanes_[__builtin_ctz(bits)];
                    *a.result = operation == Operation::gather ? lanes_[a.source].value : total;
                }
            }
            arrived_ = 0;
            // schedule() queues only; no caller can overwrite lanes_ here.
            for (uint32_t bits = mask_ & ~(1u << lane); bits; bits &= bits - 1)
                self.schedule(lanes_[__builtin_ctz(bits)].fiber);
        }
        return result;
    }
};

// One aligned warpgroup collective (PTX wgmma) on one cooperative kernel
// pthread. Each of the 128 lanes publishes a pointer into its own suspended
// stack plus a key naming the operation; the last arrival runs `body` over all
// published pointers before waking anyone, so results reach every caller
// before it resumes and no departure barrier is needed. Lanes reaching the
// rendezvous with different keys executed different instructions, which the
// aligned qualifier makes undefined; that is reported instead of completed.
class WarpgroupRendezvous {
public:
    static constexpr unsigned kLanes = 128;
    template <typename Body>
    void collect(unsigned lane, uint64_t key, void* argument, Body&& body) {
        if (lane >= kLanes || present_[lane])
            throw std::invalid_argument("invalid warpgroup collective participant");
        if (arrived_ && key != key_)
            throw std::invalid_argument("warpgroup lanes executed different aligned operations");
        auto self = FiberRef::active();
        fibers_[lane] = self;
        arguments_[lane] = argument;
        present_[lane] = true;
        key_ = key;
        if (++arrived_ != kLanes) {
            self.suspend();
            return;
        }
        body(arguments_);
        arrived_ = 0;
        present_.fill(false);
        // schedule() queues only; no caller can re-enter before all publish.
        for (unsigned i = 0; i < kLanes; ++i)
            if (i != lane) self.schedule(fibers_[i]);
    }
    ~WarpgroupRendezvous() { assert(!arrived_); }
private:
    std::array<FiberRef, kLanes> fibers_{};
    std::array<void*, kLanes> arguments_{};
    std::array<bool, kLanes> present_{};
    unsigned arrived_ = 0;
    uint64_t key_ = 0;
};
}
