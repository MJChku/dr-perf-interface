#pragma once
#include <boost/context/protected_fixedsize_stack.hpp>
#include <algorithm>
#include <memory>
#include <vector>
#include <sys/mman.h>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace gx {
// One pool per kernel pthread. Retain a bounded set of guarded mappings, but
// make every unused backing page reclaimable by Linux. Unlike a malloc stack
// cache, this does not pin the high-water resident footprint between launches.
class FiberStack {
    // Lane stack size: translated kernels keep every PTX register (and, with
    // the latency model, a 24-byte shadow per register) as C locals, so large
    // kernels (e.g. FlashMLA decode, ~6000 registers) overflow boost's 128 KiB
    // default. DRPERF_GPU_LANE_STACK_BYTES (or GX_FIBER_STACK_BYTES) overrides;
    // the default is 512 KiB of lazily committed, guard-page protected memory.
    static size_t lane_stack_bytes() {
        const char* text = std::getenv("DRPERF_GPU_LANE_STACK_BYTES");
        if (!text || !*text) text = std::getenv("GX_FIBER_STACK_BYTES");
        size_t value = size_t{512} * 1024u;
        if (text && *text) {
            auto end = text + std::strlen(text);
            auto [ptr, ec] = std::from_chars(text, end, value);
            if (ec != std::errc() || ptr != end || value < 65536 || value > (size_t{1} << 30))
                throw std::invalid_argument("lane stack bytes must be decimal in [65536, 1073741824]");
        }
        return std::max(value, boost::context::stack_traits::default_size());
    }
    struct Pool {
        boost::context::protected_fixedsize_stack backing{FiberStack::lane_stack_bytes()};
        std::vector<boost::context::stack_context> available;
        size_t page = boost::context::stack_traits::page_size();
        size_t mapping = ((FiberStack::lane_stack_bytes() + page - 1) / page + 1) * page;
        size_t limit;
        explicit Pool(size_t bytes) : limit(bytes / mapping) { available.reserve(limit); }
        ~Pool() { for (auto& s : available) backing.deallocate(s); }
        boost::context::stack_context allocate() {
            if (available.empty()) return backing.allocate();
            auto s = available.back(); available.pop_back(); return s;
        }
        void deallocate(boost::context::stack_context& s) noexcept {
            // Exclude the bottom PROT_NONE guard page. When MADV_FREE is not
            // supported, release the mapping rather than retaining dirty RAM.
            auto* data = static_cast<char*>(s.sp) - s.size + page;
            if (s.size == mapping && available.size() < limit &&
                madvise(data, s.size - page, MADV_FREE) == 0)
                available.push_back(s);
            else backing.deallocate(s);
        }
    };
    std::shared_ptr<Pool> pool_;
public:
    // A CUDA stream creates many worker pthreads; 4 GiB per worker can retain
    // over a million protected-stack VMAs across a modest multi-stream model.
    // Keep the previously supported 4 GiB opt-in ceiling, with a conservative
    // default that does not change the number or size of live lane stacks.
    static constexpr size_t default_cache_bytes = size_t{16} * 1024u * 1024u;
    static constexpr size_t max_cache_bytes = size_t{4} * 1024u * 1024u * 1024u;
    // Read once when a kernel worker's thread-local pool is constructed. Zero
    // releases every completed stack; it does not limit live CUDA lanes.
    static size_t cache_bytes_from_environment() {
        const char* text = std::getenv("GX_FIBER_STACK_CACHE_BYTES");
        if (!text) return default_cache_bytes;
        size_t value = 0;
        const char* end = text + std::strlen(text);
        auto parsed = std::from_chars(text, end, value);
        if (parsed.ec != std::errc{} || parsed.ptr != end || text == end ||
            value > max_cache_bytes)
            throw std::invalid_argument("GX_FIBER_STACK_CACHE_BYTES must be decimal bytes in [0,4294967296]");
        return value;
    }
    // Virtual mapping bound, not resident memory: 16 MiB per worker by default;
    // all cached data pages remain immediately reclaimable under pressure.
    explicit FiberStack(size_t cache_bytes = default_cache_bytes)
        : pool_(std::make_shared<Pool>(cache_bytes)) {}
    boost::context::stack_context allocate() { return pool_->allocate(); }
    void deallocate(boost::context::stack_context& s) noexcept { pool_->deallocate(s); }
    size_t cached() const { return pool_->available.size(); }
};
}
