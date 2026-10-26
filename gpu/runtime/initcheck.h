#pragma once
// Read-before-write detection for driver buffers marked uninitialized
// (drperf_gpu_mark_uninitialized / drperf::gpu::Buffer<T>::uninitialized).
// Every global access already reaches drperf_gpu_memory with its address and
// byte counts (ordinary PTX, TMA and bulk copies), so a per-byte written mask is
// enough: device writes set bits, host marks set or clear them, and a device read
// of an unset byte is reported as uninitialized_read_bytes at the reading site.
// On hardware such reads return whatever an earlier process or launch left there,
// which makes results depend on run order.
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace drperf_initcheck {

struct Range { uint64_t size; std::vector<uint8_t> written; std::string name; };

class Tracker {
    std::mutex mutex_;
    std::map<uint64_t, Range> ranges_;  // base -> range
    unsigned reported_ = 0;
    Range* find(uint64_t a, uint64_t& base) {
        auto it = ranges_.upper_bound(a);
        if (it == ranges_.begin()) return nullptr;
        --it;
        if (a - it->first >= it->second.size) return nullptr;
        base = it->first;
        return &it->second;
    }
public:
    bool empty() { std::lock_guard<std::mutex> g(mutex_); return ranges_.empty(); }
    void mark(uint64_t base, uint64_t size, bool initialized, const char* name) {
        std::lock_guard<std::mutex> g(mutex_);
        if (!initialized) { ranges_[base] = Range{size, std::vector<uint8_t>(size, 0), name ? name : ""}; return; }
        uint64_t b = 0;
        for (uint64_t i = 0; i < size;) {
            Range* r = find(base + i, b);
            if (!r) { ++i; continue; }
            uint64_t off = base + i - b, n = std::min<uint64_t>(size - i, r->size - off);
            std::fill(r->written.begin() + off, r->written.begin() + off + n, 1);
            i += n;
        }
    }
    void forget(uint64_t base) { std::lock_guard<std::mutex> g(mutex_); ranges_.erase(base); }
    // Returns the number of read bytes that were never written.
    uint64_t access(uint64_t address, uint64_t reads, uint64_t writes, const std::string& site) {
        std::lock_guard<std::mutex> g(mutex_);
        if (ranges_.empty()) return 0;
        uint64_t base = 0, bad = 0;
        Range* r = find(address, base);
        if (!r) return 0;
        const uint64_t off = address - base;
        const uint64_t nr = std::min<uint64_t>(reads, r->size - off), nw = std::min<uint64_t>(writes, r->size - off);
        for (uint64_t i = 0; i < nr; ++i) bad += !r->written[off + i];
        for (uint64_t i = 0; i < nw; ++i) r->written[off + i] = 1;
        if (bad && reported_ < 20) {
            ++reported_;
            std::fprintf(stderr, "drperf GPU initcheck: %llu uninitialized byte(s) read at %s offset %llu of %s\n",
                         (unsigned long long)bad, site.c_str(), (unsigned long long)off,
                         r->name.empty() ? "unnamed buffer" : r->name.c_str());
        }
        return bad;
    }
};

inline Tracker& tracker() { static Tracker t; return t; }

}  // namespace drperf_initcheck
