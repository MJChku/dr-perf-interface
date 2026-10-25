#pragma once
#include <vector>
#include <new>
#include <cstddef>
#include <cstring>
#include <sys/mman.h>

namespace gx {
// Private scratch mappings, never application allocations. A single kernel
// pthread owns the pool; return mappings only after every launch fiber exits.
// Reuse preserves the zero-initialized bytes of a fresh anonymous mmap and
// never writes the trailing guard page. Both retained bytes and entries are
// bounded, so a large launch cannot pin its high-water footprint indefinitely.
class SharedArenaPool {
    struct Entry { char* base; size_t usable, length; };
    std::vector<Entry> entries_;
    size_t bytes_ = 0;
    const size_t limit_;
public:
    explicit SharedArenaPool(size_t bytes = 32u * 1024u * 1024u) : limit_(bytes) {}
    SharedArenaPool(const SharedArenaPool&) = delete;
    SharedArenaPool& operator=(const SharedArenaPool&) = delete;
    ~SharedArenaPool() {
        for (const Entry& e : entries_) munmap(e.base, e.length);
    }
    char* take(size_t usable, size_t length) {
        for (size_t i = entries_.size(); i != 0;) {
            const Entry e = entries_[--i];
            if (e.usable != usable || e.length != length) continue;
            entries_[i] = entries_.back();
            entries_.pop_back();
            bytes_ -= length;
            std::memset(e.base, 0, usable);
            return e.base;
        }
        return nullptr;
    }
    void put(char* base, size_t usable, size_t length) noexcept {
        if (entries_.size() < 4096 && length <= limit_ - bytes_) {
            try {
                entries_.push_back({base, usable, length});
                bytes_ += length;
                return;
            } catch (const std::bad_alloc&) {
                // Destruction must still release the mapping if metadata
                // allocation fails; retaining scratch is only an optimization.
            }
        }
        munmap(base, length);
    }
    size_t cached_bytes() const { return bytes_; }
};
}
