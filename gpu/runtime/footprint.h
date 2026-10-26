// Exact unions of logical PTX addresses, independent of lane execution order.
// Disjoint inclusive intervals keep contiguous streams compact.
#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>

namespace drperf_footprint {
struct AddressUnion {
    std::map<uint64_t, uint64_t> ranges;
    uint64_t size = 0;

    void insert(uint64_t first, uint64_t last) {
        if (last < first) throw std::invalid_argument("reversed footprint interval");
        auto it = ranges.upper_bound(first);
        if (it != ranges.begin()) {
            auto prev = std::prev(it);
            if (prev->second >= first || first - prev->second == 1) it = prev;
        }
        while (it != ranges.end() &&
               (it->first <= last || it->first - last == 1)) {
            first = std::min(first, it->first);
            last = std::max(last, it->second);
            size -= it->second - it->first + 1;
            it = ranges.erase(it);
        }
        if (last - first == UINT64_MAX || last - first + 1 > UINT64_MAX - size)
            throw std::overflow_error("GPU footprint size overflow");
        ranges.emplace(first, last);
        size += last - first + 1;
    }
    void merge(const AddressUnion& other) {
        if (this == &other) return;
        for (auto [first, last] : other.ranges) insert(first, last);
    }
};

struct Footprint {
    std::map<std::string, AddressUnion> addresses;

    void touch(uint64_t address, uint64_t bytes, const std::string& space,
               const char* direction) {
        if (!bytes) return;
        if (bytes - 1 > UINT64_MAX - address)
            throw std::overflow_error("GPU footprint address overflow");
        const uint64_t last = address + bytes - 1;
        const std::string prefix = space + "_unique_" + direction;
        addresses[prefix + "_bytes"].insert(address, last);
        addresses[prefix + "_sectors"].insert(address / 32, last / 32);
        addresses[prefix + "_lines"].insert(address / 128, last / 128);
    }
    void merge(const Footprint& other) {
        if (this == &other) return;
        for (const auto& [name, values] : other.addresses) addresses[name].merge(values);
    }
    std::map<std::string, uint64_t> counts(std::map<std::string, uint64_t> result) const {
        for (const auto& [name, values] : addresses) {
            result[name] = values.size;
            unsigned granularity = 1;
            std::string suffix;
            if (name.size() >= 8 && name.substr(name.size() - 8) == "_sectors") {
                granularity = 32; suffix = "_sector_bytes";
            } else if (name.size() >= 6 && name.substr(name.size() - 6) == "_lines") {
                granularity = 128; suffix = "_line_bytes";
            }
            if (granularity != 1) {
                if (values.size > UINT64_MAX / granularity)
                    throw std::overflow_error("GPU footprint rounded size overflow");
                result[name.substr(0, name.rfind('_')) + suffix] = values.size * granularity;
            }
        }
        return result;
    }
};
}  // namespace drperf_footprint
