#pragma once
#include <cstdint>
namespace ds41 {
// Pointers refer only to this rank's session storage; no cross-device reads.
struct DecodeCacheRow {
    std::uint64_t window,window_scale,compressed,compressed_scale,pages;
    std::uint32_t position,compressed_rows,page_rows,window_stride,window_scale_stride,compressed_stride,compressed_scale_stride,reserved;
};
static_assert(sizeof(DecodeCacheRow)==72);
}
