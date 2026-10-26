// SPDX-License-Identifier: MIT
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

// Per-CTA Tensor Memory: 128 lanes x 512 columns of 32-bit cells. A TMEM
// address is (lane << 16) | column. Allocation hands out power-of-two column
// ranges (>= 32) aligned to their size.
struct GxTmemState {
    static constexpr unsigned kLanes = 128, kColumns = 512, kGranule = 32;
    std::vector<uint32_t> cells;
    uint32_t allocated = 0;  // bit i: columns [32 i, 32 i + 32) are allocated
    uint32_t& at(unsigned lane, unsigned column) {
        if (cells.empty()) cells.assign(std::size_t(kLanes) * kColumns, 0);
        return cells[std::size_t(lane) * kColumns + column];
    }
};

// Tensor memory of the CTA with the given linear block id in the current
// launch (cpu_kernels.cc).
GxTmemState& gx_cta_tmem(long linear_block);
long gx_current_linear_block();
long gx_cluster_rank_linear_block(uint32_t rank);
