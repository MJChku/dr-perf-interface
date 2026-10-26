// SPDX-License-Identifier: MIT
// Non-tensor bulk copies and reductions for translated PTX (cp.async.bulk,
// cp.reduce.async.bulk). Like the tensor forms they complete at issue: the
// bytes are moved (or reduced element-wise, atomically) and then the
// destination mbarrier's transaction count is completed. Programs wait on the
// mbarrier or bulk group before using the data, so synchronous completion is
// indistinguishable from the hardware's asynchronous completion.
// Shared addresses are 32-bit shared::cluster addresses (rank << 24 | offset);
// shared::cta operands are the executing CTA's addresses.
#include "reduce_ops.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" void drperf_gpu_memory(uint64_t,uint64_t,uint64_t,unsigned);
extern "C" void drperf_gpu_resource(const char*,uint64_t);
extern "C" uint64_t gx_cluster_shared_ptr(uint32_t address);
extern "C" void gx_mbarrier_complete_tx(uint64_t address, uint32_t bytes);

namespace {

constexpr uint32_t kOffsetMask = (1u << 24) - 1;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("cp.async.bulk: " + what); }

uint8_t* shared(uint32_t address) {
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(gx_cluster_shared_ptr(address)));
}
uint8_t* global(uint64_t address) { return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address)); }

void check_size(uint32_t size) {
    if (size % 16) fail("size must be a multiple of 16 bytes");
}

}  // namespace

// global -> shared::cta / shared::cluster, optionally multicast to every CTA
// of cta_mask (same CTA-relative offsets for data and mbarrier). The first
// ignore_left and last ignore_right bytes (.ignore_oob) are not read; their
// destination bytes are left unchanged (PTX: indeterminate).
extern "C" void gx_bulk_g2s(uint32_t dst, uint64_t src, uint32_t size, uint32_t mbar, uint32_t multicast,
                            uint32_t cta_mask, uint32_t ignore_left, uint32_t ignore_right) {
    check_size(size);
    if (ignore_left > 15 || ignore_right > 15 || ignore_left + ignore_right > size) fail("invalid .ignore_oob byte counts");
    unsigned destinations=0;
    drperf_gpu_resource("bulk_copy_instructions",1);
    drperf_gpu_resource("global_read_bytes",size-ignore_left-ignore_right);
    drperf_gpu_memory(src+ignore_left,size-ignore_left-ignore_right,0,1);
    auto deliver = [&](uint32_t rank) {
        ++destinations;
        drperf_gpu_resource("shared_write_bytes",size-ignore_left-ignore_right);
        const uint32_t base = (rank << 24) | (dst & kOffsetMask);
        std::memcpy(shared(base) + ignore_left, global(src) + ignore_left, size - ignore_left - ignore_right);
        gx_mbarrier_complete_tx(gx_cluster_shared_ptr((multicast ? rank << 24 : mbar & ~kOffsetMask) | (mbar & kOffsetMask)), size);
    };
    if (!multicast) { deliver(dst >> 24); return; }
    if (!cta_mask) fail("multicast with an empty CTA mask");
    for (uint32_t rank = 0; rank < 16; ++rank)
        if (cta_mask & (1u << rank)) deliver(rank);
}

// shared::cta -> shared::cluster (another CTA's shared memory); mbar is in the
// destination CTA.
extern "C" void gx_bulk_s2s(uint32_t dst, uint32_t src, uint32_t size, uint32_t mbar) {
    check_size(size);
    std::memmove(shared(dst), shared(src), size);
    drperf_gpu_resource("bulk_copy_instructions",1);
    drperf_gpu_resource("shared_read_bytes",size);drperf_gpu_resource("shared_write_bytes",size);
    gx_mbarrier_complete_tx(gx_cluster_shared_ptr(mbar), size);
}

// shared::cta -> global; with .cp_mask, bit i of byte_mask selects byte i of
// every 16-byte chunk.
extern "C" void gx_bulk_s2g(uint64_t dst, uint32_t src, uint32_t size, uint32_t byte_mask) {
    check_size(size);
    const uint8_t* s = shared(src);
    uint8_t* d = global(dst);
    uint64_t selected=uint64_t(size/16)*__builtin_popcount(byte_mask&0xffff);
    drperf_gpu_resource("bulk_copy_instructions",1);
    drperf_gpu_resource("shared_read_bytes",selected);drperf_gpu_resource("global_write_bytes",selected);
    if ((byte_mask & 0xffff) == 0xffff) { drperf_gpu_memory(dst,0,size,1);std::memcpy(d, s, size); return; }
    for (uint32_t i = 0; i < size; ++i)
        if ((byte_mask >> (i % 16)) & 1) { drperf_gpu_memory(dst+i,0,1,1);d[i] = s[i]; }
}

// cp.reduce.async.bulk: dst[i] = dst[i] op src[i] over `size` bytes of
// elements. dst_cluster selects a shared::cluster destination (32-bit
// address in dst, mbarrier completion on mbar in that CTA); otherwise dst is
// a global address (bulk_group completion). op/type are gx::RedOp/RedType.
extern "C" void gx_bulk_reduce(uint64_t dst, uint32_t src, uint32_t size, uint32_t dst_cluster, uint32_t mbar,
                               uint32_t op, uint32_t type) {
    check_size(size);
    const auto rop = gx::RedOp(op);
    const auto rt = gx::RedType(type);
    if (!gx::reduce_supported(rop, rt, false)) fail("unsupported cp.reduce.async.bulk op/type");
    if (dst_cluster) {
        const bool ok = (rop == gx::RedOp::add && (rt == gx::RedType::u32 || rt == gx::RedType::s32 || rt == gx::RedType::u64)) ||
            ((rop == gx::RedOp::min || rop == gx::RedOp::max) && (rt == gx::RedType::u32 || rt == gx::RedType::s32)) ||
            ((rop == gx::RedOp::inc || rop == gx::RedOp::dec) && rt == gx::RedType::u32) ||
            ((rop == gx::RedOp::and_ || rop == gx::RedOp::or_ || rop == gx::RedOp::xor_) &&
             (rt == gx::RedType::u32 || rt == gx::RedType::s32 || rt == gx::RedType::u64));
        if (!ok) fail("unsupported shared::cluster cp.reduce.async.bulk op/type");
    }
    uint8_t* d = dst_cluster ? shared(uint32_t(dst)) : global(dst);
    const uint8_t* s = shared(src);
    const unsigned bytes = gx::red_bytes(rt);
    for (uint32_t i = 0; i < size; i += bytes) {
        uint64_t v = 0;
        std::memcpy(&v, s + i, bytes);
        gx::reduce_element(d + i, v, rt, rop);
    }
    drperf_gpu_resource("bulk_reduce_instructions",1);
    drperf_gpu_resource("reduction_ops",size/bytes);
    drperf_gpu_resource("shared_read_bytes",size);
    drperf_gpu_resource(dst_cluster?"shared_read_bytes":"global_read_bytes",size);
    drperf_gpu_resource(dst_cluster?"shared_write_bytes":"global_write_bytes",size);
    if (!dst_cluster) drperf_gpu_memory(dst,size,size,1);
    if (dst_cluster) gx_mbarrier_complete_tx(gx_cluster_shared_ptr(mbar), size);
}
