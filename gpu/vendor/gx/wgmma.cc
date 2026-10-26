// SPDX-License-Identifier: MIT
// Warpgroup-level asynchronous MMA (PTX wgmma.mma_async, sm_90a) for
// translated PTX: PTX ISA 9.3 section 9.7.16 "Asynchronous Warpgroup Level
// Matrix Multiply-Accumulate Instructions", cross-checked with CUTLASS 4.8.1
// cute/atom/mma_traits_sm90_gmma.hpp (register layouts) and make_gmma_desc
// (shared-memory canonical layouts).
//
// Completion is eager: every wgmma.mma_async finishes before its rendezvous
// releases the warpgroup, so every committed wgmma-group is complete and
// wgmma.wait_group N (any N) has nothing left to wait for. Valid programs may
// not touch the accumulator or A registers, nor overwrite the source shared
// memory, between issue and the wait that covers the operation, so eager
// completion is indistinguishable from hardware completion for them.
//
// All four warps of the warpgroup (4 contiguous warps, first warp-rank a
// multiple of 4) must execute the same aligned instruction with uniform
// descriptors, scale and type operands; the last arriving lane computes the
// whole 64xN product and writes every lane's accumulator fragment into that
// lane's suspended stack before any lane resumes.
//
// Arithmetic policy (PTX leaves accumulation order, rounding and subnormal
// handling unspecified): products of decoded operands are exact in double,
// summed in double, added to the incoming accumulator and rounded once to the
// destination type. TF32 operands drop their low 13 bits before
// multiplication as PTX specifies. Integer forms accumulate exactly and wrap,
// or clamp to the s32 range with .satfinite. This is not a claim of H100 bit
// identity (FP8 hardware accumulates below single precision).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../../runtime/context.h"

long gx_current_linear_block();
extern "C" void drperf_gpu_resource(const char*,uint64_t);
extern "C" void drperf_gpu_resource_at(ThreadContext*,const char*,uint64_t);

extern "C" uint64_t gx_cluster_shared_ptr(uint32_t address);
extern "C" uint32_t gx_dev_cluster_ctarank(void);
extern "C" void gx_ptx_trap_op(const char* op);

namespace {

// ABI shared with the translator (tools/gxptx/lean/Gxptx/Translation/Statement.lean).
enum Format : uint32_t { kF16 = 0, kBF16 = 1, kE4M3 = 2, kE5M2 = 3, kTF32 = 4, kS8 = 5, kU8 = 6 };
enum DType : uint32_t { kDF16 = 0, kDF32 = 1, kDS32 = 2 };
enum Flag : uint32_t {
    kScaleD = 1, kNegA = 2, kNegB = 4, kTransA = 8, kTransB = 16, kRegisterA = 32, kSatFinite = 64,
};

[[noreturn]] void trap(const char* what) {
    gx_ptx_trap_op(what);
    throw std::runtime_error(what);  // gx_ptx_trap_op aborts; keep the compiler informed.
}

unsigned format_bits(uint32_t f) { return f <= kBF16 ? 16 : f == kTF32 ? 32 : 8; }
bool is_integer(uint32_t f) { return f == kS8 || f == kU8; }

// Legal type/shape combinations of the dense wgmma.mma_async forms.
bool valid_form(uint32_t n, uint32_t atype, uint32_t btype, uint32_t dtype, uint32_t flags) {
    if (n < 8 || n > 256 || n % 8) return false;
    const bool transposed = flags & (kTransA | kTransB);
    if ((flags & kTransA) && (flags & kRegisterA)) return false;
    if (atype <= kBF16) {
        if (btype != atype || (atype == kBF16 ? dtype != kDF32 : dtype == kDS32)) return false;
    } else {
        if (transposed) return false;
        if (atype == kTF32) { if (btype != kTF32 || dtype != kDF32) return false; }
        else if (atype == kE4M3 || atype == kE5M2) {
            if ((btype != kE4M3 && btype != kE5M2) || dtype == kDS32) return false;
        } else if (is_integer(atype)) {
            if (!is_integer(btype) || dtype != kDS32 || (flags & (kNegA | kNegB))) return false;
            if (n > 32 && n % 16) return false;
        } else return false;
    }
    if ((flags & kSatFinite) && !is_integer(atype)) return false;
    return true;
}

double decode(uint32_t format, uint32_t bits) {
    auto minifloat = [](uint32_t v, int eb, int mb, int bias, bool ieee, bool fn) {
        const uint32_t mmask = (1u << mb) - 1, emask = (1u << eb) - 1;
        const bool neg = (v >> (eb + mb)) & 1;
        const uint32_t e = (v >> mb) & emask, m = v & mmask;
        double r;
        if (ieee && e == emask) r = m ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
        else if (fn && e == emask && m == mmask) r = std::numeric_limits<double>::quiet_NaN();
        else if (e == 0) r = std::ldexp(double(m), 1 - bias - mb);
        else r = std::ldexp(double(m | (1u << mb)), int(e) - bias - mb);
        return neg ? -r : r;
    };
    switch (format) {
    case kF16: return minifloat(bits & 0xffff, 5, 10, 15, true, false);
    case kBF16: { const uint32_t u = (bits & 0xffff) << 16; float f; std::memcpy(&f, &u, 4); return f; }
    case kE4M3: return minifloat(bits & 0xff, 4, 3, 7, false, true);
    case kE5M2: return minifloat(bits & 0xff, 5, 2, 15, true, false);
    case kTF32: { const uint32_t u = bits & 0xffffe000u; float f; std::memcpy(&f, &u, 4); return f; }
    case kS8: return double(int8_t(bits & 0xff));
    default: return double(bits & 0xff);
    }
}

// sm_90 shared-memory matrix descriptor (PTX "Matrix Descriptor Format").
struct Descriptor {
    uint32_t start, lbo, sbo;
    unsigned swizzle_bits, atom, base_offset;
};

Descriptor decode_descriptor(uint64_t d) {
    // Fields: start [13:0], LBO [29:16], SBO [45:32], base offset [51:49],
    // layout [63:62]. Any other bit is reserved and has no defined meaning.
    constexpr uint64_t reserved = (UINT64_C(3) << 14) | (UINT64_C(3) << 30) |
        (UINT64_C(7) << 46) | (UINT64_C(0x3ff) << 52);
    if (d & reserved) trap("wgmma.descriptor_reserved_bits_set");
    Descriptor m{};
    m.start = uint32_t(d & 0x3fff) << 4;
    m.lbo = uint32_t((d >> 16) & 0x3fff) << 4;
    m.sbo = uint32_t((d >> 32) & 0x3fff) << 4;
    m.base_offset = unsigned(d >> 49) & 7;
    switch (unsigned(d >> 62)) {
    case 0: m.swizzle_bits = 0; m.atom = 1; break;   // no swizzle (interleave)
    case 1: m.swizzle_bits = 3; m.atom = 8; break;   // 128B
    case 2: m.swizzle_bits = 2; m.atom = 4; break;   // 64B
    default: m.swizzle_bits = 1; m.atom = 2; break;  // 32B
    }
    if (m.base_offset && !m.swizzle_bits) trap("wgmma.descriptor_base_offset_without_swizzle");
    return m;
}

// Swizzle<b,4,3> of a CTA shared byte address (bits [4,4+b) ^= bits [7,7+b));
// a non-zero base offset gives the phase of a pattern that does not start on
// its natural boundary, as in the tcgen05 descriptors.
uint32_t swizzle(uint32_t address, const Descriptor& m) {
    const uint32_t row = ((address >> 7) - m.base_offset) & ((1u << m.swizzle_bits) - 1);
    return address ^ (row << 4);
}

// Shared byte address of byte `byte` (< 32: one K slice is always 32 bytes) of
// row `row` (along M or N) of a K-major canonical layout:
//   none: ((8,m),(T,2k)):((1T,SBO),(1,LBO))   swizzled: ((8,m),(T,2k)):((WT,SBO),(1,T)).
uint32_t k_major(const Descriptor& m, unsigned row, unsigned byte) {
    const uint32_t r8 = row % 8, group = row / 8, chunk = byte / 16;
    const uint32_t linear = m.swizzle_bits
        ? m.start + (r8 * m.atom + chunk) * 16 + group * m.sbo + byte % 16
        : m.start + r8 * 16 + chunk * m.lbo + group * m.sbo + byte % 16;
    return swizzle(linear, m);
}

// MN-major canonical layout (transposed 16-bit operands), byte `mn_byte`
// along M/N of K index k:
//   none: ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))   swizzled: ((T,W,m),(8,k)):((1,T,LBO),(WT,SBO)).
uint32_t mn_major(const Descriptor& m, unsigned mn_byte, unsigned k) {
    const uint32_t chunk = mn_byte / 16, k8 = k % 8, kg = k / 8;
    const uint32_t linear = m.swizzle_bits
        ? m.start + (chunk % m.atom + k8 * m.atom) * 16 + (chunk / m.atom) * m.lbo + kg * m.sbo + mn_byte % 16
        : m.start + k8 * 16 + chunk * m.sbo + kg * m.lbo + mn_byte % 16;
    return swizzle(linear, m);
}

uint32_t shared_load(uint32_t address, unsigned bytes) {
    // The CTA window check is per element; elements never straddle 16 bytes.
    const uint32_t rank = gx_dev_cluster_ctarank();
    if (address & 0xff000000u) trap("wgmma.descriptor_address_outside_cta_window");
    const auto* p = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(gx_cluster_shared_ptr((rank << 24) | address)));
    (void)gx_cluster_shared_ptr((rank << 24) | (address + bytes - 1));
    uint32_t v = 0;
    std::memcpy(&v, p, bytes);
    return v;
}

// Element (row, k) of a descriptor operand with `rows` rows (64 for A, N for B).
uint32_t descriptor_element(const Descriptor& m, bool mn_major_layout, unsigned bits, unsigned row, unsigned k) {
    const unsigned bytes = bits / 8;
    return shared_load(mn_major_layout ? mn_major(m, row * bytes, k) : k_major(m, row, k * bytes), bytes);
}

struct Operation {
    uint32_t n, atype, btype, dtype, flags;
    uint64_t desc_a, desc_b;
    bool operator==(const Operation& o) const {
        return n == o.n && atype == o.atype && btype == o.btype && dtype == o.dtype &&
               flags == o.flags && desc_a == o.desc_a && desc_b == o.desc_b;
    }
};

struct Arrival {
    const Operation* op;
    const uint32_t* a;  // four A fragment registers (register-A form)
    uint32_t* d;        // accumulator fragment registers
    ThreadContext* context;
};

struct WarpgroupLane { long block; unsigned warpgroup, lane; };

WarpgroupLane warpgroup_lane() {
    const ThreadContext* ctx = gx_thread_context();
    if (!ctx || !current_named_barrier_manager) trap("wgmma.outside_cuda_launch");
    const dim3 t = ctx->threadIdx, b = ctx->blockDim;
    const uint64_t total = uint64_t(b.x) * b.y * b.z;
    const uint64_t linear = (uint64_t(t.z) * b.y + t.y) * b.x + t.x;
    const unsigned warpgroup = unsigned(linear / 128);
    if ((uint64_t(warpgroup) + 1) * 128 > total) trap("wgmma.requires_full_warpgroup");
    return {gx_current_linear_block(), warpgroup, unsigned(linear % 128)};
}

void compute(const Operation& op, const std::array<void*, gx::WarpgroupRendezvous::kLanes>& lanes) {
    const unsigned n = op.n, kdim = 256 / format_bits(op.atype);
    const unsigned abits = format_bits(op.atype), bbits = format_bits(op.btype);
    const bool integer = is_integer(op.atype);
    for (void* p : lanes) {
        const Arrival& arrival = *static_cast<const Arrival*>(p);
        if (!(*arrival.op == op)) trap("wgmma.nonuniform_warpgroup_operands");
    }
    // Charge one logical operand tile per issued warpgroup operation, never
    // the reference interpreter's element loops. Register A is already loaded.
    const uint64_t k=256/format_bits(op.atype);
    uint64_t shared_bits=uint64_t(op.n)*k*format_bits(op.btype);
    if(!(op.flags&kRegisterA))shared_bits+=64*k*format_bits(op.atype);
    for(void* p:lanes)drperf_gpu_resource_at(static_cast<const Arrival*>(p)->context,"shared_read_bytes",shared_bits/(8*128));
    // A collective is owned by lane zero, independent of scheduler arrival order.
    drperf_gpu_resource_at(static_cast<const Arrival*>(lanes[0])->context,"tensor_mma_instructions",1);
    static thread_local std::vector<double> a, b;
    a.assign(size_t(64) * kdim, 0.0);
    b.assign(size_t(n) * kdim, 0.0);
    // A: 64 x K, registers (Figures 148/150/152) or shared memory.
    if (op.flags & kRegisterA) {
        const unsigned per_reg = 32 / abits;
        for (unsigned lane = 0; lane < 128; ++lane) {
            const uint32_t* regs = static_cast<const Arrival*>(lanes[lane])->a;
            const unsigned warp = lane / 32, l = lane % 32;
            for (unsigned r = 0; r < 4; ++r)
                for (unsigned e = 0; e < per_reg; ++e) {
                    const unsigned row = 16 * warp + l / 4 + 8 * (r % 2);
                    const unsigned k = per_reg * (l % 4) + e + (kdim / 2) * (r / 2);
                    const uint32_t bits = abits == 32 ? regs[r] : (regs[r] >> (e * abits)) & ((1u << abits) - 1);
                    a[row * kdim + k] = decode(op.atype, bits);
                }
        }
    } else {
        const Descriptor m = decode_descriptor(op.desc_a);
        for (unsigned row = 0; row < 64; ++row)
            for (unsigned k = 0; k < kdim; ++k)
                a[row * kdim + k] = decode(op.atype, descriptor_element(m, op.flags & kTransA, abits, row, k));
    }
    const Descriptor mb = decode_descriptor(op.desc_b);
    for (unsigned col = 0; col < n; ++col)
        for (unsigned k = 0; k < kdim; ++k)
            b[col * kdim + k] = decode(op.btype, descriptor_element(mb, op.flags & kTransB, bbits, col, k));
    const double sign = (op.flags & kNegA ? -1.0 : 1.0) * (op.flags & kNegB ? -1.0 : 1.0);
    const bool scale_d = op.flags & kScaleD;
    // D fragments (Figures 149/151/153): element i of a lane is at
    // row 16*warp + lane/4 + 8*((i/2)%2), column 8*(i/4) + 2*(lane%4) + i%2.
    for (unsigned lane = 0; lane < 128; ++lane) {
        uint32_t* d = static_cast<const Arrival*>(lanes[lane])->d;
        const unsigned warp = lane / 32, l = lane % 32;
        for (unsigned i = 0; i < n / 2; ++i) {
            const unsigned row = 16 * warp + l / 4 + 8 * ((i / 2) % 2);
            const unsigned col = 8 * (i / 4) + 2 * (l % 4) + i % 2;
            const double* ar = &a[row * kdim];
            const double* bc = &b[col * kdim];
            if (integer) {
                int64_t sum = 0;
                for (unsigned k = 0; k < kdim; ++k) sum += int64_t(ar[k]) * int64_t(bc[k]);
                if (scale_d) sum += int64_t(int32_t(d[i]));
                if (op.flags & kSatFinite)
                    sum = std::min<int64_t>(std::max<int64_t>(sum, INT32_MIN), INT32_MAX);
                d[i] = uint32_t(sum);
                continue;
            }
            double products = 0;
            for (unsigned k = 0; k < kdim; ++k) products += ar[k] * bc[k];
            // The accumulation starts from +0 (or D), so an all-zero product
            // sum of negated operands yields +0 as an ordered IEEE sum would.
            double sum;
            if (op.dtype == kDF32) {
                float c = 0.0f;
                if (scale_d) std::memcpy(&c, &d[i], 4);
                sum = double(c) + sign * products;
                const float f = float(sum);
                std::memcpy(&d[i], &f, 4);
            } else {
                const unsigned shift = 16 * (i % 2);
                sum = (scale_d ? decode(kF16, d[i / 2] >> shift) : 0.0) + sign * products;
                const _Float16 h = (_Float16)sum;
                uint16_t u; std::memcpy(&u, &h, 2);
                d[i / 2] = (d[i / 2] & ~(0xffffu << shift)) | (uint32_t(u) << shift);
            }
        }
    }
}

}  // namespace

// wgmma.mma_async.sync.aligned.m64n<n>k<K>.<dtype>.<atype>.<btype>. `a` holds
// the four A fragment registers for the register-A form (else unused), `d` the
// N/2 (f32/s32) or N/4 (f16x2) accumulator registers, read when scale-d is set
// and always written.
extern "C" void gx_wgmma_mma_async(uint32_t n, uint32_t atype, uint32_t btype, uint32_t dtype, uint32_t flags,
                                   uint64_t desc_a, uint64_t desc_b, const uint32_t* a, uint32_t* d) {
    if (!valid_form(n, atype, btype, dtype, flags)) trap("wgmma.mma_async.unsupported_form");
    const WarpgroupLane self = warpgroup_lane();
    const Operation op{n, atype, btype, dtype, flags, (flags & kRegisterA) ? 0 : desc_a, desc_b};
    Arrival arrival{&op, a, d, gx_thread_context()};
    auto* rendezvous = current_named_barrier_manager->get_warpgroup_rendezvous(self.block, self.warpgroup);
    try {
        rendezvous->collect(self.lane, 4, &arrival, [&](const auto& lanes) { compute(op, lanes); });
    } catch (const std::invalid_argument&) {
        trap("wgmma.divergent_warpgroup_collective");
    }
}

// wgmma.fence / commit_group / wait_group N: aligned warpgroup collectives.
// Operations complete at issue, so they only rendezvous the warpgroup.
// `kind`: 1 fence, 2 commit_group, 3 wait_group, 4 setmaxnreg.inc, 5
// setmaxnreg.dec (a register-allocation hint with no effect on fibers).
extern "C" void gx_wgmma_collective(uint32_t kind, uint32_t count) {
    const WarpgroupLane self = warpgroup_lane();
    drperf_gpu_resource("warpgroup_collective_lane_arrivals",1);
    auto* rendezvous = current_named_barrier_manager->get_warpgroup_rendezvous(self.block, self.warpgroup);
    try {
        rendezvous->collect(self.lane, (uint64_t(kind) << 32) | count, nullptr, [](const auto&) {});
    } catch (const std::invalid_argument&) {
        trap("wgmma.divergent_warpgroup_collective");
    }
}
