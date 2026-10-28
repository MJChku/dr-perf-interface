// SPDX-License-Identifier: MIT
// tcgen05 (5th-generation tensor core) emulation for translated PTX.
//
// Asynchronous tcgen05 operations (mma, cp, commit) execute at issue on the
// issuing fiber; tcgen05.commit then arrives on its mbarrier. Programs must
// wait on that mbarrier (or tcgen05.wait) before reading results, so
// synchronous completion is indistinguishable from hardware completion.
// Operand, accumulator, scale-factor and copy layouts follow CUTLASS 4.8.1's
// SM100 atoms and the PTX ISA tcgen05 chapter (docs/BLACKWELL.md).
#include "tcgen05.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include "../../runtime/context.h"
extern "C" void drperf_gpu_resource(const char*, uint64_t);

extern "C" uint64_t gx_cluster_shared_ptr(uint32_t address);
extern "C" uint64_t gx_mbarrier_arrive(uint64_t address, uint32_t count, uint32_t expected_bytes);
extern "C" uint64_t gx_mbarrier_arrive_tensor(uint64_t address);
extern "C" void drperf_gpu_tensor_issue(uint64_t flops, uint32_t operand_bits, uint32_t cta_group);
extern "C" uint32_t gx_dev_cluster_ctarank(void);
extern "C" dim3 gx_dev_threadIdx(void);
extern "C" dim3 gx_dev_blockDim(void);

namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("tcgen05: " + what); }

unsigned linear_thread() {
    const dim3 t = gx_dev_threadIdx(), b = gx_dev_blockDim();
    return (t.z * b.y + t.y) * b.x + t.x;
}
unsigned lane_id() { return linear_thread() % 32; }

uint8_t* shared_bytes(uint32_t address) {
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(gx_cluster_shared_ptr(address)));
}

// Tensor-memory operands: only lane bits [22:16] and column bits [8:0] are
// significant (CUTLASS keeps sub-word indices in bits [31:30]).
unsigned tmem_lane(uint32_t a) { return (a >> 16) & 0x7f; }
unsigned tmem_col(uint32_t a) { return a & 0x1ff; }

// CTA v (0 = even, 1 = odd) of the executing CTA's pair, or this CTA.
uint32_t member_rank(unsigned v, unsigned cta_group) {
    return cta_group == 2 ? (gx_dev_cluster_ctarank() & ~1u) | v : gx_dev_cluster_ctarank();
}
GxTmemState& tmem_of(uint32_t rank) { return gx_cta_tmem(gx_cluster_rank_linear_block(rank)); }
GxTmemState& own_tmem() { return gx_cta_tmem(gx_current_linear_block()); }

uint32_t& cell(GxTmemState& t, unsigned lane, unsigned column) {
    if (lane >= GxTmemState::kLanes || column >= GxTmemState::kColumns)
        fail("tensor-memory access outside 128 lanes x 512 columns");
    return t.at(lane, column);
}

float as_float(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
uint32_t as_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// ---- value formats --------------------------------------------------------
double decode_minifloat(uint32_t bits, int exp_bits, int man_bits, int bias, bool ieee_special,
                        bool fn_nan, bool has_sign = true) {
    const uint32_t man_mask = (1u << man_bits) - 1, exp_mask = (1u << exp_bits) - 1;
    const bool sign = has_sign && ((bits >> (exp_bits + man_bits)) & 1);
    const uint32_t e = (bits >> man_bits) & exp_mask, m = bits & man_mask;
    double v;
    if (ieee_special && e == exp_mask) v = m ? NAN : INFINITY;
    else if (fn_nan && e == exp_mask && m == man_mask) v = NAN;
    else if (e == 0) v = std::ldexp(double(m), 1 - bias - man_bits);
    else v = std::ldexp(double(m | (1u << man_bits)), int(e) - bias - man_bits);
    return sign ? -v : v;
}

enum class Format { f16, bf16, tf32, e4m3, e5m2, e2m3, e3m2, e2m1, u8, s8 };

unsigned format_bits(Format f) {
    switch (f) {
    case Format::f16: case Format::bf16: return 16;
    case Format::tf32: return 32;
    case Format::e2m3: case Format::e3m2: return 6;
    case Format::e2m1: return 4;
    default: return 8;
    }
}

double decode(Format f, uint32_t bits) {
    switch (f) {
    case Format::f16: return decode_minifloat(bits & 0xffff, 5, 10, 15, true, false);
    case Format::bf16: return as_float((bits & 0xffff) << 16);
    case Format::tf32: return as_float(bits & 0xffffe000u);
    case Format::e4m3: return decode_minifloat(bits & 0xff, 4, 3, 7, false, true);
    case Format::e5m2: return decode_minifloat(bits & 0xff, 5, 2, 15, true, false);
    case Format::e2m3: return decode_minifloat(bits & 0x3f, 2, 3, 1, false, false);
    case Format::e3m2: return decode_minifloat(bits & 0x3f, 3, 2, 3, false, false);
    case Format::e2m1: return decode_minifloat(bits & 0xf, 2, 1, 1, false, false);
    case Format::u8: return double(bits & 0xff);
    case Format::s8: return double(int8_t(bits));
    }
    return 0;
}

// Block-scale factors: UE8M0 2^(e-127) (0xff NaN) or unsigned E4M3 (bias 7, 0x7f NaN).
double decode_scale(uint32_t code, bool ue8m0) {
    code &= 0xff;
    if (ue8m0) return code == 0xff ? NAN : std::ldexp(1.0, int(code) - 127);
    return decode_minifloat(code & 0x7f, 4, 3, 7, false, true, false);
}

uint16_t f16_bits(double v) {
    _Float16 h = (_Float16)v;
    uint16_t r; std::memcpy(&r, &h, 2); return r;
}

// ---- shared-memory matrix descriptors -------------------------------------
// Canonical SM100 UMMA layouts in 16-byte units (cute make_umma_desc):
//   K-major  NONE: ((8,n),2):((1,SBO),LBO)   SWxx: ((8,n),2):((W,SBO),1)
//   MN-major NONE: ((1,n),(8,k)):((-,SBO),(1,LBO))
//            SWxx: ((W,n),(8,k)):((1,LBO),(W,SBO))      W = 2/4/8 for 32/64/128B
// followed by the CuTe Swizzle<b,4,3> of the shared address.
struct MatrixDesc {
    uint32_t start, lbo, sbo, cta_bits;
    unsigned layout, swizzle_bits, atom_units, base_offset;
    bool lbo_absolute;   // bit 52: LBO is the absolute address of the K chunks past the 128-B row (sm_103 K=96)
};

MatrixDesc decode_desc(uint64_t d, uint32_t rank) {
    MatrixDesc m{};
    m.start = uint32_t((d & 0x3fff) << 4);
    m.lbo = uint32_t(((d >> 16) & 0x3fff) << 4);
    m.sbo = uint32_t(((d >> 32) & 0x3fff) << 4);
    const unsigned version = (d >> 46) & 3, base_offset = (d >> 49) & 7, lbo_mode = (d >> 52) & 1;
    m.layout = unsigned(d >> 61);
    if (version != 1) fail("shared-memory descriptor version is not 1 (SM100)");
    m.base_offset = base_offset;
    m.lbo_absolute = lbo_mode;
    switch (m.layout) {
    case 0: m.swizzle_bits = 0; m.atom_units = 1; break;
    case 6: m.swizzle_bits = 1; m.atom_units = 2; break;   // 32B
    case 4: m.swizzle_bits = 2; m.atom_units = 4; break;   // 64B
    case 2: m.swizzle_bits = 3; m.atom_units = 8; break;   // 128B
    default: fail("descriptor layout type " + std::to_string(m.layout) + " is unsupported");
    }
    if (m.lbo_absolute && (m.layout != 2 || base_offset))
        fail("descriptor lbo_mode=1 requires 128B swizzle and base_offset 0");
    if (base_offset && !m.swizzle_bits) fail("descriptor base_offset with a non-swizzled layout");
    m.cta_bits = rank << 24;
    return m;
}

// Swizzle<b,4,3> of a shared byte address. A non-zero base_offset gives the
// pattern phase of a matrix whose swizzle repeat does not start on its
// natural boundary: the row index is taken relative to that phase.
uint32_t swizzle(uint32_t address, unsigned bits, unsigned base_offset = 0) {
    const uint32_t row = ((address >> 7) - base_offset) & ((1u << bits) - 1);
    return address ^ (row << 4);
}

// Shared address of byte `byte` within the K extent of row `row` of a K-major
// canonical matrix (also the tcgen05.cp source rows).
uint32_t k_major_byte(const MatrixDesc& m, unsigned row, unsigned byte) {
    const uint64_t kk = byte / 16, r8 = row % 8, group = row / 8;
    if (m.lbo_absolute) {
        // K chunks that run past the 128-byte swizzle row continue at the
        // absolute LBO address (PTX matrix-descriptor lbo_mode, sm_103a).
        const uint64_t c = (m.start >> 4) & 7;
        const uint32_t linear = c + kk < 8
            ? uint32_t(m.start + r8 * 128 + kk * 16 + group * m.sbo + byte % 16)
            : uint32_t(m.lbo + r8 * 128 + (c + kk - 8) * 16 + group * m.sbo + byte % 16);
        return m.cta_bits | swizzle(linear, m.swizzle_bits);
    }
    const uint64_t unit = m.layout == 0 ? r8 + kk * (m.lbo / 16) : r8 * m.atom_units + kk;
    const uint32_t linear = uint32_t(m.start + unit * 16 + group * m.sbo + byte % 16);
    return m.cta_bits | swizzle(linear, m.swizzle_bits, m.base_offset);
}

// Element (row, k) of an MMA operand; row runs along M or N. `padded` selects
// the f8f6f4 sub-byte format: 16 packed values per 16-byte unit (8 or 12 data
// bytes, then padding); otherwise sub-byte values are packed densely.
uint32_t read_element(const MatrixDesc& m, bool mn_major, Format f, bool padded, unsigned row, unsigned k) {
    const unsigned bits = format_bits(f);
    if (!mn_major) {
        const uint64_t bit = padded && bits < 8 ? uint64_t(k / 16) * 128 + (k % 16) * bits : uint64_t(k) * bits;
        const uint8_t* p = shared_bytes(k_major_byte(m, row, unsigned(bit / 8)));
        if (bits >= 8) { uint32_t v = 0; std::memcpy(&v, p, bits / 8); return v; }
        // Sub-byte fields stay within a 16-byte chunk; FP6 may span two bytes.
        uint32_t w = p[0];
        if ((bit % 8) + bits > 8) w |= uint32_t(*shared_bytes(k_major_byte(m, row, unsigned(bit / 8 + 1)))) << 8;
        return (w >> (bit % 8)) & ((1u << bits) - 1);
    }
    if (bits < 8) fail("MN-major operands of sub-byte formats are invalid");
    if (m.lbo_absolute) fail("descriptor lbo_mode=1 requires a K-major operand");
    const uint64_t mb = uint64_t(row) * bits, mu = mb / 128, k8 = k % 8, kg = k / 8;
    uint64_t unit;
    if (m.layout == 0) unit = k8 * 16 + mu * m.sbo + kg * m.lbo;
    else unit = (mu % m.atom_units + k8 * m.atom_units) * 16 + (mu / m.atom_units) * m.lbo + kg * m.sbo;
    const uint32_t address = m.cta_bits | swizzle(uint32_t(m.start + unit + (mb % 128) / 8), m.swizzle_bits, m.base_offset);
    uint32_t v = 0;
    std::memcpy(&v, shared_bytes(address), bits / 8);
    return v;
}

// ---- instruction descriptor -----------------------------------------------
enum class Kind { f16, tf32, f8f6f4, i8, mxf8f6f4, mxf4, mxf4nvf4 };

struct Instr {
    Kind kind;
    Format a, b;
    unsigned m, n, k;   // k counts dense elements; a sparse A holds k/2 per row
    bool a_mn, b_mn, a_neg, b_neg, c_f32, c_s32, padded, scaled, ue8m0, sparse, saturate;
    unsigned a_sf_id, b_sf_id, selector, max_shift;
};

Format f16_format(unsigned v) {
    if (v == 0) return Format::f16;
    if (v == 1) return Format::bf16;
    fail("unsupported kind::f16 operand format");
}
Format f8f6f4_format(unsigned v) {
    switch (v) {
    case 0: return Format::e4m3; case 1: return Format::e5m2; case 3: return Format::e2m3;
    case 4: return Format::e3m2; case 5: return Format::e2m1;
    default: fail("unsupported f8f6f4 operand format");
    }
}

Instr decode_instr(Kind kind, uint32_t d) {
    Instr in{};
    in.kind = kind;
    in.scaled = kind == Kind::mxf8f6f4 || kind == Kind::mxf4 || kind == Kind::mxf4nvf4;
    in.sparse = (d >> 2) & 1;
    const unsigned c_format = (d >> 4) & 3, a_fmt = (d >> 7) & 7, b_fmt = (d >> 10) & 7;
    in.a_neg = (d >> 13) & 1;
    in.b_neg = (d >> 14) & 1;
    in.a_mn = (d >> 15) & 1;
    in.b_mn = (d >> 16) & 1;
    in.n = ((d >> 17) & 0x3f) << 3;
    in.m = ((d >> 24) & 0x1f) << 4;   // scaled kinds: bits [28:27] = M >> 7, bits [26:24] zero
    unsigned k_size = 0;
    if (in.scaled) {
        in.b_sf_id = (d >> 4) & 3;
        in.a_sf_id = (d >> 29) & 3;
        in.ue8m0 = (d >> 23) & 1;
        if (kind == Kind::mxf8f6f4) {
            if ((d >> 31) & 1) fail("kind::mxf8f6f4 descriptor bit 31 must be 0");
        } else {
            k_size = (((d >> 3) & 1) << 1) | ((d >> 31) & 1);   // bit 3 upper, bit 31 lower
            if ((d >> 12) & 1) fail("kind::mxf4* sparsity version must be 0");
        }
    } else {
        in.selector = d & 3;
        in.saturate = (d >> 3) & 1;
        in.max_shift = ((d >> 30) & 3) ? 4u << ((d >> 30) & 3) : 0;
        if (in.saturate && kind != Kind::i8) fail("tcgen05.mma saturate applies to kind::i8 only");
        if ((d >> 29) & 1) fail("tcgen05.mma descriptor bit 29 must be 0");
    }
    switch (kind) {
    case Kind::f16: in.a = f16_format(a_fmt); in.b = f16_format(b_fmt); in.k = 16; break;
    case Kind::tf32:
        if (a_fmt != 2 || b_fmt != 2) fail("kind::tf32 requires tf32 operands");
        in.a = in.b = Format::tf32; in.k = 8; break;
    case Kind::f8f6f4: case Kind::mxf8f6f4:
        in.a = f8f6f4_format(a_fmt); in.b = f8f6f4_format(b_fmt); in.k = 32; in.padded = true; break;
    case Kind::i8:
        in.a = a_fmt == 1 ? Format::s8 : Format::u8; in.b = b_fmt == 1 ? Format::s8 : Format::u8;
        in.k = 32; break;
    case Kind::mxf4: case Kind::mxf4nvf4:
        if (a_fmt != 1 || (b_fmt & 3) != 1) fail("kind::mxf4* requires e2m1 operands");
        if (k_size > 1) fail("kind::mxf4* K=128 dense (k_size 2) is not available on sm_100/sm_103");
        in.a = in.b = Format::e2m1; in.k = k_size ? 96 : 64;
        if (k_size && in.sparse) fail("kind::mxf4* K=96 has no sparse form");
        break;
    }
    if (in.sparse) in.k *= 2;
    if (in.scaled) in.c_f32 = true;
    else if (kind == Kind::i8) {
        if (c_format != 2) fail("kind::i8 requires an s32 accumulator");
        in.c_s32 = true;
    } else if (c_format == 1) in.c_f32 = true;
    else if (c_format != 0 || (kind != Kind::f16 && kind != Kind::f8f6f4))
        fail("unsupported accumulator format");
    if (kind == Kind::mxf8f6f4 && !in.ue8m0) fail("kind::mxf8f6f4 requires ue8m0 scales");
    if (kind == Kind::mxf4 && !in.ue8m0) fail("kind::mxf4 requires ue8m0 scales");
    return in;
}

Kind parse_kind(unsigned code) {
    switch (code) {
    case 0: return Kind::f16; case 1: return Kind::tf32; case 2: return Kind::f8f6f4;
    case 3: return Kind::i8; case 4: return Kind::mxf8f6f4; case 5: return Kind::mxf4;
    case 6: return Kind::mxf4nvf4;
    default: fail("unknown tcgen05.mma kind");
    }
}

// TMEM lane of local accumulator row `ml` and column of column n within one
// CTA, for PTX Layouts D (1-CTA M=128), F (1-CTA M=64), A (2-CTA M=256) and
// B (2-CTA M=128).
struct Placement { unsigned lane, column; };
Placement place(unsigned cta_group, unsigned m_total, unsigned n_total, unsigned ml, unsigned n) {
    if (cta_group == 1 && m_total == 128) return {ml, n};
    if (cta_group == 1 && m_total == 64) return {(ml % 16) + 32 * (ml / 16), n};
    if (cta_group == 2 && m_total == 256) return {ml, n};
    if (cta_group == 2 && m_total == 128) {
        const unsigned half = n_total / 2;
        return {ml + 64 * (n / half), n % half};
    }
    fail("unsupported tcgen05.mma M=" + std::to_string(m_total) + " for cta_group::" + std::to_string(cta_group));
}

// .32x32b, .16x64b, .16x128b, .16x256b, .16x32bx2 register-to-cell maps
// (lane offset, column offset) for thread t and register j.
Placement ld_cell(unsigned shape, unsigned t, unsigned j, unsigned imm) {
    switch (shape) {
    case 0: return {t, j};
    case 1: return {t / 4 + 8 * (t % 2), 2 * j + (t / 2) % 2};
    case 2: return {t / 4 + 8 * (j % 2), 4 * (j / 2) + t % 4};
    case 3: return {t / 4 + 8 * ((j / 2) % 2), 8 * (j / 4) + 2 * (t % 4) + j % 2};
    case 4: return {t % 16, j + (t / 16) * imm};
    default: fail("unknown tcgen05.ld/st shape");
    }
}

// Physical (lane, first column) of register j; with pack::16b the register
// covers the low halves of that column and the next. 16x32bx2's immediate
// already counts physical columns.
Placement ld_physical(unsigned shape, unsigned t, unsigned j, unsigned pack, unsigned imm) {
    if (!pack) return ld_cell(shape, t, j, imm);
    if (shape == 4) return {t % 16, 2 * j + (t / 16) * imm};
    const Placement p = ld_cell(shape, t, j, 0);
    return {p.lane, 2 * p.column};
}

}  // namespace

// ---- allocation -----------------------------------------------------------
// Warp-collective (one warp per CTA; with cta_group::2 one warp in each CTA of
// the pair, each allocating its own CTA's columns). Lane 0 performs the
// allocation and writes (lane 0, first column) to shared memory. A
// deterministic first-fit allocator gives both CTAs of a pair the same base.
extern "C" void gx_tcgen05_alloc(uint32_t dst, uint32_t columns, uint32_t cta_group) {
    if (cta_group != 1 && cta_group != 2) fail("invalid cta_group");
    if(cta_group==2) (void)gx_cluster_rank_linear_block(gx_dev_cluster_ctarank()^1u);
    if (columns < 32 || columns > 512 || (columns & (columns - 1))) fail("invalid allocation column count");
    if (lane_id() != 0) return;
    GxTmemState& t = own_tmem();
    const unsigned granules = columns / GxTmemState::kGranule;
    const uint32_t mask = granules >= 32 ? ~0u : ((1u << granules) - 1);
    for (unsigned first = 0; first + granules <= GxTmemState::kColumns / GxTmemState::kGranule; first += granules) {
        if (t.allocated & (mask << first)) continue;
        t.allocated |= mask << first;
        const uint32_t address = first * GxTmemState::kGranule;
        drperf_gpu_resource("shared_write_bytes",4);
        drperf_gpu_resource("tensor_memory_allocated_bytes",uint64_t(columns)*128*4);
        std::memcpy(shared_bytes(dst), &address, 4);
        return;
    }
    fail("tensor memory exhausted (GX gives each CTA its own 512 columns)");
}

extern "C" void gx_tcgen05_dealloc(uint32_t address, uint32_t columns, uint32_t cta_group) {
    if (cta_group != 1 && cta_group != 2) fail("invalid cta_group");
    if(cta_group==2) (void)gx_cluster_rank_linear_block(gx_dev_cluster_ctarank()^1u);
    if (lane_id() != 0) return;
    GxTmemState& t = own_tmem();
    const unsigned column = tmem_col(address), granules = columns / GxTmemState::kGranule;
    if (tmem_lane(address) || columns < 32 || (columns & (columns - 1)) || column % columns)
        fail("invalid deallocation");
    const uint32_t mask = (granules >= 32 ? ~0u : ((1u << granules) - 1)) << (column / GxTmemState::kGranule);
    if ((t.allocated & mask) != mask) fail("deallocating unallocated tensor memory");
    t.allocated &= ~mask;
}

// ---- tcgen05.ld / tcgen05.st -----------------------------------------------
// PTX 9.7.17.8.1: with tcgen05.ld/st, warp w%4 of a warpgroup can access only
// lanes 32*(w%4) .. 32*(w%4)+31. The access is placed in the executing warp's
// quarter. CUTLASS puts 32*(w%4) in the lane field; FlashMLA passes lane 0 from
// all four warps, which reaches each warp's own quarter on B300 (its outputs
// match an fp64 reference). Other lane fields outside the quarter are
// undefined in PTX and fail; so does an access running past the quarter.
// `span`: lanes of one access, 32 for .32x32b and 16 for the .16x* shapes.
static unsigned warp_quarter_lane(unsigned l0, unsigned span) {
    const unsigned quarter = (linear_thread() / 32) % 4;
    if (l0 != 0 && l0 / 32 != quarter)
        fail("tcgen05.ld/st lane " + std::to_string(l0) + " is outside warp quarter " + std::to_string(quarter));
    if (l0 % 32 + span > 32) fail("tcgen05.ld/st access crosses the warp's 32-lane quarter");
    return 32 * quarter + l0 % 32;
}
static unsigned ld_span(unsigned shape) { return shape == 0 ? 32 : 16; }

// shape: 0 .32x32b, 1 .16x64b, 2 .16x128b, 3 .16x256b, 4 .16x32bx2; `regs`
// registers per thread. pack (ld .pack::16b / st .unpack::16b) moves the low
// 16 bits of two adjacent columns per register; st keeps the high halves.
extern "C" void gx_tcgen05_ld(uint32_t shape, uint32_t address, uint32_t regs, uint32_t pack,
                              uint32_t imm, uint32_t* out) {
    GxTmemState& t = own_tmem();
    const unsigned l0 = warp_quarter_lane(tmem_lane(address), ld_span(shape)), c0 = tmem_col(address), lane = lane_id();
    drperf_gpu_resource("tmem_read_bytes", uint64_t(regs)*4);
    for (unsigned j = 0; j < regs; ++j) {
        const Placement p = ld_physical(shape, lane, j, pack, imm);
        if (!pack) out[j] = cell(t, l0 + p.lane, c0 + p.column);
        else out[j] = (cell(t, l0 + p.lane, c0 + p.column) & 0xffff) |
                      (cell(t, l0 + p.lane, c0 + p.column + 1) << 16);
    }
}

extern "C" void gx_tcgen05_st(uint32_t shape, uint32_t address, uint32_t regs, uint32_t pack,
                              uint32_t imm, const uint32_t* in) {
    GxTmemState& t = own_tmem();
    const unsigned l0 = warp_quarter_lane(tmem_lane(address), ld_span(shape)), c0 = tmem_col(address), lane = lane_id();
    drperf_gpu_resource("tmem_write_bytes", uint64_t(regs)*4);
    for (unsigned j = 0; j < regs; ++j) {
        const Placement p = ld_physical(shape, lane, j, pack, imm);
        if (!pack) { cell(t, l0 + p.lane, c0 + p.column) = in[j]; continue; }
        uint32_t& lo = cell(t, l0 + p.lane, c0 + p.column);
        uint32_t& hi = cell(t, l0 + p.lane, c0 + p.column + 1);
        lo = (lo & 0xffff0000u) | (in[j] & 0xffff);
        hi = (hi & 0xffff0000u) | (in[j] >> 16);
    }
}

// ---- tcgen05.cp ---------------------------------------------------------------
// shape: 0 .128x256b, 1 .128x128b, 2 .4x256b, 3 .64x128b.warpx2::02_13,
// 4 .64x128b.warpx2::01_23, 5 .32x128b.warpx4. decompress: 0 none,
// 1 .b8x16.b4x16_p64 (nibble -> byte << 2), 2 .b8x16.b6x16_p32 (6 bits -> byte).
// Source rows are K-major canonical rows of the descriptor; byte b of a row
// lands in column C0 + b/4. cta_group::2 copies in both CTAs of the pair, each
// from its own shared memory into its own tensor memory.
extern "C" void gx_tcgen05_cp(uint32_t shape, uint32_t decompress, uint32_t cta_group,
                              uint32_t address, uint64_t desc) {
    static const unsigned rows_of[6] = {128, 128, 4, 64, 64, 32}, bytes_of[6] = {32, 16, 32, 16, 16, 16};
    if (shape > 5 || decompress > 2) fail("unknown tcgen05.cp form");
    if (cta_group != 1 && cta_group != 2) fail("invalid cta_group");
    if(cta_group==2) (void)gx_cluster_rank_linear_block(gx_dev_cluster_ctarank()^1u);
    const unsigned rows = rows_of[shape], bytes = bytes_of[shape];
    const unsigned l0 = tmem_lane(address), c0 = tmem_col(address);
    const unsigned members = cta_group == 2 ? 2 : 1;
    if(!decompress)drperf_gpu_resource("shared_read_bytes",uint64_t(rows)*bytes*members);
    if(!decompress)drperf_gpu_resource("tmem_write_bytes",uint64_t((shape>=3 ? 128 : rows))*bytes*members);
    if(decompress)drperf_gpu_resource("unmodeled_work_events",1);
    for (unsigned v = 0; v < members; ++v) {
        const uint32_t rank = member_rank(v, cta_group);
        const MatrixDesc m = decode_desc(desc, rank);
        GxTmemState& t = tmem_of(rank);
        for (unsigned r = 0; r < rows; ++r) {
            uint8_t row[32];
            for (unsigned chunk = 0; chunk < bytes / 16; ++chunk) {
                uint8_t src[17] = {};
                for (unsigned b = 0; b < 16; ++b) src[b] = *shared_bytes(k_major_byte(m, r, chunk * 16 + b));
                uint8_t* dst = row + chunk * 16;
                if (decompress == 0) { std::memcpy(dst, src, 16); continue; }
                const unsigned w = decompress == 1 ? 4 : 6;
                for (unsigned i = 0; i < 16; ++i) {
                    const unsigned bit = i * w, raw = (src[bit / 8] | (src[bit / 8 + 1] << 8)) >> (bit % 8);
                    dst[i] = decompress == 1 ? uint8_t((raw & 0xf) << 2) : uint8_t(raw & 0x3f);
                }
            }
            unsigned lanes[4], count = 0;
            switch (shape) {
            case 0: case 1: lanes[count++] = l0 + r; break;
            case 2: lanes[count++] = l0 + 32 * r; break;
            case 3: lanes[count++] = l0 + r; lanes[count++] = l0 + r + 64; break;
            case 4: {
                const unsigned base = r < 32 ? l0 + r : l0 + (r - 32) + 64;
                lanes[count++] = base; lanes[count++] = base + 32; break;
            }
            case 5: for (unsigned s = 0; s < 4; ++s) lanes[count++] = l0 + r + 32 * s; break;
            }
            for (unsigned i = 0; i < count; ++i)
                for (unsigned c = 0; c < bytes / 4; ++c) {
                    uint32_t word;
                    std::memcpy(&word, row + 4 * c, 4);
                    cell(t, lanes[i], c0 + c) = word;
                }
        }
    }
}

// ---- tcgen05.commit ---------------------------------------------------------
// Prior asynchronous tcgen05 operations of this thread have completed at
// issue, so the arrival is immediate. Multicast arrives on the mbarrier at the
// same offset in every CTA of the mask.
extern "C" void gx_tcgen05_commit(uint32_t mbar, uint32_t multicast, uint32_t mask) {
    // The latency model times the arrival at the issuing CTA's tensor-pipe drain.
    if (!multicast) { gx_mbarrier_arrive_tensor(gx_cluster_shared_ptr(mbar)); return; }
    for (uint32_t rank = 0; rank < 16; ++rank)
        if (mask & (1u << rank)) gx_mbarrier_arrive_tensor(gx_cluster_shared_ptr((rank << 24) | (mbar & 0xffffff)));
}

// ---- tcgen05.mma ------------------------------------------------------------
namespace {

// Weight-stationary (.ws) accumulator placement: Layouts D (M=128), E (M=64,
// 2x2 datapaths) and G (M=32, 1x4 datapaths).
Placement place_ws(unsigned m_total, unsigned n_total, unsigned ml, unsigned n) {
    switch (m_total) {
    case 128: return {ml, n};
    case 64: return {ml + 64 * (n / (n_total / 2)), n % (n_total / 2)};
    case 32: return {ml + 32 * (n / (n_total / 4)), n % (n_total / 4)};
    default: fail("tcgen05.mma.ws M must be 32, 64 or 128");
    }
}

// Zero-column mask (PTX "Zero-Column Mask Descriptor"): bit n set means
// column n of B reads as zero. S = 128/M sub-masks of N/S bits, each an
// LSB-first sequence of alternating runs: skip+1 ones, use+1 zeros, starting
// with ones if fs_i, the first run shortened by sc_i.
std::vector<bool> zero_column_mask(uint64_t desc, unsigned m_total, unsigned n_total) {
    std::vector<bool> mask(n_total, false);
    if (!((desc >> 39) & 1)) return mask;
    const unsigned subs = 128 / m_total, width = n_total / subs;
    const unsigned skip = unsigned((desc >> 40) & 0xff) + 1, use = unsigned((desc >> 48) & 0xff) + 1;
    for (unsigned i = 0; i < subs; ++i) {
        bool ones = (desc >> (32 + i)) & 1;
        const unsigned sc = unsigned((desc >> (8 * i)) & 0xff);
        unsigned run = ones ? skip : use;
        if (sc >= run) fail("zero-column mask start count exceeds its first span");
        run -= sc;
        for (unsigned j = 0; j < width; ++j) {
            mask[i * width + j] = ones;
            if (--run == 0) { ones = !ones; run = ones ? skip : use; }
        }
    }
    return mask;
}

// Sparse A metadata nibble for row m (local to the CTA, M=128 per CTA) and
// dense-K group g (PTX "Sparse matrices" metadata layouts).
unsigned sparse_nibble(GxTmemState& t, uint32_t meta, const Instr& in, unsigned m, unsigned g) {
    const unsigned l0 = tmem_lane(meta), c0 = tmem_col(meta);
    unsigned lane, col, nib;
    if (in.kind == Kind::f16 || in.kind == Kind::tf32) {
        lane = l0 + 32 * (m / 32) + 16 * ((m % 32) / 16) + m % 8 + 8 * (g / 4);
        col = c0 + in.selector;
        nib = 4 * ((m / 8) % 2) + g % 4;
    } else {
        lane = l0 + m; col = c0 + g / 8; nib = g % 8;
    }
    return (cell(t, lane, col) >> (4 * nib)) & 0xf;
}

// One validated tcgen05.mma (both implementations below start from it).
struct MmaOp {
    Kind kind;
    Instr in;
    unsigned cta_group, members, rows, cols, vs, nsf, kc, shift;
    bool sp, ws, ashift;
    std::vector<bool> zero_col;
};

MmaOp prepare_mma(uint32_t kind_code, uint32_t cta_group, uint32_t a_is_tmem, uint32_t idesc,
                  uint32_t sfa_address, uint32_t sfb_address, uint32_t block, const uint32_t* lane_mask,
                  uint32_t mask_words, uint32_t scale_d, uint32_t flags, uint64_t zcm_desc) {
    if (cta_group != 1 && cta_group != 2) fail("invalid cta_group");
    if(cta_group==2) (void)gx_cluster_rank_linear_block(gx_dev_cluster_ctarank()^1u);
    const bool sp = flags & 1, ws = flags & 2, ashift = flags & 4, has_zcm = flags & 8;
    const Kind kind = parse_kind(kind_code);
    const Instr in = decode_instr(kind, idesc);
    if (in.sparse != sp) fail("idesc sparsity does not match the .sp qualifier");
    const unsigned members = cta_group == 2 ? 2 : 1;
    if (cta_group == 2 && (gx_dev_cluster_ctarank() & 1)) fail("cta_group::2 tcgen05.mma issued by the odd CTA");
    if (lane_mask && mask_words != 4 * members) fail("disable-output-lane needs 4 words per CTA");
    if (ws) {
        if (cta_group != 1 || in.scaled) fail("tcgen05.mma.ws is cta_group::1 without block scaling");
        if (in.n != 64 && in.n != 128 && in.n != 256) fail("tcgen05.mma.ws N must be 64, 128 or 256");
        if (sp && in.m != 128) fail("sparse tcgen05.mma.ws metadata layouts for M < 128 are not specified");
        (void)place_ws(in.m, in.n, 0, 0);
    } else {
        (void)place(cta_group, in.m, in.n, 0, 0);  // validates the shape
        if (in.max_shift) fail("idesc max_shift applies to tcgen05.mma.ws only");
    }
    if (scale_d && (scale_d > 15 || (kind != Kind::f16 && kind != Kind::tf32)))
        fail("scale-input-d is 0..15 and only for kind::f16 / kind::tf32");
    if (ashift && (!a_is_tmem || (in.m != 128 && in.m != 256)))
        fail(".ashift requires A in tensor memory and M = 128 or 256");
    if (sp && in.m / members != 128) fail("sparse tcgen05.mma metadata layouts for 64 rows per CTA are not specified");
    const unsigned rows = in.m / members, cols = in.n / members;
    const unsigned vs = !in.scaled ? 1 : (block ? block : (kind == Kind::mxf4nvf4 ? 16 : 32)) * (sp ? 2 : 1);
    if (in.scaled && block && block != 16 && block != 32) fail("invalid block-scale vector size");
    if (in.scaled && (!sfa_address && !sfb_address)) fail("block-scaled tcgen05.mma without scale factors");
    const unsigned nsf = in.scaled ? in.k / vs : 0;
    if (nsf > 8) fail("too many scale factors per row");
    const unsigned kc = sp ? in.k / 2 : in.k;   // A elements stored per row

    // Zero-column mask and column shift (.ws only).
    std::vector<bool> zero_col(in.n, false);
    unsigned shift = 0;
    if (has_zcm) {
        if (!ws) fail("zero-column-mask-desc is a tcgen05.mma.ws operand");
        if ((zcm_desc >> 36) & 7) fail("zero-column mask reserved bits are set");
        zero_col = zero_column_mask(zcm_desc, in.m, in.n);
        shift = unsigned((zcm_desc >> 56) & 0x3f);
        if (shift > (in.m == 32 ? 16u : 32u)) fail("zero-column mask shift exceeds its maximum");
    }
    return MmaOp{kind, in, cta_group, members, rows, cols, vs, nsf, kc, shift, sp, ws, ashift, std::move(zero_col)};
}

// tcgen05.shift / .ashift ("shift down", measured on B300): within each 32-lane
// subpartition, lane r takes the cells of lane r + 1 and the subpartition's
// last lane keeps its contents. Lanes [l0, l0 + lanes), columns [c0, c0 + ncols).
void shift_rows_down(GxTmemState& t, unsigned l0, unsigned lanes, unsigned c0, unsigned ncols) {
    for (unsigned r = 0; r + 1 < lanes; ++r) {
        const unsigned lane = l0 + r;
        if (lane % 32 == 31) continue;
        for (unsigned c = 0; c < ncols; ++c) cell(t, lane, c0 + c) = cell(t, lane + 1, c0 + c);
    }
}

// The instruction element by element, straight from shared and tensor
// memory. It defines the results: the fast path below must match it bit for
// bit (GX_MMA_CHECK=1 compares the two on every instruction;
// GX_MMA_REFERENCE=1 runs only this one).
void mma_reference(const MmaOp& op, uint32_t d_address, uint64_t a_operand, uint32_t a_is_tmem,
                   uint64_t b_desc, uint32_t accumulate, uint32_t sfa_address, uint32_t sfb_address,
                   const uint32_t* lane_mask, uint32_t scale_d, uint32_t sp_meta) {
    const Kind kind = op.kind;
    const Instr& in = op.in;
    const unsigned cta_group = op.cta_group, members = op.members, rows = op.rows, cols = op.cols;
    const unsigned vs = op.vs, nsf = op.nsf, kc = op.kc, shift = op.shift;
    const bool sp = op.sp, ws = op.ws, ashift = op.ashift;
    const std::vector<bool>& zero_col = op.zero_col;

    // Decode B once: columns n in [0, N), each from its owning CTA.
    std::vector<double> bvals(size_t(in.n) * in.k);
    for (unsigned vb = 0; vb < members; ++vb) {
        const MatrixDesc bd = decode_desc(b_desc, member_rank(vb, cta_group));
        for (unsigned nl = 0; nl < cols; ++nl) {
            const unsigned n = vb * cols + nl;
            if (zero_col[n]) continue;
            for (unsigned k = 0; k < in.k; ++k) {
                const double y = decode(in.b, read_element(bd, in.b_mn, in.b, in.padded, nl + shift, k));
                bvals[size_t(n) * in.k + k] = in.b_neg ? -y : y;
            }
        }
    }

    const unsigned dl = tmem_lane(d_address), dc = tmem_col(d_address);
    std::vector<uint32_t> abits(kc), dense(in.k);
    std::vector<double> avals(in.k);
    double sfa[8], sfb[8];
    for (unsigned v = 0; v < members; ++v) {
        const uint32_t rank = member_rank(v, cta_group);
        GxTmemState& t = tmem_of(rank);
        const MatrixDesc ad = a_is_tmem ? MatrixDesc{} : decode_desc(a_operand, rank);
        const unsigned a_bits = in.padded ? 8 : format_bits(in.a);
        const unsigned al = tmem_lane(uint32_t(a_operand)), ac = tmem_col(uint32_t(a_operand));
        for (unsigned ml = 0; ml < rows; ++ml) {
            const Placement row = ws ? place_ws(in.m, in.n, ml, 0) : place(cta_group, in.m, in.n, ml, 0);
            // A in tensor memory is read by the datapath that computes the
            // output: lane al + row lane, which for .ws M=64/32 (Layouts E/G,
            // 2x2 and 1x4 datapaths) depends on the N group of the output.
            unsigned loaded_lane = ~0u;
            auto load_a = [&](unsigned a_lane) {
                if (a_lane == loaded_lane) return;
                loaded_lane = a_lane;
                for (unsigned k = 0; k < kc; ++k) {
                    if (!a_is_tmem) { abits[k] = read_element(ad, in.a_mn, in.a, in.padded, ml, k); continue; }
                    // A in tensor memory: K-major, packed into 32-bit cells; f8f6f4
                    // sub-byte formats use 8-bit containers (e2m1 in bits 5:2).
                    const uint32_t c = cell(t, a_lane, ac + (k * a_bits) / 32);
                    uint32_t bits = a_bits == 32 ? c : (c >> ((k * a_bits) % 32)) & ((1u << a_bits) - 1);
                    if (in.padded && in.a == Format::e2m1) bits >>= 2;
                    abits[k] = bits;
                }
                if (!sp) {
                    for (unsigned k = 0; k < in.k; ++k) dense[k] = abits[k];
                } else {
                    // Decompress: each metadata nibble names the two kept units
                    // (idx0 = bits 1:0, idx1 = bits 3:2) of a group of four.
                    std::fill(dense.begin(), dense.end(), 0u);
                    if (kind == Kind::tf32) {           // unit = 16-bit half of a tf32
                        for (unsigned g = 0; g < in.k / 2; ++g) {
                            const unsigned nib = sparse_nibble(t, sp_meta, in, ml, g);
                            const unsigned idx[2] = {nib & 3, nib >> 2};
                            for (unsigned u = 0; u < 2; ++u) {
                                const unsigned cu = 2 * g + u;
                                const uint32_t half = (abits[cu / 2] >> (16 * (cu % 2))) & 0xffff;
                                dense[2 * g + idx[u] / 2] |= half << (16 * (idx[u] % 2));
                            }
                        }
                    } else if (in.a == Format::e2m1 && !in.padded) {   // unit = pair of e2m1
                        for (unsigned g = 0; g < in.k / 8; ++g) {
                            const unsigned nib = sparse_nibble(t, sp_meta, in, ml, g);
                            const unsigned idx[2] = {nib & 3, nib >> 2};
                            for (unsigned u = 0; u < 2; ++u)
                                for (unsigned e = 0; e < 2; ++e)
                                    dense[8 * g + 2 * idx[u] + e] = abits[2 * (2 * g + u) + e];
                        }
                    } else {                                         // unit = one element
                        for (unsigned g = 0; g < in.k / 4; ++g) {
                            const unsigned nib = sparse_nibble(t, sp_meta, in, ml, g);
                            dense[4 * g + (nib & 3)] = abits[2 * g];
                            dense[4 * g + (nib >> 2)] = abits[2 * g + 1];
                        }
                    }
                }
                for (unsigned k = 0; k < in.k; ++k) {
                    const double x = decode(in.a, dense[k]);
                    avals[k] = in.a_neg ? -x : x;
                }
            };
            auto a_lane_of = [&](unsigned n) {
                return a_is_tmem ? al + (ws ? place_ws(in.m, in.n, ml, n).lane : row.lane) : 0u;
            };
            load_a(a_lane_of(0));
            const unsigned replica = 32 * ((row.lane / 32) % 4);
            for (unsigned j = 0; j < nsf; ++j) {
                const unsigned sid = in.a_sf_id + j;
                const uint32_t c = cell(t, tmem_lane(sfa_address) + ml % 32 + replica,
                                        tmem_col(sfa_address) + ml / 32 + 4 * (sid / 4));
                sfa[j] = decode_scale(c >> (8 * (sid % 4)), in.ue8m0);
            }
            for (unsigned n = 0; n < in.n; ++n) {
                const Placement p = ws ? place_ws(in.m, in.n, ml, n) : place(cta_group, in.m, in.n, ml, n);
                const unsigned lane = dl + p.lane;
                if (lane_mask && ((lane_mask[4 * v + lane / 32] >> (lane % 32)) & 1)) continue;
                load_a(a_lane_of(n));
                const double* b = &bvals[size_t(n) * in.k];
                double sum = 0;
                if (nsf) {
                    const unsigned group = in.n > 128 ? 8 : 4;
                    for (unsigned j = 0; j < nsf; ++j) {
                        const unsigned sid = in.b_sf_id + j;
                        const uint32_t c = cell(t, tmem_lane(sfb_address) + n % 32 + 32 * ((p.lane / 32) % 4),
                                                tmem_col(sfb_address) + n / 32 + group * (sid / 4));
                        sfb[j] = decode_scale(c >> (8 * (sid % 4)), in.ue8m0);
                    }
                    for (unsigned k = 0; k < in.k; ++k) sum += (avals[k] * sfa[k / vs]) * (b[k] * sfb[k / vs]);
                } else {
                    for (unsigned k = 0; k < in.k; ++k) sum += avals[k] * b[k];
                }
                uint32_t& c = cell(t, lane, dc + p.column);
                if (in.c_s32) {
                    const int64_t r = (accumulate ? int64_t(int32_t(c)) : 0) + int64_t(sum);
                    c = uint32_t(int32_t(in.saturate ? (r > INT32_MAX ? INT32_MAX : r < INT32_MIN ? INT32_MIN : r) : r));
                } else if (in.c_f32) {
                    const double prior = accumulate ? std::ldexp(double(as_float(c)), -int(scale_d)) : 0.0;
                    c = as_bits(float(prior + sum));
                } else {
                    // f16 accumulator in the low 16 bits of the cell.
                    const double prior = accumulate ? std::ldexp(decode(Format::f16, c & 0xffff), -int(scale_d)) : 0.0;
                    c = (c & 0xffff0000u) | f16_bits(prior + sum);
                }
            }
        }
        // .ashift: A moves after this MMA has used it (B300-measured).
        if (ashift) shift_rows_down(t, al, rows, ac, (kc * a_bits + 31) / 32);
    }
}

// ---- tcgen05.mma fast path --------------------------------------------------
// The reference's arithmetic, reorganized. Operands are gathered once per
// instruction (whole 16-byte shared-memory units, table decodes of the same
// values `decode` returns); block scales are applied to the operands in the
// reference's parenthesization, (a * sfa) * (b * sfb); every output is still
// the sum of its products from +0 in increasing k, in double, and is rounded
// to D exactly as before. Outputs are independent, so rows run on worker
// threads and columns in SIMD lanes; no output's k-sum is split or reordered.
// All shared and tensor memory is read before D is written, so instructions
// whose other tensor-memory operands share columns with D take the reference.

// Exact decode of every code of a format of at most 8 bits, or null for
// f16, bf16 and tf32 (decoded directly).
const double* decode_table(Format f) {
    auto build = [](Format g) {
        std::vector<double> v(size_t(1) << format_bits(g));
        for (uint32_t i = 0; i < v.size(); ++i) v[i] = decode(g, i);
        return v;
    };
    static const std::vector<double> tables[] = {
        {}, {}, {}, build(Format::e4m3), build(Format::e5m2), build(Format::e2m3),
        build(Format::e3m2), build(Format::e2m1), build(Format::u8), build(Format::s8)};
    const std::vector<double>& t = tables[unsigned(f)];
    return t.empty() ? nullptr : t.data();
}

// f16 without a table: (1024 | m) * 2^(e - 25), or m * 2^-24 if subnormal.
// Both factors are exact and so is their product, so this is the value
// decode's ldexp returns; exponent 31 (Inf/NaN) goes to decode.
constexpr double pow2(int n) {
    double v = 1;
    for (; n > 0; --n) v *= 2;
    for (; n < 0; ++n) v /= 2;
    return v;
}
constexpr double kF16Scale[31] = {
    pow2(-24), pow2(-24), pow2(-23), pow2(-22), pow2(-21), pow2(-20), pow2(-19), pow2(-18), pow2(-17), pow2(-16),
    pow2(-15), pow2(-14), pow2(-13), pow2(-12), pow2(-11), pow2(-10), pow2(-9), pow2(-8), pow2(-7), pow2(-6),
    pow2(-5), pow2(-4), pow2(-3), pow2(-2), pow2(-1), pow2(0), pow2(1), pow2(2), pow2(3), pow2(4), pow2(5)};
double decode_f16(uint32_t bits) {
    const uint32_t e = (bits >> 10) & 31, m = bits & 1023;
    if (e == 31) return decode(Format::f16, bits);
    const double v = double(e ? m | 1024 : m) * kF16Scale[e];
    return bits & 0x8000 ? -v : v;
}

// decode_scale of every UE8M0 / UE4M3 code.
double scale_of(uint32_t code, bool ue8m0) {
    static const auto tables = [] {
        std::vector<double> t(512);
        for (uint32_t i = 0; i < 256; ++i) { t[i] = decode_scale(i, false); t[256 + i] = decode_scale(i, true); }
        return t;
    }();
    return tables[(ue8m0 ? 256 : 0) + (code & 0xff)];
}

struct Decoder {
    Format f;
    const double* table;
    uint32_t mask;
    explicit Decoder(Format g)
        : f(g), table(decode_table(g)), mask(table ? (1u << format_bits(g)) - 1 : 0) {}
    // decode(f, bits) depends only on the format's low bits.
    double operator()(uint32_t bits) const {
        return table ? table[bits & mask] : f == Format::f16 ? decode_f16(bits) : decode(f, bits);
    }
};

// Host view of one CTA's shared window. Once the runtime has resolved (and
// bounds-checked) an offset, every lower offset of that CTA maps linearly
// from the same base, so later accesses below it skip the runtime call.
struct SharedView {
    uint32_t rank = ~0u, limit = 0;
    const uint8_t* base = nullptr;
    const uint8_t* at(uint32_t address) {
        const uint32_t r = address >> 24, offset = address & 0xffffff;
        if (r == rank && offset < limit) return base + offset;
        const uint8_t* p = shared_bytes(address);
        if (r != rank) { rank = r; limit = 0; }
        base = p - offset;
        limit = std::max(limit, offset + 1);
        return p;
    }
};

// Codes of elements [0, count) of operand row `row`: read_element for each k.
// A K-major row is copied in whole 16-byte units first: swizzling permutes
// 16-byte units only, so a unit's bytes are contiguous, and shared windows
// are page-rounded, so a unit whose first byte is in the window is in it.
void read_row(SharedView& view, const MatrixDesc& m, bool mn_major, Format f, bool padded,
              unsigned row, unsigned count, uint32_t* out) {
    const unsigned bits = format_bits(f);
    if (!count) return;
    if (!mn_major) {
        auto bit_of = [&](unsigned k) {
            return padded && bits < 8 ? uint64_t(k / 16) * 128 + (k % 16) * bits : uint64_t(k) * bits;
        };
        const unsigned units = unsigned((bit_of(count - 1) + bits - 1) / 128) + 1;
        uint8_t bytes[8 * 16 + 2] = {};
        if (units > 8) fail("operand row longer than 128 bytes");
        for (unsigned u = 0; u < units; ++u) std::memcpy(bytes + 16 * u, view.at(k_major_byte(m, row, 16 * u)), 16);
        for (unsigned k = 0; k < count; ++k) {
            const uint64_t bit = bit_of(k);
            const uint8_t* p = bytes + bit / 8;
            if (bits >= 8) { uint32_t v = 0; std::memcpy(&v, p, bits / 8); out[k] = v; continue; }
            uint32_t w = p[0];
            if ((bit % 8) + bits > 8) w |= uint32_t(p[1]) << 8;
            out[k] = (w >> (bit % 8)) & ((1u << bits) - 1);
        }
        return;
    }
    if (bits < 8) fail("MN-major operands of sub-byte formats are invalid");
    if (m.lbo_absolute) fail("descriptor lbo_mode=1 requires a K-major operand");
    for (unsigned k = 0; k < count; ++k) {
        const uint64_t mb = uint64_t(row) * bits, mu = mb / 128, k8 = k % 8, kg = k / 8;
        uint64_t unit;
        if (m.layout == 0) unit = k8 * 16 + mu * m.sbo + kg * m.lbo;
        else unit = (mu % m.atom_units + k8 * m.atom_units) * 16 + (mu / m.atom_units) * m.lbo + kg * m.sbo;
        const uint32_t address = m.cta_bits | swizzle(uint32_t(m.start + unit + (mb % 128) / 8), m.swizzle_bits, m.base_offset);
        uint32_t v = 0;
        std::memcpy(&v, view.at(address), bits / 8);
        out[k] = v;
    }
}

// x * y and x + y with x as the first source operand. When both operands
// are NaN, x86 returns the first one's payload, so the fast path fixes the
// operand order the reference compiles to: products A side first, sums
// product first (then the running sum), D rounding prior first.
inline double mul_first(double x, double y) { asm("mulsd %1, %0" : "+x"(x) : "xm"(y)); return x; }
inline double add_first(double x, double y) { asm("addsd %1, %0" : "+x"(x) : "xm"(y)); return x; }

// sum + a * b as the reference's loop computes it: (a * b) + sum.
inline double mac(double sum, double a, double b) {
    asm("mulsd %1, %0\n\taddsd %2, %0" : "+x"(a) : "xm"(b), "xm"(sum));
    return a;
}

// out[n] = sum over k of a[k] * bt[k * ld + n] for n in [n0, n1): each sum
// starts at +0 and adds its products in increasing k, as the reference does;
// SIMD lanes hold different n. Separate multiplies and adds (no FMA), so
// every lane rounds exactly like the reference's scalar code.
void row_dot_scalar(const double* a, const double* bt, size_t ld, unsigned K, unsigned n0, unsigned n1,
                    double* out) {
    for (unsigned n = n0; n < n1; ++n) {
        double sum = 0;
        for (unsigned k = 0; k < K; ++k) sum = mac(sum, a[k], bt[k * ld + n]);
        out[n] = sum;
    }
}

__attribute__((target("avx2"))) inline __m256d mac4(__m256d sum, __m256d a, __m256d b) {
    __m256d p;
    asm("vmulpd %2, %1, %0\n\tvaddpd %3, %0, %0" : "=&x"(p) : "x"(a), "xm"(b), "x"(sum));
    return p;
}

__attribute__((target("avx2")))
void row_dot_avx2(const double* a, const double* bt, size_t ld, unsigned K, unsigned n0, unsigned n1,
                  double* out) {
    unsigned n = n0;
    for (; n + 32 <= n1; n += 32) {
        __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0, s4 = s0, s5 = s0, s6 = s0, s7 = s0;
        const double* b = bt + n;
        for (unsigned k = 0; k < K; ++k, b += ld) {
            const __m256d x = _mm256_broadcast_sd(a + k);
            s0 = mac4(s0, x, _mm256_loadu_pd(b));
            s1 = mac4(s1, x, _mm256_loadu_pd(b + 4));
            s2 = mac4(s2, x, _mm256_loadu_pd(b + 8));
            s3 = mac4(s3, x, _mm256_loadu_pd(b + 12));
            s4 = mac4(s4, x, _mm256_loadu_pd(b + 16));
            s5 = mac4(s5, x, _mm256_loadu_pd(b + 20));
            s6 = mac4(s6, x, _mm256_loadu_pd(b + 24));
            s7 = mac4(s7, x, _mm256_loadu_pd(b + 28));
        }
        double* o = out + n;
        _mm256_storeu_pd(o, s0); _mm256_storeu_pd(o + 4, s1); _mm256_storeu_pd(o + 8, s2); _mm256_storeu_pd(o + 12, s3);
        _mm256_storeu_pd(o + 16, s4); _mm256_storeu_pd(o + 20, s5); _mm256_storeu_pd(o + 24, s6); _mm256_storeu_pd(o + 28, s7);
    }
    for (; n + 4 <= n1; n += 4) {
        __m256d s = _mm256_setzero_pd();
        const double* b = bt + n;
        for (unsigned k = 0; k < K; ++k, b += ld) s = mac4(s, _mm256_broadcast_sd(a + k), _mm256_loadu_pd(b));
        _mm256_storeu_pd(out + n, s);
    }
    for (; n < n1; ++n) {
        __m128d s = _mm_setzero_pd();
        for (unsigned k = 0; k < K; ++k) {
            __m128d p;
            asm("vmulsd %2, %1, %0\n\tvaddsd %3, %0, %0" : "=&x"(p) : "x"(_mm_set_sd(a[k])), "x"(_mm_set_sd(bt[k * ld + n])), "x"(s));
            s = p;
        }
        out[n] = _mm_cvtsd_f64(s);
    }
}

// out[n] = b[n] * f[n] (b first, as mul_first), n in [0, N).
void scale_row_scalar(const double* b, const double* f, unsigned N, double* out) {
    for (unsigned n = 0; n < N; ++n) out[n] = mul_first(b[n], f[n]);
}

__attribute__((target("avx2")))
void scale_row_avx2(const double* b, const double* f, unsigned N, double* out) {
    unsigned n = 0;
    for (; n + 4 <= N; n += 4) {
        __m256d p;
        asm("vmulpd %2, %1, %0" : "=x"(p) : "x"(_mm256_loadu_pd(b + n)), "xm"(_mm256_loadu_pd(f + n)));
        _mm256_storeu_pd(out + n, p);
    }
    scale_row_scalar(b + n, f + n, N - n, out + n);
}

void scale_row(const double* b, const double* f, unsigned N, double* out) {
    static const bool avx2 = __builtin_cpu_supports("avx2");
    if (avx2) scale_row_avx2(b, f, N, out);
    else scale_row_scalar(b, f, N, out);
}

void row_dot(const double* a, const double* bt, size_t ld, unsigned K, unsigned n0, unsigned n1, double* out) {
    static const bool avx2 = __builtin_cpu_supports("avx2");
    if (avx2) row_dot_avx2(a, bt, ld, K, n0, n1, out);
    else row_dot_scalar(a, bt, ld, K, n0, n1, out);
}

// Worker threads for the rows of large MMAs (GX_MMA_THREADS, default up to
// 8; 0 or 1 disables them). A job is pure arithmetic on arrays the issuing
// thread gathered beforehand: workers never touch fibers, GX thread-local
// state or shared/tensor memory, and run() returns only after every worker
// has finished the job. Workers take the issuing thread's MXCSR. Jobs from
// several issuing threads run one at a time.
class RowPool {
public:
    using Job = void (*)(void* ctx, unsigned index);

    // Null when disabled. A forked child starts a new pool on first use.
    static RowPool* get() {
        static const unsigned workers = [] {
            const char* e = std::getenv("GX_MMA_THREADS");
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            const unsigned n = e && *e ? unsigned(std::strtoul(e, nullptr, 10)) : std::min(8u, std::max(1u, hw / 4));
            pthread_atfork(nullptr, nullptr, [] { instance() = nullptr; });
            return n > 1 ? n - 1 : 0;   // the issuing thread works too
        }();
        if (!workers) return nullptr;
        if (!instance()) instance() = new RowPool(workers);   // never freed: workers run until exit
        return instance();
    }

    void run(unsigned count, Job fn, void* ctx) {
        std::lock_guard<std::mutex> one_job(run_mutex_);
        std::unique_lock<std::mutex> lock(mutex_);
        fn_ = fn; ctx_ = ctx; count_ = count; mxcsr_ = _mm_getcsr();
        next_.store(0, std::memory_order_relaxed);
        active_ = workers_;
        ++generation_;
        lock.unlock();
        wake_.notify_all();
        work();
        lock.lock();
        done_.wait(lock, [&] { return active_ == 0; });
    }

private:
    explicit RowPool(unsigned workers) {
        try {
            for (; workers_ < workers; ++workers_) std::thread([this] { loop(); }).detach();
        } catch (const std::system_error&) {}   // run with the workers that started
    }
    static RowPool*& instance() { static RowPool* p = nullptr; return p; }

    void work() {
        for (unsigned i; (i = next_.fetch_add(1, std::memory_order_relaxed)) < count_;) fn_(ctx_, i);
    }
    void loop() {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            wake_.wait(lock, [&] { return generation_ != seen; });
            seen = generation_;
            lock.unlock();
            _mm_setcsr(mxcsr_);
            work();
            lock.lock();
            if (--active_ == 0) done_.notify_one();
        }
    }

    unsigned workers_ = 0;
    std::mutex run_mutex_, mutex_;
    std::condition_variable wake_, done_;
    uint64_t generation_ = 0;
    unsigned active_ = 0, count_ = 0, mxcsr_ = 0;
    Job fn_ = nullptr;
    void* ctx_ = nullptr;
    std::atomic<unsigned> next_{0};
};

// Products per instruction from which rows go to the worker threads
// (GX_MMA_PARALLEL_PRODUCTS, default 2^23, about 0.4 ms of arithmetic on one
// thread). Below that, waking workers and moving the rows' arrays between
// cores costs more than it saves: a 128x128x64 block-scaled MMA (2^20
// products) runs faster on the issuing thread alone.
size_t parallel_products() {
    static const size_t products = [] {
        const char* e = std::getenv("GX_MMA_PARALLEL_PRODUCTS");
        return e && *e ? size_t(std::strtoull(e, nullptr, 10)) : size_t(1) << 23;
    }();
    return products;
}

// Whether a tensor-memory operand other than D (A, scale factors, sparse
// metadata) may share columns with D's [C, C + N).
bool operands_meet_d(const MmaOp& op, uint32_t d_address, uint64_t a_operand, uint32_t a_is_tmem,
                     uint32_t sfa_address, uint32_t sfb_address, uint32_t sp_meta) {
    const Instr& in = op.in;
    const unsigned d0 = tmem_col(d_address), d1 = d0 + in.n;
    auto meets = [&](unsigned lo, unsigned hi) { return lo < d1 && d0 < hi; };
    if (a_is_tmem) {
        const unsigned a_bits = in.padded ? 8 : format_bits(in.a), ac = tmem_col(uint32_t(a_operand));
        if (meets(ac, ac + (op.kc * a_bits + 31) / 32)) return true;
    }
    if (op.nsf) {
        const unsigned ca = tmem_col(sfa_address), cb = tmem_col(sfb_address), group = in.n > 128 ? 8 : 4;
        if (meets(ca, ca + (op.rows - 1) / 32 + 4 * ((in.a_sf_id + op.nsf - 1) / 4) + 1)) return true;
        if (meets(cb, cb + (in.n - 1) / 32 + group * ((in.b_sf_id + op.nsf - 1) / 4) + 1)) return true;
    }
    if (op.sp) {
        const unsigned c0 = tmem_col(sp_meta);
        if (op.kind == Kind::f16 || op.kind == Kind::tf32 ? meets(c0 + in.selector, c0 + in.selector + 1)
                                                         : meets(c0, c0 + in.k / 32 + 1)) return true;
    }
    return false;
}

// Per-thread buffers of the fast path, reused across instructions.
struct FastScratch {
    std::vector<double> b, a, bt, sfa, sfb, sum;
    std::vector<uint32_t> abits, codes, prior, d;
    std::vector<unsigned> row_lane, col_lane, col, offset;
    std::vector<uint8_t> table, have;
    uint8_t alias[8];   // SFB table -> the B table it uses (tables with equal factors share one)
};

struct DotJob {
    FastScratch* s;
    Decoder da;
    unsigned K, N, vs, nsf, scale_d;
    bool a_neg, accumulate, c_s32, c_f32, saturate;
};

// Row i: A decoded with its sign (and scaled: avals[k] * sfa[k / vs]), then
// its sums of products.
void sum_row(const DotJob& j, unsigned i) {
    FastScratch& s = *j.s;
    const uint32_t* codes = &s.codes[size_t(i) * j.K];
    double* a = &s.a[size_t(i) * j.K];
    for (unsigned k = 0; k < j.K; ++k) {
        const double x = j.da(codes[k]);
        a[k] = j.a_neg ? -x : x;
    }
    // k / vs = f for k in [f vs, (f + 1) vs); K is a multiple of vs.
    const double* sfa = &s.sfa[size_t(i) * 8];
    if (j.nsf)
        for (unsigned f = 0, k = 0; k < j.K; ++f)
            for (unsigned e = 0; e < j.vs && k < j.K; ++e, ++k) a[k] = mul_first(a[k], sfa[f]);
    double* out = &s.sum[size_t(i) * j.N];
    if (!j.nsf) { row_dot(a, s.b.data(), j.N, j.K, 0, j.N, out); return; }
    // Columns of one SFB table at a time.
    const uint8_t* table = &s.table[size_t(i) * j.N];
    for (unsigned n0 = 0, n1; n0 < j.N; n0 = n1) {
        const uint8_t bt = s.alias[table[n0]];
        for (n1 = n0 + 1; n1 < j.N && s.alias[table[n1]] == bt; ++n1) {}
        row_dot(a, &s.bt[size_t(bt) * j.K * j.N], j.N, j.K, n0, n1, out);
    }
}

// Row i: its sums, then D's new bits from the gathered prior bits with the
// reference's roundings (ldexp(x, 0) is x exactly). For an f16 accumulator
// d holds the low 16 bits.
void dot_rows(void* ctx, unsigned i) {
    const DotJob& j = *static_cast<const DotJob*>(ctx);
    sum_row(j, i);
    const FastScratch& s = *j.s;
    const double* sums = &s.sum[size_t(i) * j.N];
    const uint32_t* prior_bits = &s.prior[size_t(i) * j.N];
    uint32_t* d = &j.s->d[size_t(i) * j.N];
    for (unsigned n = 0; n < j.N; ++n) {
        const double sum = sums[n];
        const uint32_t c = prior_bits[n];
        if (j.c_s32) {
            const int64_t r = (j.accumulate ? int64_t(int32_t(c)) : 0) + int64_t(sum);
            d[n] = uint32_t(int32_t(j.saturate ? (r > INT32_MAX ? INT32_MAX : r < INT32_MIN ? INT32_MIN : r) : r));
        } else if (j.c_f32) {
            double prior = j.accumulate ? double(as_float(c)) : 0.0;
            if (j.accumulate && j.scale_d) prior = std::ldexp(prior, -int(j.scale_d));
            d[n] = as_bits(float(add_first(prior, sum)));
        } else {
            double prior = j.accumulate ? decode_f16(c & 0xffff) : 0.0;
            if (j.accumulate && j.scale_d) prior = std::ldexp(prior, -int(j.scale_d));
            d[n] = f16_bits(add_first(prior, sum));
        }
    }
}

void mma_fast(const MmaOp& op, uint32_t d_address, uint64_t a_operand, uint32_t a_is_tmem,
              uint64_t b_desc, uint32_t accumulate, uint32_t sfa_address, uint32_t sfb_address,
              const uint32_t* lane_mask, uint32_t scale_d, uint32_t sp_meta) {
    const Kind kind = op.kind;
    const Instr& in = op.in;
    const unsigned cta_group = op.cta_group, members = op.members, rows = op.rows, cols = op.cols;
    const unsigned K = in.k, N = in.n, R = members * rows, vs = op.vs, nsf = op.nsf, kc = op.kc;
    const bool sp = op.sp, ws = op.ws;
    const unsigned dl = tmem_lane(d_address), dc = tmem_col(d_address);
    static thread_local FastScratch s;

    // Output placement. Every layout of place / place_ws puts (ml, n) at lane
    // row_lane[ml] + col_lane[n] and column col[n] of the CTA's D. Outputs
    // that would leave tensor memory take the reference, which fails at the
    // same output as before (nothing has been changed yet).
    s.row_lane.resize(rows);
    s.col_lane.resize(N);
    s.col.resize(N);
    auto placed = [&](unsigned ml, unsigned n) {
        return ws ? place_ws(in.m, in.n, ml, n) : place(cta_group, in.m, in.n, ml, n);
    };
    unsigned max_row = 0, max_lane = 0, max_col = 0;
    for (unsigned ml = 0; ml < rows; ++ml) max_row = std::max(max_row, s.row_lane[ml] = placed(ml, 0).lane);
    for (unsigned n = 0; n < N; ++n) {
        const Placement p = placed(0, n);
        max_lane = std::max(max_lane, s.col_lane[n] = p.lane - s.row_lane[0]);
        max_col = std::max(max_col, s.col[n] = p.column);
    }
    if (dl + max_row + max_lane >= GxTmemState::kLanes || dc + max_col >= GxTmemState::kColumns) {
        mma_reference(op, d_address, a_operand, a_is_tmem, b_desc, accumulate, sfa_address, sfb_address,
                      lane_mask, scale_d, sp_meta);
        return;
    }
    auto masked = [&](unsigned v, unsigned lane) {
        return lane_mask && ((lane_mask[4 * v + lane / 32] >> (lane % 32)) & 1);
    };

    // B with its sign, transposed: b[k * N + n], zero for zero-masked columns.
    s.b.assign(size_t(N) * K, 0.0);
    s.abits.resize(std::max(K, kc));
    const Decoder db(in.b), da(in.a);
    for (unsigned vb = 0; vb < members; ++vb) {
        const MatrixDesc bd = decode_desc(b_desc, member_rank(vb, cta_group));
        SharedView view;
        for (unsigned nl = 0; nl < cols; ++nl) {
            const unsigned n = vb * cols + nl;
            if (op.zero_col[n]) continue;
            read_row(view, bd, in.b_mn, in.b, in.padded, nl + op.shift, K, s.abits.data());
            for (unsigned k = 0; k < K; ++k) {
                const double y = db(s.abits[k]);
                s.b[size_t(k) * N + n] = in.b_neg ? -y : y;
            }
        }
    }

    // A codes, codes[(v * rows + ml) * K + k] (decoded by the row jobs), and
    // each row's SFA factors. Scaled kinds also record, per output, which SFB
    // factors it uses (table 4 v + subpartition replica: CTA v's tensor
    // memory) and read the factors of every (table, column) an unmasked
    // output uses: sfb[(table * nsf + j) * N + n].
    s.a.resize(size_t(R) * K);
    s.codes.resize(size_t(R) * K);
    s.sfa.resize(size_t(R) * 8);
    if (nsf) {
        s.sfb.assign(size_t(4) * members * nsf * N, 0.0);
        s.have.assign(size_t(4) * members * N, 0);
        s.table.resize(size_t(R) * N);
    }
    std::vector<uint32_t>& abits = s.abits;
    for (unsigned v = 0; v < members; ++v) {
        const uint32_t rank = member_rank(v, cta_group);
        GxTmemState& t = tmem_of(rank);
        const MatrixDesc ad = a_is_tmem ? MatrixDesc{} : decode_desc(a_operand, rank);
        const unsigned a_bits = in.padded ? 8 : format_bits(in.a);
        const unsigned al = tmem_lane(uint32_t(a_operand)), ac = tmem_col(uint32_t(a_operand));
        SharedView view;
        for (unsigned ml = 0; ml < rows; ++ml) {
            const unsigned row_lane = s.row_lane[ml];
            // Dense codes go straight to the row; sparse ones via abits.
            uint32_t* dense = &s.codes[(size_t(v) * rows + ml) * K];
            uint32_t* codes = sp ? abits.data() : dense;
            if (!a_is_tmem) {
                read_row(view, ad, in.a_mn, in.a, in.padded, ml, kc, codes);
            } else {
                for (unsigned k = 0; k < kc; ++k) {
                    const uint32_t c = cell(t, al + (ws ? ml : row_lane), ac + (k * a_bits) / 32);
                    uint32_t bits = a_bits == 32 ? c : (c >> ((k * a_bits) % 32)) & ((1u << a_bits) - 1);
                    if (in.padded && in.a == Format::e2m1) bits >>= 2;
                    codes[k] = bits;
                }
            }
            if (sp) {
                // Decompress (as in the reference).
                std::fill(dense, dense + K, 0u);
                if (kind == Kind::tf32) {
                    for (unsigned g = 0; g < K / 2; ++g) {
                        const unsigned nib = sparse_nibble(t, sp_meta, in, ml, g);
                        const unsigned idx[2] = {nib & 3, nib >> 2};
                        for (unsigned u = 0; u < 2; ++u) {
                            const unsigned cu = 2 * g + u;
                            const uint32_t half = (abits[cu / 2] >> (16 * (cu % 2))) & 0xffff;
                            dense[2 * g + idx[u] / 2] |= half << (16 * (idx[u] % 2));
                        }
                    }
                } else if (in.a == Format::e2m1 && !in.padded) {
                    for (unsigned g = 0; g < K / 8; ++g) {
                        const unsigned nib = sparse_nibble(t, sp_meta, in, ml, g);
                        const unsigned idx[2] = {nib & 3, nib >> 2};
                        for (unsigned u = 0; u < 2; ++u)
                            for (unsigned e = 0; e < 2; ++e)
                                dense[8 * g + 2 * idx[u] + e] = abits[2 * (2 * g + u) + e];
                    }
                } else {
                    for (unsigned g = 0; g < K / 4; ++g) {
                        const unsigned nib = sparse_nibble(t, sp_meta, in, ml, g);
                        dense[4 * g + (nib & 3)] = abits[2 * g];
                        dense[4 * g + (nib >> 2)] = abits[2 * g + 1];
                    }
                }
            }
            if (!nsf) continue;
            const unsigned replica = 32 * ((row_lane / 32) % 4);
            double* sfa = &s.sfa[(size_t(v) * rows + ml) * 8];
            for (unsigned j = 0; j < nsf; ++j) {
                const unsigned sid = in.a_sf_id + j;
                const uint32_t c = cell(t, tmem_lane(sfa_address) + ml % 32 + replica,
                                        tmem_col(sfa_address) + ml / 32 + 4 * (sid / 4));
                sfa[j] = scale_of(c >> (8 * (sid % 4)), in.ue8m0);
            }
            // Lane offsets of the layouts are multiples of 32, so an output's
            // subpartition is (row_lane / 32 + col_lane[n] / 32) % 4: rows
            // of one row_lane / 32 share their tables, and without a lane
            // mask also the factors they need.
            uint8_t* table = &s.table[(size_t(v) * rows + ml) * N];
            const unsigned q = row_lane / 32 % 4;
            if (ml && !lane_mask && s.row_lane[ml - 1] / 32 % 4 == q) {
                std::memcpy(table, table - N, N);
                continue;
            }
            const unsigned group = N > 128 ? 8 : 4;
            for (unsigned n = 0; n < N; ++n) {
                const unsigned r = (q + s.col_lane[n] / 32) % 4, tb = 4 * v + r;
                table[n] = uint8_t(tb);
                if (masked(v, dl + row_lane + s.col_lane[n]) || s.have[size_t(tb) * N + n]) continue;
                s.have[size_t(tb) * N + n] = 1;
                for (unsigned j = 0; j < nsf; ++j) {
                    const unsigned sid = in.b_sf_id + j;
                    const uint32_t c = cell(t, tmem_lane(sfb_address) + n % 32 + 32 * r,
                                            tmem_col(sfb_address) + n / 32 + group * (sid / 4));
                    s.sfb[(size_t(tb) * nsf + j) * N + n] = scale_of(c >> (8 * (sid % 4)), in.ue8m0);
                }
            }
        }
        // .ashift: A moves after this MMA has gathered it (B300-measured).
        if (op.ashift) shift_rows_down(t, al, rows, ac, (kc * a_bits + 31) / 32);
    }

    // Scaled kinds: one B table per SFB table above, the factor applied as
    // (b * sfb): bt[(table * K + k) * N + n]. SFB replicas are normally
    // equal, and tables with bitwise equal factors share one B table.
    if (nsf) {
        bool used[8] = {};
        for (unsigned tb = 0; tb < 4 * members; ++tb)
            for (unsigned n = 0; n < N && !used[tb]; ++n) used[tb] = s.have[size_t(tb) * N + n];
        const size_t factors = size_t(nsf) * N;
        s.bt.resize(size_t(4) * members * K * N);
        for (unsigned tb = 0; tb < 4 * members; ++tb) {
            s.alias[tb] = uint8_t(tb);
            if (!used[tb]) continue;
            for (unsigned other = 0; other < tb; ++other)
                if (used[other] && s.alias[other] == other &&
                    !std::memcmp(&s.sfb[tb * factors], &s.sfb[other * factors], factors * sizeof(double))) {
                    s.alias[tb] = uint8_t(other);
                    break;
                }
            if (s.alias[tb] != tb) continue;
            for (unsigned j = 0, k = 0; k < K; ++j)
                for (unsigned e = 0; e < vs && k < K; ++e, ++k)
                    scale_row(&s.b[size_t(k) * N], &s.sfb[(size_t(tb) * nsf + j) * N], N,
                              &s.bt[(size_t(tb) * K + k) * N]);
        }
    }

    // D's prior bits of every unmasked output when accumulating (all cells
    // were bounds-checked above).
    s.prior.resize(size_t(R) * N);
    s.d.resize(size_t(R) * N);
    s.offset.resize(N);
    for (unsigned n = 0; n < N; ++n) s.offset[n] = s.col_lane[n] * GxTmemState::kColumns + s.col[n];
    uint32_t* cells[2] = {nullptr, nullptr};   // allocated on first use, as by cell()
    // fn(cell, index) for every unmasked output (v, ml, n), index (v rows + ml) N + n.
    auto for_outputs = [&](auto&& fn) {
        const unsigned* row_lanes = s.row_lane.data();
        const unsigned* col_lanes = s.col_lane.data();
        const unsigned* offsets = s.offset.data();
        for (unsigned v = 0; v < members; ++v) {
            GxTmemState& t = tmem_of(member_rank(v, cta_group));
            for (unsigned ml = 0; ml < rows; ++ml) {
                const size_t i = (size_t(v) * rows + ml) * N;
                const unsigned row_lane = dl + row_lanes[ml];
                if (!lane_mask) {
                    if (!cells[v]) cells[v] = &t.at(0, 0);
                    uint32_t* row = cells[v] + size_t(row_lane) * GxTmemState::kColumns + dc;
                    for (unsigned n = 0; n < N; ++n) fn(row[offsets[n]], i + n);
                    continue;
                }
                for (unsigned n = 0; n < N; ++n) {
                    if (masked(v, row_lane + col_lanes[n])) continue;
                    if (!cells[v]) cells[v] = &t.at(0, 0);
                    fn(cells[v][size_t(row_lane) * GxTmemState::kColumns + dc + offsets[n]], i + n);
                }
            }
        }
    };
    uint32_t* prior = s.prior.data();
    const uint32_t* d = s.d.data();
    if (accumulate) for_outputs([prior](uint32_t& c, size_t i) { prior[i] = c; });

    // A decodes, sums of products and D's new bits, one row per job.
    s.sum.resize(size_t(R) * N);
    DotJob job{&s, da, K, N, vs, nsf, scale_d, in.a_neg, accumulate != 0, in.c_s32, in.c_f32, in.saturate};
    RowPool* pool = size_t(R) * N * K >= parallel_products() ? RowPool::get() : nullptr;
    if (pool) pool->run(R, dot_rows, &job);
    else for (unsigned i = 0; i < R; ++i) dot_rows(&job, i);

    // D, in the reference's order.
    if (in.c_s32 || in.c_f32) for_outputs([d](uint32_t& c, size_t i) { c = d[i]; });
    else for_outputs([d](uint32_t& c, size_t i) { c = (c & 0xffff0000u) | d[i]; });
}

// GX_MMA_REFERENCE=1: reference only. GX_MMA_CHECK=1: both on every
// instruction, from the same tensor-memory state; any difference in the
// resulting tensor memory aborts with the instruction and the first
// differing cell. Default: the fast path.
enum class MmaMode { fast, reference, check };

MmaMode mma_mode() {
    static const MmaMode mode = [] {
        auto on = [](const char* name) { const char* e = std::getenv(name); return e && *e && *e != '0'; };
        return on("GX_MMA_CHECK") ? MmaMode::check : on("GX_MMA_REFERENCE") ? MmaMode::reference : MmaMode::fast;
    }();
    return mode;
}

struct CheckCounts {
    unsigned long long fast = 0, reference = 0;
    // Every f16 code, once, before the first comparison.
    CheckCounts() {
        if (mma_mode() != MmaMode::check) return;
        for (uint32_t i = 0; i < (1u << 16); ++i) {
            const double x = decode_f16(i), y = decode(Format::f16, i);
            if (std::memcmp(&x, &y, 8)) {
                std::fprintf(stderr, "gx: GX_MMA_CHECK: f16 code %04x decodes differently\n", i);
                std::abort();
            }
        }
    }
    ~CheckCounts() {
        if (fast + reference)
            std::fprintf(stderr, "gx: GX_MMA_CHECK: %llu tcgen05.mma bit-identical to the reference (%llu took the reference)\n",
                         fast + reference, reference);
    }
};
CheckCounts check_counts;

}  // namespace

// D[m][n] (+)= sum_k A[m][k] * B[n][k]; block-scaled kinds multiply each
// product by SFA[m][k/VS] * SFB[n][k/VS]. Exact decoded products are summed in
// double and added to the accumulator with one rounding per instruction.
// With cta_group::2 the even CTA of a pair issues; CTA v supplies A rows
// [v M/2, (v+1) M/2) and B columns [v N/2, (v+1) N/2) from its own shared
// memory (same descriptors) and receives its rows of D at the same address.
// Scale factor s (= sf_id + j) of row/column i sits at lane i%32 (replicated
// per subpartition; the replica of D's subpartition is read), column
// C + i/32 + G*(s/4), byte s%4, with G = 4 (8 for SFB with N > 128).
//
// flags: bit 0 .sp, bit 1 .ws, bit 2 .ashift, bit 3 zero-column mask given.
// lane_mask: disable-output-lane words (4, or 8 with cta_group::2: words 0-3
// for the even CTA, 4-7 for the odd one), bit b of word w = TMEM lane 32w+b.
// scale_d: D = A*B + D * 2^-scale_d (kind::f16 / kind::tf32).
extern "C" void gx_tcgen05_mma(uint32_t kind_code, uint32_t cta_group, uint32_t d_address,
                               uint64_t a_operand, uint32_t a_is_tmem, uint64_t b_desc,
                               uint32_t idesc, uint32_t accumulate,
                               uint32_t sfa_address, uint32_t sfb_address, uint32_t block,
                               const uint32_t* lane_mask, uint32_t mask_words, uint32_t scale_d,
                               uint32_t sp_meta, uint32_t flags, uint64_t zcm_desc) {
    const MmaOp op = prepare_mma(kind_code, cta_group, a_is_tmem, idesc, sfa_address, sfb_address, block,
                                 lane_mask, mask_words, scale_d, flags, zcm_desc);
    // One issued MMA covers the descriptor tile, including padded work. Sparse
    // counts are dense-equivalent; metadata traffic remains explicitly unmodeled.
    const auto& in=op.in;
    drperf_gpu_resource(kind_code==3 ? "tensor_integer_ops" : "tensor_float_flops", uint64_t(2)*in.m*in.n*in.k);
    drperf_gpu_resource("tensor_mma_instructions",1);
    drperf_gpu_tensor_issue(uint64_t(2)*in.m*in.n*in.k, format_bits(in.a), cta_group);
    const uint64_t ab=(uint64_t(in.m)*op.kc*format_bits(in.a)+7)/8;
    const uint64_t bb=(uint64_t(in.n)*in.k*format_bits(in.b)+7)/8;
    drperf_gpu_resource(a_is_tmem ? "tmem_read_bytes" : "shared_read_bytes",ab);
    drperf_gpu_resource("shared_read_bytes",bb);
    const uint64_t db=uint64_t(in.m)*in.n*(in.c_f32||in.c_s32 ? 4 : 2);
    if(accumulate)drperf_gpu_resource("tmem_read_bytes",db);
    drperf_gpu_resource("tmem_write_bytes",db);
    if(in.scaled)drperf_gpu_resource("tmem_read_bytes",uint64_t(op.nsf)*(in.m+in.n));
    // Plain .ws executes the full M x N x K tile with the same logical operand
    // footprints as dense MMA; only the TMEM accumulator placement differs, so
    // its counts above are complete. A zero-column mask that zeroes any column
    // remains unmodeled (an all-zero mask descriptor masks nothing).
    if(op.sp||(op.ws&&std::find(op.zero_col.begin(),op.zero_col.end(),true)!=op.zero_col.end())||op.ashift||lane_mask)drperf_gpu_resource("unmodeled_work_events",1);
    const MmaMode mode = mma_mode();
    // The fast path gathers one A row per output row; .ws M=64/32 with A in
    // tensor memory reads a different A row lane per N group (reference only).
    const bool fast = mode != MmaMode::reference && !(op.ws && a_is_tmem && op.in.m < 128) &&
                      !operands_meet_d(op, d_address, a_operand, a_is_tmem, sfa_address, sfb_address, sp_meta);
    if (mode != MmaMode::check || !fast) {
        if (mode == MmaMode::check) ++check_counts.reference;
        (fast ? mma_fast : mma_reference)(op, d_address, a_operand, a_is_tmem, b_desc, accumulate,
                                          sfa_address, sfb_address, lane_mask, scale_d, sp_meta);
        return;
    }
    std::vector<GxTmemState> before, expected;
    for (unsigned v = 0; v < op.members; ++v) before.push_back(tmem_of(member_rank(v, cta_group)));
    mma_reference(op, d_address, a_operand, a_is_tmem, b_desc, accumulate, sfa_address, sfb_address,
                  lane_mask, scale_d, sp_meta);
    for (unsigned v = 0; v < op.members; ++v) {
        GxTmemState& t = tmem_of(member_rank(v, cta_group));
        expected.push_back(t);
        t = before[v];
    }
    mma_fast(op, d_address, a_operand, a_is_tmem, b_desc, accumulate, sfa_address, sfb_address,
             lane_mask, scale_d, sp_meta);
    for (unsigned v = 0; v < op.members; ++v) {
        GxTmemState& t = tmem_of(member_rank(v, cta_group));
        if (t.allocated == expected[v].allocated && t.cells == expected[v].cells) continue;
        size_t i = 0;
        while (i < t.cells.size() && i < expected[v].cells.size() && t.cells[i] == expected[v].cells[i]) ++i;
        std::fprintf(stderr, "gx: GX_MMA_CHECK: tcgen05.mma kind %u cta_group %u idesc 0x%08x flags %u differs "
                     "from the reference in CTA %u at lane %zu column %zu: 0x%08x, reference 0x%08x\n",
                     kind_code, cta_group, idesc, flags, v, i / GxTmemState::kColumns, i % GxTmemState::kColumns,
                     i < t.cells.size() ? t.cells[i] : 0u, i < expected[v].cells.size() ? expected[v].cells[i] : 0u);
        std::abort();
    }
    ++check_counts.fast;
}

// ---- tcgen05.shift ------------------------------------------------------------
// 32-byte elements (columns C0..C0+7) shift "down" by one row: on B300, in all
// four 32-lane subpartitions, lane r takes lane r + 1 and the last lane of each
// subpartition keeps its contents, whatever the (32-aligned) taddr lane (lanes
// 0, 32 and 96 measured). cta_group::2 shifts both CTAs of the pair.
extern "C" void gx_tcgen05_shift(uint32_t address, uint32_t cta_group) {
    if (cta_group != 1 && cta_group != 2) fail("invalid cta_group");
    if(cta_group==2) (void)gx_cluster_rank_linear_block(gx_dev_cluster_ctarank()^1u);
    const unsigned l0 = tmem_lane(address), c0 = tmem_col(address);
    // 31 moved lanes in each of the four 32-lane subpartitions, 8 columns.
    drperf_gpu_resource("tmem_read_bytes",uint64_t(4*31)*8*4*cta_group);
    drperf_gpu_resource("tmem_write_bytes",uint64_t(4*31)*8*4*cta_group);
    if (l0 % 32) fail("tcgen05.shift lane must be aligned to 32");
    for (unsigned v = 0; v < (cta_group == 2 ? 2u : 1u); ++v)
        shift_rows_down(tmem_of(member_rank(v, cta_group)), 0, GxTmemState::kLanes, c0, 8);
}

// ---- tcgen05.ld.red (sm_103) -------------------------------------------------
// A tcgen05.ld whose thread additionally receives the min/max of the values it
// loaded. op: 0 min, 1 max; type: 0 f32, 1 u32, 2 s32. For f32, .abs reduces
// |x|, -0 < +0, NaN inputs are skipped unless .NaN (canonical NaN result).
extern "C" uint32_t gx_tcgen05_ld_red(uint32_t shape, uint32_t address, uint32_t regs, uint32_t imm,
                                      uint32_t* out, uint32_t op, uint32_t type, uint32_t abs, uint32_t nan) {
    if (regs < 2) fail("tcgen05.ld.red needs .num >= .x2");
    drperf_gpu_resource(type==0 ? "fp32_flops" : "integer_ops",regs-1);
    gx_tcgen05_ld(shape, address, regs, 0, imm, out);
    if (type == 0) {
        bool seen = false, any_nan = false;
        float best = 0;
        for (unsigned j = 0; j < regs; ++j) {
            float x = as_float(out[j]);
            if (abs) x = std::fabs(x);
            if (std::isnan(x)) { any_nan = true; continue; }
            const bool better = !seen || (op ? (x > best || (x == best && !std::signbit(x) && std::signbit(best)))
                                             : (x < best || (x == best && std::signbit(x) && !std::signbit(best))));
            if (better) { best = x; seen = true; }
        }
        if ((nan && any_nan) || !seen) return 0x7fffffffu;
        return as_bits(best);
    }
    uint32_t best = out[0];
    for (unsigned j = 1; j < regs; ++j) {
        const bool less = type == 2 ? int32_t(out[j]) < int32_t(best) : out[j] < best;
        const bool greater = type == 2 ? int32_t(out[j]) > int32_t(best) : out[j] > best;
        if (op ? greater : less) best = out[j];
    }
    return best;
}
