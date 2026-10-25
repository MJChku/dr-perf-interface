// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The GX Project Authors
// Extracted math collectives from GX kernel_apis.cc.
#include "../../runtime/context.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdio>
#include <algorithm>
namespace {
double mma_decode(unsigned fmt, uint32_t bits) {
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
    switch (fmt) {
    case 0: return minifloat(bits & 0xffff, 5, 10, 15, true, false);
    case 1: { const uint32_t u = (bits & 0xffff) << 16; float f; std::memcpy(&f, &u, 4); return f; }
    case 2: return minifloat(bits & 0xff, 4, 3, 7, false, true);
    case 3: return minifloat(bits & 0xff, 5, 2, 15, true, false);
    default: { const uint32_t u = bits & 0xffffe000u; float f; std::memcpy(&f, &u, 4); return f; }
    }
}
// Exact mma_decode of every e4m3 / e5m2 code (NaN payloads included), or
// null for f16, bf16 and tf32, which decode directly.
const double* mma_decode_table(unsigned fmt) {
    auto build = [](unsigned f) {
        std::vector<double> v(256);
        for (uint32_t i = 0; i < 256; ++i) v[i] = mma_decode(f, i);
        return v;
    };
    static const std::vector<double> e4m3 = build(2), e5m2 = build(3);
    return fmt == 2 ? e4m3.data() : fmt == 3 ? e5m2.data() : nullptr;
}

// f16 without a table: (1024 | m) * 2^(e - 25), or m * 2^-24 if subnormal.
// Both factors are exact and so is their product, so this is the value
// mma_decode's ldexp returns; exponent 31 (Inf/NaN) goes to mma_decode.
constexpr double mma_pow2(int n) {
    double v = 1;
    for (; n > 0; --n) v *= 2;
    for (; n < 0; ++n) v /= 2;
    return v;
}
constexpr double mma_f16_scale[31] = {
    mma_pow2(-24), mma_pow2(-24), mma_pow2(-23), mma_pow2(-22), mma_pow2(-21), mma_pow2(-20), mma_pow2(-19),
    mma_pow2(-18), mma_pow2(-17), mma_pow2(-16), mma_pow2(-15), mma_pow2(-14), mma_pow2(-13), mma_pow2(-12),
    mma_pow2(-11), mma_pow2(-10), mma_pow2(-9), mma_pow2(-8), mma_pow2(-7), mma_pow2(-6), mma_pow2(-5),
    mma_pow2(-4), mma_pow2(-3), mma_pow2(-2), mma_pow2(-1), mma_pow2(0), mma_pow2(1), mma_pow2(2),
    mma_pow2(3), mma_pow2(4), mma_pow2(5)};
inline double mma_decode_f16(uint32_t bits) {
    const uint32_t e = (bits >> 10) & 31, m = bits & 1023;
    if (e == 31) return mma_decode(0, bits);
    const double v = double(e ? m | 1024 : m) * mma_f16_scale[e];
    return bits & 0x8000 ? -v : v;
}

// The fast path's decode of one code; bitwise equal to mma_decode (bf16 and
// tf32 as there: the float widened).
inline double mma_decode_fast(unsigned fmt, const double* table, uint32_t bits) {
    if (table) return table[bits & 0xff];
    if (fmt == 0) return mma_decode_f16(bits);
    const uint32_t u = fmt == 1 ? (bits & 0xffff) << 16 : bits & 0xffffe000u;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// x * y and x + y with x as the first source operand: when both operands are
// NaN, x86 returns the first one's payload. The fast path keeps the operand
// order the original loop compiles to: B * A, then product + sum, then
// prior + sum.
inline double mma_mul_first(double x, double y) { asm("mulsd %1, %0" : "+x"(x) : "xm"(y)); return x; }
inline double mma_add_first(double x, double y) { asm("addsd %1, %0" : "+x"(x) : "xm"(y)); return x; }

// GX_MMA_REFERENCE=1 computes mma.sync element by element as originally
// written; GX_MMA_CHECK=1 computes both ways and aborts on any difference.
enum class MmaSyncMode { fast, reference, check };
MmaSyncMode mma_sync_mode() {
    static const MmaSyncMode mode = [] {
        auto on = [](const char* name) { const char* e = getenv(name); return e && *e && *e != '0'; };
        return on("GX_MMA_CHECK") ? MmaSyncMode::check : on("GX_MMA_REFERENCE") ? MmaSyncMode::reference
                                                                                 : MmaSyncMode::fast;
    }();
    return mode;
}
// Scratch words of one warp's mma.sync in its barrier state: A and B
// fragments (as before), then C fragments, D results and a flag that the
// first lane released by the publication barrier computes D for the warp.
constexpr unsigned kMmaC = 512, kMmaD = 640, kMmaDone = 768, kMmaWords = 769;

// Codes of A element (row, kk) and B element (kk, col) per the PTX fragment
// tables, from the published fragment words.
inline uint32_t mma_field(uint32_t word, unsigned index, unsigned bits) {
    return bits == 32 ? word : (word >> (index * bits)) & ((1u << bits) - 1);
}
inline uint32_t mma_a_code(const uint32_t* values, unsigned k, bool f8, bool tf32, unsigned row, unsigned kk) {
    const unsigned g = row % 8, h = row / 8;
    unsigned t, i, reg, bits;
    if (f8) { t = (kk % 16) / 4; i = (kk & 3) + 4 * h + 8 * (kk / 16); reg = i / 4; bits = 8; }
    else if (tf32) { t = kk % 4; i = h + 2 * (kk / 4); reg = i; bits = 32; }
    else if (k == 8) { t = kk / 2; i = (kk & 1) + 2 * h; reg = i / 2; bits = 16; }
    else { t = (kk % 8) / 2; i = (kk & 1) + 2 * h + 4 * (kk / 8); reg = i / 2; bits = 16; }
    return mma_field(values[(g * 4 + t) * 4 + reg], bits == 32 ? 0 : i % (32 / bits), bits);
}
inline uint32_t mma_b_code(const uint32_t* values, unsigned k, bool f8, bool tf32, unsigned kk, unsigned col) {
    unsigned t, i, reg, bits;
    if (f8) { t = (kk % 16) / 4; i = (kk & 3) + 4 * (kk / 16); reg = i / 4; bits = 8; }
    else if (tf32) { t = kk % 4; reg = kk / 4; i = 0; bits = 32; }
    else if (k == 8) { t = kk / 2; i = kk & 1; reg = 0; bits = 16; }
    else { t = (kk % 8) / 2; i = (kk & 1) + 2 * (kk / 8); reg = i / 2; bits = 16; }
    return mma_field(values[256 + (col * 4 + t) * 2 + reg], bits == 32 ? 0 : i % (32 / bits), bits);
}

// D of the whole warp into values[kMmaD + lane * 4 + r], with each lane's
// arithmetic: A (16 x k) and B (k x 8) decoded once, every element the sum
// of its products in increasing kk from +0, one rounding of prior + sum.
void mma_sync_warp(uint32_t* values, unsigned k, unsigned atype, unsigned btype, bool c_f32, bool d_f32) {
    static thread_local double av[16][32], bv[8][32];
    const bool f8 = k == 32, tf32 = atype == 4;
    const double* at = mma_decode_table(atype);
    const double* bt = mma_decode_table(btype);
    for (unsigned row = 0; row < 16; ++row)
        for (unsigned kk = 0; kk < k; ++kk) av[row][kk] = mma_decode_fast(atype, at, mma_a_code(values, k, f8, tf32, row, kk));
    for (unsigned col = 0; col < 8; ++col)
        for (unsigned kk = 0; kk < k; ++kk) bv[col][kk] = mma_decode_fast(btype, bt, mma_b_code(values, k, f8, tf32, kk, col));
    std::fill(values + kMmaD, values + kMmaD + 128, 0u);
    for (unsigned row = 0; row < 16; ++row) {
        double sum[8] = {};
        for (unsigned kk = 0; kk < k; ++kk) {
            const double x = av[row][kk];
            for (unsigned col = 0; col < 8; ++col) sum[col] = mma_add_first(mma_mul_first(bv[col][kk], x), sum[col]);
        }
        for (unsigned col = 0; col < 8; ++col) {
            // Element i of lane (row % 8) * 4 + col / 2.
            const unsigned lane = (row % 8) * 4 + col / 2, i = 2 * (row / 8) + (col & 1);
            const uint32_t* c = values + kMmaC + lane * 4;
            const double prior = c_f32 ? double(*reinterpret_cast<const float*>(&c[i]))
                                       : mma_decode(0, c[i / 2] >> (16 * (i % 2)));
            const double r = mma_add_first(prior, sum[col]);
            uint32_t* out = values + kMmaD + lane * 4;
            if (d_f32) { const float f = float(r); std::memcpy(&out[i], &f, 4); }
            else { _Float16 h = (_Float16)r; uint16_t u; std::memcpy(&u, &h, 2); out[i / 2] |= uint32_t(u) << (16 * (i % 2)); }
        }
    }
}

struct MmaSyncChecks {
    unsigned long long lanes = 0;
    // Every f16 code, once, before the first comparison.
    MmaSyncChecks() {
        if (mma_sync_mode() != MmaSyncMode::check) return;
        for (uint32_t i = 0; i < (1u << 16); ++i) {
            const double x = mma_decode_f16(i), y = mma_decode(0, i);
            if (std::memcmp(&x, &y, 8)) {
                fprintf(stderr, "gx: GX_MMA_CHECK: f16 code %04x decodes differently\n", i);
                std::abort();
            }
        }
    }
    ~MmaSyncChecks() {
        if (lanes) fprintf(stderr, "gx: GX_MMA_CHECK: %llu mma.sync lane results bit-identical to the reference\n", lanes);
    }
} mma_sync_checks;
}  // namespace

extern "C" void gx_mma_sync(uint32_t k, uint32_t atype, uint32_t btype, uint32_t c_f32, uint32_t d_f32,
                            const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* d) {
    const auto* ctx = gx_thread_context();
    if (!ctx || !current_named_barrier_manager)
        throw std::runtime_error("mma.sync outside a CUDA launch");
    const dim3 dim = ctx->blockDim;
    const unsigned tid = (ctx->threadIdx.z * dim.y + ctx->threadIdx.y) * dim.x + ctx->threadIdx.x;
    const unsigned lane = tid % 32, warp = tid / 32;
    if ((warp + 1) * 32 > uint64_t(dim.x) * dim.y * dim.z)
        throw std::runtime_error("mma.sync requires a full warp");
    const bool f8 = k == 32, tf32 = atype == 4;
    if (!((k == 8 && atype <= 1) || (k == 8 && tf32) || (k == 16 && atype <= 1) || (f8 && atype >= 2 && atype <= 3)) ||
        (f8 ? !(btype == 2 || btype == 3) : btype != atype))
        throw std::runtime_error("unsupported mma.sync shape/type combination");
    const unsigned na = k == 8 && !tf32 ? 2 : 4, nb = k == 8 && !tf32 ? 1 : 2;
    const int name = 120000000 + get_block_id() * 1000 + int(warp);
    GxBarScope scope(name, 32);
    auto* state = current_named_barrier_manager->get_barrier(name, 32);
    if (state->shuffle_values.size() < kMmaWords) state->shuffle_values.resize(kMmaWords);
    auto& values = state->shuffle_values;   // lane * 4 + r: A words; 256 + lane * 2 + r: B words
    for (unsigned r = 0; r < na; ++r) values[lane * 4 + r] = a[r];
    for (unsigned r = 0; r < nb; ++r) values[256 + lane * 2 + r] = b[r];
    for (unsigned r = 0; r < (c_f32 ? 4u : 2u); ++r) values[kMmaC + lane * 4 + r] = c[r];
    values[kMmaDone] = 0;
    record_barrier_call();
    BARRIER_WAIT(state->barrier);
    // A element (row, kk) and B element (kk, col) per the PTX fragment tables.
    auto a_at = [&](unsigned row, unsigned kk) { return mma_decode(atype, mma_a_code(values.data(), k, f8, tf32, row, kk)); };
    auto b_at = [&](unsigned kk, unsigned col) { return mma_decode(btype, mma_b_code(values.data(), k, f8, tf32, kk, col)); };
    const unsigned g = lane / 4, t = lane % 4;
    const MmaSyncMode mode = mma_sync_mode();
    uint32_t out[4] = {0, 0, 0, 0};
    // Reference: every product decoded where it is used.
    if (mode != MmaSyncMode::fast) {
        for (unsigned i = 0; i < 4; ++i) {
            const unsigned row = g + 8 * (i / 2), col = t * 2 + (i & 1);
            double sum = 0;
            for (unsigned kk = 0; kk < k; ++kk) sum += a_at(row, kk) * b_at(kk, col);
            const double prior = c_f32 ? double(*reinterpret_cast<const float*>(&c[i]))
                                       : mma_decode(0, c[i / 2] >> (16 * (i % 2)));
            const double r = prior + sum;
            if (d_f32) { const float f = float(r); std::memcpy(&out[i], &f, 4); }
            else { _Float16 h = (_Float16)r; uint16_t u; std::memcpy(&u, &h, 2); out[i / 2] |= uint32_t(u) << (16 * (i % 2)); }
        }
    }
    if (mode != MmaSyncMode::reference) {
        // Fast path: the first lane released computes the warp's D once.
        if (!values[kMmaDone]) {
            mma_sync_warp(values.data(), k, atype, btype, c_f32, d_f32);
            values[kMmaDone] = 1;
        }
        uint32_t fast[4];
        std::memcpy(fast, &values[kMmaD + lane * 4], sizeof fast);
        if (mode == MmaSyncMode::check) {
            if (std::memcmp(fast, out, sizeof out)) {
                fprintf(stderr, "gx: GX_MMA_CHECK: mma.sync k=%u atype %u btype %u lane %u differs from the reference: "
                        "%08x %08x %08x %08x, reference %08x %08x %08x %08x\n", k, atype, btype, lane,
                        fast[0], fast[1], fast[2], fast[3], out[0], out[1], out[2], out[3]);
                std::abort();
            }
            ++mma_sync_checks.lanes;
        }
        std::memcpy(out, fast, sizeof out);
    }
    for (unsigned r = 0; r < (d_f32 ? 4u : 2u); ++r) d[r] = out[r];
    record_barrier_call();
    BARRIER_WAIT(state->barrier);
}

// PTX permits a target-specific WMMA fragment mapping. GX maps A/B to two
// copies of a logical row-major 16x16 matrix, packed in 8 b32 words per lane;
// C/D have 8 f32 values per lane. Caller arrays preserve all register values.
// Two real warp barriers protect publication and scratch reuse, independently
// of the optional relaxed bar.warp setting. Every CUDA warp has separate state.
static void wmma_f32(bool bf16, unsigned m, unsigned n,
                     const uint32_t* a, const uint32_t* b,
                     const float* c, float* d) {
    const auto* ctx = gx_thread_context();
    if (!ctx || !current_named_barrier_manager)
        throw std::runtime_error("WMMA outside a CUDA launch");
    const dim3 dim = ctx->blockDim;
    const unsigned tid = (ctx->threadIdx.z * dim.y + ctx->threadIdx.y) * dim.x + ctx->threadIdx.x;
    const unsigned lane = tid % 32, warp = tid / 32;
    if ((warp + 1) * 32 > uint64_t(dim.x) * dim.y * dim.z)
        throw std::runtime_error("WMMA requires a full warp");
    const int block = get_block_id();
    const int name = 120000000 + block * 1000 + warp;
    GxBarScope scope(name, 32);
    auto* state = current_named_barrier_manager->get_barrier(name, 32);
    // No yield before resize/publication; this manager is local to the
    // cooperative executor thread, and the vector remains stable until return.
    const unsigned na = bf16 ? m / 4 : 8, nb = bf16 ? n / 4 : 8;
    const unsigned bbase = 32 * na, words = 32 * (na + nb);
    if (state->shuffle_values.size() != words) state->shuffle_values.resize(words);
    auto& values = state->shuffle_values;
    for (unsigned i = 0; i < na; ++i) values[lane * na + i] = a[i];
    for (unsigned i = 0; i < nb; ++i) values[bbase + lane * nb + i] = b[i];
    record_barrier_call();
    BARRIER_WAIT(state->barrier);
    auto half = [&](unsigned base, unsigned index) {
        const uint16_t bits = uint16_t(values[base + index / 2] >> (16 * (index % 2)));
        if (bf16) {
            const uint32_t wide = uint32_t(bits) << 16;
            float value;
            std::memcpy(&value, &wide, sizeof value);
            return value;
        }
        // Exact IEEE binary16 decode, preserving subnormals, signs and specials.
        const unsigned e = (bits >> 10) & 31, m = bits & 1023;
        float value;
        if (e == 31) value = m ? std::numeric_limits<float>::quiet_NaN()
                              : std::numeric_limits<float>::infinity();
        else value = std::ldexp(float(e ? 1024 + m : m), e ? int(e) - 25 : -24);
        return bits & 0x8000 ? -value : value;
    };
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned row = (lane * 8 + i) / n, col = (lane * 8 + i) % n;
        float acc = c[i];
        for (unsigned k = 0; k < 16; ++k)
            acc = std::fma(half(0, row * 16 + k), half(bbase, k * n + col), acc);
        d[i] = acc;
    }
    record_barrier_call();
    BARRIER_WAIT(state->barrier);
}

extern "C" void gx_wmma_m16n16k16_f16(const uint32_t* a, const uint32_t* b,
                                      const float* c, float* d) {
    wmma_f32(false, 16, 16, a, b, c, d);
}

extern "C" void gx_wmma_bf16(unsigned m, unsigned n, const uint32_t* a,
                            const uint32_t* b, const float* c, float* d) {
    if (!((m == 16 && n == 16) || (m == 8 && n == 32) || (m == 32 && n == 8)))
        throw std::runtime_error("unsupported BF16 WMMA tile");
    wmma_f32(true, m, n, a, b, c, d);
}


// Integer MMA has the PTX int8 fragment mapping and exact integer arithmetic.
// Saturation is applied to the full dot-plus-prior; ordinary s32 wraps modulo 2^32.
extern "C" void gx_mma_sync_i8(unsigned k, bool signed_a, bool signed_b, bool sat,
                               const uint32_t* a, const uint32_t* b,
                               const uint32_t* c, uint32_t* d) {
    const auto* ctx = gx_thread_context();
    if (!ctx || !current_named_barrier_manager || (k != 16 && k != 32))
        throw std::runtime_error("invalid integer MMA launch or shape");
    const dim3 dim = ctx->blockDim;
    const unsigned tid = (ctx->threadIdx.z * dim.y + ctx->threadIdx.y) * dim.x + ctx->threadIdx.x;
    const unsigned lane = tid % 32, warp = tid / 32;
    if ((warp + 1) * 32 > uint64_t(dim.x) * dim.y * dim.z)
        throw std::runtime_error("integer MMA requires a full warp");
    const int name = 130000000 + get_block_id() * 1000 + int(warp);
    GxBarScope scope(name, 32);
    auto* state = current_named_barrier_manager->get_barrier(name, 32);
    if (state->shuffle_values.size() < 192) state->shuffle_values.resize(192);
    auto& values = state->shuffle_values;
    for (unsigned i = 0; i < k/8; ++i) values[lane*4+i] = a[i];
    for (unsigned i = 0; i < k/16; ++i) values[128+lane*2+i] = b[i];
    record_barrier_call(); BARRIER_WAIT(state->barrier);
    auto byte = [](uint32_t word, unsigned element, bool sign) -> int64_t {
        const unsigned value = (word >> (8 * (element % 4))) & 255;
        return sign && value >= 128 ? int(value) - 256 : int(value);
    };
    for (unsigned i = 0; i < 4; ++i) {
        const unsigned row = lane/4 + 8*(i/2), col = 2*(lane%4) + i%2;
        int32_t prior; std::memcpy(&prior, c+i, 4);
        int64_t sum = prior;
        for (unsigned kk = 0; kk < k; ++kk) {
            const unsigned t = (kk%16)/4;
            const unsigned ai = kk%4 + 4*(row/8) + 8*(kk/16);
            const unsigned bi = kk%4 + 4*(kk/16);
            const auto av = byte(values[((row%8)*4+t)*4+ai/4], ai, signed_a);
            const auto bv = byte(values[128+(col*4+t)*2+bi/4], bi, signed_b);
            sum += av*bv;
        }
        if (sat) sum = std::min<int64_t>(2147483647, std::max<int64_t>(-2147483648LL, sum));
        d[i] = uint32_t(sum);
    }
    record_barrier_call(); BARRIER_WAIT(state->barrier);
}
