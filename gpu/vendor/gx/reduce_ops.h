// SPDX-License-Identifier: MIT
// Element-wise atomic reductions shared by cp.reduce.async.bulk and
// cp.reduce.async.bulk.tensor (PTX ISA §9.7.9.25.4.x). Each element update
// is an atomic read-modify-write (relaxed); integer inc/dec follow atom
// semantics, float adds round to nearest even (f16/bf16 keep subnormals,
// f32_ftz flushes inputs and result), float min/max treat -0 < +0 and return
// the non-NaN operand.
#pragma once
#include <cstdint>
#include <cstring>

namespace gx {

enum class RedOp : int { add = 0, min = 1, max = 2, inc = 3, dec = 4, and_ = 5, or_ = 6, xor_ = 7 };
enum class RedType : int { u32, s32, u64, s64, f32, f32_ftz, f64, f16, bf16 };

inline unsigned red_bytes(RedType t) {
    switch (t) {
    case RedType::u64: case RedType::s64: case RedType::f64: return 8;
    case RedType::f16: case RedType::bf16: return 2;
    default: return 4;
    }
}

// Valid op/type pairs. tensor: cp.reduce.async.bulk.tensor table; otherwise
// the global-destination cp.reduce.async.bulk table (callers check the
// smaller shared::cluster table themselves).
inline bool reduce_supported(RedOp op, RedType t, bool tensor) {
    const bool is_int32 = t == RedType::u32 || t == RedType::s32, is_64 = t == RedType::u64 || t == RedType::s64;
    const bool half = t == RedType::f16 || t == RedType::bf16;
    switch (op) {
    case RedOp::add: return is_int32 || t == RedType::u64 || t == RedType::f32 || t == RedType::f32_ftz || half ||
                            (!tensor && t == RedType::f64);
    case RedOp::min: case RedOp::max: return is_int32 || is_64 || half;
    case RedOp::inc: case RedOp::dec: return t == RedType::u32;
    default: return is_int32 || is_64;   // and/or/xor on b32/b64 (bit types travel as u32/u64)
    }
}

namespace detail {
inline float f32_of(uint64_t b) { uint32_t u = uint32_t(b); float f; std::memcpy(&f, &u, 4); return f; }
inline uint64_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline double f64_of(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
inline uint64_t bits_of(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }
inline double half_of(uint64_t b) { _Float16 h; uint16_t u = uint16_t(b); std::memcpy(&h, &u, 2); return double(h); }
inline uint64_t half_bits(double d) { _Float16 h = (_Float16)d; uint16_t u; std::memcpy(&u, &h, 2); return u; }
inline double bf16_of(uint64_t b) { return double(f32_of(uint64_t(uint32_t(b & 0xffff) << 16))); }
// Single round-to-nearest-even from double to bfloat16.
inline uint64_t bf16_bits(double d) {
    if (d != d) return 0x7fc0u;
    const uint16_t sign = __builtin_signbit(d) ? 0x8000u : 0u;
    const double a = __builtin_fabs(d);
    if (a == 0) return sign;
    int e; const double m = __builtin_frexp(a, &e);
    int exp = e - 1 + 127;
    double q;
    if (exp >= 1) q = __builtin_nearbyint(__builtin_ldexp(m, 8));
    else { q = __builtin_nearbyint(__builtin_ldexp(a, 133)); exp = 0; }
    if (exp >= 1 && q == 256.0) { q = 128.0; ++exp; }
    if (exp == 0 && q >= 128.0) exp = 1;
    if (exp >= 255) return uint16_t(sign | 0x7f80u);
    const uint16_t mant = exp ? uint16_t(uint32_t(q) - 128u) : uint16_t(q);
    return uint16_t(sign | (exp << 7) | mant);
}
inline float flush(float f) { return __builtin_fpclassify(0, 1, 2, 3, 4, f) == 3 ? __builtin_copysignf(0.f, f) : f; }
template <class T> T fmin_num(T a, T b) {
    if (a != a) return b;
    if (b != b) return a;
    if (a == b) return __builtin_signbit(double(a)) ? a : b;
    return a < b ? a : b;
}
template <class T> T fmax_num(T a, T b) {
    if (a != a) return b;
    if (b != b) return a;
    if (a == b) return __builtin_signbit(double(a)) ? b : a;
    return a > b ? a : b;
}

inline uint64_t apply(RedOp op, RedType t, uint64_t r, uint64_t s) {
    const unsigned bytes = red_bytes(t);
    const uint64_t mask = bytes == 8 ? ~0ull : (1ull << (8 * bytes)) - 1;
    r &= mask; s &= mask;
    switch (t) {
    case RedType::f32: case RedType::f32_ftz: {
        float a = f32_of(r), b = f32_of(s);
        if (t == RedType::f32_ftz) { a = flush(a); b = flush(b); }
        float o = op == RedOp::add ? a + b : op == RedOp::min ? fmin_num(a, b) : fmax_num(a, b);
        if (t == RedType::f32_ftz) o = flush(o);
        return bits_of(o);
    }
    case RedType::f64: return bits_of(f64_of(r) + f64_of(s));
    case RedType::f16: case RedType::bf16: {
        const double a = t == RedType::f16 ? half_of(r) : bf16_of(r), b = t == RedType::f16 ? half_of(s) : bf16_of(s);
        const double o = op == RedOp::add ? a + b : op == RedOp::min ? fmin_num(a, b) : fmax_num(a, b);
        if (op != RedOp::add) return o == a && (a != 0 || __builtin_signbit(o) == __builtin_signbit(a)) ? r : s;
        return t == RedType::f16 ? half_bits(o) : bf16_bits(o);
    }
    default: break;
    }
    const bool sgn = t == RedType::s32 || t == RedType::s64;
    const int64_t sr = bytes == 4 ? int64_t(int32_t(r)) : int64_t(r), ss = bytes == 4 ? int64_t(int32_t(s)) : int64_t(s);
    switch (op) {
    case RedOp::add: return (r + s) & mask;
    case RedOp::min: return sgn ? (sr < ss ? r : s) : (r < s ? r : s);
    case RedOp::max: return sgn ? (sr > ss ? r : s) : (r > s ? r : s);
    case RedOp::inc: return r >= s ? 0 : r + 1;
    case RedOp::dec: return (r == 0 || r > s) ? s : r - 1;
    case RedOp::and_: return r & s;
    case RedOp::or_: return r | s;
    case RedOp::xor_: return r ^ s;
    }
    return r;
}

template <class T> void rmw(void* dst, RedOp op, RedType t, uint64_t s) {
    T* p = static_cast<T*>(dst);
    T old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(p, &old, T(apply(op, t, uint64_t(old), s)), true,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}
}  // namespace detail

// *dst = dst op src for one element (src passed as raw bits).
inline void reduce_element(void* dst, uint64_t src, RedType t, RedOp op) {
    switch (red_bytes(t)) {
    case 2: detail::rmw<uint16_t>(dst, op, t, src); break;
    case 4: detail::rmw<uint32_t>(dst, op, t, src); break;
    default: detail::rmw<uint64_t>(dst, op, t, src); break;
    }
}

}  // namespace gx
