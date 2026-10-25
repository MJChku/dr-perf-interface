// SPDX-License-Identifier: MIT
// Floating-point add/sub/mul/fma with PTX rounding, .ftz and .sat modifiers
// for translated PTX. The host rounding mode is switched only around the
// single operation (fibers never yield inside), and volatile temporaries keep
// the compiler from folding or moving the operation across fesetround.
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

enum : uint32_t { kRn = 0, kRz = 1, kRm = 2, kRp = 3, kFtz = 4, kSat = 8 };

int host_mode(uint32_t flags) {
    switch (flags & 3) {
    case kRz: return FE_TOWARDZERO;
    case kRm: return FE_DOWNWARD;
    case kRp: return FE_UPWARD;
    default: return FE_TONEAREST;
    }
}

template <class T> T flush(T v, bool ftz) {
    return ftz && std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(T(0), v) : v;
}

template <class T> T compute(uint32_t op, T a, T b, T c, uint32_t flags) {
    const bool ftz = flags & kFtz;
    a = flush(a, ftz); b = flush(b, ftz); c = flush(c, ftz);
    const int old = fegetround();
    fesetround(host_mode(flags));
    volatile T va = a, vb = b, vc = c;
    volatile T r;
    switch (op) {
    case 0: r = va + vb; break;
    case 1: r = va - vb; break;
    case 2: r = va * vb; break;
    default: r = std::fma(T(va), T(vb), T(vc)); break;
    }
    fesetround(old);
    T out = flush(T(r), ftz);
    if (flags & kSat) out = std::isnan(out) ? T(0) : out < T(0) ? T(0) : out > T(1) ? T(1) : out;
    return out;
}

}  // namespace

// op: 0 add, 1 sub, 2 mul, 3 fma; flags: rounding (0 rn, 1 rz, 2 rm, 3 rp) | 4 ftz | 8 sat.
extern "C" uint32_t gx_farith32(uint32_t op, uint32_t a, uint32_t b, uint32_t c, uint32_t flags) {
    float fa, fb, fc;
    std::memcpy(&fa, &a, 4); std::memcpy(&fb, &b, 4); std::memcpy(&fc, &c, 4);
    const float r = compute<float>(op, fa, fb, fc, flags);
    uint32_t out; std::memcpy(&out, &r, 4);
    return out;
}

extern "C" uint64_t gx_farith64(uint32_t op, uint64_t a, uint64_t b, uint64_t c, uint32_t flags) {
    double fa, fb, fc;
    std::memcpy(&fa, &a, 8); std::memcpy(&fb, &b, 8); std::memcpy(&fc, &c, 8);
    const double r = compute<double>(op, fa, fb, fc, flags);
    uint64_t out; std::memcpy(&out, &r, 8);
    return out;
}
