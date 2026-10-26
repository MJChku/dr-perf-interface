#pragma once
#include <cstdint>

__device__ inline float ds41_bf16_float(std::uint16_t v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }
__device__ inline std::uint16_t ds41_float_bf16(float v) {
    unsigned bits = __float_as_uint(v);
    if ((bits & 0x7fffffff) > 0x7f800000) return static_cast<std::uint16_t>((bits >> 16) | 64);
    return static_cast<std::uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
__device__ inline float ds41_e4m3_float(unsigned code) {
    unsigned mag = code & 127, exponent = mag >> 3, mantissa = mag & 7;
    float value = exponent ? __uint_as_float(((exponent + 120) << 23) | (mantissa << 20)) : mantissa * 0x1p-9f;
    return (code & 128) ? -value : value;
}
__device__ inline unsigned ds41_float_e4m3(float value) {
    const unsigned sign = (__float_as_uint(value) >> 24) & 128;
    const float a = fminf(fabsf(value),448.0f);
    unsigned magnitude;
    if (a < 0x1p-6f) {
        const float units = a * 512.0f;
        magnitude = static_cast<unsigned>(units);
        const float remainder = units - magnitude;
        magnitude += remainder > .5f || (remainder == .5f && (magnitude & 1));
    } else {
        unsigned bits = __float_as_uint(a);
        magnitude = ((bits + 0x7ffff + ((bits >> 20) & 1)) >> 20) - 960;
        magnitude = min(magnitude,126u);
    }
    return sign | magnitude;
}
__device__ inline float ds41_e2m1_float(unsigned code) {
    float value;
    switch (code & 7) {
        case 0:value=0;break; case 1:value=.5f;break; case 2:value=1;break;
        case 3:value=1.5f;break; case 4:value=2;break; case 5:value=3;break;
        case 6:value=4;break; default:value=6;
    }
    return (code & 8) ? -value : value;
}
__device__ inline unsigned ds41_float_e2m1(float value) {
    const float a = fminf(fabsf(value),6.0f);
    // Midpoint ties choose the even representable significand/code.
    unsigned code = a <= .25f ? 0 : a < .75f ? 1 : a <= 1.25f ? 2 :
                    a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5 ? 6 : 7;
    return code | ((__float_as_uint(value) >> 28) & 8);
}
__device__ inline unsigned ds41_ceil_power_scale(float value) {
    const unsigned bits = __float_as_uint(value);
    return ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
}
