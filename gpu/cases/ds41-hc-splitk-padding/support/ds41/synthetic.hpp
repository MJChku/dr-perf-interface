#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define DS41_HD __host__ __device__
#else
#define DS41_HD
#endif

namespace ds41::synthetic {
inline constexpr std::uint64_t seed = 0x41345eed20261024ull;
DS41_HD inline std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}
inline std::uint64_t tensor_id(const char* name) {
    std::uint64_t value = 0xcbf29ce484222325ull;
    for (; *name; ++name) {
        value ^= static_cast<unsigned char>(*name);
        value *= 0x100000001b3ull;
    }
    return value;
}
DS41_HD inline std::uint8_t fp8(std::uint64_t id, std::uint64_t offset) {
    const auto h = mix(seed ^ id ^ offset);
    return static_cast<std::uint8_t>(((h >> 56) & 128) | ((4 + (h >> 16) % 5) << 3) | (h & 7));
}
DS41_HD inline std::uint8_t fp4(std::uint64_t id, std::uint64_t offset) {
    const auto h = mix(seed ^ id ^ offset);
    const auto lo = ((h & 7) % 7 + 1) | (h & 8);
    const auto hi = (((h >> 8) & 7) % 7 + 1) | ((h >> 8) & 8);
    return static_cast<std::uint8_t>(lo | (hi << 4));
}
// Compute the base exponent with integer operations to avoid platform libm
// differences. Caller supplies positive fan_in and checks exponent range.
inline int scale_exponent(std::uint64_t fan_in) {
    int ceil_log2 = 0;
    for (auto n = fan_in - 1; n; n >>= 1) ++ceil_log2;
    return -(ceil_log2 + 1) / 2;
}
DS41_HD inline std::uint8_t scale(std::uint64_t scale_id, std::uint64_t offset, int exponent) {
    const auto h = mix(seed ^ scale_id ^ offset);
    return static_cast<std::uint8_t>(127 + exponent + static_cast<int>((h >> 32) % 3) - 1);
}
// All offsets are global byte offsets, including partial scalar writes. FP32
// promoted-BF16 parameters preserve the checkpoint's lossless promotion rule;
// native FP32 HC parameters retain all 23 mantissa bits.
enum Kind : unsigned { fp8_kind,fp4_kind,scale_kind,bf16_kind,fp32_kind,
                       bf16_one_kind,fp32_one_kind,fp32_from_bf16_kind,hc_scale_kind };
DS41_HD inline std::uint8_t byte(unsigned kind,std::uint64_t id,std::uint64_t offset,int exponent){
    if(kind==fp8_kind)return fp8(id,offset);
    if(kind==fp4_kind)return fp4(id,offset);
    if(kind==scale_kind)return scale(id,offset,exponent);
    const bool short_scalar=kind==bf16_kind||kind==bf16_one_kind;
    const unsigned width=short_scalar ? 2:4;
    const auto h=mix(seed^id^(offset/width));
    const auto sign=static_cast<std::uint32_t>((h>>63)<<31);
    const auto exp=static_cast<std::uint32_t>(127+exponent+static_cast<int>((h>>32)%3)-1);
    std::uint32_t bits=sign|(exp<<23)|static_cast<std::uint32_t>(h&0x7fffff);
    if(kind==fp32_from_bf16_kind)bits&=0xffff0000u;
    if(kind==bf16_one_kind||kind==fp32_one_kind)bits=0x3f800000u;
    if(kind==hc_scale_kind)bits=0x3c23d70au; // FP32 0.01
    if(short_scalar)bits>>=16;
    return static_cast<std::uint8_t>(bits>>((offset%width)*8));
}
}
#undef DS41_HD
