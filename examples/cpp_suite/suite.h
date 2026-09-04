// Shared helpers for the drperf C++ suite.  Programs take "key=value" argv
// items (drperf run appends the --state values that way).
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include "../../perfmark/perfmark.h"

static inline long arg_value(int argc, char **argv, const char *key, long def)
{
    size_t kl = strlen(key);
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=')
            return atol(argv[i] + kl + 1);
    return def;
}

static uint64_t g_sink;

// Exactly 3 instructions per iteration (add, dec, jnz); n >= 1.
__attribute__((noinline)) static uint64_t loop3(long n)
{
    uint64_t acc = 0;
    long cnt = n;
    __asm__ __volatile__("1:\n\tadd %2, %0\n\tdec %1\n\tjnz 1b"
                         : "+r"(acc), "+r"(cnt) : "r"((uint64_t)7) : "cc");
    return acc;
}

// Exactly 5 instructions per iteration (add, xor, add, dec, jnz); n >= 1.
__attribute__((noinline)) static uint64_t loop5(long n)
{
    uint64_t acc = 0, tmp = 1;
    long cnt = n;
    __asm__ __volatile__("1:\n\tadd %2, %0\n\txor %1, %0\n\tadd %0, %1\n\tdec %3\n\tjnz 1b"
                         : "+r"(acc), "+r"(tmp), "+r"(tmp), "+r"(cnt) : : "cc");
    return acc;
}

// C loop: 1 asm add + compiler loop control (4/iteration at -O2, as in tests/ctest.c)
__attribute__((noinline)) static uint64_t loop_c(long n)
{
    uint64_t acc = 0;
    for (long i = 0; i < n; i++)
        __asm__ __volatile__("add %1, %0" : "+r"(acc) : "r"((uint64_t)i));
    return acc;
}
