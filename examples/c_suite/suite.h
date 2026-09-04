/* Shared helpers for the drperf C suite (examples/c_suite). */
#ifndef SUITE_H
#define SUITE_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../perfmark/perfmark.h"

static long arg_value(int argc, char **argv, const char *key, long def)
{
    int i;
    size_t kl = strlen(key);
    for (i = 1; i < argc; i++)
        if (strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=')
            return atol(argv[i] + kl + 1);
    return def;
}

static const char *arg_str(int argc, char **argv, const char *key, const char *def)
{
    int i;
    size_t kl = strlen(key);
    for (i = 1; i < argc; i++)
        if (strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=')
            return argv[i] + kl + 1;
    return def;
}

/* A counted loop with exactly 4 instructions per iteration at -O2
 * (op, add i, cmp, jne); `op` differs per function so -fipa-icf cannot
 * fold two loops into one block. */
#define DEFINE_LOOP(name, op) \
__attribute__((noinline)) uint64_t name(long n) \
{ \
    long i; uint64_t acc = 0; \
    for (i = 0; i < n; i++) \
        __asm__ __volatile__(op " %1, %0" : "+r"(acc) : "r"((uint64_t)i)); \
    return acc; \
}

static uint64_t sink;
#endif
