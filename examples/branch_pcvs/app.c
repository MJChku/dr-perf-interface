/* Same loop, first annotated with raw inputs, then with conditional PCVs.
 * Run via: python3 examples/branch_pcvs/validate.py
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../../perfmark/perfmark.h"

__attribute__((noinline)) static uint64_t expensive_step(uint64_t value)
{
    uint64_t result = 0;
    for (int j = 0; j < 32; ++j) {
        result += value;
        /* Keep the work from being replaced by a single multiplication. */
        __asm__ __volatile__("" : "+r"(result));
    }
    return result;
}

__attribute__((noinline)) static uint64_t cheap_step(uint64_t value)
{
    __asm__ __volatile__("" : "+r"(value));
    return value;
}

__attribute__((noinline)) static uint64_t process(long x, long y)
{
    uint64_t result = 0;
    for (long i = 0; i < x; ++i) {
        if (i < y)
            result += expensive_step((uint64_t)i + 1);
        else
            result += cheap_step((uint64_t)i + 1);
    }
    return result;
}

/* Both annotations wrap exactly the same call site and implementation. */
__attribute__((noinline)) static uint64_t measure(
    const char *region, const char *const *names, const int64_t *values,
    long x, long y)
{
    perfmark_begin_v(region, 2, names, values);
    uint64_t result = process(x, y);
    perfmark_end(region);
    return result;
}

int main(void)
{
    const long xs[] = {1000, 2000, 4000, 8000};
    const long ys[] = {500, 1500, 3000, 6000, 12000};
    const char *raw_names[] = {"x", "y"};
    const char *conditional_names[] = {"expensive_iters", "cheap_iters"};
    uint64_t checksum = 0;
    int cases = 0;
    assert(process(10, 3) == 55 + 31 * 6);
    for (unsigned i = 0; i < sizeof(xs) / sizeof(xs[0]); ++i) {
        for (unsigned j = 0; j < sizeof(ys) / sizeof(ys[0]); ++j) {
            long x = xs[i], y = ys[j];
            /* These branches are evaluated before entering either region. */
            long expensive = x <= y ? x : y;
            long cheap = x <= y ? 0 : x - y;
            int64_t raw[] = {x, y};
            int64_t conditional[] = {expensive, cheap};
            uint64_t expected = (uint64_t)x * (x + 1) / 2
                + 31 * (uint64_t)expensive * (expensive + 1) / 2;
            for (int rep = 0; rep < 3; ++rep) {
                uint64_t a = measure("raw", raw_names, raw, x, y);
                uint64_t b = measure("conditional", conditional_names, conditional, x, y);
                assert(a == expected && b == expected);
                checksum += a + b;
            }
            ++cases;
        }
    }
    printf("PASS: %d input pairs, 3 repetitions, identical checked results; checksum=%llu\n",
           cases, (unsigned long long)checksum);
    return 0;
}
