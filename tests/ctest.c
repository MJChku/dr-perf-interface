/* Deterministic C test for drperf: regions with known instruction structure.
 * Usage: ctest [n]   (default n = 1000)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../perfmark/perfmark.h"

static uint64_t sink;

__attribute__((noinline)) static void loop_work(long n)
{
    /* One add + one loop-control pair per iteration, nothing else. */
    long i;
    uint64_t acc = 0;
    for (i = 0; i < n; i++)
        __asm__ __volatile__("add %1, %0" : "+r"(acc) : "r"((uint64_t)i));
    sink += acc;
}

__attribute__((noinline)) static void rep_work(unsigned char *buf, long n)
{
    /* rep stosb: 1 instruction, n iterations. */
    __asm__ __volatile__("rep stosb" : "+D"(buf), "+c"(n) : "a"(0) : "memory");
}

int main(int argc, char **argv)
{
    long n = 1000;
    if (argc > 1) {
        const char *eq = strchr(argv[1], '=');
        n = atol(eq != NULL ? eq + 1 : argv[1]);
    }
    unsigned char *buf = malloc(1 << 20);
    int rep;
    /* warm-up outside any region */
    loop_work(10);
    rep_work(buf, 64);
    for (rep = 0; rep < 3; rep++) {
        perfmark_begin("loop", "n", n);
        loop_work(n);
        perfmark_end("loop");
    }
    perfmark_begin("rep", "bytes", n);
    rep_work(buf, n);
    perfmark_end("rep");
    perfmark_begin("outer", "n", n);
    loop_work(n);
    perfmark_begin("inner", "n", n);
    loop_work(2 * n);
    perfmark_end("inner");
    loop_work(n);
    perfmark_end("outer");
    /* unbalanced: an inner region left open, closed by the outer end */
    perfmark_begin("unbal_outer", "", 0);
    perfmark_begin("unbal_inner", "", 0);
    loop_work(n);
    perfmark_end("unbal_outer");
    /* end without begin */
    perfmark_end("never_opened");
    /* memcpy through libc (exercises rep movsb / vector paths) */
    perfmark_begin("memcpy", "bytes", n);
    memcpy(buf + (1 << 19), buf, n);
    perfmark_end("memcpy");
    printf("ctest done n=%ld sink=%llu\n", n, (unsigned long long)sink);
    return 0;
}
