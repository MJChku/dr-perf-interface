/* Case 7, a delta between two builds.
 * feature: base = work(n) [4*n]; head adds extra3(n), a loop of exactly 3
 * instructions per iteration (add, sub, jnz).
 * Expected: derive A_head - A_base = 3 exactly; drperf diff shows the slope change. */
#include "suite.h"
DEFINE_LOOP(work, "add")

#ifdef HEAD
__attribute__((noinline)) uint64_t extra3(long n)
{
    uint64_t acc = 0;
    __asm__ __volatile__("1:\n\tadd %1, %0\n\tsub $1, %1\n\tjnz 1b" : "+r"(acc), "+r"(n) : : "cc");
    return acc;
}
#endif

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    sink += work(3);
    perfmark_begin("feature", "n", n);
    sink += work(n);
#ifdef HEAD
    sink += extra3(n);
#endif
    perfmark_end("feature");
    printf("c7 n=%ld sink=%llu\n", n, (unsigned long long)sink);
    return 0;
}
