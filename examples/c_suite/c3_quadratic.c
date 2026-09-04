/* Case 3, non-affine dependence (SPEC N3).
 * quad: 5*n*n (nested loops, 5 instructions per inner iteration) + 4*n (a linear
 * loop) + outer-loop overhead.
 * Expected: the n*n blocks are reported irregular, not forced into a line;
 * the linear loop and the outer loop overhead are affine. */
#include "suite.h"
DEFINE_LOOP(work_lin, "add")

__attribute__((noinline)) uint64_t quad(long n)
{
    long i, j; uint64_t acc = 0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            __asm__ __volatile__("xor %1, %0" : "+r"(acc) : "r"((uint64_t)j));
    return acc;
}

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 100);
    sink += quad(3) + work_lin(3);
    perfmark_begin("quad", "n", n);
    sink += quad(n);
    sink += work_lin(n);
    perfmark_end("quad");
    printf("c3 n=%ld sink=%llu\n", n, (unsigned long long)sink);
    return 0;
}
