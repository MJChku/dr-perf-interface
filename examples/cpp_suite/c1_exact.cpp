// Case 1: exact affine (P1).  outer = loop3(n) + [inner = loop5(2n)] + loop3(n).
// Expected: inner = 10*n + d1, outer (inclusive of inner) = 16*n + d2, exactly,
// and byte-identical block counts across repeats.
#include "suite.h"
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    g_sink += loop3(16) + loop5(16);              // warm-up outside any region
    for (int rep = 0; rep < 2; rep++) {
        perfmark_begin("outer", "n", n);
        g_sink += loop3(n);
        perfmark_begin("inner", "n", n);
        g_sink += loop5(2 * n);
        perfmark_end("inner");
        g_sink += loop3(n);
        perfmark_end("outer");
    }
    printf("c1 n=%ld sink=%llu\n", n, (unsigned long long)g_sink);
    return 0;
}
