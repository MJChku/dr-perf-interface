// Case 5: two code paths chosen by n < 1000.  Expected: two regimes,
// n < 1000: 3*n + d, n >= 1000: 5*n + 3*500 + d'.
#include "suite.h"
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    g_sink += loop3(16) + loop5(16);
    perfmark_begin("regime", "n", n);
    if (n < 1000)
        g_sink += loop3(n);
    else
        g_sink += loop5(n) + loop3(500);
    perfmark_end("regime");
    printf("c5 n=%ld sink=%llu\n", n, (unsigned long long)g_sink);
    return 0;
}
