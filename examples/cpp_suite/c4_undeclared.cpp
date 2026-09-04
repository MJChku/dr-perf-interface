// Case 4 (N2): the region declares n but the cost depends on m.
//  hidden_hash:  m = (n*7919) % 1000 + 100 -> expect irregular.  Use non-round n
//                (137, 251, ...): for multiples of 100 this m equals 1100 - n, which
//                is affine, and derive then correctly reports -3*n + 3,314.
//  hidden_cycle: m cycles over {100,300,500,700} across 4 triggers at each n
//                -> per-value means are equal at every n -> classified constant (N4)
//  hidden_2n:    m = 2n -> attributed to n with coefficient 2*3 = 6 per n (N2)
#include "suite.h"
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    g_sink += loop3(16);
    long m = (n * 7919) % 1000 + 100;
    perfmark_begin("hidden_hash", "n", n);
    g_sink += loop3(m);
    perfmark_end("hidden_hash");
    static const long cyc[4] = { 100, 300, 500, 700 };
    for (int k = 0; k < 4; k++) {
        perfmark_begin("hidden_cycle", "n", n);
        g_sink += loop3(cyc[k]);
        perfmark_end("hidden_cycle");
    }
    perfmark_begin("hidden_2n", "n", n);
    g_sink += loop3(2 * n);
    perfmark_end("hidden_2n");
    printf("c4 n=%ld m=%ld sink=%llu\n", n, m, (unsigned long long)g_sink);
    return 0;
}
