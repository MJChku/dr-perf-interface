// Case 8: delta.  base: work = loop_c(n) + loop5(n).  head (-DHEAD): adds loop3(n).
// Expected: derive(head).A - derive(base).A = 3 exactly; drperf diff shows the
// added instructions in loop3.
#include "suite.h"
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    g_sink += loop_c(16) + loop5(16) + loop3(16);
    perfmark_begin("work", "n", n);
    g_sink += loop_c(n);
    g_sink += loop5(n);
#ifdef HEAD
    g_sink += loop3(n);
#endif
    perfmark_end("work");
    printf("c8 n=%ld sink=%llu\n", n, (unsigned long long)g_sink);
    return 0;
}
