/* Case 4, cost depending on an undeclared variable (SPEC N2).
 * dep: work_n(n) + work_m(m); only n is declared.
 *   mode=indep : m from a table indexed by n/1000 (no relation to n) -> work_m irregular
 *   mode=corr  : m = 2 * n                    (co-varies with n) -> work_m attributed to n: A = 4 + 8
 * Expected: indep gives A = 4 with the work_m block irregular; corr gives A = 12, nothing irregular. */
#include "suite.h"
DEFINE_LOOP(work_n, "add")
DEFINE_LOOP(work_m, "sub")

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    const char *mode = arg_str(argc, argv, "mode", "indep");
    static const long table[] = { 700, 2600, 300, 4100, 900, 3300, 150, 2000 };
    long m = strcmp(mode, "corr") == 0 ? 2 * n : table[(n / 1000) % 8];
    sink += work_n(3) + work_m(3);
    perfmark_begin("dep", "n", n);
    sink += work_n(n);
    sink += work_m(m);
    perfmark_end("dep");
    printf("c4 n=%ld m=%ld mode=%s sink=%llu\n", n, m, mode, (unsigned long long)sink);
    return 0;
}
