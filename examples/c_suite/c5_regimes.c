/* Case 5, two regimes.
 * reg: n < 100 -> small(n)            = 4*n
 *      n >= 100 -> big(10*n) + fix(75) = 40*n + 300
 * Expected: derive splits at 100 with slopes 4 and 40 and the 300 in the upper constant. */
#include "suite.h"
DEFINE_LOOP(small_path, "add")
DEFINE_LOOP(big_path, "sub")
DEFINE_LOOP(fix_path, "xor")

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 10);
    sink += small_path(3) + big_path(3) + fix_path(3);
    perfmark_begin("reg", "n", n);
    if (n < 100)
        sink += small_path(n);
    else
        sink += big_path(10 * n) + fix_path(75);
    perfmark_end("reg");
    printf("c5 n=%ld sink=%llu\n", n, (unsigned long long)sink);
    return 0;
}
