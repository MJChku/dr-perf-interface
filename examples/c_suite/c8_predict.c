/* Case 8, prediction and extrapolation (SPEC N5).
 * lin: 4*n, exactly affine everywhere -> predicts any n.
 * cap: 4*min(n, 800): affine on the derivation range 100..500, flat beyond;
 *      the derived line extrapolates and is wrong at 1000 and 4000. */
#include "suite.h"
DEFINE_LOOP(lin_work, "add")
DEFINE_LOOP(cap_work, "sub")

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 100);
    sink += lin_work(3) + cap_work(3);
    perfmark_begin("lin", "n", n);
    sink += lin_work(n);
    perfmark_end("lin");
    perfmark_begin("cap", "n", n);
    sink += cap_work(n < 800 ? n : 800);
    perfmark_end("cap");
    printf("c8 n=%ld sink=%llu\n", n, (unsigned long long)sink);
    return 0;
}
