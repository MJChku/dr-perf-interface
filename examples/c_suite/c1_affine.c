/* Case 1, exact affine (SPEC P1) and determinism.
 * single: 4*n + c              (one loop)
 * inner : 8*n + c              (loop of 2n, nested in outer)
 * outer : self 8*n + c, inclusive 16*n + c
 * Expected: derive gives exactly these slopes, 0 irregular; --repeat 3 gives
 * byte-identical block counts.  With multi=1 the regions run at n, 2n, 3n
 * and 5n inside one process (all state values in one trace). */
#include "suite.h"
DEFINE_LOOP(work_a, "add")
DEFINE_LOOP(work_b, "sub")

int main(int argc, char **argv)
{
    long n0 = arg_value(argc, argv, "n", 1000);
    long multi = arg_value(argc, argv, "multi", 0);
    static const long mult[] = { 1, 2, 3, 5 };
    int k;
    sink += work_a(10) + work_b(10);                    /* warm-up, outside regions */
    for (k = 0; k < (multi ? 4 : 1); k++) {
    long n = n0 * mult[k];
    perfmark_begin("single", "n", n);
    sink += work_a(n);
    perfmark_end("single");
    perfmark_begin("outer", "n", n);
    sink += work_a(n);
    perfmark_begin("inner", "n", n);
    sink += work_b(2 * n);
    perfmark_end("inner");
    sink += work_a(n);
    perfmark_end("outer");
    }
    printf("c1 n=%ld multi=%ld sink=%llu\n", n0, multi, (unsigned long long)sink);
    return 0;
}
