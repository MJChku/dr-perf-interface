// Case 9: the same source at -O0 and -O2 (a toolchain change).  Region "sum":
// dot product of two int vectors of length n.  Expected: both affine in n with
// very different coefficients; derive reports each, diff reports the change.
#include "suite.h"
#include <vector>
__attribute__((noinline)) long dot(const std::vector<int> &a, const std::vector<int> &b)
{
    long s = 0;
    for (size_t i = 0; i < a.size(); i++) s += (long)a[i] * b[i];
    return s;
}
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    std::vector<int> a(n), b(n);
    for (long i = 0; i < n; i++) { a[i] = (int)(i % 7); b[i] = (int)(i % 5); }
    g_sink += dot(a, b);
    perfmark_begin("sum", "n", n);
    g_sink += dot(a, b);
    perfmark_end("sum");
    printf("c9 n=%ld sink=%llu\n", n, (unsigned long long)g_sink);
    return 0;
}
