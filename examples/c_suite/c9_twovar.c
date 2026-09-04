/* c9_twovar.c: one region declaring two states, n and m (perfmark_begin_v).
 *
 * Expected (SPEC, two variables): derive gives cost = A_n*n + A_m*m + D
 * exactly with 0 irregular, A_n = instructions per iteration of work_n,
 * A_m = instructions per iteration of work_m.  With `one=1` the region
 * declares n only: the m-dependent blocks cannot follow a line in n and are
 * reported irregular.  With `prod=1` the body is n*m: irregular in (n, m).
 *
 *   drperf run --blocks -o X --state n=100,200,300,500 --state m=50,150,400 -- ./c9_twovar
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../perfmark/perfmark.h"

static long arg(int argc, char **argv, const char *key, long def)
{
    size_t kl = strlen(key);
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=')
            return atol(argv[i] + kl + 1);
    return def;
}

__attribute__((noinline)) static long work_n(long n)
{
    long s = 0;
    for (long i = 0; i < n; i++) {
        s += i;
        __asm__ __volatile__("" : "+r"(s));
    }
    return s;
}

__attribute__((noinline)) static long work_m(long m)
{
    long s = 1;
    for (long i = 0; i < m; i++) {
        s ^= i;
        s += 3;
        __asm__ __volatile__("" : "+r"(s));
    }
    return s;
}

__attribute__((noinline)) static long work_prod(long n, long m)
{
    long s = 0;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < m; j++) {
            s += j;
            __asm__ __volatile__("" : "+r"(s));
        }
    return s;
}

int main(int argc, char **argv)
{
    long n = arg(argc, argv, "n", 100), m = arg(argc, argv, "m", 50);
    long one = arg(argc, argv, "one", 0), prod = arg(argc, argv, "prod", 0);
    const char *names[2] = { "n", "m" };
    int64_t vals[2] = { n, m };
    long acc = 0;
    for (int rep = 0; rep < 3; rep++) {
        if (one)
            perfmark_begin("twovar", "n", n);
        else
            perfmark_begin_v("twovar", 2, names, vals);
        if (prod)
            acc += work_prod(n, m);
        else
            acc += work_n(n) + work_m(m);
        perfmark_end("twovar");
    }
    printf("c9_twovar n=%ld m=%ld acc=%ld\n", n, m, acc);
    return 0;
}
