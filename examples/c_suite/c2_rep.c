/* Case 2, rep-string expansion.
 * stos: rep stosb of n bytes; movs: rep movsb of n bytes.
 * Expected: with expansion (default) 1 instruction per byte + constant;
 * with --no-rep-expand the rep instruction counts as 1 (slope 0). */
#include "suite.h"

__attribute__((noinline)) void do_stos(unsigned char *p, long n)
{
    __asm__ __volatile__("rep stosb" : "+D"(p), "+c"(n) : "a"(1) : "memory");
}
__attribute__((noinline)) void do_movs(unsigned char *d, const unsigned char *s, long n)
{
    __asm__ __volatile__("rep movsb" : "+D"(d), "+S"(s), "+c"(n) : : "memory");
}

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    unsigned char *buf = malloc(1 << 20);
    do_stos(buf, 64); do_movs(buf + 4096, buf, 64);      /* warm-up */
    perfmark_begin("stos", "bytes", n);
    do_stos(buf, n);
    perfmark_end("stos");
    perfmark_begin("movs", "bytes", n);
    do_movs(buf + (1 << 19), buf, n);
    perfmark_end("movs");
    printf("c2 n=%ld %d\n", n, buf[n - 1]);
    return 0;
}
