/* Marker throughput under contention: T threads each do N begin/end pairs
 * around a tiny body.  Usage: cmarkers [threads=T] [n=N]  */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "../perfmark/perfmark.h"
static long arg_value(int argc, char **argv, const char *key, long def) {
    size_t kl = strlen(key);
    for (int i = 1; i < argc; i++)
        if (strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=') return atol(argv[i] + kl + 1);
    return def;
}
static long N; static volatile uint64_t sink[64];
static void *worker(void *p) {
    int idx = (int)(intptr_t)p; uint64_t acc = 0;
    for (long i = 0; i < N; i++) {
        perfmark_begin("body", "", 0);
        for (int j = 0; j < 20; j++) __asm__ __volatile__("add %1, %0" : "+r"(acc) : "r"((uint64_t)j));
        perfmark_end("body");
    }
    sink[idx] = acc; return NULL;
}
int main(int argc, char **argv) {
    int T = (int)arg_value(argc, argv, "threads", 1); N = arg_value(argc, argv, "n", 100000);
    pthread_t th[64]; struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i = 0; i < T; i++) pthread_create(&th[i], NULL, worker, (void *)(intptr_t)i);
    for (int i = 0; i < T; i++) pthread_join(th[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);
    double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
    printf("threads=%d pairs/thread=%ld wall=%.3fs -> %.0f ns per begin/end pair (per thread)\n", T, N, s, s * 1e9 / N);
    return 0;
}
