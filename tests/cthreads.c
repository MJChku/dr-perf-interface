/* Threaded test for drperf: work done on worker threads while a region is
 * open on the main thread must be attributed to that region.
 * Usage: cthreads [n=N] [threads=T]
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../perfmark/perfmark.h"

static uint64_t sink[64];

__attribute__((noinline)) static uint64_t loop_work(long n)
{
    long i;
    uint64_t acc = 0;
    for (i = 0; i < n; i++)
        __asm__ __volatile__("add %1, %0" : "+r"(acc) : "r"((uint64_t)i));
    return acc;
}

struct arg { long n; int idx; int own_region; };

static void *worker(void *p)
{
    struct arg *a = p;
    if (a->own_region)
        perfmark_begin("worker_loop", "n", a->n);
    sink[a->idx] += loop_work(a->n);
    if (a->own_region)
        perfmark_end("worker_loop");
    return NULL;
}

static long arg_value(int argc, char **argv, const char *key, long def)
{
    int i;
    size_t kl = strlen(key);
    for (i = 1; i < argc; i++) {
        if (strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=')
            return atol(argv[i] + kl + 1);
    }
    return def;
}

static void run(long n, int T, int own_region, const char *region)
{
    pthread_t th[64];
    struct arg args[64];
    int i;
    perfmark_begin(region, "threads", T);
    for (i = 0; i < T; i++) {
        args[i].n = n; args[i].idx = i; args[i].own_region = own_region;
        pthread_create(&th[i], NULL, worker, &args[i]);
    }
    for (i = 0; i < T; i++)
        pthread_join(th[i], NULL);
    perfmark_end(region);
}

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 100000);
    int T = (int)arg_value(argc, argv, "threads", 4);
    if (T > 64) T = 64;
    run(100, T, 0, "warmup");
    run(n, T, 0, "parallel");         /* workers follow the main thread's region */
    run(n, T, 1, "parallel_own");     /* workers open their own nested regions */
    perfmark_begin("serial", "n", n);
    sink[0] += loop_work(n);
    perfmark_end("serial");
    printf("cthreads done n=%ld threads=%d sink=%llu\n", n, T, (unsigned long long)sink[0]);
    return 0;
}
