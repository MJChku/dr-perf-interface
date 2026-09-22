#include <stdint.h>
typedef void (*body_fn)(void *, int);
#ifdef EMULATOR
__attribute__((noinline)) void gxvm_gpu_native_run(unsigned device, body_fn body, void *arg) {
    body(arg, (int)device);
}
__attribute__((noinline)) void cudaFake(unsigned device, body_fn body, void *arg) {
    gxvm_gpu_native_run(device, body, arg);
}
#else
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "../perfmark/perfmark.h"
extern void cudaFake(unsigned device, body_fn body, void *arg);
static void body(void *arg, int device) {
    volatile uint64_t *sum = arg;
    /* Native work may use application TLS and grow the allocator's heap. */
    char *buffers[256];
    for (int i = 0; i < 256; ++i) {
        buffers[i] = malloc(8192);
        if (!buffers[i]) abort();
        buffers[i][0] = (char)i;
    }
    for (int i = 0; i < 10000; ++i) *sum += (unsigned)i + device;
    for (int i = 0; i < 256; ++i) free(buffers[i]);
}
__attribute__((noinline)) static void host_work(int n, volatile uint64_t *sum) {
    for (int i = 0; i < n; ++i) *sum += i;
}
static void *worker(void *arg) {
    (void)arg;
    for (int n = 1; n <= 4; ++n) {
        volatile uint64_t sum = 0;
        perfmark_begin("emulated", "n", n);
        cudaFake(3, body, (void *)&sum);
        perfmark_end("emulated");
        if (sum != 50025000) return (void *)1;
        perfmark_begin("after", "n", n);
        host_work(n * 100, &sum);
        perfmark_end("after");
    }
    return NULL;
}
int main(void) {
    pthread_t thread;
    void *result;
    /* Attach before starting the worker in late mode. */
    perfmark_begin("setup", "n", 1);
    perfmark_end("setup");
    if (pthread_create(&thread, NULL, worker, NULL)) return 1;
    void *own = worker(NULL);
    if (pthread_join(thread, &result) || result || own) return 2;
    puts("GX_NATIVE_PASS");
    return 0;
}
#endif
