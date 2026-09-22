/* Reproduce an allocator worker which blocks the late-takeover signal. */
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include "../perfmark/perfmark.h"

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static int ready, go;

__attribute__((noinline)) static void work(int n) {
    volatile uint64_t sum = 0;
    for (int i = 0; i < n; ++i) sum += i;
}

static void *worker(void *arg) {
    sigset_t set;
    sigfillset(&set);
    if (arg && pthread_sigmask(SIG_BLOCK, &set, NULL)) return (void *)1;
    pthread_mutex_lock(&lock);
    ++ready;
    pthread_cond_broadcast(&changed);
    while (!go) pthread_cond_wait(&changed, &lock);
    pthread_mutex_unlock(&lock);
    for (int n = 100; n <= 400; n += 100) {
        perfmark_begin("worker", "n", n);
        work(n);
        perfmark_end("worker");
    }
    return NULL;
}

int main(void) {
    pthread_t threads[2];
    for (int i = 0; i < 2; ++i)
        if (pthread_create(&threads[i], NULL, worker, (void *)(uintptr_t)i)) return 1;
    pthread_mutex_lock(&lock);
    while (ready != 2) pthread_cond_wait(&changed, &lock);
    pthread_mutex_unlock(&lock);
    /* Both workers exist, and one blocks SIGILL, before the first marker. */
    perfmark_begin("attach", "n", 1);
    perfmark_end("attach");
    pthread_mutex_lock(&lock);
    go = 1;
    pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&lock);
    for (int i = 0; i < 2; ++i) {
        void *result;
        if (pthread_join(threads[i], &result) || result) return 2;
    }
    puts("BLOCKED_WORKERS_PASS");
    return 0;
}
