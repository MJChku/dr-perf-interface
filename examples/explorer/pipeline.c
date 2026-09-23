/* drperf explorer demo: concurrent producer/consumer, affine and unexplained work.
 * ./run.sh measures the real program and exports a source-linked model.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../perfmark/perfmark.h"

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static long queued;
static int available, consumed, done;
static long items_scale = 1;
static volatile uint64_t producer_sink, consumer_sink;

__attribute__((noinline)) static uint64_t work(long n) {
    uint64_t value = 0;
    for (long i = 0; i < n; ++i)
        __asm__ __volatile__("add %1, %0" : "+r"(value) : "r"((uint64_t)i));
    return value;
}

static void *produce(void *unused) {
    (void)unused;
    for (long round = 0; round < 24; ++round) {
        int burst = 1 + round % 3;
        for (int b = 0; b < burst; ++b) {
            long items = items_scale * (7 + (round * 17 + b * 11) % 53);
            perfmark_begin("enqueue", "items", items);
            producer_sink += work(items * 16);
            pthread_mutex_lock(&mutex);
            queued += items;
            pthread_mutex_unlock(&mutex);
            perfmark_end("enqueue");
        }
        pthread_mutex_lock(&mutex);
        available = 1;
        pthread_cond_broadcast(&changed);
        while (!consumed) pthread_cond_wait(&changed, &mutex);
        consumed = 0;
        pthread_mutex_unlock(&mutex);
    }
    pthread_mutex_lock(&mutex);
    done = 1;
    pthread_cond_broadcast(&changed);
    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main(int argc, char **argv) {
    pthread_t producer;
    int refined_pcvs = argc > 2 && argv[2][0] == '1';
    long token_multiplier = 2;
    if (argc > 1) {
        char *end;
        items_scale = strtol(argv[1], &end, 10);
        if (*end || items_scale < 1 || items_scale > 8) {
            fprintf(stderr, "usage: %s [items_scale: 1..8]\n", argv[0]);
            return 2;
        }
    }
    if (argc > 3) {
        char *end;
        token_multiplier = strtol(argv[3], &end, 10);
        if (*end || token_multiplier < 1 || token_multiplier > 8) return 2;
    }
    /* Establish instrumentation before the producer begins its first region. */
    perfmark_begin("startup", "once", 1);
    perfmark_end("startup");
    if (pthread_create(&producer, NULL, produce, NULL)) return 1;
    for (;;) {
        pthread_mutex_lock(&mutex);
        while (!available && !done) pthread_cond_wait(&changed, &mutex);
        if (done && !available) { pthread_mutex_unlock(&mutex); break; }
        long items = queued;
        queued = 0;
        available = 0;
        pthread_mutex_unlock(&mutex);

        perfmark_begin("dequeue", "items", items);
        consumer_sink += work(items * 32);
        perfmark_end("dequeue");

        long tokens = token_multiplier * items;
        perfmark_begin("decode", "tokens", tokens);
        consumer_sink += work(tokens * 64);
        perfmark_end("decode");

        long bytes = 8 * tokens;
        perfmark_begin("copy", "bytes", bytes);
        consumer_sink += work(bytes * 8);
        perfmark_end("copy");

        /* The initial PCV misses quadratic growth. Refine the interface without
         * changing the region body: pair count is its natural unit of work. */
        if (refined_pcvs)
            perfmark_begin("lookup", "pairs", items * items);
        else
            perfmark_begin("lookup", "entries", items);
        consumer_sink += work(items * items * 32);
        perfmark_end("lookup");

        perfmark_begin("dispatch", "items", items);
        consumer_sink += work(items < 70 ? items * 128 : items * 32);
        perfmark_end("dispatch");

        pthread_mutex_lock(&mutex);
        consumed = 1;
        pthread_cond_broadcast(&changed);
        pthread_mutex_unlock(&mutex);
    }
    pthread_join(producer, NULL);
    printf("pipeline complete: %llu / %llu\n", (unsigned long long)producer_sink,
           (unsigned long long)consumer_sink);
    return 0;
}
