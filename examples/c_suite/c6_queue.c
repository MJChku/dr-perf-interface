/* Case 6b, relationship across threads (SPEC R1).
 * produce (thread A) pushes m items per call in bursts of 1..3 calls, then
 * wakes the consumer; consume (thread B) drains the queue, declaring q = items
 * found.  Every produce has ended before a consume begins, so
 *     consume.q = cum(produce.m) - cum(consume.q)
 * must hold exactly at every consume; q != last(produce.m) because of the bursts.
 * consume's cost is 4*8 = 32 instructions per item + constant. */
#include <pthread.h>
#include "suite.h"
DEFINE_LOOP(per_item, "add")

static long ring[1 << 16];
static long head, tail;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_full = PTHREAD_COND_INITIALIZER, cv_drained = PTHREAD_COND_INITIALIZER;
static int ready, drained, done;
static long ROUNDS;

static void produce(long m)
{
    long i;
    perfmark_begin("produce", "m", m);
    pthread_mutex_lock(&mu);
    for (i = 0; i < m; i++) ring[tail++ & 0xFFFF] = i;
    pthread_mutex_unlock(&mu);
    perfmark_end("produce");
}

static void consume(void)
{
    long q, i;
    pthread_mutex_lock(&mu);
    q = tail - head;
    pthread_mutex_unlock(&mu);
    perfmark_begin("consume", "q", q);
    for (i = 0; i < q; i++) sink += per_item(8) + ring[head++ & 0xFFFF];
    perfmark_end("consume");
}

static void *producer(void *p)
{
    long k = 0;
    (void)p;
    while (k < ROUNDS) {
        int burst = 1 + (int)(k % 3), b;
        for (b = 0; b < burst && k < ROUNDS; b++, k++)
            produce(5 + 20 * (k % 4));
        pthread_mutex_lock(&mu); ready = 1; pthread_cond_signal(&cv_full);
        while (!drained) pthread_cond_wait(&cv_drained, &mu);
        drained = 0; pthread_mutex_unlock(&mu);
    }
    pthread_mutex_lock(&mu); done = 1; ready = 1; pthread_cond_signal(&cv_full); pthread_mutex_unlock(&mu);
    return NULL;
}

int main(int argc, char **argv)
{
    pthread_t th;
    ROUNDS = arg_value(argc, argv, "rounds", 12);
    pthread_create(&th, NULL, producer, NULL);
    for (;;) {
        pthread_mutex_lock(&mu);
        while (!ready) pthread_cond_wait(&cv_full, &mu);
        ready = 0;
        if (done && tail == head) { pthread_mutex_unlock(&mu); break; }
        pthread_mutex_unlock(&mu);
        consume();
        pthread_mutex_lock(&mu); drained = 1; pthread_cond_signal(&cv_drained); pthread_mutex_unlock(&mu);
    }
    pthread_join(th, NULL);
    printf("c6q rounds=%ld consumed=%ld sink=%llu\n", ROUNDS, head, (unsigned long long)sink);
    return 0;
}
