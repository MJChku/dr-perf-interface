/* playground/system.c: a small log-processing system to play with drperf.
 *
 *   ingest -> queue -> handle (parse each line, emit) -> output
 *
 * Three marked regions, each declaring integer states.  All declared states
 * form the aggregation key, and the cost formula is derived in all of them:
 *
 *   ingest   batch    lines pushed this round
 *            bytes    batch * len
 *   handle   q        lines popped this round
 *            inflight queue length when the round began
 *   parse    len      bytes in this line         (per line, inside handle)
 *            entries  words in the lookup table  (sets the hash chain length)
 *
 * Expected: ingest affine in (batch, bytes), one instruction per byte from
 * rep movsb; parse affine in (len, entries); handle affine in q, with the
 * per-line cost of snprintf visible in libc; and, learned from the trace,
 * handle.inflight = cum(ingest.batch) - cum(handle.q), the queue invariant.
 *
 * bin/head is this file built with -DDELTA: parse gets one extra pass over
 * every byte, so its per-byte coefficient rises by exactly that pass.
 *
 * Arguments (K=V, as `drperf run --state` appends them):
 *   reqs=600 batch=16 len=64 cap=12 entries=4096 threads=1
 */
#include <pthread.h>
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

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void)
{
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(rng_state >> 33);
}

/* ------------------------------------------------------- lookup table */

#define WORD 7
#define BUCKETS 256
#define LOOKUPS 4            /* words looked up per line, fixed */

typedef struct node { char word[WORD + 1]; struct node *next; } node_t;
static node_t *buckets[BUCKETS];
static char *vocab;
static long entries, len;

static uint32_t hash_word(const char *w)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < WORD; i++)
        h = (h ^ (unsigned char)w[i]) * 16777619u;
    return h;
}

static void table_init(long n)
{
    entries = n;
    vocab = malloc((size_t)n * (WORD + 1));
    for (long i = 0; i < n; i++) {
        char *w = vocab + i * (WORD + 1);
        for (int j = 0; j < WORD; j++)
            w[j] = (char)('a' + rnd() % 26);
        w[WORD] = '\0';
        node_t *nd = malloc(sizeof(*nd));
        memcpy(nd->word, w, WORD + 1);
        uint32_t b = hash_word(w) % BUCKETS;
        nd->next = buckets[b];
        buckets[b] = nd;
    }
}

/* Not marked: a marker pair costs a few hundred instructions, far more than
 * one chain walk.  Its cost is counted inside `parse`, where it belongs. */
static int lookup_word(const char *w)
{
    for (node_t *nd = buckets[hash_word(w) % BUCKETS]; nd != NULL; nd = nd->next)
        if (memcmp(nd->word, w, WORD) == 0)
            return 1;
    return 0;
}

/* ------------------------------------------------------------ queue */

static char *queue, *scratch;
static long *llen;                /* bytes in each queued line */
static long qcap, qhead, qtail;   /* qtail - qhead = lines in flight */

static void make_line(char *dst, long n)
{
    long pos = 0;
    while (pos + WORD + 1 <= n) {
        memcpy(dst + pos, vocab + (rnd() % entries) * (WORD + 1), WORD);
        dst[pos + WORD] = ' ';
        pos += WORD + 1;
    }
    while (pos < n)
        dst[pos++] = ' ';
}

/* region 1 --------------------------------------------------- ingest */

static long line_len(long i)      /* lines are not all the same length */
{
    static const long mult[4] = { 1, 2, 3, 4 };
    return (len / 4) * mult[i % 4];
}

__attribute__((noinline)) static void ingest(long batch, long bytes)
{
    const char *names[2] = { "batch", "bytes" };
    int64_t vals[2] = { batch, bytes };
    perfmark_begin_v("ingest", 2, names, vals);
    for (long i = 0; i < batch; i++) {
        long n = llen[qtail % qcap];
        make_line(scratch, n);
        char *d = queue + (qtail % qcap) * len;
        const char *s = scratch;
        /* rep movsb: one counted instruction per byte under drperf */
        __asm__ __volatile__("rep movsb" : "+D"(d), "+S"(s), "+c"(n) : : "memory");
        qtail++;
    }
    perfmark_end("ingest");
}

/* region 2 ---------------------------------------------------- parse */

static char *out;
static long out_bytes, out_cap, checksum;
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;

__attribute__((noinline)) static long parse(char *line, long n)
{
    long sum = 0;
    const char *names[2] = { "len", "entries" };
    int64_t vals[2] = { n, entries };
    perfmark_begin_v("parse", 2, names, vals);
#ifdef DELTA
    /* the change under test: sanitise control characters before parsing */
    for (long i = 0; i < n; i++)
        if ((unsigned char)line[i] < 32)
            line[i] = ' ';
#endif
    for (long i = 0; i < n; i++)
        sum += (unsigned char)line[i];
    for (int k = 0; k < LOOKUPS; k++)
        sum += lookup_word(line + (long)k * (WORD + 1) % (n - WORD));
    perfmark_end("parse");
    return sum;
}

static void emit(const char *line, long n, long sum)
{
    char hdr[32];
    int hl = snprintf(hdr, sizeof(hdr), "%06ld ", sum % 1000000);
    pthread_mutex_lock(&out_lock);
    if (out_bytes + hl + n + 1 <= out_cap) {
        memcpy(out + out_bytes, hdr, (size_t)hl);
        memcpy(out + out_bytes + hl, line, (size_t)n);
        out[out_bytes + hl + n] = '\n';
        out_bytes += hl + n + 1;
    }
    checksum += sum;
    pthread_mutex_unlock(&out_lock);
}

/* region 3 --------------------------------------------------- handle */

typedef struct { long first, count; } job_t;

static void *worker(void *p)
{
    job_t *j = p;
    for (long i = 0; i < j->count; i++) {
        long idx = (j->first + i) % qcap;
        char *line = queue + idx * len;
        emit(line, llen[idx], parse(line, llen[idx]));
    }
    return NULL;
}

__attribute__((noinline)) static void handle(long q, long inflight, long threads)
{
    const char *names[2] = { "q", "inflight" };
    int64_t vals[2] = { q, inflight };
    perfmark_begin_v("handle", 2, names, vals);
    if (threads <= 1) {
        job_t j = { qhead, q };
        worker(&j);
    } else {
        pthread_t th[64];
        job_t jobs[64];
        long per = (q + threads - 1) / threads, first = qhead;
        int nt = 0;
        for (long t = 0; t < threads && first < qhead + q; t++, nt++) {
            jobs[nt].first = first;
            jobs[nt].count = (first + per <= qhead + q) ? per : qhead + q - first;
            first += jobs[nt].count;
            pthread_create(&th[nt], NULL, worker, &jobs[nt]);
        }
        for (int t = 0; t < nt; t++)
            pthread_join(th[t], NULL);
    }
    qhead += q;
    out_bytes = 0;              /* the output is drained every round */
    perfmark_end("handle");
}

/* ------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    long reqs = arg(argc, argv, "reqs", 600), batch = arg(argc, argv, "batch", 16);
    long cap = arg(argc, argv, "cap", 12), threads = arg(argc, argv, "threads", 1);
    len = arg(argc, argv, "len", 64);
    if (len < 2 * (WORD + 1)) len = 2 * (WORD + 1);
    if (threads > 64) threads = 64;
    table_init(arg(argc, argv, "entries", 4096));
    qcap = reqs + 4 * batch;
    queue = malloc((size_t)qcap * len);
    llen = malloc((size_t)qcap * sizeof(long));
    for (long i = 0; i < qcap; i++)
        llen[i] = line_len(i);
    scratch = malloc((size_t)len);
    out_cap = (long)(2 * cap + 8) * (len + 40);
    out = malloc((size_t)out_cap);

    long produced = 0, rounds = 0;
    while (produced < reqs || qtail > qhead) {
        if (produced < reqs) {
            long b = batch + (rounds % 3) * (batch / 2);     /* bursty */
            if (b > reqs - produced) b = reqs - produced;
            long bytes = 0;
            for (long i = 0; i < b; i++)
                bytes += llen[(qtail + i) % qcap];
            ingest(b, bytes);
            produced += b;
        }
        long inflight = qtail - qhead;
        long room = cap - (rounds % 4) * (cap / 8);          /* the pop size moves */
        long q = inflight > room ? room : inflight;
        if (q > 0)
            handle(q, inflight, threads);
        rounds++;
    }
    printf("playground: %ld lines of %ld bytes in %ld rounds, %ld entries, %ld threads (checksum %ld)%s\n",
           reqs, len, rounds, entries, threads, checksum,
#ifdef DELTA
           " [head: sanitise pass]"
#else
           " [base]"
#endif
    );
    return 0;
}
