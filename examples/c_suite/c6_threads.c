/* Case 6a, all-thread attribution.
 * parallel: 4 workers each run work(n/4) while the main thread's region is open.
 * Expected: total per call = 4*n + thread start/join overhead (constant); derive A = 4. */
#include <pthread.h>
#include "suite.h"
DEFINE_LOOP(work, "add")

struct arg { long n; uint64_t out; };
static void *worker(void *p) { struct arg *a = p; a->out = work(a->n); return NULL; }

static void run(long n, int T, const char *region)
{
    pthread_t th[8]; struct arg args[8]; int i;
    perfmark_begin(region, "n", n);
    for (i = 0; i < T; i++) { args[i].n = n / T; pthread_create(&th[i], NULL, worker, &args[i]); }
    for (i = 0; i < T; i++) { pthread_join(th[i], NULL); sink += args[i].out; }
    perfmark_end(region);
}

int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 4000);
    run(400, 4, "warmup");
    run(n, 4, "parallel");
    printf("c6 n=%ld sink=%llu\n", n, (unsigned long long)sink);
    return 0;
}
