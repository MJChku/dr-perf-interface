// Case 6: OpenMP parallel for over n iterations under the main thread's
// region (each iteration: loop3(200)).  Expected: all-thread work affine in n
// with the same total at 4 and 8 threads; libgomp spin-waiting excluded and
// reported as waiting.
#include "suite.h"
#include <omp.h>
#include <vector>
int main(int argc, char **argv)
{
    long n = arg_value(argc, argv, "n", 1000);
    std::vector<uint64_t> part(omp_get_max_threads(), 0);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < 64; i++) part[omp_get_thread_num()] += loop3(8);      // warm-up: start the team
    for (int rep = 0; rep < 2; rep++) {
        perfmark_begin("omp_for", "n", n);
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < n; i++)
            part[omp_get_thread_num()] += loop3(200);
        perfmark_end("omp_for");
    }
    for (auto x : part) g_sink += x;
    printf("c6 n=%ld threads=%d sink=%llu\n", n, omp_get_max_threads(), (unsigned long long)g_sink);
    return 0;
}
