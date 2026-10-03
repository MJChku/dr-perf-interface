
#include "ditto_runtime.h"
#include "perfmark.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>
#include <vector>
int main(int argc, char **argv) {
    assert(argc == 2);
    void* lib = dlopen(argv[1], RTLD_NOW); assert(lib);
    auto create = (decltype(&ditto_context_create))dlsym(lib, "ditto_context_create");
    auto destroy = (decltype(&ditto_context_destroy))dlsym(lib, "ditto_context_destroy");
    auto prefetch = (decltype(&ditto_candidates_prefetch))dlsym(lib, "ditto_candidates_prefetch");
    auto get = (decltype(&ditto_candidates_get))dlsym(lib, "ditto_candidates_get");
    perfmark_begin("capture", "n", 1); perfmark_end("capture");
    ditto_context_options options{2, 2}; ditto_report report{};
    auto context = create(&options, &report); assert(context);
    uint64_t checksum = 1469598103934665603ull;
    // Fifth batch revisits an evicted key: the event must get a fresh identity.
    for (int n : {8, 64, 256, 1024, 8}) {
        std::vector<int32_t> tokens(n), result(n * 7);
        for (int i = 0; i < n; ++i) tokens[i] = i % 17;
        perfmark_begin("candidates.prefetch", "tokens", n);
        assert(prefetch(context, tokens.data(), n, 3, &report) == DITTO_OK);
        perfmark_end("candidates.prefetch");
        // Baseline deliberately allows publication before get: a premature
        // waited marker looks valid until the automatic delay probe runs.
        usleep(200000);
        for (int cached : {0, 1}) {
            const char *names[] = {"tokens", "cached"}; int64_t values[] = {n, cached};
            perfmark_begin_v("candidates.get", 2, names, values);
            assert(get(context, tokens.data(), n, 3, result.data(), result.size(), &report) == DITTO_OK);
            perfmark_end("candidates.get");
            for (int32_t value : result) { checksum ^= (uint32_t)value; checksum *= 1099511628211ull; }
        }
    }
    destroy(context);
    printf("checksum=%llu\n", (unsigned long long)checksum);
}
