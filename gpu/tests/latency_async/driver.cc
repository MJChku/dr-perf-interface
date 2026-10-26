#include "drperf_gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
int main(int argc, char** argv) {
  drperf::gpu::Module m(argv[1]);
  if (argc > 2 && !std::strcmp(argv[2], "spin")) {
    drperf::gpu::Buffer<uint64_t> chain(4, 0), out(2, 0);
    chain[0] = uint64_t(&chain[1]); chain[1] = uint64_t(&chain[2]); chain[2] = 42; chain[3] = 7;
    uint64_t* cp = chain.data(); uint64_t* op = out.data(); void* a[] = {&cp, &op};
    { drperf::gpu::Region r("spin"); m.launch("spin", dim3(1), dim3(2), a); }
    if (out[0] != 7 || out[1] != 42) { std::fprintf(stderr, "bad result %llu %llu\n", (unsigned long long)out[0], (unsigned long long)out[1]); return 1; }
  }
  return 0;
}
