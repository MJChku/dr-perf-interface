#include "drperf_gpu.h"
#include <cstdio>
int main(int, char** argv) {
  drperf::gpu::Module m(argv[1]);
  drperf::gpu::Buffer<float> out(1, 0.0f);
  float* op = out.data(); void* a[] = {&op};
  { drperf::gpu::Region r("big_local"); m.launch("big_local", dim3(2), dim3(32), a); }
  if (out[0] != 3.0f) { std::fprintf(stderr, "bad result %g\n", out[0]); return 1; }
  return 0;
}
