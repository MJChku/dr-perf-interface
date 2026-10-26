#include "drperf_gpu.h"
#include <cstdio>
int main(int, char** argv) {
  drperf::gpu::Module m(argv[1]);
  drperf::gpu::Buffer<float> in(4, 2.0f), out(1, 0.0f);
  float* ip = in.data(); float* op = out.data(); unsigned n = 9; void* a[] = {&ip, &op, &n};
  { drperf::gpu::Region r("guarded"); m.launch("guarded", dim3(1), dim3(1), a); }
  if (out[0] != 4.0f) { std::fprintf(stderr, "bad result %g\n", out[0]); return 1; }
  return 0;
}
