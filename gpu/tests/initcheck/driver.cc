#include "drperf_gpu.h"
#include <cstdio>
int main(int argc, char** argv) {
  drperf::gpu::Module m(argv[1]);
  for (int n : {64, 128, 256}) for (int written : {0, n / 2, n}) {
    auto in = drperf::gpu::Buffer<float>::uninitialized(n, "in");
    for (int i = 0; i < n; ++i) in[i] = float(i);
    in.host_written();
    auto scratch = drperf::gpu::Buffer<float>::uninitialized(n, "scratch");
    auto out = drperf::gpu::Buffer<float>::uninitialized(n, "out");
    float* sp = scratch.data(); float* ip = in.data(); float* op = out.data();
    void* pa[] = {&sp, &written};
    void* ca[] = {&ip, &sp, &op, &n};
    { drperf::gpu::Region r("partial_write", {{"n", n}, {"written", written}}); m.launch("partial_write", dim3((n + 63) / 64), dim3(64), pa); }
    { drperf::gpu::Region r("consume", {{"n", n}, {"written", written}}); m.launch("consume", dim3((n + 63) / 64), dim3(64), ca); }
  }
  return 0;
}
