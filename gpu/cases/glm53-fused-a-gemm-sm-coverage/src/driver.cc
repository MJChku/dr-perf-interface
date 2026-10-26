#include "drperf_gpu.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
static std::string kernel(const char* ptx, const std::string& want) {
  std::ifstream f(ptx); std::string s;
  while (std::getline(f, s)) { auto p = s.find(".visible .entry "); if (p != std::string::npos && s.find(want) != std::string::npos) { s = s.substr(p + 16); return s.substr(0, s.find('(')); } }
  throw std::runtime_error("kernel " + want);
}
static uint16_t bf(float x) { uint32_t u; std::memcpy(&u, &x, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static float fb(uint16_t h) { uint32_t u = uint32_t(h) << 16; float x; std::memcpy(&x, &u, 4); return x; }
int main(int, char** argv) {
  drperf::gpu::Module mod(argv[1]);
  struct S { int M, K, TM; };
  for (S s : {S{64,512,16}, S{64,1024,16}, S{64,2048,16}, S{128,512,16}, S{128,1024,16}, S{128,2048,16}, S{256,512,16}, S{256,1024,16}, S{64,512,32}, S{128,1024,32}, S{256,2048,32}})
    for (int t = 1; t <= 4; ++t) {
      const int stages = std::min(s.K / 256, 1024 * 192 / ((s.TM + 8) * 256 * 2));
      const int barrier = (stages * 16 + 1023) / 1024 * 1024;
      const int smem = ((s.TM * 2 + 8 * 2) * 256 * stages + barrier + 1023) / 1024 * 1024;
      // mangled: fused_a_gemm_kernel<1, M, K, TM, 8, 256, stages>
      char want[128]; std::snprintf(want, sizeof want, "fused_a_gemm_kernelILi1ELi%dELi%dELi%dELi8ELi256ELi%dE", s.M, s.K, s.TM, stages);
      auto name = kernel(argv[2], want);
      drperf::gpu::Buffer<uint16_t> w(size_t(s.M) * s.K), x(size_t(t) * s.K), out(size_t(t) * s.M + 8, 0xdead);
      for (size_t i = 0; i < w.size(); ++i) w[i] = bf(std::sin(0.01f * float(i % 997)));
      for (size_t i = 0; i < x.size(); ++i) x[i] = bf(std::cos(0.02f * float(i % 991)));
      // kernel(output, mat_a = weight [M,K], mat_b = input [t,K], gemm_n = t, row stride K)
      uint16_t* op = out.data(); const uint16_t* wp = w.data(); const uint16_t* xp = x.data(); int n = t, stride = s.K;
      void* args[] = {&op, &wp, &xp, &n, &stride};
      { drperf::gpu::Region r("fused_a_gemm", {{"M", s.M}, {"K", s.K}, {"tile_m", s.TM}, {"tokens", t}, {"MK", s.M * s.K}, {"ctas", s.M / s.TM}, {"stages", stages}, {"tK", t * s.K}, {"tM", t * s.M}, {"Kctas", s.K * (s.M / s.TM)}});
        mod.launch(name.c_str(), dim3(s.M / s.TM, 1), dim3(256), args, smem); }
      for (int i = 0; i < t; ++i) for (int j = 0; j < s.M; ++j) {
        double ref = 0; for (int k = 0; k < s.K; ++k) ref += double(fb(w[size_t(j) * s.K + k])) * fb(x[size_t(i) * s.K + k]);
        double got = fb(out[size_t(i) * s.M + j]);
        if (std::fabs(got - ref) > 0.02 * std::fabs(ref) + 0.05) { std::fprintf(stderr, "mismatch M=%d K=%d t=%d (%d,%d) %g vs %g\n", s.M, s.K, t, i, j, got, ref); return 1; }
      }
      if (out[size_t(t) * s.M] != 0xdead) { std::fprintf(stderr, "guard\n"); return 1; }
    }
  std::puts("fused_a_gemm small shapes PASS");
  return 0;
}
