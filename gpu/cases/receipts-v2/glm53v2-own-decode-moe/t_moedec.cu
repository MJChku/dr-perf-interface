// gx check: decode NVFP4 MoE (routing + gemm1/SwiGLU/requant + gemm2/finalize) vs a CPU reference of the same
// semantics (moe_decode.h). Random fp4 weights/activations and e4m3 scales; each case runs twice (the kernels must
// leave their reduction slots zeroed).
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "../common/cuda_utils.h"
#include "../kernels/moe_decode.h"

namespace {
uint32_t rng = 12345;
uint32_t nxt() { return rng = rng * 1664525u + 1013904223u; }
float ur() { return (nxt() >> 8) * (1.0f / 16777216.0f); }

const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
float fp4(uint8_t n) { return (n & 8 ? -1.f : 1.f) * kE2M1[n & 7]; }
float e4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(m / 8.f, -6);
  return s ? -v : v;
}
uint8_t to_e4m3(float v) {  // round to nearest even, saturate (v >= 0)
  int best = 0;
  float bd = 1e30f;
  for (int b = 0; b < 127; ++b) {  // 0x7f is NaN
    const float d = std::fabs(e4m3(uint8_t(b)) - v);
    if (d < bd || (d == bd && (b & 1) == 0)) bd = d, best = b;
  }
  return uint8_t(best);
}
float rn_e2m1(float v) {
  const float a = std::fabs(v);
  int best = 0;
  float bd = 1e30f;
  for (int i = 0; i < 8; ++i) {
    const float d = std::fabs(kE2M1[i] - a);
    if (d < bd || (d == bd && (i & 1) == 0)) bd = d, best = i;
  }
  return v < 0 ? -kE2M1[best] : kE2M1[best];
}
template <class T>
T* up(const std::vector<T>& h) {
  T* d;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}
}  // namespace

int main() {
  moedec::Shape sh;
  sh.H = 1024, sh.I = 256, sh.experts = 256, sh.local_offset = 64, sh.local_num = 64, sh.top_k = 8;
  const int H = sh.H, I = sh.I, E = sh.local_num, NE = sh.experts;
  auto rnd_fp4 = [](size_t n) {
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = uint8_t(nxt() >> 24);
    return v;
  };
  auto rnd_sf = [](size_t n) {  // e4m3 in ~[2^-4, 2^-1]
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = uint8_t(0x18 + (nxt() >> 27) % 24);
    return v;
  };
  auto w13 = rnd_fp4(size_t(E) * 2 * I * H / 2), w13s = rnd_sf(size_t(E) * 2 * I * H / 16);
  auto w2 = rnd_fp4(size_t(E) * H * I / 2), w2s = rnd_sf(size_t(E) * H * I / 16);
  std::vector<float> bias(NE), g1a(E), g1c(E), g2a(E);
  for (auto& x : bias) x = (ur() - 0.5f) * 0.2f;
  for (int e = 0; e < E; ++e) g1a[e] = 0.5f + ur(), g1c[e] = 0.5f + ur(), g2a[e] = 0.5f + ur();
  uint8_t *dw13 = up(w13), *dw13s = up(w13s), *dw2 = up(w2), *dw2s = up(w2s);
  float *dbias = up(bias), *dg1a = up(g1a), *dg1c = up(g1c), *dg2a = up(g2a);
  moedec::Work wk;
  moedec::init(wk, sh, 0);
  int fails = 0;
  for (int T : {1, 3, 4, 8}) {
    auto x4 = rnd_fp4(size_t(T) * H / 2), xs = rnd_sf(size_t(T) * H / 16);
    std::vector<float> logits(size_t(T) * NE);
    for (auto& l : logits) l = (ur() - 0.5f) * 4.f;
    uint8_t *dx4 = up(x4), *dxs = up(xs);
    float* dl = up(logits);
    __nv_bfloat16* dout;
    CUDA_CHECK(cudaMalloc(&dout, size_t(T) * H * 2));
    // CPU reference
    std::vector<double> ref(size_t(T) * H, 0.0);
    int pairs = 0;
    for (int t = 0; t < T; ++t) {
      std::vector<float> sc(NE), ch(NE);
      for (int e = 0; e < NE; ++e) sc[e] = 1.f / (1.f + std::exp(-logits[t * NE + e])), ch[e] = sc[e] + bias[e];
      std::vector<int> sel;
      float sum = 0.f;
      for (int k = 0; k < sh.top_k; ++k) {
        int bi = -1;
        for (int e = 0; e < NE; ++e)
          if (std::find(sel.begin(), sel.end(), e) == sel.end() && (bi < 0 || ch[e] > ch[bi])) bi = e;
        sel.push_back(bi);
        sum += sc[bi];
      }
      std::vector<float> x(H);
      for (int k = 0; k < H; ++k) x[k] = fp4((x4[t * H / 2 + k / 2] >> (4 * (k & 1))) & 15) * e4m3(xs[t * H / 16 + k / 16]);
      for (int e0 : sel) {
        const int e = e0 - sh.local_offset;
        if (e < 0 || e >= E) continue;
        ++pairs;
        const float wt = sc[e0] / (sum + 1e-20f) * sh.routed_scaling;
        std::vector<float> act(I);
        auto wv = [&](int row, int k) {
          const size_t r = size_t(e) * 2 * I + row;
          return fp4((w13[r * H / 2 + k / 2] >> (4 * (k & 1))) & 15) * e4m3(w13s[r * H / 16 + k / 16]);
        };
        for (int j = 0; j < I; ++j) {
          double gs = 0, us = 0;
          for (int k = 0; k < H; ++k) gs += double(x[k]) * wv(j, k), us += double(x[k]) * wv(I + j, k);
          const float gv = float(gs) * g1a[e], uv = float(us) * g1c[e];
          act[j] = gv / (1.f + std::exp(-gv)) * uv;
        }
        for (int b = 0; b < I / 16; ++b) {
          float amax = 0.f;
          for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(act[b * 16 + j]));
          const float sfv = e4m3(to_e4m3(amax * (1.0f / 6.0f)));
          const float os = sfv == 0.f ? 0.f : 1.0f / sfv;
          for (int j = 0; j < 16; ++j) act[b * 16 + j] = rn_e2m1(act[b * 16 + j] * os) * sfv;
        }
        for (int h = 0; h < H; ++h) {
          const size_t r = size_t(e) * H + h;
          double acc = 0;
          for (int k = 0; k < I; ++k)
            acc += double(act[k]) * fp4((w2[r * I / 2 + k / 2] >> (4 * (k & 1))) & 15) * e4m3(w2s[r * I / 16 + k / 16]);
          ref[size_t(t) * H + h] += double(wt) * g2a[e] * acc;
        }
      }
    }
    for (int rep = 0; rep < 2; ++rep) {
      moedec::run(sh, wk, T, dl, dbias, dx4, dxs, dw13, dw13s, dw2, dw2s, dg1a, dg1c, dg2a, dout, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<__nv_bfloat16> o(size_t(T) * H);
      CUDA_CHECK(cudaMemcpy(o.data(), dout, o.size() * 2, cudaMemcpyDeviceToHost));
      double rms = 0;
      for (double v : ref) rms += v * v;
      rms = std::sqrt(rms / ref.size()) + 1e-30;
      int bad = 0;
      double maxrel = 0;
      for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(double(__bfloat162float(o[i])) - ref[i]);
        maxrel = std::max(maxrel, d / rms);
        bad += d > 0.02 * rms + 0.008 * std::fabs(ref[i]);
      }
      const bool ok = bad <= int(ref.size() / 200);
      fails += !ok;
      printf("%s moedec T=%d rep=%d local_pairs=%d mismatches=%d/%zu max|d|/rms=%.4f rms=%.4g\n", ok ? "PASS" : "FAIL", T,
             rep, pairs, bad, ref.size(), maxrel, rms);
    }
  }
  return fails ? 1 : 0;
}
