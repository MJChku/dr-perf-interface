// A/B for GLM53_ENABLE_DECODE_PRENORM_BATCHED_LOADS: moe_front_prenorm (NVFP4
// quant + 256-expert router) and attention_iw_prenorm, rows 1..4, bitwise.
#include "drperf_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
static const char* K_FRONT = "_ZN5glm5317moe_front_prenormEPK13__nv_bfloat16PKfPhS5_Pfiiiif";
static const char* K_IW = "_ZN5glm5320attention_iw_prenormEPK13__nv_bfloat16PKfPfiii";
static uint16_t bf(float v) { uint32_t u; std::memcpy(&u, &v, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }
int main(int, char** argv) {
  drperf::gpu::Module A(argv[1]), B(argv[2]);
  const int H = 6144, E = 256, IH = 32; int bad = 0, cases = 0;
  std::vector<float> router(size_t(E) * H), iwm(size_t(IH) * H);
  for (size_t i = 0; i < router.size(); ++i) router[i] = std::sin(0.013f * float(i % 7919)) / 64.f;
  for (size_t i = 0; i < iwm.size(); ++i) iwm[i] = std::cos(0.017f * float(i % 6007)) / 64.f;
  for (int w = 1; w <= 4; ++w) {
    std::vector<uint16_t> xb(size_t(w) * H); for (size_t i = 0; i < xb.size(); ++i) xb[i] = bf(std::sin(0.37f * float(i)) * 1.5f);
    auto front = [&](drperf::gpu::Module& m, std::vector<uint8_t>& packed, std::vector<uint8_t>& scales, std::vector<float>& logits) {
      packed.assign(size_t(w) * H / 2, 0); scales.assign(size_t(w) * H / 16, 0); logits.assign(size_t(w) * E, -1);
      auto* px = xb.data(); const float* pr = router.data(); auto *pp = packed.data(), *ps = scales.data(); float* pl = logits.data();
      int rows = w, width = H, ex = E, sl = 8; float ts = 1.f / 448.f; void* a[] = {&px, &pr, &pp, &ps, &pl, &rows, &width, &ex, &sl, &ts};
      drperf::gpu::Region rg("ab.moe_front_prenorm", {{"rows", w}}); m.launch(K_FRONT, dim3(w * 8 + E), dim3(128), a, H * 4); };
    auto iw = [&](drperf::gpu::Module& m, std::vector<float>& out) {
      out.assign(size_t(w) * IH, -1); auto* px = xb.data(); const float* pw = iwm.data(); float* po = out.data(); int rows = w, heads = IH, width = H;
      void* a[] = {&px, &pw, &po, &rows, &heads, &width};
      drperf::gpu::Region rg("ab.attention_iw_prenorm", {{"rows", w}}); m.launch(K_IW, dim3(IH), dim3(128), a, H * 4); };
    std::vector<uint8_t> pa, sa, pb, sb; std::vector<float> la, lb, ia, ib;
    front(A, pa, sa, la); front(B, pb, sb, lb); iw(A, ia); iw(B, ib); cases += 2;
    if (pa != pb || sa != sb || std::memcmp(la.data(), lb.data(), la.size() * 4)) { ++bad; std::fprintf(stderr, "front mismatch w=%d\n", w); }
    if (std::memcmp(ia.data(), ib.data(), ia.size() * 4)) { ++bad; std::fprintf(stderr, "iw mismatch w=%d\n", w); }
  }
  std::printf("prenorm A/B cases %d, mismatches %d\n", cases, bad);
  return bad ? 1 : 0;
}
