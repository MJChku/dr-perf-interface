// moe_front_prenorm alone (NVFP4 quantize + 256-expert router on the BF16
// pre-normalized rows) at production width, rows 1..4, for latency attribution.
#include "drperf_gpu.h"
#include <cstdint>
#include <cstring>
#include <vector>
static const char* K_FRONT = "_ZN5glm5317moe_front_prenormEPK13__nv_bfloat16PKfPhS5_Pfiiiif";
static uint16_t bf(float v) { uint32_t u; std::memcpy(&u, &v, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }
int main(int, char** argv) {
  drperf::gpu::Module mod(argv[1]);
  const int H = 6144, E = 256;
  std::vector<float> router(size_t(E) * H); for (size_t i = 0; i < router.size(); ++i) router[i] = float(int(i % 31) - 15) / 1024.f;
  for (int w = 1; w <= 4; ++w) {
    std::vector<uint16_t> xb(size_t(w) * H); for (size_t i = 0; i < xb.size(); ++i) xb[i] = bf(float(int(i % 17) - 8) * 0.1f);
    std::vector<uint8_t> packed(size_t(w) * H / 2), scales(size_t(w) * H / 16); std::vector<float> logits(size_t(w) * E);
    auto* px = xb.data(); const float* pr = router.data(); auto *pp = packed.data(), *ps = scales.data(); float* pl = logits.data();
    int rows = w, width = H, ex = E, sl = 8; float ts = 1.f / 448.f;
    void* a[] = {&px, &pr, &pp, &ps, &pl, &rows, &width, &ex, &sl, &ts};
    drperf::gpu::Region rg("moe_front_prenorm", {{"rows", w}}); mod.launch(K_FRONT, dim3(w * 8 + E), dim3(128), a, H * 4);
  }
  return 0;
}
