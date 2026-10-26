// A/B equivalence for GLM53_ENABLE_DECODE_EAGER_CHECKS in the kv-post body
// (attention_kv_post_mlp_strided; the same body runs inside
// attention_qrank_kv_ik{,_pk}): identical kl/kr/ka outputs, cache bytes and
// error codes across every validation failure path. argv: original.so eager.so
#include "drperf_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
struct PagedFp8LayerView {
  uint8_t* latent = nullptr; uint8_t* rotary = nullptr; const int32_t* page_table = nullptr; const uint64_t* page_epochs = nullptr;
  uint64_t lease_epoch = 0; int32_t page_tokens = 256, table_pages = 0, physical_pages = 0;
  int32_t logical_length = 0, reserved_tokens = 0, latent_dim = 512, rotary_dim = 64; uint8_t* index = nullptr; int32_t index_dim = 128;
};
static const char* K = "_ZN5glm5329attention_kv_post_mlp_stridedEPK13__nv_bfloat16iPfS3_S3_S3_PKfPKiS5_PNS_17PagedFp8LayerViewES7_Piiiif";
static uint16_t bf(float v) { uint32_t u; std::memcpy(&u, &v, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }
int main(int, char** argv) {
  drperf::gpu::Module A(argv[1]), B(argv[2]);
  const int REQ = 4, PT = 16, PPR = 8, PAGES = REQ * PPR, LAT = 512, ROPE = 64, W = LAT + ROPE;
  int cases = 0, failures = 0, faulted = 0;
  for (int scenario = 0; scenario < 12; ++scenario) for (int rows = 1; rows <= 4; ++rows) {
    struct Out { std::vector<float> ka, klr, kl, kr; std::vector<int> err; std::vector<uint8_t> lat, rot; };
    auto run = [&](drperf::gpu::Module& m) {
      Out o{std::vector<float>(size_t(rows) * W, -9), std::vector<float>(size_t(rows) * LAT, -9), std::vector<float>(size_t(rows) * LAT, -9),
            std::vector<float>(size_t(rows) * ROPE, -9), std::vector<int>(REQ, 0),
            std::vector<uint8_t>(size_t(PAGES) * PT * LAT, 0x11), std::vector<uint8_t>(size_t(PAGES) * PT * ROPE, 0x22)};
      std::vector<int32_t> table(size_t(REQ) * PPR); std::vector<uint64_t> epochs(PAGES);
      for (int r = 0; r < REQ; ++r) for (int p = 0; p < PPR; ++p) { int ph = (p * REQ + r) % PAGES; table[size_t(r) * PPR + p] = ph; epochs[ph] = 100 + r; }
      std::vector<PagedFp8LayerView> views(REQ);
      for (int r = 0; r < REQ; ++r) { auto& v = views[r]; v.latent = o.lat.data(); v.rotary = o.rot.data(); v.page_table = table.data() + size_t(r) * PPR;
        v.page_epochs = epochs.data(); v.lease_epoch = 100 + r; v.page_tokens = PT; v.table_pages = PPR; v.physical_pages = PAGES;
        v.logical_length = 60; v.reserved_tokens = 100; }
      const int t = rows - 1; auto& v = views[t];
      std::vector<int> pos(4); for (int r = 0; r < 4; ++r) pos[r] = 40 + 9 * r;
      std::vector<int> rv = {0, 1, 2, 3};
      switch (scenario) {
        case 1: v.latent = nullptr; break; case 2: v.latent_dim = 576; break; case 3: v.rotary = nullptr; break;
        case 4: v.rotary_dim = 32; break; case 5: pos[t] = 150; break; case 6: pos[t] = -3; break;
        case 7: table[size_t(t) * PPR + pos[t] / PT] = PAGES + 2; break; case 8: epochs[table[size_t(t) * PPR + pos[t] / PT]] = 5; break;
        case 9: v.table_pages = 1; break; case 10: v.lease_epoch = 0; break; case 11: v.page_tokens = 0; break; }
      std::vector<uint16_t> kab(size_t(rows) * W); for (size_t i = 0; i < kab.size(); ++i) kab[i] = bf(std::sin(0.3f * float(i)) * 2.f);
      std::vector<float> klw(LAT, 1.f), freq(32); for (int i = 0; i < 32; ++i) freq[i] = 1.f / float(1 << (i % 10));
      const uint16_t* pk = kab.data(); int stride = W; float *pka = o.ka.data(), *pklr = o.klr.data(), *pkl = o.kl.data(), *pkr = o.kr.data();
      const float* pw = klw.data(); const int* pp = pos.data(); const float* pf = freq.data(); PagedFp8LayerView* pv = views.data();
      const int* prv = rv.data(); int* pe = o.err.data(); int tokens = rows, lat = LAT, rope = ROPE; float eps = 1e-6f;
      void* a[] = {&pk, &stride, &pka, &pklr, &pkl, &pkr, &pw, &pp, &pf, &pv, &prv, &pe, &tokens, &lat, &rope, &eps};
      { drperf::gpu::Region rg("ab.kv_post", {{"rows", rows}}); m.launch(K, dim3(rows), dim3(128), a); }
      return o;
    };
    Out a = run(A), b = run(B); ++cases;
    for (int e : a.err) faulted += e != 0;
    if (a.ka != b.ka || a.klr != b.klr || a.kl != b.kl || a.kr != b.kr || a.err != b.err || a.lat != b.lat || a.rot != b.rot) {
      ++failures; std::fprintf(stderr, "MISMATCH scenario=%d rows=%d err %d/%d %d/%d %d/%d %d/%d\n", scenario, rows,
                               a.err[0], b.err[0], a.err[1], b.err[1], a.err[2], b.err[2], a.err[3], b.err[3]); }
  }
  std::printf("kv-post A/B cases %d, mismatches %d, nonzero error cells %d\n", cases, failures, faulted);
  return failures ? 1 : 0;
}
