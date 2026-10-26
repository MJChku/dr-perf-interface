// A/B equivalence for GLM53_ENABLE_DECODE_EAGER_CHECKS: run the original and
// eager builds of dc_selected_address and decode_gather_v32 on identical inputs,
// including every validation failure path, and require identical outputs and
// error codes. argv: original.so eager.so
#include "drperf_gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
struct PagedFp8LayerView {
  uint8_t* latent = nullptr; uint8_t* rotary = nullptr; const int32_t* page_table = nullptr; const uint64_t* page_epochs = nullptr;
  uint64_t lease_epoch = 0; int32_t page_tokens = 256, table_pages = 0, physical_pages = 0;
  int32_t logical_length = 0, reserved_tokens = 0, latent_dim = 512, rotary_dim = 64; uint8_t* index = nullptr; int32_t index_dim = 128;
};
static const char* K_ADDR = "_ZN5glm5319dc_selected_addressEPKNS_17PagedFp8LayerViewEPKiS4_S4_PiS5_S5_iiiiib";
static const char* K_GATHER = "_ZN5glm5313flashmla_glue17decode_gather_v32EPKNS_17PagedFp8LayerViewEPKiS5_PhPiPjiiiiii";
int main(int, char** argv) {
  drperf::gpu::Module A(argv[1]), B(argv[2]);
  const int REQ = 4, PT = 16, PPR = 8, PAGES = REQ * PPR, K = 32;
  std::vector<uint8_t> latent(size_t(PAGES) * PT * 512), rotary(size_t(PAGES) * PT * 64);
  for (size_t i = 0; i < latent.size(); ++i) latent[i] = uint8_t(i * 13 + 7);
  for (size_t i = 0; i < rotary.size(); ++i) rotary[i] = uint8_t(i * 5 + 1);
  std::vector<int32_t> table(size_t(REQ) * PPR); std::vector<uint64_t> epochs(PAGES);
  int cases = 0, failures = 0;
  // 0 valid, 1 null latent, 2 wrong latent dim, 3 small reserved, 4 bad table entry,
  // 5 epoch mismatch, 6 table_pages too small, 7 zero lease epoch, 8 page_tokens 0, 9 bad row view
  for (int scenario = 0; scenario < 10; ++scenario) for (int w = 1; w <= 4; ++w) for (int dense = 0; dense <= 1; ++dense) {
    for (int r = 0; r < REQ; ++r) for (int p = 0; p < PPR; ++p) { int ph = (p * REQ + r) % PAGES; table[size_t(r) * PPR + p] = ph; epochs[ph] = 100 + r; }
    std::vector<PagedFp8LayerView> views(REQ);
    for (int r = 0; r < REQ; ++r) { auto& v = views[r]; v.latent = latent.data(); v.rotary = rotary.data(); v.page_table = table.data() + size_t(r) * PPR;
      v.page_epochs = epochs.data(); v.lease_epoch = 100 + r; v.page_tokens = PT; v.table_pages = PPR; v.physical_pages = PAGES;
      v.logical_length = 60 + 7 * r; v.reserved_tokens = 100; }
    auto& t = views[w - 1];
    if (scenario == 1) t.latent = nullptr; if (scenario == 2) t.latent_dim = 576; if (scenario == 3) t.reserved_tokens = 20;
    if (scenario == 4) table[size_t(w - 1) * PPR + 1] = PAGES + 3; if (scenario == 5) epochs[table[size_t(w - 1) * PPR + 2]] = 7;
    if (scenario == 6) t.table_pages = 2; if (scenario == 7) t.lease_epoch = 0; if (scenario == 8) t.page_tokens = 0;
    std::vector<int> rv = {0, 1, 2, 3}; if (scenario == 9) rv[w - 1] = 9;
    std::vector<int> pos(4), lens(4);
    for (int r = 0; r < 4; ++r) { pos[r] = 50 + 3 * r; lens[r] = dense ? 20 + r : 200; }
    std::vector<int> ids0(size_t(w) * K);
    for (int i = 0; i < w * K; ++i) ids0[i] = (i * 37) % 90 - 5;  // negatives, future and in-range
    struct Out { std::vector<int> ids, phys, err, idx; std::vector<uint8_t> kv; std::vector<unsigned> gerr; };
    auto run = [&](drperf::gpu::Module& m) {
      Out o{ids0, std::vector<int>(size_t(w) * K, -9), std::vector<int>(4, 0), std::vector<int>(size_t(w) * K, -9),
            std::vector<uint8_t>(size_t(w) * K * 656, 0xEE), std::vector<unsigned>(4, 0)};
      const PagedFp8LayerView* pv = views.data(); const int *prv = rv.data(), *ppos = pos.data(), *pl = lens.data();
      int *pid = o.ids.data(), *pph = o.phys.data(), *perr = o.err.data(); int rows = w, req = REQ, cnt = K, ld = 512, rd = 64; bool dn = dense;
      void* a[] = {&pv, &prv, &ppos, &pl, &pid, &pph, &perr, &rows, &req, &cnt, &ld, &rd, &dn};
      { drperf::gpu::Region rg("ab.dc_selected_address", {{"rows", w}}); m.launch(K_ADDR, dim3(unsigned((size_t(w) * K / 4 + 255) / 256)), dim3(256), a); }
      const int* gph = o.phys.data(); auto* pkv = o.kv.data(); int* pidx = o.idx.data(); auto* pge = o.gerr.data(); int sel = K, topk = K;
      void* g[] = {&pv, &prv, &gph, &pkv, &pidx, &pge, &rows, &req, &sel, &topk, &ld, &rd};
      { drperf::gpu::Region rg("ab.decode_gather_v32", {{"rows", w}}); m.launch(K_GATHER, dim3(unsigned((size_t(w) * K * 32 + 255) / 256)), dim3(256), g); }
      return o;
    };
    Out a = run(A), b = run(B);
    ++cases;
    if (a.ids != b.ids || a.phys != b.phys || a.err != b.err || a.idx != b.idx || a.kv != b.kv || a.gerr != b.gerr) {
      ++failures; std::fprintf(stderr, "MISMATCH scenario=%d w=%d dense=%d errA=%d,%d,%d,%d errB=%d,%d,%d,%d\n", scenario, w, dense,
                               a.err[0], a.err[1], a.err[2], a.err[3], b.err[0], b.err[1], b.err[2], b.err[3]);
    }
  }
  std::printf("A/B cases %d, mismatches %d\n", cases, failures);
  return failures ? 1 : 0;
}
