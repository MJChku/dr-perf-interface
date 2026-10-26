// A/B for GLM53_ENABLE_DECODE_EAGER_CHECKS in lib_indexer decode_tables_kernel:
// block tables, context lengths and error codes identical across faults.
// argv: original.so original.ptx eager.so eager.ptx
#include "drperf_gpu.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
struct PagedFp8LayerView {
  uint8_t* latent = nullptr; uint8_t* rotary = nullptr; const int32_t* page_table = nullptr; const uint64_t* page_epochs = nullptr;
  uint64_t lease_epoch = 0; int32_t page_tokens = 256, table_pages = 0, physical_pages = 0;
  int32_t logical_length = 0, reserved_tokens = 0, latent_dim = 512, rotary_dim = 64; uint8_t* index = nullptr; int32_t index_dim = 128;
};
static std::string find(const char* ptx) { std::ifstream f(ptx); std::string s; while (std::getline(f, s)) { auto p = s.find(".entry "); if (p != std::string::npos && s.find("decode_tables_kernel") != std::string::npos) { s = s.substr(p + 7); return s.substr(0, s.find('(')); } } return ""; }
int main(int, char** argv) {
  drperf::gpu::Module A(argv[1]), B(argv[3]); const std::string ka = find(argv[2]), kb = find(argv[4]);
  const int REQ = 4, PPR = 8, PAGES = REQ * PPR, MB = 16; int cases = 0, bad = 0, faulted = 0;
  std::vector<uint8_t> index(16);
  for (int scenario = 0; scenario < 9; ++scenario) for (int rows = 1; rows <= 4; ++rows) {
    struct Out { std::vector<int> bt, cl, err; };
    auto run = [&](drperf::gpu::Module& m, const std::string& k) {
      std::vector<int32_t> table(size_t(REQ) * PPR); std::vector<uint64_t> epochs(PAGES);
      for (int r = 0; r < REQ; ++r) for (int p = 0; p < PPR; ++p) { int ph = (p * REQ + r) % PAGES; table[size_t(r) * PPR + p] = ph; epochs[ph] = 100 + r; }
      std::vector<PagedFp8LayerView> views(REQ);
      for (int r = 0; r < REQ; ++r) { auto& v = views[r]; v.index = index.data(); v.page_table = table.data() + size_t(r) * PPR; v.page_epochs = epochs.data();
        v.lease_epoch = 100 + r; v.table_pages = PPR; v.physical_pages = PAGES; }
      auto& v = views[rows - 1];
      std::vector<int> pos = {700, 1200, 1900, 300}, rv = {0, 1, 2, 3};
      switch (scenario) { case 1: v.index = nullptr; break; case 2: v.index_dim = 64; break; case 3: v.page_tokens = 128; break;
        case 4: table[size_t(rows - 1) * PPR + 2] = PAGES + 1; break; case 5: epochs[table[size_t(rows - 1) * PPR + 1]] = 3; break;
        case 6: v.table_pages = 2; break; case 7: v.lease_epoch = 0; break; case 8: v.physical_pages = 4; break; }
      Out o{std::vector<int>(size_t(rows) * MB, -9), std::vector<int>(rows, -9), std::vector<int>(REQ, 0)};
      const PagedFp8LayerView* pv = views.data(); const int *prv = rv.data(), *pp = pos.data(); int r = rows, mb = MB; int *pbt = o.bt.data(), *pcl = o.cl.data(), *pe = o.err.data();
      void* a[] = {&pv, &prv, &pp, &r, &mb, &pbt, &pcl, &pe};
      drperf::gpu::Region rg("ab.decode_tables", {{"rows", rows}}); m.launch(k.c_str(), dim3(rows), dim3(256), a); return o; };
    Out a = run(A, ka), b = run(B, kb); ++cases; for (int e : a.err) faulted += e != 0;
    if (a.bt != b.bt || a.cl != b.cl || a.err != b.err) { ++bad; std::fprintf(stderr, "MISMATCH scenario=%d rows=%d\n", scenario, rows); }
  }
  std::printf("tables A/B cases %d, mismatches %d, nonzero error cells %d\n", cases, bad, faulted);
  return bad ? 1 : 0;
}
