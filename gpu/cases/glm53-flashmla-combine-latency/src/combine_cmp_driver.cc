// drperf comparison: original combine (64 padded heads, MAX_SPLITS 64) vs wide
// combine (32 valid heads, all split loads in flight). Bitwise output check on
// the 32 consumed heads; PCVs w (requests) and splits per request.
#include "drperf_gpu.h"
#include "flash_params.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
static std::string kernel(const char* ptx, const std::string& part) {
  std::ifstream f(ptx); std::string s;
  while (std::getline(f, s)) { auto p = s.find(".visible .entry "); if (p != std::string::npos && s.find(part) != std::string::npos) { s = s.substr(p + 16); return s.substr(0, s.find('(')); } }
  throw std::runtime_error("kernel " + part);
}
int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: combine_cmp.so combine_cmp.ptx\n"); return 1; }
  drperf::gpu::Module m(argv[1]);
  auto orig = kernel(argv[2], "flash_fwd_mla_combine_kernelIN7cutlass10bfloat16_tELi512ELi8ELi64ELi256");
  auto wide = kernel(argv[2], "combine_wide_kernel");
  const int H = 64, DV = 512, VALID = 32;
  int checked = 0;
  auto meta = kernel(argv[2], "get_mla_metadata_kernel");
  // Real schedules: production (w, parts, overhead) at topk 2048, plus the
  // uniform synthetic split counts.
  struct Case { int w, parts, overhead, sp; };
  std::vector<Case> cases;
  for (Case c : {Case{1,16,5,0}, Case{2,24,1,0}, Case{3,48,5,0}, Case{4,48,1,0}}) cases.push_back(c);
  for (int w = 1; w <= 4; ++w) for (int sp : {2, 3, 5, 8, 11}) cases.push_back({w, 0, 0, sp});
  for (const Case& cs : cases) {
    const int w = cs.w;
    drperf::gpu::Buffer<int> splits(w + 1);
    if (cs.parts) {
      std::vector<DecodingSchedMeta> sched(cs.parts);
      GetDecodeSchedMetaParams mp{}; mp.b = w; mp.s_q = 1; mp.block_size_n = 64; mp.fixed_overhead_num_blocks = cs.overhead;
      mp.topk = 2048; mp.tile_scheduler_metadata_ptr = sched.data(); mp.num_splits_ptr = splits.data(); mp.num_sm_parts = cs.parts;
      void* ma[] = {&mp};
      { drperf::gpu::Region r("combine.schedule", {{"w", w}, {"parts", cs.parts}}); m.launch(meta.c_str(), dim3(1), dim3(32), ma, 4 * (5 * w + 1)); }
    } else for (int r = 0; r <= w; ++r) splits[r] = r * cs.sp;
    int sp = 0; for (int r = 0; r < w; ++r) sp = std::max(sp, splits[r + 1] - splits[r]);
    const int total = splits[w];
    std::printf("case w=%d parts=%d splits:", w, cs.parts); for (int r = 0; r <= w; ++r) std::printf(" %d", splits[r]); std::printf("\n");
    drperf::gpu::Buffer<float> oacc(size_t(total) * H * DV), lacc(size_t(total) * H);
    for (size_t i = 0; i < oacc.size(); ++i) oacc[i] = std::sin(0.37f * float(i % 9973)) * 3.0f;
    for (size_t i = 0; i < lacc.size(); ++i) lacc[i] = std::cos(0.11f * float(i)) * 4.0f;
    drperf::gpu::Buffer<uint16_t> out_a(size_t(w) * H * DV, 0xdead), out_b(size_t(w) * H * DV, 0xbeef);
    drperf::gpu::Buffer<float> lse_a(size_t(w) * H, -7), lse_b(size_t(w) * H, -7);
    auto make = [&](int h_q, void* out, float* lse, int parts) {
      CombineParams c{};
      c.b = w; c.s_q = 1; c.h_q = h_q; c.d_v = DV; c.lse = lse; c.out = out;
      c.stride_lse_b = H; c.stride_lse_s_q = H; c.stride_o_b = H * DV; c.stride_o_s_q = H * DV; c.stride_o_h_q = DV;
      c.lse_accum = lacc.data(); c.o_accum = oacc.data();
      c.stride_lse_accum_split = H; c.stride_lse_accum_s_q = H;
      c.stride_o_accum_split = H * DV; c.stride_o_accum_s_q = H * DV; c.stride_o_accum_h_q = DV;
      c.num_splits_ptr = splits.data(); c.num_sm_parts = parts; c.attn_sink = nullptr;
      return c;
    };
    CombineParams a = make(H, out_a.data(), lse_a.data(), 48), b = make(VALID, out_b.data(), lse_b.data(), sp);
    void* aa[] = {&a}; void* ba[] = {&b};
    { drperf::gpu::Region r("combine.original_h64_m64", {{"w", w}, {"splits", sp}, {"wsplits", total}});
      m.launch(orig.c_str(), dim3(w, 1, H / 8), dim3(256), aa); }
    { drperf::gpu::Region r("combine.wide_h32", {{"w", w}, {"splits", sp}, {"wsplits", total}});
      m.launch(wide.c_str(), dim3(w, 1, VALID / 2), dim3(256), ba); }
    for (int r = 0; r < w; ++r) for (int h = 0; h < VALID; ++h) {
      if (std::memcmp(&out_a[(size_t(r) * H + h) * DV], &out_b[(size_t(r) * H + h) * DV], DV * 2) ||
          std::memcmp(&lse_a[r * H + h], &lse_b[r * H + h], 4)) { std::fprintf(stderr, "MISMATCH w=%d sp=%d r=%d h=%d\n", w, sp, r, h); return 1; }
      ++checked;
    }
  }
  std::printf("bitwise equal on %d (request, head) rows\n", checked);
  return 0;
}
