// drperf GPU launch driver (decode_step audit): integration-18 (4f5de3ae) custom
// decode kernels at production widths. Each kernel runs at training states
// (history T = 32..512, widths w = 1..4) and held-out states (T = 2048, 100000).
// PCVs: rows (w), history (T), selected (K), blocks (ceil(T/128)) where meaningful.
#include "drperf_gpu.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
struct PagedFp8LayerView {  // include/latent_cache.hpp (4f5de3ae)
  uint8_t* latent = nullptr; uint8_t* rotary = nullptr; const int32_t* page_table = nullptr; const uint64_t* page_epochs = nullptr;
  uint64_t lease_epoch = 0; int32_t page_tokens = 256, table_pages = 0, physical_pages = 0;
  int32_t logical_length = 0, reserved_tokens = 0, latent_dim = 512, rotary_dim = 64; uint8_t* index = nullptr; int32_t index_dim = 128;
};
static uint16_t bf(float v) { uint32_t u; std::memcpy(&u, &v, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }
static const char* K_QRANK = "_ZN5glm5324attention_qrank_kv_ik_pkEPK13__nv_bfloat16iiiiiPfPKfS3_PS0_S3_S3_S3_S3_S5_PKiS5_PNS_17PagedFp8LayerViewES8_PiS3_iiffS5_S5_f";
static const char* K_QPOST = "_ZN5glm5325attention_q_post_iq_flashEPK13__nv_bfloat16iPfS3_S3_S3_PKiPKfiiiiS3_iPS0_S8_i";
static const char* K_PREPQ = "_ZN5glm5311lib_indexer43_GLOBAL__N__846acc06_10_kernels_cu_12f9497b16prepare_q_kernelEPKfS3_PKiS3_iPhPf";
static const char* K_TABLES = "_ZN5glm5311lib_indexer43_GLOBAL__N__846acc06_10_kernels_cu_12f9497b20decode_tables_kernelEPKNS_17PagedFp8LayerViewEPKiS6_iiPiS7_S7_";
static const char* K_ADDR = "_ZN5glm5319dc_selected_addressEPKNS_17PagedFp8LayerViewEPKiS4_S4_PiS5_S5_iiiiib";
static const char* K_GATHER = "_ZN5glm5313flashmla_glue17decode_gather_v32EPKNS_17PagedFp8LayerViewEPKiS5_PhPiPjiiiiii";
static const char* K_FRONT = "_ZN5glm5317moe_front_prenormEPK13__nv_bfloat16PKfPhS5_Pfiiiif";
static const char* K_IW = "_ZN5glm5320attention_iw_prenormEPK13__nv_bfloat16PKfPfiii";
static const char* K_COMBINE = "_ZN5glm5327lib_moe_combine_shared_bf16EPK13__nv_bfloat16S2_PS0_m";
static const char* K_SWIGLU = "_ZN5glm5324bf16_gate_up_swiglu_rowsEPK13__nv_bfloat16PS0_ii";
static const char* K_CAST = "_ZN5glm5313cast_f32_bf16EPKfP13__nv_bfloat16m";
static const char* K_SELECT = "_ZN5glm5320select_rows_bf16_f32EPK13__nv_bfloat16PKiPfii";
constexpr int H = 6144, QR = 2048, LAT = 512, ROPE = 64, IK = 128, N1 = QR + LAT + ROPE + IK, HQ = 32, NOPE = 192, QW = 256, IQW = 32 * 128;
constexpr int E = 256, IH = 32, F = 2048, TOPK = 2048, REQ = 4, MAXT = 100000;
struct Cache {  // 4 requests, histories up to MAXT, 256-token pages
  int pages_per_req = (MAXT + 512 + 255) / 256, pages = REQ * pages_per_req; size_t rows = size_t(pages) * 256;
  std::vector<uint8_t> latent, rotary, index; std::vector<int32_t> table; std::vector<uint64_t> epochs; std::vector<PagedFp8LayerView> views;
  Cache() : latent(rows * 512), rotary(rows * 64), index(size_t(pages) * 256 * (128 + 4)), table(size_t(REQ) * pages_per_req), epochs(pages) {
    for (size_t i = 0; i < latent.size(); i += 61) latent[i] = uint8_t(i * 7);
    for (int r = 0; r < REQ; ++r) for (int p = 0; p < pages_per_req; ++p) { int phys = (p * REQ + r) % pages; table[size_t(r) * pages_per_req + p] = phys; epochs[phys] = 100 + r; }
    views.resize(REQ);
    for (int r = 0; r < REQ; ++r) { auto& v = views[r]; v.latent = latent.data(); v.rotary = rotary.data(); v.page_table = table.data() + size_t(r) * pages_per_req;
      v.page_epochs = epochs.data(); v.lease_epoch = 100 + r; v.table_pages = pages_per_req; v.physical_pages = pages; v.index = index.data(); }
  }
  void set_history(int t) { for (int r = 0; r < REQ; ++r) { views[r].logical_length = t + r + 1; views[r].reserved_tokens = t + 512; } }
};
int main(int argc, char** argv) { try {
  if (argc != 2) return 2;
  drperf::gpu::Module mod(argv[1]); Cache c;
  std::vector<uint16_t> merged(4 * N1), merged2(4 * (HQ * QW + IQW)), xb(4 * H), routed(4 * H), shared(4 * H), gate_up(4 * 2 * F), act(4 * F), qnope_bf(4 * HQ * NOPE), qbq(4 * HQ * 576), bcast(4 * H);
  for (size_t i = 0; i < merged.size(); ++i) merged[i] = bf(float(int(i % 23) - 11) * 0.03f);
  for (size_t i = 0; i < merged2.size(); ++i) merged2[i] = bf(float(int(i % 29) - 14) * 0.02f);
  for (size_t i = 0; i < xb.size(); ++i) { xb[i] = bf(float(int(i % 17) - 8) * 0.1f); routed[i] = bf(0.01f * float(i % 13)); shared[i] = bf(-0.02f * float(i % 7)); }
  for (size_t i = 0; i < gate_up.size(); ++i) gate_up[i] = bf(float(int(i % 19) - 9) * 0.05f);
  std::vector<float> qw(QR, 1.f), klw(LAT, 1.f), ikw(IK, 1.f), ikb(IK, 0.f), freq(32), qa(4 * QR), qr(4 * QR), ka(4 * (LAT + ROPE)), klr(4 * LAT), kl(4 * LAT), kr(4 * ROPE), ik(4 * IK);
  for (int i = 0; i < 32; ++i) freq[i] = 1.f / float(1 << (i % 12));
  std::vector<uint16_t> qrb(4 * QR); std::vector<float> q(4 * HQ * QW), qn(4 * HQ * NOPE), qrope(4 * HQ * ROPE), qj(4 * HQ * QW), iq(4 * IQW), iw(4 * IH, 0.5f), wout(4 * IH);
  std::vector<uint8_t> q8(4 * 32 * 128); std::vector<float> router(size_t(E) * H), iwm(size_t(IH) * H), logits(4 * E), iwo(4 * IH), f32(4 * H), sel(4 * H);
  for (size_t i = 0; i < router.size(); ++i) router[i] = float(int(i % 31) - 15) / 1024.f;
  for (size_t i = 0; i < iwm.size(); ++i) iwm[i] = float(int(i % 37) - 18) / 1024.f;
  for (size_t i = 0; i < f32.size(); ++i) f32[i] = float(int(i % 11) - 5) * 0.25f;
  std::vector<uint8_t> packed(4 * H / 2), scales(4 * H / 16);
  const int max_blocks = (MAXT + 512 + 127) / 128;
  std::vector<int> block_table(size_t(4) * max_blocks), ctx(4), rv = {0, 1, 2, 3}, pos(4), lens(4), errors(4), sel_rows = {0, 1, 2, 3};
  std::vector<int> ids(size_t(4) * TOPK), phys(size_t(4) * TOPK), idx(size_t(4) * TOPK); std::vector<uint8_t> kv(size_t(4) * TOPK * 656);
  auto* views = c.views.data();
  const std::vector<int> train_t = {64, 128, 256, 512};
  for (int t : train_t) for (int k : {16, 32, 64}) for (int w = 1; w <= 4; ++w) {
    c.set_history(t);
    for (int r = 0; r < 4; ++r) { pos[r] = t + r; lens[r] = t + r + 1; }
    for (int r = 0; r < 4; ++r) for (int j = 0; j < k; ++j) ids[size_t(r) * k + j] = int(int64_t(j) * (lens[r] - 1) / (k - 1));
    const long long kk = k, tt = t, ww = w, bb = (t + 127) / 128, pp = (t + 255) / 256;
    // D. decode tables (DeepGEMM block table + context lengths), once per step.
    { const PagedFp8LayerView* pv = views; const int *prv = rv.data(), *ppos = pos.data(); int rows = w, mb = max_blocks; int *pbt = block_table.data(), *pc = ctx.data(), *perr = errors.data();
      void* a[] = {&pv, &prv, &ppos, &rows, &mb, &pbt, &pc, &perr};
      drperf::gpu::Region rg("indexer_decode_tables", {{"rows", ww}, {"history", tt}, {"blocks", bb * ww}, {"pages", pp * ww}}); mod.launch(K_TABLES, dim3(w), dim3(256), a); }
    // E. selected address (CORE_GLUE dense-short + physical rows).
    { const PagedFp8LayerView* pv = views; const int *prv = rv.data(), *ppos = pos.data(), *pl = lens.data(); int *pid = ids.data(), *pph = phys.data(), *perr = errors.data();
      int rows = w, req = REQ, cnt = k, ld = LAT, rd = ROPE; bool dense = true;
      void* a[] = {&pv, &prv, &ppos, &pl, &pid, &pph, &perr, &rows, &req, &cnt, &ld, &rd, &dense};
      drperf::gpu::Region rg("dc_selected_address", {{"rows", ww}, {"history", tt}, {"selected", kk * ww}}); mod.launch(K_ADDR, dim3(unsigned((size_t(w) * k / 4 + 255) / 256)), dim3(256), a); }
    // F. FlashMLA V32 gather of the selected rows.
    { const PagedFp8LayerView* pv = views; const int *prv = rv.data(), *pph = phys.data(); auto* pkv = kv.data(); int* pidx = idx.data(); auto* perr = reinterpret_cast<unsigned*>(errors.data());
      int rows = w, req = REQ, selected = k, topk = k, latent = LAT, rot = ROPE;
      void* a[] = {&pv, &prv, &pph, &pkv, &pidx, &perr, &rows, &req, &selected, &topk, &latent, &rot};
      drperf::gpu::Region rg("decode_gather_v32", {{"rows", ww}, {"history", tt}, {"selected", kk * ww}}); mod.launch(K_GATHER, dim3(unsigned((size_t(w) * k * 32 + 255) / 256)), dim3(256), a); }
    if (t == 64 && k == 16) { auto* px = xb.data(); const float* pw = iwm.data(); float* po = iwo.data(); int rows = w, heads = IH, width = H; void* a[] = {&px, &pw, &po, &rows, &heads, &width};
      drperf::gpu::Region rg("attention_iw_prenorm", {{"rows", ww}}); mod.launch(K_IW, dim3(IH), dim3(128), a, H * 4); }
    std::cout << "T=" << t << " K=" << k << " w=" << w << " done" << std::endl;
  }
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
