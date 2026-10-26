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
static const char* K_PREPQ = "_ZN5glm5311lib_indexer43_GLOBAL__N__c0645384_10_kernels_cu_12f9497b16prepare_q_kernelEPKfS3_PKiS3_iPhPf";
static const char* K_TABLES = "_ZN5glm5311lib_indexer43_GLOBAL__N__c0645384_10_kernels_cu_12f9497b20decode_tables_kernelEPKNS_17PagedFp8LayerViewEPKiS6_iiPiS7_S7_";
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
  const std::vector<int> train_t = {32, 64, 96, 128, 256, 512}, held_t = {2048, 100000};
  for (int pass = 0; pass < 1; ++pass) /* small states only */ for (int t : (pass ? held_t : train_t)) for (int w = 1; w <= 4; ++w) {
    c.set_history(t);
    for (int r = 0; r < 4; ++r) { pos[r] = t + r; lens[r] = t + r + 1; }
    // Selected IDs: K = min(2048, length) distinct positions spread over the history.
    const int k = TOPK;
    for (int r = 0; r < 4; ++r) for (int j = 0; j < k; ++j) ids[size_t(r) * k + j] = (lens[r] > k) ? int(int64_t(j) * (lens[r] - 1) / (k - 1)) : (j < lens[r] ? j : -1);
    const long long kk = k, tt = t, ww = w, bb = (t + 127) / 128;
    // A. q-rank norm + kv post (+ KV cache write) + indexer-K preparation (FUSE_PREPK).
    { int stride = N1, qrank = QR, latent = LAT, rope = ROPE, idim = IK, rows = w, qs = 4; float eps = 1e-5f;
      auto* m = merged.data(); auto *pqa = qa.data(), *pqr = qr.data(), *pka = ka.data(), *pklr = klr.data(), *pkl = kl.data(), *pkr = kr.data(), *pik = ik.data();
      const float *pqw = qw.data(), *pklw = klw.data(), *pf = freq.data(), *pikw = ikw.data(), *pikb = ikb.data(); auto* pqrb = qrb.data(); const int* ppos = pos.data();
      PagedFp8LayerView* pv = views; const int* prv = rv.data(); int* perr = errors.data();
      void* a[] = {&m, &stride, &qrank, &latent, &rope, &idim, &pqa, &pqw, &pqr, &pqrb, &pka, &pklr, &pkl, &pkr, &pklw, &ppos, &pf, &pv, &prv, &perr, &pik, &rows, &qs, &eps, &eps, &pikw, &pikb, &eps};
      drperf::gpu::Region rg("qrank_kv_ik_pk", {{"rows", ww}, {"history", tt}}); mod.launch(K_QRANK, dim3(w, qs + 2), dim3(128), a); }
    // B. q post + indexer-Q copy + FlashMLA query layout (CORE_GLUE).
    { int stride = HQ * QW + IQW, tokens = w, heads = HQ, nope = NOPE, rope = ROPE, iqw = IQW, latent = LAT; auto* m = merged2.data();
      float *pq = q.data(), *pqn = qn.data(), *pqro = qrope.data(), *pqj = qj.data(), *piq = iq.data(); const int* ppos = pos.data(); const float* pf = freq.data();
      auto *pa = qnope_bf.data(), *pb = qbq.data(); void* a[] = {&m, &stride, &pq, &pqn, &pqro, &pqj, &ppos, &pf, &tokens, &heads, &nope, &rope, &piq, &iqw, &pa, &pb, &latent};
      const unsigned blocks = unsigned((size_t(w) * HQ * QW + size_t(w) * IQW + 127) / 128);
      drperf::gpu::Region rg("q_post_iq_flash", {{"rows", ww}, {"history", tt}}); mod.launch(K_QPOST, dim3(blocks), dim3(128), a); }
    // C. indexer Q preparation.
    { const float *piq = iq.data(), *piw = iw.data(), *pf = freq.data(); const int* ppos = pos.data(); int rows = w; auto* pq8 = q8.data(); auto* pw = wout.data();
      void* a[] = {&piq, &piw, &ppos, &pf, &rows, &pq8, &pw};
      drperf::gpu::Region rg("indexer_prepare_q", {{"rows", ww}, {"history", tt}}); mod.launch(K_PREPQ, dim3((w * 32 + 3) / 4), dim3(128), a); }
    // D. decode tables (DeepGEMM block table + context lengths), once per step.
    { const PagedFp8LayerView* pv = views; const int *prv = rv.data(), *ppos = pos.data(); int rows = w, mb = max_blocks; int *pbt = block_table.data(), *pc = ctx.data(), *perr = errors.data();
      void* a[] = {&pv, &prv, &ppos, &rows, &mb, &pbt, &pc, &perr};
      drperf::gpu::Region rg("indexer_decode_tables", {{"rows", ww}, {"history", tt}, {"blocks", bb * ww}}); mod.launch(K_TABLES, dim3(w), dim3(256), a); }
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
    if (t != 32 && !(pass && t == 100000)) continue;  // width-only kernels: one training history (+ one held-out check at T=100000)
    // G. MoE front (pre-normalized input): NVFP4 quant + router.
    { auto* px = xb.data(); const float* pr = router.data(); auto *pp = packed.data(), *ps = scales.data(); float* pl = logits.data(); int rows = w, width = H, ex = E, sl = 8; float ts = 1.f / 448.f;
      void* a[] = {&px, &pr, &pp, &ps, &pl, &rows, &width, &ex, &sl, &ts};
      drperf::gpu::Region rg("moe_front_prenorm", {{"rows", ww}}); mod.launch(K_FRONT, dim3(w * 8 + E), dim3(128), a, H * 4); }
    // H. indexer weights from the pre-normalized input.
    { auto* px = xb.data(); const float* pw = iwm.data(); float* po = iwo.data(); int rows = w, heads = IH, width = H; void* a[] = {&px, &pw, &po, &rows, &heads, &width};
      drperf::gpu::Region rg("attention_iw_prenorm", {{"rows", ww}}); mod.launch(K_IW, dim3(w * IH), dim3(128), a, H * 4); }
    // I. BF16 combine, SwiGLU, casts.
    { auto *pr = routed.data(), *ps = shared.data(), *po = bcast.data(); size_t n = size_t(w) * H; void* a[] = {&pr, &ps, &po, &n};
      drperf::gpu::Region rg("combine_shared_bf16", {{"rows", ww}}); mod.launch(K_COMBINE, dim3(unsigned((n + 127) / 128)), dim3(128), a); }
    { auto* pg = gate_up.data(); auto* pa = act.data(); int rows = w, half = F; void* a[] = {&pg, &pa, &rows, &half};
      drperf::gpu::Region rg("gate_up_swiglu_rows", {{"rows", ww}}); mod.launch(K_SWIGLU, dim3((F + 127) / 128, w), dim3(128), a); }
    { auto* pf = f32.data(); auto* pb = bcast.data(); size_t n = size_t(w) * H; void* a[] = {&pf, &pb, &n};
      drperf::gpu::Region rg("cast_f32_bf16", {{"rows", ww}}); mod.launch(K_CAST, dim3(unsigned((n + 127) / 128)), dim3(128), a); }
    { auto* px = xb.data(); const int* pr = sel_rows.data(); float* py = sel.data(); int n = w, width = H; void* a[] = {&px, &pr, &py, &n, &width};
      drperf::gpu::Region rg("select_rows_bf16_f32", {{"rows", ww}}); mod.launch(K_SELECT, dim3(unsigned((size_t(w) * H + 127) / 128)), dim3(128), a); }
    std::cout << "T=" << t << " w=" << w << " done" << std::endl;
  }
  for (int r = 0; r < 4; ++r) if (errors[r]) std::cout << "error cell r=" << r << " = " << errors[r] << std::endl;
  return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
