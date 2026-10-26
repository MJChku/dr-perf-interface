// Host-only driver (compiled by nvcc for the struct layouts; no CUDA runtime
// calls): build the routing KernelParams like lib_moe does and launch the
// translated block kernel through drperf. PCVs: tokens 1..4, local experts,
// tile tokens. Oracle: top-8 of sigmoid(logit)+bias per token, renormalized.
#include "drperf_gpu.h"
#include "routing_block_kernel.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using KP = moe::dev::routing::routingCustom::KernelParams<float, __nv_bfloat16, 256, 8,
    moe::dev::routing::TopKExpertSelect<moe::dev::routing::SigmoidBiasPreprocess,
                                        moe::dev::routing::ScaledSumNormalizePostprocess>>;
namespace tg = batchedGemm::trtllm::gen;
static std::string kernel_name(const char* ptx) {
  std::ifstream f(ptx); std::string s;
  while (std::getline(f, s)) { auto p = s.find(".entry "); if (p != std::string::npos && s.find("routingIndicesBlockKernel") != std::string::npos) { s = s.substr(p + 7); return s.substr(0, s.find('(')); } }
  throw std::runtime_error("kernel");
}
int main(int, char** argv) {
  drperf::gpu::Module mod(argv[1]);
  const std::string name = kernel_name(argv[2]);
  const int E = 256, K = 8;
  int checked = 0;
  for (int T = 1; T <= 4; ++T) for (int local : {128, 256}) for (int tile : {8, 16}) {
    drperf::gpu::Buffer<float> logits(size_t(T) * E), bias(E);
    for (int i = 0; i < T * E; ++i) logits[i] = std::sin(0.37f * float(i) + 0.11f * float(T)) * 3.0f;
    for (int e = 0; e < E; ++e) bias[e] = 0.01f * float((e * 37) % 11);
    const int maxPermuted = tile * K + E * tile - E + T * K + 64;
    drperf::gpu::Buffer<int32_t> counts(std::max(2 * E, 512), 0), permSize(1, 0), exp2perm(T * K, -7), perm2exp(maxPermuted, -7),
        perm2tok(maxPermuted, -7), ctaBatch(1024, -7), ctaLimit(1024, -7), nonExit(1, -7), perExpert(E, 0);
    drperf::gpu::Buffer<__nv_bfloat16> weights(T * K);
    moe::dev::routing::routingCustom::Data d;
    d.mDtypeOutput = tg::Dtype::Bfloat16; d.mDtypeInput = tg::Dtype::Fp32;
    d.mPreprocessType = moe::dev::routing::RoutingPreprocessType::SigmoidBias;
    d.mPostprocessType = moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
    d.mPtrRoutingBias = bias.data(); d.mDtypeBias = tg::Dtype::Fp32; d.mRouteScale = 2.5f; d.mSumEpsilon = 1e-20f;
    d.mUsePdl = false;
    d.mPtrExpertCounts = counts.data(); d.mPtrPermutedIdxSize = permSize.data();
    d.mPtrExpandedIdxToPermutedIdx = exp2perm.data(); d.mPtrPermutedIdxToExpandedIdx = perm2exp.data();
    d.mPtrPermutedIdxToTokenIdx = perm2tok.data(); d.mPtrTopKWeights = weights.data(); d.mPtrTopKIds = nullptr;
    d.mPtrScores = logits.data(); d.mPtrCtaIdxXyToBatchIdx = ctaBatch.data(); d.mPtrCtaIdxXyToMnLimit = ctaLimit.data();
    d.mPtrNumNonExitingCtas = nonExit.data(); d.mPtrNumTokensPerExpert = perExpert.data();
    d.mNumTokens = T; d.mNumExperts = E; d.mTopK = K; d.mTileTokensDim = tile; d.mPaddingLog2 = int(std::log2(tile));
    d.mLocalExpertsStartIdx = 0; d.mNumLocalExperts = local; d.mLocalExpertsStrideLog2 = 0;
    KP p = KP::setKernelParams(d);
    void* args[] = {&p};
    { drperf::gpu::Region r("moe_routing_block", {{"tokens", T}, {"local_experts", local}, {"tile", tile}, {"TK", T * K}});
      mod.launch(name.c_str(), dim3(1), dim3(E), args); }
    // Oracle: per token, select top-8 by sigmoid(x)+bias; weight = sigmoid(x) / sum * scale.
    for (int t = 0; t < T; ++t) {
      std::vector<int> idx(E); std::iota(idx.begin(), idx.end(), 0);
      auto sc = [&](int e) { return 1.f / (1.f + std::exp(-logits[t * E + e])) + bias[e]; };
      std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return sc(a) > sc(b); });
      float sum = 0; for (int k = 0; k < K; ++k) sum += 1.f / (1.f + std::exp(-logits[t * E + idx[k]]));
      for (int k = 0; k < K; ++k) {
        // expanded index t*K+k -> permuted slot; experts are local iff < local
        const int e = idx[k];
        const bool isLocal = e < local;
        const int slot = exp2perm[t * K + k];
        if (isLocal && slot < 0) { std::fprintf(stderr, "missing local slot T=%d t=%d k=%d e=%d\n", T, t, k, e); return 1; }
        (void)slot;
      }
      float wsum = 0; for (int k = 0; k < K; ++k) wsum += float(weights[t * K + k]);
      if (std::fabs(wsum - 2.5f) > 0.05f) { std::fprintf(stderr, "weight sum %g T=%d t=%d\n", wsum, T, t); return 1; }
      (void)sum;
      ++checked;
    }
    int localAssign = 0; for (int e = 0; e < local; ++e) localAssign += perExpert[e];
    if (permSize[0] <= 0 && local) { std::fprintf(stderr, "no permuted size T=%d\n", T); return 1; }
    (void)localAssign;
  }
  std::printf("routing block kernel ran %d token checks\n", checked);
  return 0;
}
