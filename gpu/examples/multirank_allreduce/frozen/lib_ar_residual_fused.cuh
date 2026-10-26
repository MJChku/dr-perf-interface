// GLM53_ENABLE_DECODE_AR_RESIDUAL_FUSED: project kernel (custom arithmetic in the
// epilogue only) = FlashInfer 0.6.18 one-shot Lamport all-reduce (NRanks 2, the
// library kernel's exchange, sentinel, clear and flag protocol, unchanged) with
// the decoder's residual kernel folded into its epilogue:
//   dtype 0 (attention, BF16 partials, FP32 accumulation):
//     out = round_bf16(residual + float(bf16(float(x0) + float(x1))))   (lib_ar_residual_bf16)
//   dtype 1 (MoE, FP32 partials):
//     out = round_bf16(residual + round_bf16(x0 + x1))                  (lib_ar_residual_f32)
// Same per-element expressions as the library all-reduce followed by the
// residual kernel, so the FP32 hidden state is bitwise identical to that chain
// (tests/lib_ar_residual_single.cu, strict GXPTX). The RMSNorm stays in the next
// block's front kernel (moe_front_v5 / attention_norm_iw_v5), which also runs the
// router / indexer-weight dot CTAs in the same launch; see eval/lanes/decode_step.md.
// Communicating kernel: never priced from profiled time (GX: zero, disclosed).
// Included by src/lib_allreduce_fusion.cu (the FlashInfer header defines
// non-inline functions, so both kernels live in that one translation unit).
#pragma once
#include <cuda_bf16.h>
#include <type_traits>
#include <cstdint>

namespace glm53_ar_fused {
using namespace flashinfer::trtllm_allreduce_fusion;
using flashinfer::vec_t;
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }  // = model_ops round_bf16

template <typename T, bool Fp32Acc>
__global__ void ar_residual_oneshot(AllReduceFusionParams<T> params, const float* residual, float* out) {  // out may alias residual (in place)
  static constexpr int VEC_SIZE = details::kBytesPerAccess / sizeof(T);
  IndexHelper<T> index_helper(params);
  const int access_id = index_helper.access_id, access_stride = index_helper.access_stride, tot_access = index_helper.tot_access;
  vec_t<T, VEC_SIZE> clear_vec;
  clear_vec.fill(neg_zero_v<T>);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  LamportComm<2> comm(params.workspace, params.rank);
  const int clear_access = comm.clear_size / VEC_SIZE;
  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> val;
    val.load(reinterpret_cast<T*>(params.allreduce_in) + idx * VEC_SIZE);
    remove_neg_zero<T, VEC_SIZE>(val);
#pragma unroll
    for (int r = 0; r < 2; ++r) val.store(reinterpret_cast<T*>(comm.data_bufs[r]) + (params.rank * tot_access + idx) * VEC_SIZE);
  }
  for (int idx = access_id; idx < clear_access; idx += access_stride) clear_vec.store(reinterpret_cast<T*>(comm.clear_buf) + idx * VEC_SIZE);
  for (int idx = access_id; idx < tot_access; idx += access_stride) {
    vec_t<T, VEC_SIZE> vals[2];
    bool done = false;
    while (!done) {
      done = true;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        vals[r].load_global_volatile(reinterpret_cast<T*>(comm.data_bufs[params.rank]) + (r * tot_access + idx) * VEC_SIZE);
        done &= !has_neg_zero<T, VEC_SIZE>(vals[r]);
      }
    }
    vec_t<T, VEC_SIZE> sum = allreduce_sum<T, VEC_SIZE, 2, Fp32Acc>(vals);
    if (params.allreduce_out) sum.store(reinterpret_cast<T*>(params.allreduce_out) + idx * VEC_SIZE);  // optional (tests)
    const size_t e0 = size_t(idx) * VEC_SIZE;
    float res[VEC_SIZE], o[VEC_SIZE];
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 4) *reinterpret_cast<float4*>(res + i) = *reinterpret_cast<const float4*>(residual + e0 + i);
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      if constexpr (std::is_same_v<T, float>) o[i] = rbf(res[i] + rbf(sum[i]));
      else o[i] = rbf(res[i] + __bfloat162float(sum[i]));
    }
#pragma unroll
    for (int i = 0; i < VEC_SIZE; i += 4) *reinterpret_cast<float4*>(out + e0 + i) = *reinterpret_cast<const float4*>(o + i);
  }
  comm.update(params.size * 2);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename T, bool Acc>
int run_fused(void* const* ws, int rank, const void* in, const float* residual, float* out, void* ar_out, int tokens, int hidden,
              cudaStream_t st, bool pdl) {
  AllReduceFusionParams<T> p{};
  p.nranks = 2; p.rank = rank; p.size = tokens * hidden; p.hidden_dim = hidden;
  p.workspace = const_cast<void**>(ws);
  p.allreduce_in = const_cast<void*>(in); p.allreduce_out = ar_out;
  p.use_oneshot = true; p.pattern = AllReduceFusionPattern::kAllReduce; p.trigger_completion_at_end = true; p.stream = st;
  // Launch configuration: identical rule to run_t in src/lib_allreduce_fusion.cu
  // (library one-shot launcher for decode shapes, portable cluster size 8).
  constexpr int VEC = int(details::kBytesPerAccess / sizeof(T));
  if (hidden % VEC != 0 || tokens > details::kOneShotMaxToken) return int(cudaErrorInvalidValue);
  const int threads_per_token = hidden / VEC;
  int cluster = 8;
  while (threads_per_token % cluster != 0 && cluster > 1) cluster /= 2;
  int threads = threads_per_token / cluster;
  while (threads < 128 && cluster >= 2) { threads *= 2; cluster /= 2; }
  if (threads * cluster != threads_per_token || threads > 1024) return int(cudaErrorInvalidValue);
  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attr[2];
  cfg.gridDim = dim3(unsigned(tokens * cluster)); cfg.blockDim = dim3(unsigned(threads)); cfg.dynamicSmemBytes = 0; cfg.stream = st;
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  attr[1].id = cudaLaunchAttributeClusterDimension;
  attr[1].val.clusterDim.x = unsigned(cluster); attr[1].val.clusterDim.y = 1; attr[1].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = 2;
  return int(cudaLaunchKernelEx(&cfg, ar_residual_oneshot<T, Acc>, p, residual, out));
}
}  // namespace glm53_ar_fused

extern "C" int glm53_lib_ar_residual_run(void* const* ws, int dtype, int rank, const void* in, const float* residual, float* out,
                                         void* allreduce_out, int tokens, int hidden, cudaStream_t st, int pdl) {
  if (rank < 0 || rank > 1 || tokens < 1 || hidden < 1 || !residual || !out) return int(cudaErrorInvalidValue);
  using namespace glm53_ar_fused;
  return dtype ? run_fused<float, false>(ws, rank, in, residual, out, allreduce_out, tokens, hidden, st, pdl != 0)
               : run_fused<__nv_bfloat16, true>(ws, rank, in, residual, out, allreduce_out, tokens, hidden, st, pdl != 0);
}
