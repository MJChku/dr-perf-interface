#pragma once
// Latency-oriented variant of FlashMLA's smxx::decode::flash_fwd_mla_combine_kernel
// (pinned 6bc49418) for the decode shapes GLM-5.3 runs: <= 16 splits per request.
//
// The original assigns one warp per head and walks the splits with a
// prefetch-next loop (#pragma unroll 1), so each warp waits for ~11 dependent
// 2 KB partial loads in sequence; only b * h_q/8 CTAs exist (16 at w4/H32).
// Here four warps share a head (128 dims each) and every warp issues all of its
// split loads before accumulating, so the per-warp chain is one load latency and
// the grid has 4x the CTAs. The LSE reduction and the per-split accumulation
// (result += scale * data in split order) are the same operations in the same
// order, so outputs are bitwise identical to the original kernel.
#include <math_constants.h>
#include <cutlass/numeric_types.h>
#include "params.h"

namespace glm53_flashmla {

template <typename ElementT, int MAX_SPLITS, int MAX_LOADS>
__global__ void __launch_bounds__(256)
combine_wide_kernel(__grid_constant__ const CombineParams params) {
  constexpr int HEAD_DIM_V = 512, HEADS_PER_CTA = 2, WARPS_PER_HEAD = 4;
  const int batch_idx = blockIdx.x, s_q_idx = blockIdx.y;
  const int warp_idx = threadIdx.x / 32, lane_idx = threadIdx.x % 32;
  const int head = blockIdx.z * HEADS_PER_CTA + warp_idx / WARPS_PER_HEAD;
  const int quarter = warp_idx % WARPS_PER_HEAD;
  if (head >= params.h_q) return;

  const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
  const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
  const int my_num_splits = end_split_idx - start_split_idx;
  if (my_num_splits <= 1) return;

  const float* lse_accum = params.lse_accum + start_split_idx * params.stride_lse_accum_split +
                           s_q_idx * params.stride_lse_accum_s_q + head;

  // Wait for the previous kernel (the MLA kernel) to finish
  cudaGridDependencySynchronize();

  // All split partials of this warp's 128 dims in flight at once.
  const float* oaccum_ptr = params.o_accum + start_split_idx * params.stride_o_accum_split +
                            s_q_idx * params.stride_o_accum_s_q + head * params.stride_o_accum_h_q +
                            quarter * (HEAD_DIM_V / WARPS_PER_HEAD) + lane_idx * 4;
  float4 datas[MAX_LOADS];
#pragma unroll
  for (int split = 0; split < MAX_LOADS; ++split)
    if (split < my_num_splits) datas[split] = *(const float4*)(oaccum_ptr + split * params.stride_o_accum_split);

  // LSE: identical to the original kernel's warp reduction for this MAX_SPLITS.
  __shared__ float smem_buf[HEADS_PER_CTA * WARPS_PER_HEAD][MAX_SPLITS];
  {
    constexpr int NUM_LSE_PER_THREAD = (MAX_SPLITS + 31) / 32;
    float local_lse[NUM_LSE_PER_THREAD];
#pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
      const int split_idx = i * 32 + lane_idx;
      local_lse[i] = split_idx < my_num_splits ? lse_accum[split_idx * params.stride_lse_accum_split] : -INFINITY;
    }
    float max_lse = -INFINITY;
#pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) max_lse = max(max_lse, local_lse[i]);
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
    max_lse = max_lse == -INFINITY ? 0.0f : max_lse;
    float sum_lse = 0;
#pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
#pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);
    float global_lse = (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
    if (lane_idx == 0 && quarter == 0)
      params.lse[batch_idx * params.stride_lse_b + s_q_idx * params.stride_lse_s_q + head] = global_lse / (float)M_LOG2E;
    if (params.attn_sink != nullptr) {
      float attn_sink = __ldg(params.attn_sink + head);
      if (global_lse != INFINITY) global_lse += log2f(1 + exp2f(attn_sink * CUDART_L2E_F - global_lse));
      else global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink * CUDART_L2E_F;
    }
#pragma unroll
    for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) smem_buf[warp_idx][i * 32 + lane_idx] = exp2f(local_lse[i] - global_lse);
  }
  __syncwarp();

  float4 result = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int split = 0; split < MAX_LOADS; ++split) {
    if (split < my_num_splits) {
      const float lse_scale = smem_buf[warp_idx][split];
      result.x += lse_scale * datas[split].x;
      result.y += lse_scale * datas[split].y;
      result.z += lse_scale * datas[split].z;
      result.w += lse_scale * datas[split].w;
    }
  }
  ElementT* o_ptr = (ElementT*)params.out + batch_idx * params.stride_o_b + s_q_idx * params.stride_o_s_q +
                    head * params.stride_o_h_q + quarter * (HEAD_DIM_V / WARPS_PER_HEAD) + lane_idx * 4;
  ElementT converted[4] = {(ElementT)result.x, (ElementT)result.y, (ElementT)result.z, (ElementT)result.w};
  *(uint64_t*)o_ptr = *(uint64_t*)converted;
}

// Returns false (caller falls back to the original combine) when a request may
// exceed 16 splits or the head dimension differs.
template <typename ElementT>
bool run_combine_wide(CombineParams& params, int max_splits) {
  if (params.d_v != 512 || max_splits > 16 || max_splits < 1) return false;
  auto kernel = &combine_wide_kernel<ElementT, 32, 16>;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {dim3(params.b, params.s_q, (params.h_q + 1) / 2), dim3(256, 1, 1), 0, params.stream,
                               attribute, 1};
  return cudaLaunchKernelEx(&config, kernel, params) == cudaSuccess;
}

}  // namespace glm53_flashmla
