/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Custom routing: entry point, kernel definitions, and launch wrappers.
//
// Kernel inventory:
//   1. routingIndicesBlockKernel      — single-block fused kernel (≤4 tokens)
//   2. routingIndicesClusterKernel    — single-cluster fused kernel (≤256 tokens, SM90+)
//   3. routingIndicesHistogramScoresKernel — TopK + histogram from raw scores
//   4. routingIndicesCoopKernel       — cooperative histogram + offsets (defined in
//   RoutingKernel.cuh)
//   5. routingInitExpertCounts        — zero expert counts (defined in RoutingKernel.cuh)
//   6. routingIndicesHistogramKernel  — histogram from packed TopK (defined in RoutingKernel.cuh)
//   7. routingIndicesOffsetsKernel    — prefix-scan + permutation (defined in RoutingKernel.cuh)

#include <cstdlib>
#include <string>
#include <type_traits>

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"

namespace moe::dev::routing {
namespace routingCustom {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 1. Block kernel — single-block fused kernel for ≤4 tokens.
//    Fuses TopK, histogram, prefix-scan, and permutation in one block.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingIndicesBlockKernel(KernelParams params) {
  // types used in this kernel
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  // When MaxNumExperts > 1024, cap actual thread count at 1024 and let each thread handle
  // multiple experts. This is needed because CUDA blocks support at most 1024 threads.
  static constexpr int NumThreadsBlock = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsBlock;
  static_assert(MaxNumExperts % NumThreadsBlock == 0,
                "MaxNumExperts must be a multiple of NumThreadsBlock");

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  auto scoreOffset = warpIdx * params.mNumExperts;
  bool validToken = warpIdx < params.mNumTokens;

  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
  static constexpr int totalExpertCounts = BlockKernelMaxNumTokens * MaxNumExperts;
  __shared__ int8_t __attribute((aligned(128))) smemOffset[totalExpertCounts];
  __shared__ int8_t __attribute((aligned(128))) smemKIdx[totalExpertCounts];

  using Scan = cub::BlockScan<int32_t, NumThreadsBlock>;
  __shared__ typename Scan::TempStorage tempStorage;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  for (int i = threadIdx.x; i < totalExpertCounts; i += blockDim.x) {
    smemOffset[i] = int8_t{-1};
    smemKIdx[i] = int8_t{-1};
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // then wait on primary grid
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (params.mPtrTopKIds != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
        auto expertIdx = params.mPtrTopKIds[expandedIdx];
        if (expertIdx > -1 && expertIdx < params.mNumExperts) {
          int offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        }
      }
    }
  } else if (params.mPtrScores != nullptr) {
    // in this case, each warp represents a token
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

    if (validToken) {
      KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                       KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);

      if (laneIdx < params.mTopK) {
        int offset = warpIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx];
        smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[warpIdx * params.mTopK + laneIdx] =
              OutputT{warpTopKScore[laneIdx]};
        }
        // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
        // -- same indexing as mPtrTopKPacked / mPtrTopKWeights.
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[warpIdx * params.mTopK + laneIdx] =
              static_cast<int16_t>(warpTopKExpertIdx[laneIdx]);
        }
      }
    }  // end if (validToken)
  } else if (params.mPtrTopKPacked != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
        // Pre-initialize to -1: when duplicate expert IDs appear, multiple
        // lanes race on the same smemKIdx slot and only one wins.  Losing
        // lanes would skip the else-branch below, leaving their entry as
        // uninitialized garbage.  Writing -1 up front makes every entry safe.
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
        auto const scoreIdx = params.mPtrTopKPacked[expandedIdx];
        int const expertIdx = static_cast<int>(scoreIdx.idx);
        if (expertIdx >= 0 && expertIdx < params.mNumExperts) {
          int const offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
          if (params.mPtrTopKWeights != nullptr) {
            params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
          }
        }
      }
    }
  }
  __syncthreads();

  // Each thread handles ExpertsPerThread contiguous experts.
  // Thread i handles experts [i * ExpertsPerThread, (i+1) * ExpertsPerThread).
  // Contiguous assignment ensures prefix sum ordering is correct.
  int accExpertCount[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    auto localExpIdx = expert - params.mLocalExpertsStartIdx;
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                   (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

    // Get the count of each expert and the offset for each token
    accExpertCount[e] = 0;
    if (isLocal) {
      int offset = expert;
      for (int j = 0; j < BlockKernelMaxNumTokens; j++) {
        if (smemKIdx[offset] >= 0) {
          smemOffset[offset] = static_cast<int8_t>(accExpertCount[e]);
          accExpertCount[e]++;
        }
        offset += MaxNumExperts;
      }
    }
  }
  __syncthreads();

  // Get the number of CTAs and the offset for each CTA.
  // Use cub::BlockScan's array overload: each thread holds ExpertsPerThread items,
  // and ExclusiveSum computes the prefix sum across all NumThreadsBlock * ExpertsPerThread
  // items in thread order — exactly matching our contiguous expert assignment.
  int32_t numCtaPerExpert[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if (params.mIsPow2) {
      numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
    } else {
      numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
    }
  }
  int32_t ctaOffsetPerExpert[ExpertsPerThread];
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCtaPerExpert, ctaOffsetPerExpert, numNonExitingCtas);
  __syncthreads();  // Required barrier before reusing TempStorage for the next BlockScan

  // Compute padded expert scan counts (same array-overload pattern)
  int32_t tmpCountPerExpert[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if (params.mIsPow2) {
      tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
    } else {
      tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
    }
  }
  int32_t expertScanCountsPerExpert[ExpertsPerThread];
  Scan(tempStorage).ExclusiveSum(tmpCountPerExpert, expertScanCountsPerExpert);
  __syncthreads();

  // Write CTA configs for each expert this thread handles
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    auto localExpIdx = expert - params.mLocalExpertsStartIdx;
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                   (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

    if (isLocal) {
      for (int cta = 0; cta < numCtaPerExpert[e]; ++cta) {
        int32_t const mappedLocalIdx =
            (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
        // Write CTA-level MnLimits using ctaTile = cgaTile / clusterSize
        int32_t mnLimit1;
        int32_t mnLimit2;
        if (params.mIsPow2) {
          mnLimit1 = mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mPaddingLog2);
          mnLimit2 =
              mulLog2<int32_t>(ctaOffsetPerExpert[e], params.mPaddingLog2) + accExpertCount[e];
        } else {
          mnLimit1 = mulTileN<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mTileTokensDim);
          mnLimit2 =
              mulTileN<int32_t>(ctaOffsetPerExpert[e], params.mTileTokensDim) + accExpertCount[e];
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
      }
    }
  }

  // at this point, we can write out padded count
  if (threadIdx.x == 0) {
    int32_t permutedIdxSize;
    if (params.mIsPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  for (int tokenIdx = 0; tokenIdx < params.mNumTokens; tokenIdx++) {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      int offset = tokenIdx * MaxNumExperts + expert;
      if (smemKIdx[offset] >= 0) {
        auto localExpIdx = expert - params.mLocalExpertsStartIdx;
        auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
        auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                       (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

        int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
        // Only load smemOffset for local experts; the histogram phase only
        // writes it for local experts, so remote entries are uninitialized.
        int const permutedIdx =
            isLocal ? expertScanCountsPerExpert[e] + static_cast<int>(smemOffset[offset])
                    : int32_t{-1};

        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        }
        if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocal) {
          params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
        }
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocal) {
          params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Trigger the secondary kernel AFTER all global memory writes (including permutation indices).
  // The downstream kernels depend on all routing outputs being visible.
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}
}  // namespace routingCustom
}  // namespace moe::dev::routing
