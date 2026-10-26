#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"

#include <cuda_runtime_api.h>
#include <cutlass/fast_math.h>
#include <kerutils/kerutils.cuh>

#include "utils.h"

namespace smxx::decode {

__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
    int *seqlens_k_ptr = params.seqlens_k_ptr;
    DecodingSchedMeta *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.b;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem; // [batch_size]
    int* num_splits_shared = shared_mem + batch_size; // [batch_size+1]
    int* seqlens_k_shared = shared_mem + batch_size*2+1; // [batch_size]
    int* first_block_idx_shared = shared_mem + batch_size*3+1; // [batch_size]
    int* last_block_idx_shared = shared_mem + batch_size*4+1; // [batch_size]

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32) {
        int cur_s_k;
        if (params.topk == -1) {
            // Dense model, cur_s_k = actual s_k
            cur_s_k = __ldg(seqlens_k_ptr + i);
        } else {
            // Sparse model, cur_s_k = topk (+ extra topk)
            cur_s_k = params.topk_length ? __ldg(params.topk_length + i) : params.topk;
            if (cur_s_k == 0) cur_s_k = 1;  // Ensure the main loop will never be empty
            if (params.extra_topk) {
                cur_s_k = ku::ceil(cur_s_k, block_size_n);
                cur_s_k += params.extra_topk_length ? __ldg(params.extra_topk_length + i) : params.extra_topk;
            }
        }
        seqlens_k_shared[i] = cur_s_k;
        int first_token_idx = 0;
        int last_token_idx = max(cur_s_k-1, 0);
        int cur_first_block_idx = first_token_idx / block_size_n;
        int cur_last_block_idx = last_token_idx / block_size_n;
        // NOTE Should attend to tokens [first_token_idx, last_token_idx], i.e. blocks [cur_first_block_idx, cur_last_block_idx]
        // NOTE if seqlens_k is 0, then first_token_idx == last_token_idx == cur_first_block_idx == cur_last_block_idx == 0. So the sequence will have 1 block. We will correct this later in this kernel.
        int num_blocks = cur_last_block_idx - cur_first_block_idx + 1;
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
        first_block_idx_shared[i] = cur_first_block_idx;
        last_block_idx_shared[i] = cur_last_block_idx;
    }
    for (int offset = 16; offset >= 1; offset /= 2) {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0) {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

        int now_req_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i) {
            DecodingSchedMeta cur_meta;
            cur_meta.begin_req_idx = now_req_idx;
            cur_meta.begin_block_idx = now_block + first_block_idx_shared[now_req_idx];
            cur_meta.begin_split_idx = now_n_split_idx;
            cur_meta.is_first_req_splitted = (now_block != 0);
            int remain_payload = payload;
            while (now_req_idx < batch_size) {
                int num_blocks = num_blocks_shared[now_req_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks) {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_req_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_req_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                } else {
                    if (remain_payload - fixed_overhead_num_blocks > 0) {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            cur_meta.end_req_idx = now_block > 0 ? now_req_idx : now_req_idx - 1;
            cur_meta.end_block_idx = now_block > 0 ? now_block + first_block_idx_shared[now_req_idx] : (seqlens_k_shared[now_req_idx-1] == 0 ? 0 : last_block_idx_shared[now_req_idx-1] + 1);
            cur_meta.is_last_req_splitted = cur_meta.end_block_idx != last_block_idx_shared[cur_meta.end_req_idx] + 1 && seqlens_k_shared[cur_meta.end_req_idx] != 0;
            if (cur_meta.begin_req_idx == cur_meta.end_req_idx) {
                cur_meta.is_first_req_splitted = cur_meta.is_last_req_splitted = cur_meta.is_first_req_splitted || cur_meta.is_last_req_splitted;
            }
            tile_scheduler_metadata_ptr[i] = cur_meta;
        }
        FLASH_DEVICE_ASSERT(now_req_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32) {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

}
#include "smxx/decode/combine/combine.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"

using namespace cute;

namespace smxx::decode {

template<typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_mla_combine_kernel(__grid_constant__ const CombineParams params) {
    // grid_shape: [batch_size, s_q, h_q/BLOCK_SIZE_M]
    // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result
    static_assert(NUM_THREADS/32 == BLOCK_SIZE_M); // The number of warps == block_size_m
    const int batch_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int h_block_idx = blockIdx.z;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;

    int num_valid_heads = std::min(BLOCK_SIZE_M, params.h_q - BLOCK_SIZE_M*h_block_idx);
    if (warp_idx >= num_valid_heads) {
        return;
    }

    const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
    const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
    const int my_num_splits = end_split_idx - start_split_idx;
    if (my_num_splits <= 1) {
        return;
    }
    
    FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);
    
    Tensor gLseAccum = make_tensor(
        make_gmem_ptr((float*)params.lse_accum + start_split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + h_block_idx*BLOCK_SIZE_M),
        Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
        make_stride(params.stride_lse_accum_split, _1{})
    );
    Tensor gLse = make_tensor(
        make_gmem_ptr((float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + h_block_idx*BLOCK_SIZE_M),
        Shape<Int<BLOCK_SIZE_M>>{},
        Stride<_1>{}
    );
    
    __shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS];

    // Wait for the previous kernel (the MLA kernel) to finish
    cudaGridDependencySynchronize();

    // Prefetch
    static_assert(HEAD_DIM_V % (32*4) == 0);
    constexpr int ELEMS_PER_THREAD = HEAD_DIM_V / (32*4);
    float* oaccum_ptr = params.o_accum + start_split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + (h_block_idx*BLOCK_SIZE_M + warp_idx)*params.stride_o_accum_h_q;
    float4 datas[ELEMS_PER_THREAD];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        datas[i] = *(float4*)(oaccum_ptr + lane_idx*4 + i*128); // NOTE We don't use __ldg here since it is incompatible with PDL
    }

    // Warp #i gathers LseAccum for seq #i
    {
        constexpr int NUM_LSE_PER_THREAD = cute::ceil_div(MAX_SPLITS, 32);
        float local_lse[NUM_LSE_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            local_lse[i] = split_idx < my_num_splits ? gLseAccum(split_idx, warp_idx) : -INFINITY;
        }

        float max_lse = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            max_lse = max(max_lse, local_lse[i]);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
        if (lane_idx == 0)
            gLse(warp_idx) = global_lse / (float)M_LOG2E;
        
        if (params.attn_sink != nullptr) {
            int q_head_idx = h_block_idx*BLOCK_SIZE_M + warp_idx;
            float attn_sink = __ldg(params.attn_sink + q_head_idx);
            if (global_lse != INFINITY) {
                // If attn_sink is +inf, global_lse will be +inf and scale factors will be exp2f(local_lse - inf) = 0 (since local_lse never becomes +inf)
                // If attn_sink is -inf, this has no effect on global_lse
                global_lse += log2f(1 + exp2f(attn_sink*CUDART_L2E_F - global_lse));
            } else {
                // We have no tokens to attend, so global lse should be attn_sink*CUDART_L2E_F (+inf if it's -inf or +inf)
                global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink*CUDART_L2E_F;
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            smem_buf[warp_idx][split_idx] = exp2f(local_lse[i] - global_lse);
        }
    }

    __syncwarp();

    // Warp #i accumulates activation for seq #i
    {
        float4 result[ELEMS_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            result[i] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll 1
        for (int split = 0; split < my_num_splits; ++split) {
            float lse_scale = smem_buf[warp_idx][split];
            // if (lse_scale != 0.f) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                result[i].x += lse_scale * datas[i].x;
                result[i].y += lse_scale * datas[i].y;
                result[i].z += lse_scale * datas[i].z;
                result[i].w += lse_scale * datas[i].w;
                if (split != my_num_splits-1) {
                    datas[i] = *(float4*)(oaccum_ptr + (split+1)*params.stride_o_accum_split + lane_idx*4 + i*128);
                }
            }
            // }
        }
        
        const int h_q_idx = h_block_idx*BLOCK_SIZE_M + warp_idx;
        ElementT* o_ptr = (ElementT*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + h_q_idx*params.stride_o_h_q;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float4 data = result[i];
            ElementT data_converted[4];
            data_converted[0] = (ElementT)(data.x);
            data_converted[1] = (ElementT)(data.y);
            data_converted[2] = (ElementT)(data.z);
            data_converted[3] = (ElementT)(data.w);
            static_assert(sizeof(ElementT) == 2);
            *(uint64_t*)(o_ptr + lane_idx*4 + i*128) = *(uint64_t*)data_converted;
        }
    }
}


template __global__ void flash_fwd_mla_combine_kernel<cutlass::bfloat16_t,512,8,32,256>(__grid_constant__ const CombineParams);
template __global__ void flash_fwd_mla_combine_kernel<cutlass::bfloat16_t,512,8,64,256>(__grid_constant__ const CombineParams);
}
