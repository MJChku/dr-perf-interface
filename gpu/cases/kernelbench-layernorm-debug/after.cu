
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////
// Kernel #1 : each block processes a slice of one sample and writes a single
//             (sum, sumsq) pair to the workspace – NO global atomics
////////////////////////////////////////////////////////////////////////////////
__global__ __launch_bounds__(256, 4)     // ≤ 4 CTAs/SM, 256 threads/CTA
void layernorm_partial_sum_kernel(const float* __restrict__ input,
                                  float*       __restrict__ partial,  // [N, B, 2]
                                  int D,
                                  int num_blocks_per_sample)
{
    const int global_block = blockIdx.x;              // 0 .. N*B-1
    const int tid          = threadIdx.x;
    const int n            = global_block / num_blocks_per_sample;
    const int b_in_sample  = global_block % num_blocks_per_sample;

    const int D4    = D >> 2;             // elements handled as float4
    const int base  = n * D;              // scalar offset
    const int base4 = n * D4;             // float4 offset

    float local_sum   = 0.f;
    float local_sumsq = 0.f;

    // ------------------------------------------------------------------
    // Vectorised loop (float4)
    // ------------------------------------------------------------------
    const float4* input4 = reinterpret_cast<const float4*>(input);

    for (int idx4 = b_in_sample * blockDim.x + tid;
         idx4 < D4;
         idx4 += blockDim.x * num_blocks_per_sample)
    {
        float4 v4 = input4[base4 + idx4];

        float v0 = v4.x, v1 = v4.y, v2 = v4.z, v3 = v4.w;

        local_sum   += v0 + v1 + v2 + v3;
        local_sumsq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    }

    // ------------------------------------------------------------------
    // Tail loop (0–3 remaining scalars, if any)
    // ------------------------------------------------------------------
    for (int idx = D4 * 4 + b_in_sample * blockDim.x + tid;
         idx < D;
         idx += blockDim.x * num_blocks_per_sample)
    {
        float v = input[base + idx];
        local_sum   += v;
        local_sumsq += v * v;
    }

    // ------------------------------------------------------------------
    // Block-wide reduction of (sum, sumsq) – no global atomics
    // ------------------------------------------------------------------
    float2 val;
    val.x = local_sum;
    val.y = local_sumsq;

    const unsigned int FULL_MASK = 0xffffffff;

    // Intra-warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val.x += __shfl_down_sync(FULL_MASK, val.x, offset);
        val.y += __shfl_down_sync(FULL_MASK, val.y, offset);
    }

    // Shared memory to accumulate per-warp results
    __shared__ float2 warp_red[32];                       // supports up to 1024 threads
    const int warp_id = threadIdx.x >> 5;                 // 0..31
    const int lane_id = threadIdx.x & 31;

    if (lane_id == 0) {                                   // first lane writes
        warp_red[warp_id] = val;
    }
    __syncthreads();

    // First warp accumulates the per-warp partials
    if (warp_id == 0) {
        val = (threadIdx.x < ((blockDim.x + 31) >> 5)) ? warp_red[lane_id] : make_float2(0.f, 0.f);

        // Reduce within first warp
        for (int offset = 16; offset > 0; offset >>= 1) {
            val.x += __shfl_down_sync(FULL_MASK, val.x, offset);
            val.y += __shfl_down_sync(FULL_MASK, val.y, offset);
        }

        if (lane_id == 0) {
            const int partial_idx = ((n * num_blocks_per_sample) + b_in_sample) * 2;
            partial[partial_idx    ] = val.x;   // single, non-contended store
            partial[partial_idx + 1] = val.y;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel #2 : one block per sample – reduces the partial sums to
//             final mean and inv_std  (unchanged)
////////////////////////////////////////////////////////////////////////////////
__global__ void layernorm_finalize_kernel(const float* __restrict__ partial,
                                          float*       __restrict__ mean,
                                          float*       __restrict__ inv_std,
                                          int D,
                                          int num_blocks_per_sample,
                                          float eps)
{
    const int n = blockIdx.x;  // sample id
    if (threadIdx.x == 0) {
        float sum   = 0.f;
        float sumsq = 0.f;
        for (int b = 0; b < num_blocks_per_sample; ++b) {
            const int idx = ((n * num_blocks_per_sample) + b) * 2;
            sum   += partial[idx    ];
            sumsq += partial[idx + 1];
        }
        float m   = sum / D;
        float var = sumsq / D - m * m;
        mean[n]    = m;
        inv_std[n] = rsqrtf(var + eps);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Kernel #3 : normalise + optional affine transform (float4 I/O)
////////////////////////////////////////////////////////////////////////////////
__global__ __launch_bounds__(256, 4)     // ≤ 4 CTAs/SM, 256 threads/CTA
void layernorm_apply_kernel(const float* __restrict__ input,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            const float* __restrict__ mean,
                            const float* __restrict__ inv_std,
                            float*       __restrict__ output,
                            int D,
                            int num_blocks_per_sample)
{
    const int global_block = blockIdx.x;              // 0 .. N*B-1
    const int tid          = threadIdx.x;
    const int n            = global_block / num_blocks_per_sample;
    const int b_in_sample  = global_block % num_blocks_per_sample;

    const int D4    = D >> 2;
    const int base  = n * D;
    const int base4 = n * D4;

    __shared__ float stats[2];
    if (tid == 0) { stats[0] = mean[n]; stats[1] = inv_std[n]; }
    __syncthreads();
    const float m = stats[0];
    const float inv_s = stats[1];

    const float4* input4  = reinterpret_cast<const float4*>(input);
    float4*       output4 = reinterpret_cast<float4*>(output);

    const float4* gamma4  = gamma != nullptr ? reinterpret_cast<const float4*>(gamma) : nullptr;
    const float4* beta4   = beta  != nullptr ? reinterpret_cast<const float4*>(beta)  : nullptr;

    // ------------------------------------------------------------------
    // Vectorised loop (float4)
    // ------------------------------------------------------------------
    for (int idx4 = b_in_sample * blockDim.x + tid;
         idx4 < D4;
         idx4 += blockDim.x * num_blocks_per_sample)
    {
        float4 v4 = input4[base4 + idx4];

        float4 y4;
        // component-wise normalisation
        y4.x = (v4.x - m) * inv_s;
        y4.y = (v4.y - m) * inv_s;
        y4.z = (v4.z - m) * inv_s;
        y4.w = (v4.w - m) * inv_s;

        if (gamma4 != nullptr) {
            float4 g4 = gamma4[idx4];
            y4.x *= g4.x;  y4.y *= g4.y;  y4.z *= g4.z;  y4.w *= g4.w;
        }
        if (beta4 != nullptr) {
            float4 b4 = beta4[idx4];
            y4.x += b4.x;  y4.y += b4.y;  y4.z += b4.z;  y4.w += b4.w;
        }

        output4[base4 + idx4] = y4;
    }

    // ------------------------------------------------------------------
    // Tail loop (scalar)
    // ------------------------------------------------------------------
    for (int idx = D4 * 4 + b_in_sample * blockDim.x + tid;
         idx < D;
         idx += blockDim.x * num_blocks_per_sample)
    {
        float y = (input[base + idx] - m) * inv_s;
        if (gamma != nullptr) y *= gamma[idx];
        if (beta  != nullptr) y += beta[idx];
        output[base + idx] = y;
    }
}

////////////////////////////////////////////////////////////////////////////////
// C++/CUDA launcher (unchanged)
////////////////////////////////////////////////////////////////////////////////
