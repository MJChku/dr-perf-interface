
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ __launch_bounds__(1024) void softmax_forward_kernel_vec4(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const int num_features) {
    extern __shared__ float sdata[];

    const int  row    = blockIdx.x;      // Each block handles one row
    const int  tid    = threadIdx.x;
    const int  stride = blockDim.x;

    const float* row_in  = input  + row * num_features;
    float*       row_out = output + row * num_features;

    // Cache up to 8 float4s per thread. Stream remaining vectors through
    // the original temporary-output path; keep the original scalar-tail path. The branch is uniform across the block.
    if ((num_features & 3) == 0 && stride == 1024) {
        float4 cached[8];
        const float4* in4 = reinterpret_cast<const float4*>(row_in);
        float4* out4 = reinterpret_cast<float4*>(row_out);
        const int num_vec = num_features / 4;
        float thread_max = -FLT_MAX;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            int idx = tid + slot * 1024;
            if (idx < num_vec) {
                float4 v = in4[idx];
                cached[slot] = v;
                thread_max = fmaxf(thread_max, v.x);
                thread_max = fmaxf(thread_max, v.y);
                thread_max = fmaxf(thread_max, v.z);
                thread_max = fmaxf(thread_max, v.w);
            }
        }
        for (int idx = 8192 + tid; idx < num_vec; idx += stride) {
            float4 v = in4[idx];
            thread_max = fmaxf(thread_max, v.x); thread_max = fmaxf(thread_max, v.y);
            thread_max = fmaxf(thread_max, v.z); thread_max = fmaxf(thread_max, v.w);
        }
        sdata[tid] = thread_max;
        __syncthreads();
        for (int offset = stride >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float other = sdata[tid + offset];
                if (other > sdata[tid]) sdata[tid] = other;
            }
            __syncthreads();
        }
        float row_max = sdata[0];
        __syncthreads();
        float thread_sum = 0;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            if (tid + slot * 1024 < num_vec) {
                float4 v = cached[slot];
                float4 ev = {__expf(v.x-row_max), __expf(v.y-row_max),
                             __expf(v.z-row_max), __expf(v.w-row_max)};
                cached[slot] = ev;
                thread_sum += ev.x + ev.y + ev.z + ev.w;
            }
        }
        for (int idx = 8192 + tid; idx < num_vec; idx += stride) {
            float4 v = in4[idx];
            float4 ev = {__expf(v.x-row_max), __expf(v.y-row_max),
                         __expf(v.z-row_max), __expf(v.w-row_max)};
            out4[idx] = ev;
            thread_sum += ev.x + ev.y + ev.z + ev.w;
        }
        sdata[tid] = thread_sum;
        __syncthreads();
        for (int offset = stride >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) sdata[tid] += sdata[tid + offset];
            __syncthreads();
        }
        float sum_exp = sdata[0] + 1e-8f;
        #pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            int idx = tid + slot * 1024;
            if (idx < num_vec) {
                float4 ev = cached[slot];
                ev.x /= sum_exp; ev.y /= sum_exp;
                ev.z /= sum_exp; ev.w /= sum_exp;
                out4[idx] = ev;
            }
        }
        for (int idx = 8192 + tid; idx < num_vec; idx += stride) {
            float4 ev = out4[idx];
            ev.x /= sum_exp; ev.y /= sum_exp; ev.z /= sum_exp; ev.w /= sum_exp;
            out4[idx] = ev;
        }
        return;
    }

    // Cast to float4 pointers for vectorized access
    const float4* in4  = reinterpret_cast<const float4*>(row_in);
    float4*       out4 = reinterpret_cast<float4*>(row_out);

    const int num_vec = num_features >> 2;      // num_features / 4
    const int rem     = num_features & 3;       // leftover elements (0-3)

    // ------------------------------------------------------------------
    // 1. Find maximum value in the row (for numerical stability)
    // ------------------------------------------------------------------
    float thread_max = -FLT_MAX;
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += stride) {
        float4 v = in4[vec_idx];
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
    }

    // Handle leftovers (if any)
    if (tid == 0 && rem) {
        int base = num_vec << 2;        // num_vec * 4
        for (int i = 0; i < rem; ++i) {
            thread_max = fmaxf(thread_max, row_in[base + i]);
        }
    }

    // Reduce within block to get the row maximum
    sdata[tid] = thread_max;
    __syncthreads();
    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = sdata[tid + offset];
            if (other > sdata[tid])
                sdata[tid] = other;
        }
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads(); // All threads must read the maximum before sdata is reused.

    // ------------------------------------------------------------------
    // 2. Compute exponentials and their sum
    // ------------------------------------------------------------------
    float thread_sum = 0.0f;
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += stride) {
        float4 v  = in4[vec_idx];
        float4 ev = {__expf(v.x - row_max),
                     __expf(v.y - row_max),
                     __expf(v.z - row_max),
                     __expf(v.w - row_max)};

        out4[vec_idx] = ev;
        thread_sum += ev.x + ev.y + ev.z + ev.w;
    }

    // Leftovers
    if (tid == 0 && rem) {
        int base = num_vec << 2;
        for (int i = 0; i < rem; ++i) {
            float val = __expf(row_in[base + i] - row_max);
            row_out[base + i] = val;
            thread_sum += val;
        }
    }

    // Block-wide reduction to obtain the sum of exponentials
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0] + 1e-8f;      // Avoid divide-by-zero

    // ------------------------------------------------------------------
    // 3. Normalize to obtain softmax output
    // ------------------------------------------------------------------
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += stride) {
        float4 ev = out4[vec_idx];
        ev.x /= sum_exp;
        ev.y /= sum_exp;
        ev.z /= sum_exp;
        ev.w /= sum_exp;
        out4[vec_idx] = ev;
    }

    if (tid == 0 && rem) {
        int base = num_vec << 2;
        for (int i = 0; i < rem; ++i) {
            row_out[base + i] /= sum_exp;
        }
    }
}

