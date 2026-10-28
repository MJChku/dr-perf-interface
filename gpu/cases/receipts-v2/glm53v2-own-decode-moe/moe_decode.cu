#include <cuda_fp16.h>
#include <cmath>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "moe_decode.h"

namespace moedec {
namespace {
constexpr int kWarps = 16;      // one block per SM
constexpr int kStep = 128;      // K values per warp step: lane t of a row group reads 16 B = 32 values at 32t
constexpr int kS1 = 4;          // gemm1 K slices per (slot, 16-row tile)
constexpr int kU1 = 2, kU2 = 4; // steps per load round (gemm1 reads 4 rows per lane, gemm2 2)

__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void red2(float* p, float a, float b) {  // (red.global.add.v2.f32: not in GX's translator)
  atomicAdd(p, a);
  atomicAdd(p + 1, b);
}
__device__ __forceinline__ __half2 sf_h2(uint8_t s) {
  const __half h(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
  return __half2half2(h);
}
// 16 B of packed e2m1 (32 values, low nibble first) times its two block scales -> 16 f16x2 (values 2i, 2i+1)
__device__ __forceinline__ void dq32(uint4 w, uint16_t sc, uint32_t* r) {
  const __half2 h0 = sf_h2(uint8_t(sc & 0xff)), h1 = sf_h2(uint8_t(sc >> 8));
  const uint32_t wd[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t((wd[i >> 2] >> (8 * (i & 3))) & 0xff),
                                                      __NV_E2M1);
    __half2 v = __half2(hr);
    v = __hmul2(v, i < 8 ? h0 : h1);
    r[i] = *reinterpret_cast<uint32_t*>(&v);
  }
}
__device__ __forceinline__ uint8_t f2e4m3(float v) { return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3); }

// Dequantize 32 fp4 values (16 B) + 2 scales into 64 B of f16 at dst.
__device__ __forceinline__ void dq_store(const uint8_t* q, const uint8_t* sf, __half* dst) {
  uint32_t r[16];
  dq32(*reinterpret_cast<const uint4*>(q), uint16_t(sf[0] | (sf[1] << 8)), r);
  uint4* d = reinterpret_cast<uint4*>(dst);
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = make_uint4(r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]);
}
}  // namespace
}  // namespace moedec

using namespace moedec;

// One block, warp m handles token m: top-k of sigmoid(logit) + bias (ties to the lower index), weights normalized
// and scaled; thread 0 then lists the distinct local experts (slots) and the per-token coefficients. If no local
// expert is selected, nothing else writes the output, so this block zeroes it.
extern "C" __global__ void __launch_bounds__(256) glm_moe_dec_route_k(const float* logits, const float* bias,
                                                                      const float* g2a, int M, int E, int local_off,
                                                                      int local_num, int topk, float scaling,
                                                                      int* n_sel, int* slot_e, float* coef,
                                                                      __nv_bfloat16* out, int H) {
  __shared__ int ns_s;
  __shared__ int sel[kMaxT][16];
  __shared__ float wts[kMaxT][16];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp < M) {
    float sc[8], ch[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = lane + 32 * i;
      if (e < E) {
        const float l = logits[size_t(warp) * E + e];
        sc[i] = 1.f / (1.f + expf(-l));
        ch[i] = sc[i] + bias[e];
      } else {
        sc[i] = 0.f;
        ch[i] = -INFINITY;
      }
    }
    float sum = 0.f;
    for (int k = 0; k < topk; ++k) {
      float bv = -INFINITY;
      int bi = 1 << 30;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (ch[i] > bv) bv = ch[i], bi = lane + 32 * i;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
      }
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        if (lane + 32 * i == bi) s = sc[i], ch[i] = -INFINITY;
      s = __shfl_sync(0xffffffffu, s, bi & 31);
      sum += s;
      if (lane == 0) sel[warp][k] = bi, wts[warp][k] = s;
    }
    __syncwarp();
    if (lane < topk) wts[warp][lane] = wts[warp][lane] / (sum + 1e-20f) * scaling;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int n = 0;
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < topk; ++k) {
        const int e = sel[m][k] - local_off;
        if (e < 0 || e >= local_num) continue;
        int j = 0;
        while (j < n && slot_e[j] != e) ++j;
        if (j == n) {
          slot_e[n++] = e;
          for (int t = 0; t < kMaxT; ++t) coef[j * kMaxT + t] = 0.f;
        }
        coef[j * kMaxT + m] += wts[m][k] * g2a[e];
      }
    *n_sel = n;
    ns_s = n;
  }
  __syncthreads();
  if (ns_s == 0)
    for (int i = threadIdx.x; i < M * H; i += blockDim.x) out[i] = __float2bfloat16(0.f);
}

// gemm1: tasks = slot x (I/16 row tiles) x kS1 K slices, split evenly over the blocks; a warp streams the tile's
// 16 gate and 16 up rows (lane (g, t): rows g and g+8, 32 values at 32t of every 128-value step, dequantized to
// f16 straight into mma A fragments) against x in shared memory (same K permutation as the B fragments). Partial
// 16x8 tiles go to ws1 with vector atomics; the warp completing a tile's last slice applies SwiGLU, re-quantizes the
// 16 values per token (one NVFP4 block), stores their f16 value and re-zeroes its slot.
extern "C" __global__ void __launch_bounds__(kWarps * 32, 1)
    glm_moe_dec_gemm1_k(const uint8_t* __restrict__ x4, const uint8_t* __restrict__ x_sf,
                        const uint8_t* __restrict__ w13, const uint8_t* __restrict__ w13_sf,
                        const float* __restrict__ g1a, const float* __restrict__ g1c, const int* __restrict__ n_sel,
                        const int* __restrict__ slot_e, float* ws1, int* cnt1, __half* __restrict__ act, int M, int H,
                        int I) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* xs = reinterpret_cast<__half*>(smem);  // [M][H]
  for (int i = threadIdx.x; i < M * H / 32; i += blockDim.x) {
    const int m = i / (H / 32), c = i % (H / 32);
    dq_store(x4 + size_t(m) * H / 2 + c * 16, x_sf + size_t(m) * H / 16 + c * 2, xs + size_t(m) * H + c * 32);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int ns = *n_sel, tiles = I / 16, steps = H / kStep, per = steps / kS1;
  const int total = ns * tiles * kS1;
  const int b0 = int((long long)total * blockIdx.x / gridDim.x), b1 = int((long long)total * (blockIdx.x + 1) / gridDim.x);
  const bool va = g < M;
  const __half* xr = xs + size_t(va ? g : 0) * H + 32 * t;
  const size_t wrow = H / 2, srow = H / 16;
  for (int task = b0 + warp; task < b1; task += kWarps) {
    const int slice = task % kS1, unit = task / kS1, tile = unit % tiles, slot = unit / tiles;
    const int e = slot_e[slot];
    const uint8_t* wb = w13 + size_t(e) * 2 * I * wrow + 16 * t;
    const uint8_t* sb = w13_sf + size_t(e) * 2 * I * srow + 2 * t;
    const int rows[4] = {16 * tile + g, 16 * tile + g + 8, I + 16 * tile + g, I + 16 * tile + g + 8};
    float acc[4][4] = {};
    for (int s = slice * per; s < (slice + 1) * per; s += kU1) {
      uint4 wv[kU1][4];
      uint16_t sv[kU1][4];
#pragma unroll
      for (int u = 0; u < kU1; ++u)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          wv[u][r] = __ldcs(reinterpret_cast<const uint4*>(wb + rows[r] * wrow + (s + u) * 64));
          sv[u][r] = __ldcs(reinterpret_cast<const unsigned short*>(sb + rows[r] * srow + (s + u) * 8));
        }
#pragma unroll
      for (int u = 0; u < kU1; ++u) {
        uint32_t xb[16];
        if (va) {
          const uint4* xp = reinterpret_cast<const uint4*>(xr + (s + u) * kStep);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 v = xp[i];
            xb[4 * i] = v.x, xb[4 * i + 1] = v.y, xb[4 * i + 2] = v.z, xb[4 * i + 3] = v.w;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) xb[i] = 0u;
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {  // gate tile, up tile
          uint32_t ra[16], rb[16];
          dq32(wv[u][2 * h], sv[u][2 * h], ra);
          dq32(wv[u][2 * h + 1], sv[u][2 * h + 1], rb);
#pragma unroll
          for (int j = 0; j < 8; ++j)
            mma(acc[2 * h + (j & 1)], ra[2 * j], rb[2 * j], ra[2 * j + 1], rb[2 * j + 1], xb[2 * j], xb[2 * j + 1]);
        }
      }
    }
    float* wsu = ws1 + size_t(unit) * 256;  // [gate|up][16][8]
    if (2 * t < M) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float* a0 = acc[2 * h];
        const float* a1 = acc[2 * h + 1];
        red2(wsu + h * 128 + g * 8 + 2 * t, a0[0] + a1[0], a0[1] + a1[1]);
        red2(wsu + h * 128 + (g + 8) * 8 + 2 * t, a0[2] + a1[2], a0[3] + a1[3]);
      }
    }
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) last = atomicAdd(cnt1 + unit, 1) == kS1 - 1;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) continue;
    __threadfence();
    const int r = lane & 15, hf = lane >> 4;
    const float ga = g1a[e], gc = g1c[e];
    for (int m0 = 0; m0 < M; m0 += 2) {
      const int m = m0 + hf;
      float a = 0.f;
      if (m < M) {
        const float gv = __ldcg(wsu + r * 8 + m) * ga, uv = __ldcg(wsu + 128 + r * 8 + m) * gc;
        a = gv / (1.f + expf(-gv)) * uv;
      }
      float amax = fabsf(a);
#pragma unroll
      for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      const uint8_t s8 = f2e4m3(amax * __frcp_rn(6.f));
      const float sfv = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3)));
      const float os = sfv == 0.f ? 0.f : __frcp_rn(sfv);
      const __nv_fp4_storage_t q = __nv_cvt_float_to_fp4(a * os, __NV_E2M1, cudaRoundNearest);
      const float dq = __half2float(__half(__nv_cvt_fp4_to_halfraw(q, __NV_E2M1))) * sfv;
      if (m < M) act[(size_t(slot) * kMaxT + m) * I + 16 * tile + r] = __float2half_rn(dq);
    }
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; ++i) wsu[lane * 8 + i] = 0.f;
    if (lane == 0) cnt1[unit] = 0;
  }
}

// gemm2 + finalize: tasks = slot x (H/16 row tiles) over the full I, split evenly over the blocks (a block's range
// spans at most two slots, whose f16 activations it stages in shared memory). Each task's 16x8 partial, scaled per
// token by coef (routing weight x g2_alpha), is added into ws2; the warp completing a row tile's last slot writes
// the bf16 rows and re-zeroes it.
extern "C" __global__ void __launch_bounds__(kWarps * 32, 1)
    glm_moe_dec_gemm2_k(const __half* __restrict__ act, const uint8_t* __restrict__ w2,
                        const uint8_t* __restrict__ w2_sf, const int* __restrict__ n_sel,
                        const int* __restrict__ slot_e, const float* __restrict__ coef, float* ws2, int* cnt2,
                        __nv_bfloat16* __restrict__ out, int M, int H, int I) {
  extern __shared__ __align__(16) unsigned char smem[];
  __half* as = reinterpret_cast<__half*>(smem);  // [2][M][I]
  const int ns = *n_sel, tiles = H / 16, steps = I / kStep;
  const int total = ns * tiles;
  const int b0 = int((long long)total * blockIdx.x / gridDim.x), b1 = int((long long)total * (blockIdx.x + 1) / gridDim.x);
  const int sA = b0 / tiles, sB = b1 > b0 ? (b1 - 1) / tiles : sA;
  for (int i = threadIdx.x; i < (sB - sA + 1) * M * I / 8; i += blockDim.x) {
    const int q = i / (M * I / 8), rem = i % (M * I / 8), m = rem / (I / 8), c = rem % (I / 8);
    reinterpret_cast<uint4*>(as + (size_t(q) * M + m) * I)[c] =
        reinterpret_cast<const uint4*>(act + (size_t(sA + q) * kMaxT + m) * I)[c];
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const bool va = g < M;
  const size_t wrow = I / 2, srow = I / 16;
  for (int task = b0 + warp; task < b1; task += kWarps) {
    const int tile = task % tiles, slot = task / tiles;
    const int e = slot_e[slot];
    const uint8_t* wb = w2 + (size_t(e) * H + 16 * tile + g) * wrow + 16 * t;
    const uint8_t* sb = w2_sf + (size_t(e) * H + 16 * tile + g) * srow + 2 * t;
    const __half* ar = as + (size_t(slot - sA) * M + (va ? g : 0)) * I + 32 * t;
    float acc[2][4] = {};
    for (int s = 0; s < steps; s += kU2) {
      uint4 wv[kU2][2];
      uint16_t sv[kU2][2];
#pragma unroll
      for (int u = 0; u < kU2; ++u)
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const bool in = s + u < steps;
          wv[u][r] = in ? __ldcs(reinterpret_cast<const uint4*>(wb + r * 8 * wrow + (s + u) * 64)) : make_uint4(0, 0, 0, 0);
          sv[u][r] = in ? __ldcs(reinterpret_cast<const unsigned short*>(sb + r * 8 * srow + (s + u) * 8)) : 0;
        }
#pragma unroll
      for (int u = 0; u < kU2; ++u) {
        if (s + u >= steps) break;
        uint32_t xb[16];
        if (va) {
          const uint4* xp = reinterpret_cast<const uint4*>(ar + (s + u) * kStep);
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const uint4 v = xp[i];
            xb[4 * i] = v.x, xb[4 * i + 1] = v.y, xb[4 * i + 2] = v.z, xb[4 * i + 3] = v.w;
          }
        } else {
#pragma unroll
          for (int i = 0; i < 16; ++i) xb[i] = 0u;
        }
        uint32_t ra[16], rb[16];
        dq32(wv[u][0], sv[u][0], ra);
        dq32(wv[u][1], sv[u][1], rb);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          mma(acc[j & 1], ra[2 * j], rb[2 * j], ra[2 * j + 1], rb[2 * j + 1], xb[2 * j], xb[2 * j + 1]);
      }
    }
    float* wst = ws2 + size_t(tile) * 128;
    if (2 * t < M) {
      const float c0 = coef[slot * kMaxT + 2 * t], c1 = coef[slot * kMaxT + 2 * t + 1];
      red2(wst + g * 8 + 2 * t, (acc[0][0] + acc[1][0]) * c0, (acc[0][1] + acc[1][1]) * c1);
      red2(wst + (g + 8) * 8 + 2 * t, (acc[0][2] + acc[1][2]) * c0, (acc[0][3] + acc[1][3]) * c1);
    }
    __threadfence();
    __syncwarp();
    int last = 0;
    if (lane == 0) last = atomicAdd(cnt2 + tile, 1) == ns - 1;
    last = __shfl_sync(0xffffffffu, last, 0);
    if (!last) continue;
    __threadfence();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int o = lane * 4 + i, r = o >> 3, m = o & 7;
      if (m < M) out[size_t(m) * H + 16 * tile + r] = __float2bfloat16(__ldcg(wst + o));
      wst[o] = 0.f;
    }
    if (lane == 0) cnt2[tile] = 0;
  }
}

namespace moedec {
void init(Work& w, const Shape& sh, cudaStream_t s) {
  const size_t n1 = size_t(kMaxSlots) * (sh.I / 16);
  cudaMalloc(&w.n_sel, 4);
  cudaMalloc(&w.slot_e, kMaxSlots * 4);
  cudaMalloc(&w.coef, kMaxSlots * kMaxT * 4);
  cudaMalloc(&w.ws1, n1 * 256 * 4);
  cudaMalloc(&w.cnt1, n1 * 4);
  cudaMalloc(&w.act, size_t(kMaxSlots) * kMaxT * sh.I * 2);
  cudaMalloc(&w.ws2, size_t(sh.H / 16) * 128 * 4);
  cudaMalloc(&w.cnt2, size_t(sh.H / 16) * 4);
  cudaMemsetAsync(w.n_sel, 0, 4, s);
  cudaMemsetAsync(w.ws1, 0, n1 * 256 * 4, s);
  cudaMemsetAsync(w.cnt1, 0, n1 * 4, s);
  cudaMemsetAsync(w.act, 0, size_t(kMaxSlots) * kMaxT * sh.I * 2, s);
  cudaMemsetAsync(w.ws2, 0, size_t(sh.H / 16) * 128 * 4, s);
  cudaMemsetAsync(w.cnt2, 0, size_t(sh.H / 16) * 4, s);
}

void run(const Shape& sh, const Work& w, int T, const float* logits, const float* bias, const uint8_t* x4,
         const uint8_t* x_sf, const uint8_t* w13, const uint8_t* w13_sf, const uint8_t* w2, const uint8_t* w2_sf,
         const float* g1_alpha, const float* g1_scale_c, const float* g2_alpha, __nv_bfloat16* out, cudaStream_t s) {
  static int sms = 0;
  if (!sms) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaFuncSetAttribute(glm_moe_dec_gemm1_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
    cudaFuncSetAttribute(glm_moe_dec_gemm2_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
  }
  glm_moe_dec_route_k<<<1, 256, 0, s>>>(logits, bias, g2_alpha, T, sh.experts, sh.local_offset, sh.local_num,
                                       sh.top_k, sh.routed_scaling, w.n_sel, w.slot_e, w.coef, out, sh.H);
  glm_moe_dec_gemm1_k<<<sms, kWarps * 32, size_t(T) * sh.H * 2, s>>>(x4, x_sf, w13, w13_sf, g1_alpha, g1_scale_c,
                                                                     w.n_sel, w.slot_e, w.ws1, w.cnt1,
                                                                     static_cast<__half*>(w.act), T, sh.H, sh.I);
  glm_moe_dec_gemm2_k<<<sms, kWarps * 32, size_t(2) * T * sh.I * 2, s>>>(static_cast<const __half*>(w.act), w2,
                                                                         w2_sf, w.n_sel, w.slot_e, w.coef, w.ws2,
                                                                         w.cnt2, out, T, sh.H, sh.I);
}
}  // namespace moedec
