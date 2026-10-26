// Copied GX tests/blackwell/tma_test.cu with C-linkage entry names.
// TMA (cp.async.bulk.tensor) under GX strict PTX: tile loads with every
// swizzle span, out-of-bounds fill, element strides, packed 4-bit data,
// multicast across a cluster with cta_group::2 mbarrier routing, and tile
// stores. The host computes each expected shared-memory image from the
// cuTensorMapEncodeTiled contract documented in cuda.h. Exit 0 = all pass.
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ void bar_init(uint64_t* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}
__device__ void bar_expect(uint64_t* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ void bar_wait(uint64_t* bar, unsigned parity) {
  asm volatile("{ .reg .pred p; WAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra WAIT; }"
               :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}

constexpr int kSmem = 16384;

// One CTA loads one box per launch and dumps the raw shared image.
extern "C" __global__ void load_kernel(const __grid_constant__ CUtensorMap map, int c0, int c1, int c2,
                            unsigned bytes, uint8_t* out) {
  __shared__ alignas(1024) uint8_t smem[kSmem];
  __shared__ alignas(8) uint64_t bar;
  for (int i = threadIdx.x; i < kSmem; i += blockDim.x) smem[i] = 0xa5;
  if (threadIdx.x == 0) { bar_init(&bar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  __syncthreads();
  if (threadIdx.x == 0) {
    bar_expect(&bar, bytes);
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
                 :: "r"(smem_u32(smem)), "l"(&map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(&bar)) : "memory");
  }
  bar_wait(&bar, 0);
  for (int i = threadIdx.x; i < kSmem; i += blockDim.x) out[i] = smem[i];
}

// Fill shared memory with a pattern and store one box.
extern "C" __global__ void store_kernel(const __grid_constant__ CUtensorMap map, int c0, int c1, int c2) {
  __shared__ alignas(1024) uint8_t smem[kSmem];
  for (int i = threadIdx.x; i < kSmem; i += blockDim.x) smem[i] = uint8_t(i * 7 + 3);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, {%1, %2, %3}], [%4];"
                 :: "l"(&map), "r"(c0), "r"(c1), "r"(c2), "r"(smem_u32(smem)) : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  }
}

// Cluster of 4: rank 0 multicasts one box to ranks in `mask`. With
// cta_group::2 the transaction bytes land on the even CTA of each pair.
extern "C" __global__ void multicast_kernel(const __grid_constant__ CUtensorMap map, unsigned mask, int pair,
                                 unsigned bytes, uint8_t* out) {
  __shared__ alignas(1024) uint8_t smem[kSmem];
  __shared__ alignas(8) uint64_t bar;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  for (int i = threadIdx.x; i < kSmem; i += blockDim.x) smem[i] = 0x5a;
  const bool receives = mask & (1u << rank);
  // Expected transactions per barrier: own delivery, or both pair members'.
  unsigned expect = 0;
  if (!pair) expect = receives ? bytes : 0;
  else if (!(rank & 1)) expect = bytes * (((mask >> rank) & 1) + ((mask >> (rank + 1)) & 1));
  if (threadIdx.x == 0) { bar_init(&bar, 1); asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  if (threadIdx.x == 0) bar_expect(&bar, expect);
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  if (rank == 0 && threadIdx.x == 0) {
    uint32_t mbar = smem_u32(&bar) & 0xFEFFFFFFu;
    uint16_t m16 = (uint16_t)mask;
    if (pair)
      asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"
                   :: "r"(smem_u32(smem)), "l"(&map), "r"(8), "r"(3), "r"(1), "r"(mbar), "h"(m16) : "memory");
    else
      asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4}], [%5], %6;"
                   :: "r"(smem_u32(smem)), "l"(&map), "r"(8), "r"(3), "r"(1), "r"(mbar), "h"(m16) : "memory");
  }
  bar_wait(&bar, 0);
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  for (int i = threadIdx.x; i < kSmem; i += blockDim.x) out[rank * kSmem + i] = smem[i];
}

// ---- host model -------------------------------------------------------------
struct Map {
  CUtensorMapDataType type; int bits; uint64_t dims[3]; uint64_t strides[3];  // strides[0] unused
  uint32_t box[3]; uint32_t estride[3]; CUtensorMapSwizzle swz; CUtensorMapFloatOOBfill oob;
};
static uint32_t swizzle(uint32_t a, CUtensorMapSwizzle s) {
  unsigned b = s == CU_TENSOR_MAP_SWIZZLE_32B ? 1 : s == CU_TENSOR_MAP_SWIZZLE_64B ? 2 : s == CU_TENSOR_MAP_SWIZZLE_128B ? 3 : 0;
  return a ^ ((a >> 3) & (((1u << b) - 1) << 4));
}
static uint64_t sbit(const Map& m, uint64_t i) {
  return m.type == CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B ? (i / 16) * 128 + (i % 16) * 4 : i * m.bits;
}
// complete_tx bytes: the data copied (padding of ALIGN16B types is not counted).
static unsigned box_bytes(const Map& m) {
  uint64_t n = 1; for (int i = 0; i < 3; ++i) n *= i ? (m.box[i] + m.estride[i] - 1) / m.estride[i] : m.box[0];
  return unsigned((n * m.bits + 7) / 8);
}
// Expected shared image (smem offset 0 aligned to 1024) for a load at c.
static std::vector<uint8_t> expect_load(const Map& m, const std::vector<uint8_t>& g, const int* c, uint8_t init) {
  std::vector<uint8_t> s(kSmem, init);
  uint64_t n[3] = {m.box[0], (m.box[1] + m.estride[1] - 1) / m.estride[1], (m.box[2] + m.estride[2] - 1) / m.estride[2]};
  uint64_t idx = 0;
  for (uint64_t k2 = 0; k2 < n[2]; ++k2) for (uint64_t k1 = 0; k1 < n[1]; ++k1) for (uint64_t k0 = 0; k0 < n[0]; ++k0, ++idx) {
    int64_t g0 = c[0] + int64_t(k0), g1 = c[1] + int64_t(k1 * m.estride[1]), g2 = c[2] + int64_t(k2 * m.estride[2]);
    bool in = g0 >= 0 && g0 < int64_t(m.dims[0]) && g1 >= 0 && g1 < int64_t(m.dims[1]) && g2 >= 0 && g2 < int64_t(m.dims[2]);
    uint64_t v = 0;
    if (in) {
      uint64_t bit = uint64_t(g0) * m.bits + (uint64_t(g1) * m.strides[1] + uint64_t(g2) * m.strides[2]) * 8;
      if (m.bits >= 8) memcpy(&v, &g[bit / 8], m.bits / 8); else v = (g[bit / 8] >> (bit % 8)) & 0xf;
    } else if (m.oob == CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA) v = m.bits == 16 ? 0x7fff : 0x7fffffff;
    uint64_t b = sbit(m, idx), at = swizzle(uint32_t(b / 8), m.swz);
    if (m.bits >= 8) memcpy(&s[at], &v, m.bits / 8);
    else s[at] = uint8_t((s[at] & ~(0xf << (b % 8))) | (v << (b % 8)));
  }
  return s;
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  void* fn = nullptr; cudaDriverEntryPointQueryResult q;
  CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q));
  return (PFN_cuTensorMapEncodeTiled_v12000)fn;
}
static CUtensorMap encode(const Map& m, void* base) {
  CUtensorMap t;
  cuuint64_t dims[3] = {m.dims[0], m.dims[1], m.dims[2]}, strides[2] = {m.strides[1], m.strides[2]};
  cuuint32_t box[3] = {m.box[0], m.box[1], m.box[2]}, es[3] = {m.estride[0], m.estride[1], m.estride[2]};
  CUresult r = encode_fn()(&t, m.type, 3, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, m.swz,
                           CU_TENSOR_MAP_L2_PROMOTION_NONE, m.oob);
  if (r != CUDA_SUCCESS) { printf("encode failed %d\n", int(r)); exit(2); }
  return t;
}

static int failures = 0;
static void report(const char* name, bool ok) {
  printf("{\"case\":\"%s\",\"status\":\"%s\"}\n", name, ok ? "pass" : "fail");
  failures += !ok;
}
static bool same(const std::vector<uint8_t>& a, const uint8_t* b, size_t n, const char* name) {
  for (size_t i = 0; i < n; ++i) if (a[i] != b[i]) {
    printf("  %s: first difference at byte %zu: got %02x want %02x\n", name, i, b[i], a[i]); return false;
  }
  return true;
}

int main() {
  // A padded 3D tensor: dims {104 elements, 37, 3}, row pitch 256 bytes.
  const size_t gbytes = 256 * 37 * 3 + 256;
  std::vector<uint8_t> host(gbytes);
  for (size_t i = 0; i < gbytes; ++i) host[i] = uint8_t(i * 131 + (i >> 8) * 17 + 1);
  uint8_t *dg, *dout;
  CHECK(cudaMalloc(&dg, gbytes)); CHECK(cudaMalloc(&dout, 4 * kSmem));
  CHECK(cudaMemcpy(dg, host.data(), gbytes, cudaMemcpyHostToDevice));
  std::vector<uint8_t> got(4 * kSmem);

  struct Case { const char* name; Map m; int c[3]; };
  const Case cases[] = {
    {"bf16 128B swizzle interior", {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 16, {104, 37, 3}, {0, 256, 256 * 37}, {64, 16, 2}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE}, {8, 5, 0}},
    {"bf16 128B swizzle OOB corner", {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 16, {104, 37, 3}, {0, 256, 256 * 37}, {64, 16, 2}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE}, {72, 30, 2}},
    {"f16 64B swizzle negative coords NaN fill", {CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 16, {104, 37, 3}, {0, 256, 256 * 37}, {32, 8, 1}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA}, {-8, -3, 1}},
    {"u8 32B swizzle", {CU_TENSOR_MAP_DATA_TYPE_UINT8, 8, {208, 37, 3}, {0, 256, 256 * 37}, {32, 24, 3}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_32B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE}, {16, 2, 0}},
    {"f32 no swizzle element strides", {CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 32, {52, 37, 3}, {0, 256, 256 * 37}, {12, 15, 3}, {1, 3, 2}, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE}, {4, 1, 0}},
    {"u4 align8B 128B swizzle", {CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 4, {416, 37, 3}, {0, 256, 256 * 37}, {256, 8, 1}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE}, {64, 7, 2}},
    {"u4 align16B 128B swizzle", {CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, 4, {512, 37, 3}, {0, 256, 256 * 37}, {128, 8, 1}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE}, {128, 4, 1}},
  };
  for (const Case& t : cases) {
    CUtensorMap map = encode(t.m, dg);
    const unsigned bytes = box_bytes(t.m);
    load_kernel<<<1, 128>>>(map, t.c[0], t.c[1], t.c[2], bytes, dout);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(), dout, kSmem, cudaMemcpyDeviceToHost));
    report(t.name, same(expect_load(t.m, host, t.c, 0xa5), got.data(), kSmem, t.name));
  }

  // Store: bf16 128B swizzle box at a partially out-of-bounds corner.
  {
    Map m{CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 16, {104, 37, 3}, {0, 256, 256 * 37}, {64, 16, 2}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE};
    const int c[3] = {56, 28, 1};
    CHECK(cudaMemcpy(dg, host.data(), gbytes, cudaMemcpyHostToDevice));
    store_kernel<<<1, 128>>>(encode(m, dg), c[0], c[1], c[2]);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    std::vector<uint8_t> want = host, out(gbytes);
    std::vector<uint8_t> smem(kSmem); for (int i = 0; i < kSmem; ++i) smem[i] = uint8_t(i * 7 + 3);
    uint64_t idx = 0;
    for (int k2 = 0; k2 < 2; ++k2) for (int k1 = 0; k1 < 16; ++k1) for (int k0 = 0; k0 < 64; ++k0, ++idx) {
      int g0 = c[0] + k0, g1 = c[1] + k1, g2 = c[2] + k2;
      if (g0 >= 104 || g1 >= 37 || g2 >= 3) continue;
      memcpy(&want[size_t(g0) * 2 + size_t(g1) * 256 + size_t(g2) * 256 * 37], &smem[swizzle(uint32_t(idx * 2), m.swz)], 2);
    }
    CHECK(cudaMemcpy(out.data(), dg, gbytes, cudaMemcpyDeviceToHost));
    report("bf16 128B swizzle store OOB corner", same(want, out.data(), gbytes, "store"));
    CHECK(cudaMemcpy(dg, host.data(), gbytes, cudaMemcpyHostToDevice));
  }

  // Multicast within a 4-CTA cluster, plain and cta_group::2.
  Map m{CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 16, {104, 37, 3}, {0, 256, 256 * 37}, {64, 32, 1}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE};
  const int c[3] = {8, 3, 1};
  CUtensorMap map = encode(m, dg);
  for (int pair = 0; pair < 2; ++pair) for (unsigned mask : {0xbu, 0xfu, 0x6u}) {
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(4); cfg.blockDim = dim3(128);
    cudaLaunchAttribute attr[1]; attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 4; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    CHECK(cudaLaunchKernelEx(&cfg, multicast_kernel, map, mask, pair, box_bytes(m), dout));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(), dout, 4 * kSmem, cudaMemcpyDeviceToHost));
    bool ok = true;
    std::vector<uint8_t> loaded = expect_load(m, host, c, 0x5a), untouched(kSmem, 0x5a);
    for (int r = 0; r < 4; ++r) ok &= same((mask >> r) & 1 ? loaded : untouched, &got[r * kSmem], kSmem, "multicast");
    char name[96]; snprintf(name, sizeof name, "multicast mask 0x%x%s", mask, pair ? " cta_group::2" : "");
    report(name, ok);
  }
  CHECK(cudaFree(dg)); CHECK(cudaFree(dout));
  return failures != 0;
}
