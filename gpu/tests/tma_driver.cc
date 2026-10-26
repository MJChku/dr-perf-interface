// Host oracle adapted from GX tests/blackwell/tma_test.cu. No CUDA runtime.
#include "drperf_gpu.h"
#include "drperf_gpu_tensor_map.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
using namespace drperf_gpu_tensor_abi;
constexpr int kSmem=16384;
#define CHECK(x) do { if((x)!=0)throw std::runtime_error("host fixture operation failed"); } while(0)
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
static int cudaMemcpy(void* d,const void* s,size_t n,int){std::memcpy(d,s,n);return 0;}
static int cudaGetLastError(){return 0;} static int cudaDeviceSynchronize(){return 0;}
static int cudaFree(void* p){std::free(p);return 0;}
template<class T>static int cudaMalloc(T** p,size_t n){*p=static_cast<T*>(std::aligned_alloc(128,((n+127)/128)*128));return *p?0:1;}
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

static CUtensorMap encode(const Map& m, void* base) {
  CUtensorMap t;
  cuuint64_t dims[3] = {m.dims[0], m.dims[1], m.dims[2]}, strides[2] = {m.strides[1], m.strides[2]};
  cuuint32_t box[3] = {m.box[0], m.box[1], m.box[2]}, es[3] = {m.estride[0], m.estride[1], m.estride[2]};
  CUresult r = gx_tensor_map_encode_tiled(&t, m.type, 3, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, m.swz,
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


static uint64_t source_bytes(const Map& m,int x,int y,int z){
 uint64_t count=0;
 for(unsigned k2=0;k2<m.box[2];k2+=m.estride[2])for(unsigned k1=0;k1<m.box[1];k1+=m.estride[1])for(unsigned k0=0;k0<m.box[0];++k0){
  int64_t a=int64_t(x)+k0,b=int64_t(y)+k1,c=int64_t(z)+k2;
  if(a>=0&&uint64_t(a)<m.dims[0]&&b>=0&&uint64_t(b)<m.dims[1]&&c>=0&&uint64_t(c)<m.dims[2])count+=m.bits;
 }return (count+7)/8;
}
static void launch_load(drperf::gpu::Module& module,const Map& m,CUtensorMap map,int x,int y,int z,unsigned bytes,uint8_t* out){
 void* args[]={&map,&x,&y,&z,&bytes,&out};
 drperf::gpu::Region region("tma.load",{{"source_bytes",int64_t(source_bytes(m,x,y,z))},{"destination_bytes",bytes}});
 module.launch("load_kernel",1,128,args);
}
static void launch_store(drperf::gpu::Module& module,const Map& m,CUtensorMap map,int x,int y,int z){
 void* args[]={&map,&x,&y,&z};auto bytes=source_bytes(m,x,y,z);
 drperf::gpu::Region region("tma.store",{{"written_bytes",int64_t(bytes)}});
 module.launch("store_kernel",1,128,args);
}
static void launch_multicast(drperf::gpu::Module& module,const Map& m,CUtensorMap map,unsigned mask,int pair,unsigned bytes,uint8_t* out){
 void* args[]={&map,&mask,&pair,&bytes,&out};int destinations=__builtin_popcount(mask);
 drperf::gpu::Region region("tma.multicast",{{"source_bytes",int64_t(source_bytes(m,8,3,1))},{"destinations",destinations},{"destination_bytes",int64_t(bytes)*destinations}});
 module.launch_cluster("multicast_kernel",4,128,4,args);
}
int main(int argc,char** argv) {try{
  if(argc!=2)return 2;drperf::gpu::Module module(argv[1]);
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
    launch_load(module,t.m,map, t.c[0], t.c[1], t.c[2], bytes, dout);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(got.data(), dout, kSmem, cudaMemcpyDeviceToHost));
    report(t.name, same(expect_load(t.m, host, t.c, 0xa5), got.data(), kSmem, t.name));
  }

  // Store: bf16 128B swizzle box at a partially out-of-bounds corner.
  {
    Map m{CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 16, {104, 37, 3}, {0, 256, 256 * 37}, {64, 16, 2}, {1, 1, 1}, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE};
    const int c[3] = {56, 28, 1};
    CHECK(cudaMemcpy(dg, host.data(), gbytes, cudaMemcpyHostToDevice));
    launch_store(module,m,encode(m, dg), c[0], c[1], c[2]);
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
    launch_multicast(module,m,map,mask,pair,box_bytes(m),dout);
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
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
