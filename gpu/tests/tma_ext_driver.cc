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
#include <random>
static CUtensorMap map2d(CUtensorMapDataType type, unsigned esize, void* base, uint64_t w, uint64_t h, uint32_t bw, uint32_t bh) {
  CUtensorMap t;
  cuuint64_t dims[2] = {w, h}, strides[1] = {w * esize};
  cuuint32_t box[2] = {bw, bh}, es[2] = {1, 1};
  CUresult r = gx_tensor_map_encode_tiled(&t, type, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("encode failed %d\n", int(r)); exit(2); }
  return t;
}
template <class T> static T* dev(const std::vector<T>& v) {
  T* p; CHECK(cudaMalloc(&p, v.size() * sizeof(T))); CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
template <class T> static std::vector<T> host(const T* p, size_t n) {
  std::vector<T> v(n); CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
static int report(const char* name, int bad) {
  printf("{\"case\":\"%s\",\"status\":\"%s\",\"mismatches\":%d}\n", name, bad ? "fail" : "pass", bad);
  return bad;
}


static void launch_reduce(drperf::gpu::Module& m,const char* kernel,CUtensorMap map,const uint32_t* src,int x,int y){
 void* args[]={&map,&src,&x,&y};drperf::gpu::Region r(kernel);m.launch(kernel,1,128,args);
}
static void launch_k_gather_scatter(drperf::gpu::Module& m,CUtensorMap in,CUtensorMap out,int x,int r0,int r1,int r2,int r3,int q0,int q1,int q2,int q3,uint16_t* dump){
 void* args[]={&in,&out,&x,&r0,&r1,&r2,&r3,&q0,&q1,&q2,&q3,&dump};drperf::gpu::Region r("tma.gather_scatter");m.launch("k_gather_scatter",1,128,args);
}
static void launch_k_replace(drperf::gpu::Module& m,CUtensorMap map,void* other,uint32_t height,CUtensorMap* slot,uint32_t* dump){
 void* args[]={&map,&other,&height,&slot,&dump};drperf::gpu::Region r("tma.replace");m.launch("k_replace",1,128,args);
}
static void launch_k_placeholder(drperf::gpu::Module& m,CUtensorMap map,void* data,uint64_t row,CUtensorMap* slot,uint32_t* dump){
 void* args[]={&map,&data,&row,&slot,&dump};drperf::gpu::Region r("tma.placeholder");m.launch("k_placeholder",1,128,args);
}
static void launch_k_bulk(drperf::gpu::Module& m,const uint32_t* src,uint32_t* add,int32_t* max,uint8_t* masked,uint32_t mask){
 void* args[]={&src,&add,&max,&masked,&mask};drperf::gpu::Region r("tma.bulk");m.launch("k_bulk",1,128,args);
}
int main(int argc,char** argv) {try{
 if(argc!=2)return 2;drperf::gpu::Module module(argv[1]);
  for(int bytes:{0,3,16}){
   alignas(16) uint8_t input[16];for(int i=0;i<16;++i)input[i]=i+1;
   uint8_t output[50];std::memset(output,0xa5,sizeof output);auto p=input,q=output+1;void* args[]={&p,&q,&bytes};
   {drperf::gpu::Region r("async.tail",{{"source_bytes",4+(bytes?0:16)+bytes}});module.launch("k_async_tail",1,1,args);}
   if(output[0]!=0xa5||output[49]!=0xa5)throw std::runtime_error("async tail output guard");
   for(int i=0;i<48;++i){int offset=i%16;uint8_t want=(i<16?offset<4:i<32?bytes==0:offset<bytes)?input[offset]:0;
    if(q[i]!=want)throw std::runtime_error("async tail/ignore-source oracle");}
  }
  std::mt19937 rng(17);
  int total = 0;
  {  // 1a. add.f32 into a 48 x 20 tensor, box at (32, 16): columns 48.. and rows 20.. are out of bounds
    const int W = 48, H = 20, X = 32, Y = 16;
    std::vector<float> g(W * H), sv(256);
    for (auto& x : g) x = float(int(rng() % 200) - 100) * 0.25f;
    for (auto& x : sv) x = float(int(rng() % 200) - 100) * 0.125f;
    std::vector<uint32_t> s(256); memcpy(s.data(), sv.data(), 1024);
    float* dg = dev(g); uint32_t* ds = dev(s);
    launch_reduce(module,"k_reduce_f32",map2d(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, dg, W, H, 32, 8), ds, X, Y);
    CHECK(cudaDeviceSynchronize());
    auto got = host(dg, g.size());
    int bad = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        float want = g[y * W + x];
        if (x >= X && x < X + 32 && y >= Y && y < Y + 8) want += sv[(y - Y) * 32 + (x - X)];
        bad += got[y * W + x] != want;
      }
    total += report("cp.reduce.async.bulk.tensor add.f32 (partial OOB box)", bad);
  }
  {  // 1b. min.s32
    const int W = 64, H = 16, X = 16, Y = 4;
    std::vector<int32_t> g(W * H); std::vector<uint32_t> s(256);
    for (auto& x : g) x = int32_t(rng() % 2001) - 1000;
    for (auto& x : s) x = uint32_t(int32_t(rng() % 2001) - 1000);
    int32_t* dg = dev(g); uint32_t* ds = dev(s);
    launch_reduce(module,"k_reduce_s32",map2d(CU_TENSOR_MAP_DATA_TYPE_INT32, 4, dg, W, H, 32, 8), ds, X, Y);
    CHECK(cudaDeviceSynchronize());
    auto got = host(dg, g.size());
    int bad = 0;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        int32_t want = g[y * W + x];
        if (x >= X && x < X + 32 && y >= Y && y < Y + 8) want = std::min(want, int32_t(s[(y - Y) * 32 + (x - X)]));
        bad += got[y * W + x] != want;
      }
    total += report("cp.reduce.async.bulk.tensor min.s32", bad);
  }
  {  // 2. gather4 / scatter4 on a 128 x 40 u16 tensor; one requested row (r3 = 45) is out of bounds
    const int W = 128, H = 40, X = 32;
    const int r[4] = {7, 0, 31, 45}, q[4] = {3, 39, 12, 20};
    std::vector<uint16_t> g(W * H), o(W * H, 0xeeee);
    for (int i = 0; i < W * H; ++i) g[i] = uint16_t(i * 13 + 1);
    uint16_t* dg = dev(g); uint16_t* dout = dev(o); uint16_t* dd; CHECK(cudaMalloc(&dd, 512));
    launch_k_gather_scatter(module,map2d(CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, dg, W, H, 64, 1),
                                 map2d(CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, dout, W, H, 64, 1),
                                 X, r[0], r[1], r[2], r[3], q[0], q[1], q[2], q[3], dd);
    CHECK(cudaDeviceSynchronize());
    auto sm = host(dd, 256); auto out = host(dout, o.size());
    int bad = 0;
    for (int k = 0; k < 4; ++k)
      for (int c = 0; c < 64; ++c) bad += sm[k * 64 + c] != (r[k] < H ? g[r[k] * W + X + c] : 0);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        uint16_t want = 0xeeee;
        for (int k = 0; k < 4; ++k) if (y == q[k] && x >= X && x < X + 64) want = sm[k * 64 + x - X];
        bad += out[y * W + x] != want;
      }
    total += report("tile::gather4 load + tile::scatter4 store", bad);
  }
  {  // 3. tensormap.replace + cp_fenceproxy: rows 4.. of the new 6-row buffer, rows 6..11 out of bounds
    const int W = 32, H = 16, H2 = 6;
    std::vector<uint32_t> a(W * H, 1), b(W * H);
    for (int i = 0; i < W * H; ++i) b[i] = uint32_t(i * 3 + 7);
    uint32_t* da = dev(a); uint32_t* db = dev(b); uint32_t* dd; CHECK(cudaMalloc(&dd, 1024));
    CUtensorMap* slot; CHECK(cudaMalloc(&slot, sizeof(CUtensorMap)));
    launch_k_replace(module,map2d(CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, da, W, H, 32, 8), db, H2, slot, dd);
    CHECK(cudaDeviceSynchronize());
    auto got = host(dd, 256);
    int bad = 0;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 32; ++x) bad += got[y * 32 + x] != (4 + y < H2 ? b[(4 + y) * W + x] : 0u);
    total += report("tensormap.replace (address, global_dim) + cp_fenceproxy", bad);
  }
  {  // 4. plain bulk: add.u32, max.s32, cp_mask
    std::vector<uint32_t> s(64), add(64), mx(64);
    std::vector<uint8_t> m(256, 0x77);
    for (int i = 0; i < 64; ++i) { s[i] = uint32_t(int32_t(rng() % 2001) - 1000); add[i] = rng(); mx[i] = uint32_t(int32_t(rng() % 2001) - 1000); }
    const uint32_t byte_mask = 0xa5c3;
    uint32_t* ds = dev(s); uint32_t* dadd = dev(add); int32_t* dmx = reinterpret_cast<int32_t*>(dev(mx)); uint8_t* dm = dev(m);
    launch_k_bulk(module,ds, dadd, dmx, dm, byte_mask);
    CHECK(cudaDeviceSynchronize());
    auto ga = host(dadd, 64); auto gm = host(dmx, 64); auto gk = host(dm, 256);
    int bad = 0;
    for (int i = 0; i < 64; ++i) {
      bad += ga[i] != add[i] + s[i];
      bad += gm[i] != std::max(int32_t(mx[i]), int32_t(s[i]));
    }
    const uint8_t* sb = reinterpret_cast<const uint8_t*>(s.data());
    for (int i = 0; i < 256; ++i) bad += gk[i] != (((byte_mask >> (i % 16)) & 1) ? sb[i] : 0x77);
    total += report("cp.reduce.async.bulk add.u32/max.s32 + cp.async.bulk .cp_mask", bad);
  }
  {  // 5. placeholder map: null address and zero stride encode (as on the driver), completed in the kernel
    const int W = 32, H = 16;
    std::vector<uint32_t> b(W * H);
    for (int i = 0; i < W * H; ++i) b[i] = uint32_t(i * 5 + 3);
    uint32_t* db = dev(b); uint32_t* dd; CHECK(cudaMalloc(&dd, 1024));
    CUtensorMap* slot; CHECK(cudaMalloc(&slot, sizeof(CUtensorMap)));
    CUtensorMap t;
    cuuint64_t dims[2] = {W, H}, zero[1] = {0};
    cuuint32_t box[2] = {32, 8}, es[2] = {1, 1};
    int bad = 0;
    CUresult r = gx_tensor_map_encode_tiled(&t, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, nullptr, dims, zero, box, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    bad += r != CUDA_SUCCESS;
    CUtensorMap misaligned;
    bad += gx_tensor_map_encode_tiled(&misaligned, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, reinterpret_cast<char*>(db) + 4, dims, zero,
                       box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_ERROR_INVALID_VALUE;
    if (r == CUDA_SUCCESS) {
      launch_k_placeholder(module,t, db, W * 4, slot, dd);
      CHECK(cudaDeviceSynchronize());
      auto got = host(dd, 256);
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 32; ++x) bad += got[y * 32 + x] != b[(2 + y) * W + x];
    }
    total += report("placeholder map (null address, zero stride) completed by tensormap.replace", bad);
  }
  return total != 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
