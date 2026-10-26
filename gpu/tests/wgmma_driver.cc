// Adapted from GX tests/hopper/wgmma_test.cu; standalone CPU-only launch.
// gx-arch: compute_90a
// Warpgroup MMA (wgmma.mma_async, sm_90a) under GX strict PTX. Each case
// generates a PTX kernel for one instruction form, loads it through the driver
// API, and checks every accumulator register of every lane bit for bit against
// a host reference:
//   - kinds: f16 (f32 and f16 accumulators), bf16, tf32 (low 13 bits of each
//     input are set and must be ignored), e4m3/e5m2 in all four pairings
//     (f32 and f16 accumulators), s8/u8 in all four pairings (wrapping and
//     .satfinite);
//   - N from 8 to 256, A from a shared-memory descriptor or from registers;
//   - all four descriptor layouts (none/32B/64B/128B swizzle), K-major and the
//     MN-major (transposed) f16/bf16 layouts, padded LBO/SBO, a 128B pattern
//     starting off its 1024-byte boundary (descriptor base offset);
//   - scale-d as a predicate (0 with NaN-poisoned accumulators, 1), immediate
//     scale-d, imm-scale-a/b = -1;
//   - two MMAs in separate commit groups retired by wait_group 1 then 0;
//   - CTAs of two warpgroups with 1D and 3D thread blocks;
//   - cuFuncSetAttribute/cuFuncGetAttribute (dynamic shared-memory opt-in) on
//     the PTX-module functions.
// The host builds the shared-memory image from the canonical layouts of the
// PTX ISA ("Shared Memory Matrix Layout", CuTe shape:stride form) composed with
// CuTe Swizzle<B,4,3>, and the register fragments from CUTLASS's SM90 GMMA
// ALayout/CLayout tuples; none of this shares code with the emulator. Values
// are small exact numbers, so every sum is exact in the accumulator type and
// the reference is independent of the (unspecified) accumulation order.
#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>
static std::string output_directory,run_mode;
enum Fmt { F16, BF16, E4M3, E5M2, TF32, S8, U8 };
static const char* fmt_name[] = {"f16", "bf16", "e4m3", "e5m2", "tf32", "s8", "u8"};
static int bits_of(Fmt f) { return f == TF32 ? 32 : (f == F16 || f == BF16) ? 16 : 8; }
static bool is_int(Fmt f) { return f == S8 || f == U8; }
enum Layout { NONE = 0, B128 = 1, B64 = 2, B32 = 3 };  // descriptor layout_type encoding

struct Operand {   // one descriptor operand
  Layout layout;
  bool mn_major;
  uint32_t start, lbo, sbo, base_offset;
};

struct Case {
  const char* name;
  int n;
  Fmt a, b;
  int dtype;            // 0 f16, 1 f32, 2 s32
  bool reg_a;
  Operand da, db;
  int neg_a, neg_b;     // imm-scale -1?
  int scale_d;          // 0, 1 (predicate), 2 (immediate 1), 3 (immediate 0)
  bool satfinite;
  bool chain;           // second MMA in its own group, wait_group 1 then 0
  unsigned bx, by, bz;  // CTA shape (128 or 256 threads)
};

// ---- encodings of exact small values ---------------------------------------
static uint32_t f32_bits(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }
static uint32_t encode(Fmt f, double v, std::mt19937& rng) {
  if (is_int(f)) return uint32_t(int(v)) & 0xff;
  const float x = float(v);
  const uint32_t u = f32_bits(x), s = u >> 31, e = (u >> 23) & 0xff, m = u & 0x7fffff;
  if (x == 0) return f == TF32 ? (rng() & 0x1fff) : 0;  // tf32: low bits must be ignored
  switch (f) {
  case F16: return (s << 15) | ((e - 127 + 15) << 10) | (m >> 13);
  case BF16: return u >> 16;
  case E4M3: return (s << 7) | ((e - 127 + 7) << 3) | (m >> 20);
  case E5M2: return (s << 7) | ((e - 127 + 15) << 2) | (m >> 21);
  default: return u | (rng() & 0x1fff);
  }
}
static double draw(Fmt f, std::mt19937& rng, bool half_acc) {
  if (half_acc) return int(rng() % 7) - 3;  // f16 accumulators: every partial sum stays an exact integer
  switch (f) {
  case S8: return int(rng() % 256) - 128;
  case U8: return int(rng() % 256);
  case E5M2: return (int(rng() % 9) - 4) * (rng() % 2 ? 1.0 : 0.5);      // {-4..4} and halves
  case E4M3: return (int(rng() % 13) - 6) * (rng() % 2 ? 1.0 : 0.25);
  default: return (int(rng() % 17) - 8) * (rng() % 2 ? 1.0 : 0.5);
  }
}

// ---- canonical shared-memory layouts (PTX ISA table, CuTe notation) --------
// Element offset (in elements) of (mn, k) for an operand with T = 128/bits.
// K-major:  NONE ((8,m),(T,2k)):((1T,SBO),(1,LBO))   W-swizzled ((8,m),(T,2k)):((WT,SBO),(1,T))
// MN-major: NONE ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)) W-swizzled ((T,W,m),(8,k)):((1,T,LBO),(WT,SBO))
static long element_offset(const Operand& o, int bits, int mn, int k) {
  const long T = 128 / bits, eb = bits / 8;
  const long W = o.layout == B128 ? 8 : o.layout == B64 ? 4 : o.layout == B32 ? 2 : 1;
  const long SBO = o.sbo / eb, LBO = o.lbo / eb;
  if (!o.mn_major) {
    const long i0 = mn % 8, i1 = mn / 8, j0 = k % T, j1 = k / T;
    return o.layout == NONE ? i0 * T + i1 * SBO + j0 + j1 * LBO : i0 * W * T + i1 * SBO + j0 + j1 * T;
  }
  const long j0 = k % 8, j1 = k / 8;
  if (o.layout == NONE) return (mn % T) + (mn / T) * SBO + j0 * T + j1 * LBO;
  return (mn % T) + ((mn / T) % W) * T + (mn / (T * W)) * LBO + j0 * W * T + j1 * SBO;
}
// CuTe Swizzle<B,4,3>: yyy = bits [7,7+B) moved onto bits [4,4+B). A pattern
// whose repeat starts 128*base_offset bytes past its natural boundary is
// swizzled relative to its own start.
static uint32_t swizzled(const Operand& o, uint32_t address) {
  const int B = o.layout == B128 ? 3 : o.layout == B64 ? 2 : o.layout == B32 ? 1 : 0;
  if (!B) return address;
  const uint32_t phase = address - 128 * o.base_offset;
  const uint32_t mask = ((1u << B) - 1) << 7;
  return address ^ ((phase & mask) >> 3);
}
static uint32_t byte_address(const Operand& o, int bits, int mn, int k) {
  return swizzled(o, uint32_t(o.start + element_offset(o, bits, mn, k) * (bits / 8)));
}
static uint64_t descriptor(const Operand& o) {
  return uint64_t((o.start & 0x3ffff) >> 4) | (uint64_t((o.lbo & 0x3ffff) >> 4) << 16) |
         (uint64_t((o.sbo & 0x3ffff) >> 4) << 32) | (uint64_t(o.base_offset) << 49) | (uint64_t(o.layout) << 62);
}

// ---- CUTLASS SM90 GMMA register layouts (cute/atom/mma_traits_sm90_gmma.hpp)
// Index (thread, value) -> m + 64 * column, column = n (CLayout) or k (ALayout).
struct Tuple3 { int s0, s1, s2; long d0, d1, d2; };
static long apply(const Tuple3& t, int x) {
  return (x % t.s0) * t.d0 + ((x / t.s0) % t.s1) * t.d1 + (x / (t.s0 * t.s1)) * t.d2;
}
static const Tuple3 thr_c = {4, 8, 4, 128, 1, 16};                 // CLayout_64xN / ALayout_64x16 threads
static void c_coord(int lane, int v, int& m, int& n) {           // CLayout_64xN values (2,2,N/8):(64,8,512)
  const long idx = apply(thr_c, lane) + apply({2, 2, 1 << 20, 64, 8, 512}, v);
  m = int(idx % 64); n = int(idx / 64);
}
static void a_coord(Fmt f, int lane, int v, int& m, int& k) {
  long idx;
  if (bits_of(f) == 16) idx = apply(thr_c, lane) + apply({2, 2, 2, 64, 8, 512}, v);               // ALayout_64x16
  else if (bits_of(f) == 8) idx = apply({4, 8, 4, 256, 1, 16}, lane) + apply({4, 2, 2, 64, 8, 1024}, v);  // 64x32
  else idx = apply({4, 8, 4, 64, 1, 16}, lane) + apply({2, 2, 1, 8, 256, 0}, v);                 // 64x8
  m = int(idx % 64); k = int(idx / 64);
}

// ---- reference helpers -------------------------------------------------------
static double value_of(Fmt f, uint32_t code) {
  auto mini = [](uint32_t v, int eb, int mb, int bias) {
    const int s = (v >> (eb + mb)) & 1, e = (v >> mb) & ((1 << eb) - 1), m = v & ((1 << mb) - 1);
    const double r = e ? std::ldexp(1.0 + m / double(1 << mb), e - bias) : std::ldexp(m / double(1 << mb), 1 - bias);
    return s ? -r : r;
  };
  switch (f) {
  case F16: return mini(code, 5, 10, 15);
  case BF16: { float x; uint32_t u = code << 16; memcpy(&x, &u, 4); return x; }
  case E4M3: return mini(code, 4, 3, 7);
  case E5M2: return mini(code, 5, 2, 15);
  case TF32: { float x; uint32_t u = code & ~0x1fffu; memcpy(&x, &u, 4); return x; }  // PTX: low 13 bits dropped
  case S8: return double(int8_t(code));
  default: return double(code & 0xff);
  }
}
static uint16_t half_bits(double v) {  // normal-range values, round to nearest even
  if (v == 0) return std::signbit(v) ? 0x8000 : 0;
  int e; const double fr = std::frexp(std::fabs(v), &e);  // v = fr * 2^e, fr in [0.5,1)
  uint32_t mant = uint32_t(std::nearbyint(std::ldexp(fr, 11)));
  if (mant == 2048) { mant = 1024; ++e; }
  return uint16_t((v < 0 ? 0x8000 : 0) | ((e - 1 + 15) << 10) | (mant & 0x3ff));
}

static std::string ptx_for(const Case& c, int nd, uint64_t desc_a, uint64_t desc_b, unsigned image_words) {
  std::string regs, d;
  for (int i = 0; i < nd; ++i) d += (i ? "," : "") + std::string("%d") + std::to_string(i);
  const int K = 256 / bits_of(c.a);
  std::string shape = "m64n" + std::to_string(c.n) + "k" + std::to_string(K);
  const char* dt = c.dtype == 0 ? "f16" : c.dtype == 1 ? "f32" : "s32";
  std::string op = "wgmma.mma_async.sync.aligned." + shape + (c.satfinite ? ".satfinite" : "") + "." + dt + "." +
                   fmt_name[c.a] + "." + fmt_name[c.b];
  std::string scale = c.scale_d == 0 || c.scale_d == 1 ? "%sd" : c.scale_d == 2 ? "1" : "0";
  std::string tail;
  if (!is_int(c.a)) tail += std::string(", ") + (c.neg_a ? "-1" : "1") + ", " + (c.neg_b ? "-1" : "1");
  if (c.a == F16 || c.a == BF16) {
    if (!c.reg_a) tail += std::string(", ") + (c.da.mn_major ? "1" : "0");
    tail += std::string(", ") + (c.db.mn_major ? "1" : "0");
  }
  std::string mma = op + " {" + d + "}, " + (c.reg_a ? "{%a0,%a1,%a2,%a3}" : "%desca") + ", %descb, " + scale + tail + ";\n";
  std::string s;
  s += ".version 8.7\n.target sm_90a\n.address_size 64\n.extern .shared .align 1024 .b8 smem[];\n";
  s += ".visible .entry wg(.param .u64 img, .param .u64 areg, .param .u64 cin, .param .u64 dout, .param .u32 sdv) {\n";
  s += ".reg .b32 %d<" + std::to_string(nd) + ">;\n.reg .b32 %a<4>;\n.reg .b32 %r<16>;\n.reg .b64 %rd<16>;\n"
       ".reg .b64 %desca, %descb;\n.reg .pred %p<4>, %sd;\n";
  s += "mov.u32 %r0, %tid.x;\nmov.u32 %r1, %tid.y;\nmov.u32 %r2, %tid.z;\nmov.u32 %r3, %ntid.x;\nmov.u32 %r4, %ntid.y;\n"
       "mov.u32 %r5, %ntid.z;\nmad.lo.u32 %r6, %r2, %r4, %r1;\nmad.lo.u32 %r6, %r6, %r3, %r0;\n"   // linear tid
       "mul.lo.u32 %r7, %r3, %r4;\nmul.lo.u32 %r7, %r7, %r5;\n";                                     // threads
  s += "ld.param.u64 %rd0, [img];\nld.param.u64 %rd1, [areg];\nld.param.u64 %rd2, [cin];\nld.param.u64 %rd3, [dout];\n"
       "ld.param.u32 %r8, [sdv];\nsetp.ne.u32 %sd, %r8, 0;\nmov.u32 %r9, smem;\nmov.u32 %r10, %r6;\n";
  s += "COPY:\nsetp.ge.u32 %p0, %r10, " + std::to_string(image_words) + ";\n@%p0 bra COPIED;\n"
       "mul.wide.u32 %rd4, %r10, 4;\nadd.u64 %rd5, %rd0, %rd4;\nld.global.u32 %r11, [%rd5];\n"
       "shl.b32 %r12, %r10, 2;\nadd.u32 %r12, %r12, %r9;\nst.shared.u32 [%r12], %r11;\n"
       "add.u32 %r10, %r10, %r7;\nbra COPY;\nCOPIED:\nbar.sync 0;\n";
  // descriptors are relative to the 1024-aligned dynamic shared base
  s += "cvt.u64.u32 %rd6, %r9;\nshr.u64 %rd6, %rd6, 4;\n";
  char buf[160];
  snprintf(buf, sizeof buf, "add.u64 %%desca, %%rd6, %llu;\nadd.u64 %%descb, %%rd6, %llu;\n",
           (unsigned long long)desc_a, (unsigned long long)desc_b);
  s += buf;
  s += "mul.wide.u32 %rd7, %r6, " + std::to_string(4 * nd) + ";\nadd.u64 %rd8, %rd2, %rd7;\nadd.u64 %rd9, %rd3, %rd7;\n";
  for (int i = 0; i < nd; ++i) s += "ld.global.u32 %d" + std::to_string(i) + ", [%rd8+" + std::to_string(4 * i) + "];\n";
  s += "mul.wide.u32 %rd10, %r6, 16;\nadd.u64 %rd11, %rd1, %rd10;\n";
  for (int i = 0; i < 4; ++i) s += "ld.global.u32 %a" + std::to_string(i) + ", [%rd11+" + std::to_string(4 * i) + "];\n";
  s += "fence.proxy.async.shared::cta;\nwgmma.fence.sync.aligned;\n" + mma + "wgmma.commit_group.sync.aligned;\n";
  if (c.chain) {
    std::string again = mma;
    again.replace(again.find(", " + scale + tail), scale.size() + 2, ", 1");
    s += again + "wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 1;\n";
  }
  s += "wgmma.wait_group.sync.aligned 0;\n";
  for (int i = 0; i < nd; ++i) s += "st.global.u32 [%rd9+" + std::to_string(4 * i) + "], %d" + std::to_string(i) + ";\n";
  s += "ret;\n}\n";
  return s;
}

static bool run(const Case& c, std::mt19937& rng) {
  const int K = 256 / bits_of(c.a), N = c.n;
  const int nd = c.dtype == 0 ? N / 4 : N / 2;
  const unsigned threads = c.bx * c.by * c.bz, groups = threads / 128;
  std::vector<double> A(64 * K), B(N * K);
  std::vector<uint32_t> Acode(64 * K), Bcode(N * K);
  for (int i = 0; i < 64 * K; ++i) { A[i] = draw(c.a, rng, c.dtype == 0); Acode[i] = encode(c.a, A[i], rng); A[i] = value_of(c.a, Acode[i]); }
  for (int i = 0; i < N * K; ++i) { B[i] = draw(c.b, rng, c.dtype == 0); Bcode[i] = encode(c.b, B[i], rng); B[i] = value_of(c.b, Bcode[i]); }
  // shared image
  uint32_t top = 0;
  std::vector<uint8_t> image(1 << 18, 0);
  std::vector<int> owner(1 << 18, -1);
  auto place = [&](const Operand& o, int bits, int mn, int k, uint32_t code, int tag) {
    const uint32_t at = byte_address(o, bits, mn, k);
    for (int b = 0; b < bits / 8; ++b) {
      if (owner[at + b] != -1) { printf("layout collision in %s\n", c.name); exit(3); }
      owner[at + b] = tag;
      image[at + b] = uint8_t(code >> (8 * b));
    }
    top = std::max(top, at + bits / 8);
  };
  if (!c.reg_a) for (int m = 0; m < 64; ++m) for (int k = 0; k < K; ++k) place(c.da, bits_of(c.a), m, k, Acode[m * K + k], 0);
  for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) place(c.db, bits_of(c.b), n, k, Bcode[n * K + k], 1);
  const unsigned words = (top + 3) / 4;
  // A registers per lane (register-A form)
  std::vector<uint32_t> areg(threads * 4, 0);
  const int per = 32 / bits_of(c.a);
  for (unsigned t = 0; t < threads; ++t)
    for (int v = 0; v < 4 * per; ++v) {
      int m, k; a_coord(c.a, t % 128, v, m, k);
      areg[t * 4 + v / per] |= (bits_of(c.a) == 32 ? Acode[m * K + k] : Acode[m * K + k] << (bits_of(c.a) * (v % per)));
    }
  // incoming accumulators: exact small values, distinct per warpgroup; NaN
  // poison when scale-d is false (it must not be read)
  const bool read_c = c.scale_d == 1 || c.scale_d == 2;
  std::vector<uint32_t> cin(threads * nd), want(threads * nd);
  std::vector<double> cval(groups * 64 * N);
  for (unsigned g = 0; g < groups; ++g)
    for (int i = 0; i < 64 * N; ++i)
      cval[g * 64 * N + i] = c.dtype == 2 ? double(int32_t(rng() % 2001) - 1000) : (int(rng() % 33) - 16) * 0.5;
  if (c.satfinite || (c.dtype == 2 && c.name[0] == 'w'))   // drive s32 overflow: wrap or clamp
    for (auto& v : cval) v = v >= 0 ? 2147483000.0 : -2147483000.0;
  for (unsigned t = 0; t < threads; ++t) {
    const unsigned g = t / 128, lane = t % 128;
    for (int i = 0; i < nd; ++i) {
      uint32_t in = 0, out = 0;
      for (int h = 0; h < (c.dtype == 0 ? 2 : 1); ++h) {
        const int v = c.dtype == 0 ? 2 * i + h : i;
        int m, n; c_coord(lane, v, m, n);
        const double cv = cval[g * 64 * N + m * N + n];
        double sum = 0;
        for (int k = 0; k < K; ++k) sum += A[m * K + k] * B[n * K + k];
        if (c.neg_a) sum = -sum;
        if (c.neg_b) sum = -sum;
        if (c.chain) sum *= 2;
        const double r = (read_c ? cv : 0) + sum;
        if (c.dtype == 2) {
          long long q = (long long)r;
          if (c.satfinite) q = q > INT32_MAX ? INT32_MAX : q < INT32_MIN ? INT32_MIN : q;
          in = uint32_t(int32_t(cv)); out = uint32_t(q);
          if (!read_c) in = 0x7fc01234u;
        } else if (c.dtype == 1) {
          in = read_c ? f32_bits(float(cv)) : 0x7fc01234u;
          out = f32_bits(float(r));
        } else {
          in |= uint32_t(read_c ? half_bits(cv) : 0x7e12) << (16 * h);
          out |= uint32_t(half_bits(r)) << (16 * h);
        }
      }
      cin[t * nd + i] = in;
      want[t * nd + i] = out;
    }
  }
  const std::string text=ptx_for(c,nd,descriptor(c.da),descriptor(c.db),words);
  const auto path=std::filesystem::path(output_directory)/c.name;
  if(run_mode=="emit") { std::ofstream(path.string()+".ptx")<<text;return true; }
  drperf::gpu::Module module((path.string()+".so").c_str());
  auto* d_img=image.data();auto* d_areg=areg.data();auto* d_cin=cin.data();
  std::vector<uint32_t> guarded(cin.size()+2,0xaabbccddu);auto* d_out=guarded.data()+1;
  unsigned sdv=c.scale_d==1?1:0;
  void* args[]={&d_img,&d_areg,&d_cin,&d_out,&sdv};
  unsigned shared=((top+1023)/1024)*1024+1024;
  unsigned threads_run=run_mode=="partial"?32:c.bx;
  {drperf::gpu::Region region((std::string("wgmma.")+c.name).c_str(),
        {{"N",N},{"K",K},{"groups",groups},{"operations",c.chain?2:1},
         {"register_a",int(c.reg_a)},{"integer",int(is_int(c.a))}});
   module.launch("wg",1,dim3(threads_run,c.by,c.bz),args,shared);}
  if(guarded.front()!=0xaabbccddu||guarded.back()!=0xaabbccddu)throw std::runtime_error("WGMMA output guard");
  std::vector<uint32_t> got(d_out,d_out+cin.size());
  int bad = 0;
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != want[i] && bad++ < 4)
      printf("  %s: thread %zu reg %zu got %08x want %08x\n", c.name, i / nd, i % nd, got[i], want[i]);
  printf("%s %s (N=%d, %zu registers)\n", bad ? "FAIL" : "PASS", c.name, N, got.size());
  return !bad;
}

int main(int argc,char** argv) {try{
  if(argc!=3)return 2;
  run_mode=argv[1];output_directory=argv[2];
  if(run_mode=="emit")std::filesystem::create_directories(output_directory);
  // Operand placements: K-major rows of 32 bytes (one K slice).
  // swizzled K-major: SBO = 8 rows x W x 16 B (one atom), optionally padded
  auto kmaj = [](Layout l, uint32_t start, uint32_t pad = 0) {
    const uint32_t W = l == B128 ? 8 : l == B64 ? 4 : l == B32 ? 2 : 1;
    if (l == NONE) return Operand{l, false, start, 128 + pad, 256 + 2 * pad, 0};  // core matrices of 8 x 16 B
    return Operand{l, false, start, 16, 128 * W + pad, 0};
  };
  // MN-major 16-bit operand with `rows` rows (T = 8 elements per 16 B chunk)
  auto mnmaj = [](Layout l, uint32_t start, int rows) {
    const uint32_t W = l == B128 ? 8 : l == B64 ? 4 : l == B32 ? 2 : 1;
    const uint32_t chunks = (rows + 7) / 8, atoms = (chunks + W - 1) / W;
    if (l == NONE) return Operand{l, true, start, 128 * chunks, 128, 0};
    return Operand{l, true, start, 128 * W, 128 * W * atoms, 0};
  };
  const uint32_t A0 = 0, B0 = 32768;  // B placed at 32 KiB (1024-aligned)
  std::vector<Case> cases = {
    // name, N, a, b, dtype, regA, da, db, negA, negB, scaleD, sat, chain, bx, by, bz
    {"bf16_n8_sw128", 8, BF16, BF16, 1, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"bf16_n32_sw32", 32, BF16, BF16, 1, false, kmaj(B32, A0), kmaj(B32, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"bf16_n64_sw64_pad", 64, BF16, BF16, 1, false, kmaj(B64, A0, 512), kmaj(B64, B0, 1536), 1, 0, 1, false, false, 128, 1, 1},
    {"bf16_n128_none", 128, BF16, BF16, 1, false, kmaj(NONE, A0), kmaj(NONE, B0, 16), 0, 1, 1, false, false, 128, 1, 1},
    {"bf16_n256_sw128_scaled0", 256, BF16, BF16, 1, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 0, false, false, 128, 1, 1},
    {"bf16_n200_sw128_regA", 200, BF16, BF16, 1, true, kmaj(B128, A0), kmaj(B128, B0), 1, 1, 1, false, false, 128, 1, 1},
    {"bf16_n64_transA_sw128", 64, BF16, BF16, 1, false, mnmaj(B128, A0, 64), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"bf16_n64_transB_sw64", 64, BF16, BF16, 1, false, kmaj(B64, A0), mnmaj(B64, B0, 64), 0, 0, 1, false, false, 128, 1, 1},
    {"bf16_n40_transAB_sw32", 40, BF16, BF16, 1, false, mnmaj(B32, A0, 64), mnmaj(B32, B0, 40), 0, 1, 2, false, false, 128, 1, 1},
    {"bf16_n24_transAB_none", 24, BF16, BF16, 1, false, mnmaj(NONE, A0, 64), mnmaj(NONE, B0, 24), 0, 0, 1, false, false, 128, 1, 1},
    {"bf16_n128_transB_sw128_regA", 128, BF16, BF16, 1, true, kmaj(B128, A0), mnmaj(B128, B0, 128), 0, 0, 1, false, false, 128, 1, 1},
    {"f16_n16_f32_sw128", 16, F16, F16, 1, false, kmaj(B128, A0), kmaj(B128, B0), 0, 1, 1, false, false, 128, 1, 1},
    {"f16_n72_f16acc_sw64", 72, F16, F16, 0, false, kmaj(B64, A0), kmaj(B64, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"f16_n248_f16acc_transB", 248, F16, F16, 0, false, kmaj(B128, A0), mnmaj(B128, B0, 248), 1, 0, 0, false, false, 128, 1, 1},
    {"f16_n96_f16acc_regA", 96, F16, F16, 0, true, kmaj(B32, A0), kmaj(B32, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"tf32_n16_sw128", 16, TF32, TF32, 1, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"tf32_n120_none_regA", 120, TF32, TF32, 1, true, kmaj(NONE, A0), kmaj(NONE, B0), 1, 0, 1, false, false, 128, 1, 1},
    {"tf32_n256_sw64", 256, TF32, TF32, 1, false, kmaj(B64, A0), kmaj(B64, B0), 0, 1, 3, false, false, 128, 1, 1},
    {"e4m3_e4m3_n128_sw128", 128, E4M3, E4M3, 1, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"e4m3_e5m2_n64_f16acc", 64, E4M3, E5M2, 0, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"e5m2_e4m3_n8_regA", 8, E5M2, E4M3, 1, true, kmaj(B32, A0), kmaj(B32, B0), 1, 1, 1, false, false, 128, 1, 1},
    {"e5m2_e5m2_n232_none", 232, E5M2, E5M2, 1, false, kmaj(NONE, A0), kmaj(NONE, B0), 0, 1, 0, false, false, 128, 1, 1},
    {"e4m3_e4m3_n256_f16acc_regA", 256, E4M3, E4M3, 0, true, kmaj(B64, A0), kmaj(B64, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"s8_s8_n32_sw128", 32, S8, S8, 2, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"u8_u8_n48_regA", 48, U8, U8, 2, true, kmaj(B64, A0), kmaj(B64, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"s8_u8_n256_none", 256, S8, U8, 2, false, kmaj(NONE, A0), kmaj(NONE, B0), 0, 0, 0, false, false, 128, 1, 1},
    {"u8_s8_n24_sw32", 24, U8, S8, 2, false, kmaj(B32, A0), kmaj(B32, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"s8_s8_n64_satfinite", 64, S8, S8, 2, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, true, false, 128, 1, 1},
    {"wrap_u8_u8_n16", 16, U8, U8, 2, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, false, 128, 1, 1},
    {"bf16_n64_chain_wait1", 64, BF16, BF16, 1, false, kmaj(B128, A0), kmaj(B128, B0), 0, 0, 1, false, true, 128, 1, 1},
    {"e4m3_n128_chain_regA_2wg", 128, E4M3, E4M3, 1, true, kmaj(B128, A0), kmaj(B128, B0), 0, 1, 1, false, true, 256, 1, 1},
    {"bf16_n96_2wg_3d", 96, BF16, BF16, 1, false, kmaj(B64, A0), kmaj(B64, B0), 0, 0, 1, false, false, 32, 4, 2},
  };
  // 128B K-major pattern starting 384 bytes past its 1024-byte boundary
  Operand off_a = kmaj(B128, A0 + 384); off_a.base_offset = 3;
  Operand off_b = kmaj(B128, B0 + 640); off_b.base_offset = 5;
  cases.push_back({"bf16_n32_sw128_base_offset", 32, BF16, BF16, 1, false, off_a, off_b, 0, 0, 1, false, false, 128, 1, 1});
  std::mt19937 rng(20261022);
  int failures = 0;
  for (const Case& c : cases) failures += !run(c, rng);
  printf("wgmma_test: %s (%zu cases, %d failed)\n", failures ? "FAIL" : "PASS", cases.size(), failures);
  return failures ? 1 : 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
