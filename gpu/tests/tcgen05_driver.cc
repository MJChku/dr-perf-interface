#include "drperf_gpu.h"
#include <vector>
#include <cstring>
#include <stdexcept>
#include <iostream>
// Input canonicalization matches GX tests. Integer BF16 fixtures make the
// independent scalar dot products and repeated accumulation exact in FP32.
static std::vector<uint16_t> canon(const std::vector<uint16_t>&v,int rows){
 std::vector<uint16_t>o(v.size());
 for(int r=0;r<rows;r++)for(int k=0;k<16;k++)o[(r/8)*128+(k/8)*64+(r%8)*8+k%8]=v[r*16+k];return o;
}
static float value(uint16_t h){uint32_t u=uint32_t(h)<<16;float f;std::memcpy(&f,&u,4);return f;}
int main(int argc,char**argv){try{
 if(argc<2||argc>3)return 2;drperf::gpu::Module module(argv[1]);
 if(argc==3&&std::string(argv[2])=="group2"){
  drperf::gpu::Region r("unsupported-cluster");module.launch("tcgen_group2",dim3(1),dim3(128),nullptr);throw std::runtime_error("group2 unexpectedly supported");
 }
 const char*kernel=argc==3?"tcgen_bf16_mask":"tcgen_bf16";
 std::vector<uint16_t>a(128*16),b(64*16);
 for(size_t i=0;i<a.size();i++)a[i]=(i%3==0?0xbf80:0x4000);
 for(size_t i=0;i<b.size();i++)b[i]=(i%5==0?0xc000:0x3f80);
 auto pa=canon(a,128),pb=canon(b,64);auto*ap=pa.data();auto*bp=pb.data();
 for(int blocks:{1,2,3})for(int repeats:{1,2,4}){
  std::vector<float>out(size_t(blocks)*128*64+2,-999);float*op=out.data()+1;
  void*args[]={&ap,&bp,&op,&repeats};
  {drperf::gpu::Region r("tcgen-bf16",{{"mma_tiles",blocks*repeats},{"blocks",blocks},{"prior_tiles",blocks*(repeats-1)}});module.launch(kernel,dim3(blocks),dim3(128),args);}
  for(int block=0;block<blocks;block++)for(int m=0;m<128;m++)for(int n=0;n<64;n++){
   float expected=0;for(int k=0;k<16;k++)expected+=value(a[m*16+k])*value(b[n*16+k]);expected*=repeats;
   if(op[(block*128+m)*64+n]!=expected)throw std::runtime_error("tcgen BF16 scalar oracle mismatch");
  }
  if(out.front()!=-999||out.back()!=-999)throw std::runtime_error("output guard overwritten");
 }
 for(int blocks:{1,2})for(int repeats:{1,3}){
  std::vector<float>out(size_t(blocks)*128*64+2,-999);float*op=out.data()+1;
  void*args[]={&op,&repeats};
  {drperf::gpu::Region r("tcgen-nvfp4",{{"mma_tiles",blocks*repeats},{"blocks",blocks}});module.launch("tcgen_nvfp4",dim3(blocks),dim3(128),args);}
  for(size_t i=0;i<size_t(blocks)*128*64;i++)if(op[i]!=64.f*repeats)throw std::runtime_error("NVFP4 unit operands/scales oracle mismatch");
  if(out.front()!=-999||out.back()!=-999)throw std::runtime_error("NVFP4 guards");
 }
 std::cout<<"tcgen NVFP4: 4 block-scaled unit-matrix cases PASS\n";
 std::cout<<"tcgen BF16: 9 tile/repetition cases, independent oracle and guards PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
