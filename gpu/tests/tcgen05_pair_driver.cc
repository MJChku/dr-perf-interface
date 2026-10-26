#include "drperf_gpu.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <stdexcept>
static float value(uint16_t x){uint32_t b=uint32_t(x)<<16;float f;std::memcpy(&f,&b,4);return f;}
static std::vector<uint16_t> canonical(const std::vector<uint16_t>& a,int rows){
 std::vector<uint16_t> out(a.size());
 for(int r=0;r<rows;++r)for(int k=0;k<16;++k)out[(r/128)*2048+(r%128/8)*128+(k/8)*64+(r%8)*8+k%8]=a[r*16+k];return out;
}
int main(int argc,char** argv){try{
 if(argc!=2)return 2;drperf::gpu::Module module(argv[1]);
 std::vector<uint16_t> a(256*16),b(128*16);
 for(size_t i=0;i<a.size();++i)a[i]=i%3==0?0xbf80:0x4000;
 for(size_t i=0;i<b.size();++i)b[i]=i%5==0?0xc000:0x3f80;
 auto pa=canonical(a,256);
 std::vector<uint16_t> pb(b.size());
 for(int v=0;v<2;++v)for(int r=0;r<64;++r)for(int k=0;k<16;++k)
  pb[v*1024+(r/8)*128+(k/8)*64+(r%8)*8+k%8]=b[(v*64+r)*16+k];
 for(int repeats:{1,2,4}){
  std::vector<float> output(256*128+2,-999.f);auto ap=pa.data(),bp=pb.data();auto out=output.data()+1;
  void* args[]={&ap,&bp,&out,&repeats};
  {drperf::gpu::Region r("tcgen-pair",{{"repeats",repeats}});module.launch_cluster("tcgen_bf16_pair",2,128,2,args);}
  if(output.front()!=-999.f||output.back()!=-999.f)throw std::runtime_error("pair output guard");
  for(int m=0;m<256;++m)for(int n=0;n<128;++n){float expected=0;
   for(int k=0;k<16;++k)expected+=value(a[m*16+k])*value(b[n*16+k]);expected*=repeats;
   if(out[m*128+n]!=expected){std::cerr<<"pair mismatch "<<m<<","<<n<<" got "<<out[m*128+n]<<" expected "<<expected<<"\n";return 1;}
  }
 }
 std::cout<<"paired-CTA BF16 oracle: 3 cases PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
