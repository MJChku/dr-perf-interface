#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#ifndef CASE_TWO
#define CASE_TWO 0
#endif
int main(int argc,char**argv){try{
 if(argc!=4)return 2; drperf::gpu::Module mod(argv[1]);
 bool held=std::string(argv[3])=="heldout";
 std::vector<int> ds=held?std::vector<int>{128,768,4100,8192}:std::vector<int>{256,1024,16384};
 if(CASE_TWO) ds=held?std::vector<int>{4096,8192}:std::vector<int>{128,256};
 int cases=0,failures=0; double worst=0;
 for(int N:{1,2})for(int D:ds)for(int pattern=0;pattern<3;++pattern){
  int B=std::max(1,std::min(32,(D+255)/256)); if(CASE_TWO)B=4;
  size_t count=size_t(N)*D;
  drperf::gpu::Buffer<float> x(count),g(D),b(D),y(count+16,-12345.f),partial(N*B*2),mean(N),inv(N);
  std::mt19937 gen(1234+D+N+pattern); std::uniform_real_distribution<float> rnd(-2,2);
  for(int j=0;j<D;++j){g[j]=1.f+0.1f*rnd(gen);b[j]=0.1f*rnd(gen);}
  for(size_t i=0;i<count;++i)x[i]=pattern==0?rnd(gen):pattern==1?3.f:float((i%D)/1024)*3.f+0.1f*rnd(gen);
  std::vector<float> original(x.begin(),x.end()),gorig(g.begin(),g.end()),borig(b.begin(),b.end());
  auto xp=x.data(),gp=g.data(),bp=b.data(),yp=y.data(),pp=partial.data(),mp=mean.data(),ip=inv.data();float eps=1.e-5f;
  {drperf::gpu::Region reg(argv[2],{{"elements",int64_t(count)},{"rows",N},{"row_blocks",N*B}});
   if(CASE_TWO){void*args[]={&xp,&gp,&bp,&yp,&D,&eps};mod.launch("_Z22fused_layernorm_kernelPKfS0_S0_Pfif",N*B,256,args,3072);}
   else {
    void*a[]={&xp,&pp,&D,&B};mod.launch("_Z28layernorm_partial_sum_kernelPKfPfii",N*B,256,a);
    void*c[]={&pp,&mp,&ip,&D,&B,&eps};mod.launch("_Z25layernorm_finalize_kernelPKfPfS1_iif",N,1,c);
    void*d[]={&xp,&gp,&bp,&mp,&ip,&yp,&D,&B};mod.launch("_Z22layernorm_apply_kernelPKfS0_S0_S0_S0_Pfii",N*B,256,d);
   }
  }
  bool ok=true;double err=0;
  for(int n=0;n<N;++n){double m=0,v=0;for(int j=0;j<D;++j)m+=x[n*D+j];m/=D;
   for(int j=0;j<D;++j){double z=x[n*D+j]-m;v+=z*z;}v/=D;
   for(int j=0;j<D;++j){double want=(x[n*D+j]-m)/std::sqrt(v+eps)*g[j]+b[j];double got=y[n*D+j];double e=std::abs(got-want);err=std::max(err,e);if(!std::isfinite(got)||e>3e-4+3e-4*std::abs(want))ok=false;}}
  for(size_t j=count;j<y.size();++j)if(y[j]!=-12345.f)throw std::runtime_error("guard overwritten");
  if(!std::equal(original.begin(),original.end(),x.begin())||!std::equal(gorig.begin(),gorig.end(),g.begin())||!std::equal(borig.begin(),borig.end(),b.begin()))throw std::runtime_error("input modified");
  if(!ok){++failures;std::cerr<<"numerical mismatch N="<<N<<" D="<<D<<" pattern="<<pattern<<" max_error="<<err<<"\n";}
  worst=std::max(worst,err);++cases;
 }
 std::cout<<"{\"cases\":"<<cases<<",\"failures\":"<<failures<<",\"max_absolute_error\":"<<worst<<",\"passed\":"<<(failures?"false":"true")<<"}\n";
 return failures?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
