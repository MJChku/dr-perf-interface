#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
struct Shape{int N,C,H,W,O,K,S,P;};
int main(int argc,char**argv){try{
 if(argc!=4)return 2; drperf::gpu::Module mod(argv[1]);bool after=std::string(argv[2])=="after",held=std::string(argv[3])=="heldout";
 std::vector<Shape> shapes=held?std::vector<Shape>{{1,3,13,17,96,3,2,1},{2,3,9,11,96,3,1,1},{1,3,17,19,17,3,2,1},{1,3,32,32,96,11,4,2}}:std::vector<Shape>{{1,1,4,4,32,1,1,0},{1,1,4,8,64,1,1,0},{1,1,8,8,96,1,1,0},{1,1,6,6,32,3,1,0},{1,1,6,10,64,3,1,0},{1,3,6,10,96,3,1,0},{1,1,8,32,96,1,1,0}};
 double worst=0;int cases=0;
 for(auto s:shapes)for(int pattern=0;pattern<3;++pattern){
 int N=s.N,C=s.C,H=s.H,W=s.W,O=s.O,Kh=s.K,Kw=s.K,Sh=s.S,Sw=s.S,Ph=s.P,Pw=s.P;
 int Ho=(H+2*Ph-Kh)/Sh+1,Wo=(W+2*Pw-Kw)/Sw+1,M=O,NP=N*Ho*Wo,K=C*Kh*Kw;
 int TM=after?96:128,threads=after?192:256;int gx=(NP+127)/128,gy=(M+TM-1)/TM;
 drperf::gpu::Buffer<float>x(size_t(N)*C*H*W),w(size_t(O)*C*Kh*Kw),bias(O),y(size_t(N)*O*Ho*Wo+16,-12345.f);
 std::mt19937 gen(1234+NP+K+O+pattern);std::uniform_real_distribution<float>rnd(-0.25f,0.25f);
 // Binary fractions are represented exactly in FP16: oracle remains the FP32 convolution.
 for(auto&v:x)v=pattern==1?0.125f:pattern==2?rnd(gen):float(int(rnd(gen)*256))/256;
 for(auto&v:w)v=pattern==1?-0.25f:pattern==2?rnd(gen):float(int(rnd(gen)*256))/256;
 for(auto&v:bias)v=rnd(gen);
 std::vector<float>xo(x.begin(),x.end()),wo(w.begin(),w.end()),bo(bias.begin(),bias.end());
 auto xp=x.data(),wp=w.data(),bp=bias.data(),yp=y.data();void*args[]={&xp,&wp,&bp,&yp,&N,&C,&H,&W,&O,&Kh,&Kw,&Sh,&Sw,&Ph,&Pw,&Ho,&Wo,&M,&NP,&K};
 {drperf::gpu::Region r(argv[2],{{"outputs",int64_t(N)*O*Ho*Wo},{"channels",O},{"pixels",NP},{"reduction_elements",K},{"k_tiles",(K+15)/16},{"spatial_tiles",gx},{"tile_products",gx*gy*((K+15)/16)}});
 mod.launch("_Z32conv2d_implicit_gemm_wmma_kernelPKfS0_S0_Pfiiiiiiiiiiiiiiii",dim3(gx,gy),threads,args);}
 for(int n=0;n<N;++n)for(int o=0;o<O;++o)for(int h=0;h<Ho;++h)for(int j=0;j<Wo;++j){double want=bias[o];
 for(int c=0;c<C;++c)for(int kh=0;kh<Kh;++kh)for(int kw=0;kw<Kw;++kw){int ih=h*Sh-Ph+kh,iw=j*Sw-Pw+kw;if(ih>=0&&ih<H&&iw>=0&&iw<W)want+=double(x[((n*C+c)*H+ih)*W+iw])*w[((o*C+c)*Kh+kh)*Kw+kw];}
 double got=y[((n*O+o)*Ho+h)*Wo+j];double err=std::abs(got-want);worst=std::max(worst,err);if(!std::isfinite(got)||err>(pattern==2?2e-3:2e-5)*(1+std::abs(want))){std::cerr<<"O="<<O<<" K="<<K<<" NP="<<NP<<" index="<<((n*O+o)*Ho+h)*Wo+j<<" want="<<want<<" got="<<got<<'\n';throw std::runtime_error("convolution oracle failed");}}
 for(size_t i=size_t(N)*O*Ho*Wo;i<y.size();++i)if(y[i]!=-12345.f)throw std::runtime_error("guard overwritten");
 if(!std::equal(xo.begin(),xo.end(),x.begin())||!std::equal(wo.begin(),wo.end(),w.begin())||!std::equal(bo.begin(),bo.end(),bias.begin()))throw std::runtime_error("input modified");++cases;
 }
 std::cout<<"{\"cases\":"<<cases<<",\"max_absolute_error\":"<<worst<<",\"passed\":true}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
