#include "drperf_gpu.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
struct Shape{int N,C,H,W,O,K,S,P,PK,PS;};
int main(int argc,char**argv){try{
 if(argc!=4)return 2;drperf::gpu::Module mod(argv[1]);bool after=std::string(argv[2])=="after",held=std::string(argv[3])=="heldout";int cases=0;double worst=0;
 for(auto s:held?std::vector<Shape>{{1,3,9,11,17,3,1,1,3,2},{2,1,6,8,16,1,1,0,2,1},{1,3,32,32,96,11,4,2,3,2}}:std::vector<Shape>{{1,1,4,4,16,1,1,0,2,2},{1,1,4,8,32,1,1,0,2,2},{1,1,6,6,16,3,1,0,3,1},{1,3,6,8,32,3,1,1,3,2}})for(int pattern=0;pattern<3;++pattern){
 int N=s.N,C=s.C,H=s.H,W=s.W,O=s.O,Kh=s.K,Kw=s.K,Sh=s.S,Sw=s.S,Ph=s.P,Pw=s.P,PKh=s.PK,PKw=s.PK,PSh=s.PS,PSw=s.PS;
 int Ho=(H+2*Ph-Kh)/Sh+1,Wo=(W+2*Pw-Kw)/Sw+1,Hp=(Ho-PKh)/PSh+1,Wp=(Wo-PKw)/PSw+1,M=O,NP=N*Ho*Wo,K=C*Kh*Kw;size_t count=size_t(N)*O*Hp*Wp;
 drperf::gpu::Buffer<float>x(size_t(N)*C*H*W),w(size_t(O)*C*Kh*Kw),bias(O),y(count+16,-12345.f);std::fill(y.begin(),y.begin()+count,after?0.f:-FLT_MAX);
 std::mt19937 gen(813+NP+O+pattern);std::uniform_real_distribution<float>rnd(-.25f,.25f);
 for(auto&v:x)v=pattern==0?float(int(rnd(gen)*256))/256:.125f;for(auto&v:w)v=pattern==0?float(int(rnd(gen)*256))/256:.125f;for(auto&v:bias)v=pattern==0?rnd(gen):pattern==1?-8.f:.125f;
 // Separate mathematical oracle computes original FP32 convolution and pooling.
 std::vector<double>conv(size_t(N)*O*Ho*Wo),reference(count,0.);long long updates=0,positive=0;
 for(int n=0;n<N;++n)for(int o=0;o<O;++o)for(int h=0;h<Ho;++h)for(int j=0;j<Wo;++j){double v=bias[o];for(int c=0;c<C;++c)for(int kh=0;kh<Kh;++kh)for(int kw=0;kw<Kw;++kw){int ih=h*Sh-Ph+kh,iw=j*Sw-Pw+kw;if(ih>=0&&ih<H&&iw>=0&&iw<W)v+=double(x[((n*C+c)*H+ih)*W+iw])*w[((o*C+c)*Kh+kh)*Kw+kw];}conv[((n*O+o)*Ho+h)*Wo+j]=std::max(0.,v);}
 for(int n=0;n<N;++n)for(int o=0;o<O;++o)for(int hp=0;hp<Hp;++hp)for(int wp=0;wp<Wp;++wp){double v=0;for(int kh=0;kh<PKh;++kh)for(int kw=0;kw<PKw;++kw){double z=conv[((n*O+o)*Ho+hp*PSh+kh)*Wo+wp*PSw+kw];v=std::max(v,z);++updates;positive+=z>0;}reference[((n*O+o)*Hp+hp)*Wp+wp]=v;}
 auto xo=std::vector<float>(x.begin(),x.end()),wo=std::vector<float>(w.begin(),w.end()),bo=std::vector<float>(bias.begin(),bias.end());auto xp=x.data(),wp=w.data(),bp=bias.data(),yp=y.data();void*args[]={&xp,&wp,&bp,&yp,&N,&C,&H,&W,&O,&Kh,&Kw,&Sh,&Sw,&Ph,&Pw,&Ho,&Wo,&M,&NP,&K,&PKh,&PKw,&PSh,&PSw,&Hp,&Wp};
 {drperf::gpu::Region r(argv[2],{{"conv_outputs",int64_t(N)*O*Ho*Wo},{"pool_outputs",int64_t(count)},{"pool_updates",updates},{"positive_updates",positive},{"tile_products",int64_t((NP+127)/128)*((O+127)/128)*((K+15)/16)}});mod.launch("_Z35fused_conv_relu_maxpool_wmma_kernelPKfS0_S0_Pfiiiiiiiiiiiiiiiiiiiiii",dim3((NP+127)/128,(O+127)/128),256,args);}
 for(size_t i=0;i<count;++i){double err=std::abs(y[i]-reference[i]);worst=std::max(worst,err);if(!std::isfinite(y[i])||err>2e-5*(1+std::abs(reference[i])))throw std::runtime_error("pool oracle failed: "+std::to_string(y[i])+" expected "+std::to_string(reference[i]));}
 for(size_t i=count;i<y.size();++i)if(y[i]!=-12345.f)throw std::runtime_error("guard overwritten");if(!std::equal(xo.begin(),xo.end(),x.begin())||!std::equal(wo.begin(),wo.end(),w.begin())||!std::equal(bo.begin(),bo.end(),bias.begin()))throw std::runtime_error("input modified");++cases;
 }
 std::cout<<"{\"cases\":"<<cases<<",\"max_error\":"<<worst<<",\"passed\":true}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
