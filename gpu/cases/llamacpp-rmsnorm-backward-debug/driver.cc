#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#ifndef CASE_RMS
#define CASE_RMS 1
#endif
int main(int argc,char**argv){try{
 if(argc!=4)return 2;drperf::gpu::Module mod(argv[1]);bool held=std::string(argv[3])=="heldout";
 std::vector<int>widths=held?std::vector<int>{33,255,767,1025,4097,16380,65536}:std::vector<int>{64,256,768,1024,4096,16384};
 int cases=0;double worst=0;float eps=1.e-5f;
 for(int rows:{1,2})for(int width:widths)for(int pattern=0;pattern<3;++pattern){
 int count=rows*width,threads=width<1024?32:1024;
 drperf::gpu::Buffer<float>x(count),g(count),y(count+16,-12345.f);
 std::mt19937 gen(1234+width+rows+pattern+(held?86753:0));std::uniform_real_distribution<float>rnd(-2,2);
 for(int i=0;i<count;++i){x[i]=pattern==0?rnd(gen):pattern==1?7.f:(i%width==width/2?10.f:0.f);g[i]=pattern==1?1.f:rnd(gen);}
 auto xp=x.data(),gp=g.data(),yp=y.data();std::vector<float>xo(x.begin(),x.end()),go(g.begin(),g.end());
 {std::string region_name=std::string(argv[2])+(threads==32?".warp":".cta");drperf::gpu::Region r(region_name.c_str(),{{"elements",count},{"rows",rows},{"launched_lanes",rows*threads},{"cached_elements",rows*std::min(width,8*threads)}});
  if(CASE_RMS){void*a[]={&gp,&xp,&yp,&width,&eps};mod.launch(threads==32?"_Z17rms_norm_back_f32ILi32EEvPKfS1_Pfif":"_Z17rms_norm_back_f32ILi1024EEvPKfS1_Pfif",rows,threads,a);}
  else {void*a[]={&xp,&yp,&width,&count,&eps};mod.launch(threads==32?"_Z14group_norm_f32ILi32EEvPKfPfiif":"_Z14group_norm_f32ILi1024EEvPKfPfiif",rows,threads,a,threads==32?0:256);}}
 for(int row=0;row<rows;++row){double mean=0,sumxx=0,sumxg=0,var=0;for(int j=0;j<width;++j){double v=x[row*width+j];mean+=v;sumxx+=v*v;sumxg+=v*g[row*width+j];}mean/=width;
  for(int j=0;j<width;++j){double d=x[row*width+j]-mean;var+=d*d;}var/=width;
  double inv=1/std::sqrt(sumxx/width+eps),sx=-inv*sumxg/(sumxx+width*eps);
  for(int j=0;j<width;++j){int i=row*width+j;double want=CASE_RMS?inv*g[i]+sx*x[i]:(x[i]-mean)/std::sqrt(var+eps);double got=y[i],err=std::abs(got-want);worst=std::max(worst,err);if(!std::isfinite(got)||err>4e-5+4e-5*std::abs(want)){std::cerr<<"rows="<<rows<<" width="<<width<<" pattern="<<pattern<<" index="<<i<<" want="<<want<<" got="<<got<<"\n";throw std::runtime_error("independent numerical oracle failed");}}}
 for(size_t i=count;i<y.size();++i)if(y[i]!=-12345.f)throw std::runtime_error("output guard overwritten");
 if(!std::equal(xo.begin(),xo.end(),x.begin())||!std::equal(go.begin(),go.end(),g.begin()))throw std::runtime_error("input modified");++cases;
 }
 std::cout<<"{\"cases\":"<<cases<<",\"max_absolute_error\":"<<worst<<",\"passed\":true}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
