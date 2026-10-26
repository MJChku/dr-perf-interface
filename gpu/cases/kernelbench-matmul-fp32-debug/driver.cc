#include "drperf_gpu.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
int main(int argc,char**argv){try{
 if(argc!=4)return 2;drperf::gpu::Module mod(argv[1]);bool held=std::string(argv[3])=="heldout";int cases=0,badcases=0;double worst=0;
 for(int N:held?std::vector<int>{128,384}:std::vector<int>{128,256})for(int pattern=0;pattern<(held?4:3);++pattern){
 if(held && ((N==128 && pattern!=3)||(N==384 && pattern!=0)))continue;
 drperf::gpu::Buffer<float>a(N*N),b(N*N),c(N*N+16,-12345.f);std::mt19937 gen(82+N+pattern);std::uniform_real_distribution<float>rnd(-.25f,.25f);
 for(int i=0;i<N*N;++i){a[i]=pattern==1?.125f:pattern==2?float(int(rnd(gen)*256))/256:rnd(gen);b[i]=pattern==1?-.125f:pattern==2?float(int(rnd(gen)*256))/256:rnd(gen);}
 // Every pair has exactly cancelling hi*hi and cross products;
 // their surviving low*low term is representable in FP32.
 if(pattern==3){std::fill(a.begin(),a.end(),0.f);std::fill(b.begin(),b.end(),0.f);float e=std::ldexp(1.f,-12);for(int row=0;row<N;++row)for(int k=0;k<N;k+=2){a[row*N+k]=1+e;a[row*N+k+1]=1;}for(int k=0;k<N;k+=2)for(int col=0;col<N;++col){b[k*N+col]=1+e;b[(k+1)*N+col]=-(1+2*e);}}
 auto ac=std::vector<float>(a.begin(),a.end()),bc=std::vector<float>(b.begin(),b.end());auto ap=a.data(),bp=b.data(),cp=c.data();void*args[]={&ap,&bp,&cp,&N};
 {drperf::gpu::Region r(argv[2],{{"outputs",int64_t(N)*N},{"input_elements_per_tile_grid",int64_t(N)*N*(N/128)},{"tile_products",int64_t(N/128)*(N/128)*(N/16)}});mod.launch("_Z18matmul_kernel_wmmaPKfS0_Pfi",dim3(N/128,N/128),256,args);}
 double maxerr=0;int failures=0;for(int i=0;i<N;++i)for(int j=0;j<N;++j){double want=0;for(int k=0;k<N;++k)want+=double(a[i*N+k])*b[k*N+j];double got=c[i*N+j],err=std::abs(got-want);maxerr=std::max(maxerr,err);double tol=2e-6*(1+std::abs(want));if(!std::isfinite(got)||err>tol)++failures;}
 for(int i=N*N;i<N*N+16;++i)if(c[i]!=-12345.f)throw std::runtime_error("guard overwritten");if(!std::equal(ac.begin(),ac.end(),a.begin())||!std::equal(bc.begin(),bc.end(),b.begin()))throw std::runtime_error("input modified");
 ++cases;badcases+=failures>0;worst=std::max(worst,maxerr);std::cout<<"{\"N\":"<<N<<",\"pattern\":"<<pattern<<",\"failures\":"<<failures<<",\"max_error\":"<<maxerr<<"}\n";
 }
 std::cout<<"{\"cases\":"<<cases<<",\"failed_cases\":"<<badcases<<",\"max_error\":"<<worst<<"}\n";return badcases && !(held && std::string(argv[2])=="after")?1:0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
