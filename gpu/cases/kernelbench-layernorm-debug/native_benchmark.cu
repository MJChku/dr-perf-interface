#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>
namespace baseline {
#include "before.cu"
}
namespace optimized {
#include "after.cu"
}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{float*p;size_t n;Device(size_t n):n(n){ck(cudaMalloc(&p,n*4));}~Device(){cudaFree(p);}void put(const std::vector<float>&v){ck(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice));}std::vector<float>get(){std::vector<float>v(n);ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}};
static float value(size_t i,unsigned seed){uint32_t z=uint32_t(i)^seed;z^=z>>16;z*=0x7feb352d;z^=z>>15;z*=0x846ca68b;z^=z>>16;return (float(z&65535)/16384.f)-2.f;}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(){try{std::cout<<std::setprecision(12)<<"{\"cases\":[";bool first=true;
 for(int D:{256,16384,4194304}){int N=16,B=std::max(1,std::min(32,(D+255)/256));size_t count=size_t(N)*D;
  std::vector<float>x(count),g(D),b(D),sent(count+16,-12345.f);
  #pragma omp parallel for
  for(size_t i=0;i<count;++i)x[i]=value(i,1234);
  for(int i=0;i<D;++i){g[i]=1.f+.1f*value(i,5678);b[i]=.1f*value(i,9753);}
  Device dx(count),dg(D),db(D),dy(count+16),partial(N*B*2),mean(N),inv(N);dx.put(x);dg.put(g);db.put(b);
  auto launch=[&](bool after){float eps=1.e-5f;if(after){optimized::layernorm_partial_sum_kernel<<<N*B,256>>>(dx.p,partial.p,D,B);optimized::layernorm_finalize_kernel<<<N,1>>>(partial.p,mean.p,inv.p,D,B,eps);optimized::layernorm_apply_kernel<<<N*B,256>>>(dx.p,dg.p,db.p,mean.p,inv.p,dy.p,D,B);}else{baseline::layernorm_partial_sum_kernel<<<N*B,256>>>(dx.p,partial.p,D,B);baseline::layernorm_finalize_kernel<<<N,1>>>(partial.p,mean.p,inv.p,D,B,eps);baseline::layernorm_apply_kernel<<<N*B,256>>>(dx.p,dg.p,db.p,mean.p,inv.p,dy.p,D,B);}ck(cudaGetLastError());};
  std::vector<double>means(N),scales(N);
  #pragma omp parallel for
  for(int n=0;n<N;++n){double m=0,v=0;for(int j=0;j<D;++j)m+=x[size_t(n)*D+j];m/=D;for(int j=0;j<D;++j){double z=x[size_t(n)*D+j]-m;v+=z*z;}means[n]=m;scales[n]=1/std::sqrt(v/D+1.e-5);}
  double errors[2]={};
  for(int after=0;after<2;++after){dy.put(sent);launch(after);ck(cudaDeviceSynchronize());auto y=dy.get();double err=0;int bad=0;
   #pragma omp parallel for reduction(max:err) reduction(+:bad)
   for(size_t i=0;i<count;++i){double want=(x[i]-means[i/D])*scales[i/D]*g[i%D]+b[i%D];double e=std::abs(double(y[i])-want);err=std::max(err,e);if(!std::isfinite(y[i])||e>3.e-4+3.e-4*std::abs(want))++bad;}
   for(size_t i=count;i<y.size();++i)if(y[i]!=-12345.f)++bad;if(bad)throw std::runtime_error("native LayerNorm oracle/guard failed");errors[after]=err;
  }
  for(int i=0;i<30;++i){launch(false);launch(true);}ck(cudaDeviceSynchronize());cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));std::vector<double>times[2];
  for(int round=0;round<11;++round)for(int order=0;order<2;++order){int a=(round+order)%2;ck(cudaEventRecord(start));for(int i=0;i<30;++i)launch(a);ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));times[a].push_back(ms*1000/30);}
  ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));if(!first)std::cout<<',';first=false;std::cout<<"{\"rows\":"<<N<<",\"columns\":"<<D<<",\"passed\":true,\"baseline_us\":"<<median(times[0])<<",\"after_us\":"<<median(times[1])<<",\"max_error\":["<<errors[0]<<','<<errors[1]<<"],\"rounds_us\":[";
  for(int a=0;a<2;++a){if(a)std::cout<<',';std::cout<<'[';for(size_t j=0;j<times[a].size();++j){if(j)std::cout<<',';std::cout<<times[a][j];}std::cout<<']';}std::cout<<"]}"<<std::flush;
 }
 std::cout<<"]}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
