#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
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
#include "compatible.cu"
}
#undef WARPS_PER_BLOCK
#undef BLOCK_M_TILES_WMMA
namespace optimized {
#include "after.cu"
}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{float*p;size_t n;Device(size_t n):n(n){ck(cudaMalloc(&p,n*4));}~Device(){cudaFree(p);}void put(const std::vector<float>&v){ck(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice));}std::vector<float>get(){std::vector<float>v(n);ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}};
static float value(size_t i,unsigned seed){uint32_t z=uint32_t(i)^seed;z^=z>>16;z*=0x7feb352d;z^=z>>15;z*=0x846ca68b;z^=z>>16;return (float(z&65535)/16384.f)-2.f;}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}

int main(){try{std::cout<<std::setprecision(12)<<"{\"cases\":[";bool first=true;
 struct Shape{int N,H;};
 for(Shape sh:std::vector<Shape>{{1,32},{8,224},{100,224}}){int N=sh.N,C=3,H=sh.H,W=sh.H,O=96,Kh=11,Kw=11,Sh=4,Sw=4,Ph=2,Pw=2;
 int Ho=(H+2*Ph-Kh)/Sh+1,Wo=(W+2*Pw-Kw)/Sw+1,M=O,NP=N*Ho*Wo,K=C*Kh*Kw;
 size_t count=size_t(N)*O*Ho*Wo;
 std::vector<float>x(size_t(N)*C*H*W),w(size_t(O)*C*Kh*Kw),b(O),sent(count+16,-12345.f);
 #pragma omp parallel for
 for(size_t i=0;i<x.size();++i)x[i]=value(i,1234);
 for(size_t i=0;i<w.size();++i)w[i]=value(i,5678)*0.025f;
 for(int i=0;i<O;++i)b[i]=value(i,9753)*0.025f;
 Device dx(x.size()),dw(w.size()),db(O),dy(count+16);dx.put(x);dw.put(w);db.put(b);
 auto launch=[&](bool after){if(after)optimized::conv2d_implicit_gemm_wmma_kernel<<<dim3((NP+127)/128,1),192>>>(dx.p,dw.p,db.p,dy.p,N,C,H,W,O,Kh,Kw,Sh,Sw,Ph,Pw,Ho,Wo,M,NP,K);
 else baseline::conv2d_implicit_gemm_wmma_kernel<<<dim3((NP+127)/128,1),256>>>(dx.p,dw.p,db.p,dy.p,N,C,H,W,O,Kh,Kw,Sh,Sw,Ph,Pw,Ho,Wo,M,NP,K);ck(cudaGetLastError());};
 std::vector<float>reference(count);
 #pragma omp parallel for collapse(3)
 for(int n=0;n<N;++n)for(int o=0;o<O;++o)for(int oh=0;oh<Ho;++oh)for(int ow=0;ow<Wo;++ow){double want=b[o];
 for(int c=0;c<C;++c)for(int kh=0;kh<Kh;++kh)for(int kw=0;kw<Kw;++kw){int ih=oh*Sh-Ph+kh,iw=ow*Sw-Pw+kw;if(ih>=0&&ih<H&&iw>=0&&iw<W)want+=double(x[((n*C+c)*H+ih)*W+iw])*w[((o*C+c)*Kh+kh)*Kw+kw];}
 reference[((n*O+o)*Ho+oh)*Wo+ow]=float(want);}
 double errors[2]={};
 for(int after=0;after<2;++after){dy.put(sent);launch(after);ck(cudaDeviceSynchronize());auto y=dy.get();double err=0;int bad=0;
 #pragma omp parallel for reduction(max:err) reduction(+:bad)
 for(size_t i=0;i<count;++i){double want=reference[i],e=std::abs(double(y[i])-want);err=std::max(err,e);if(!std::isfinite(y[i])||e>3.e-3+3.e-3*std::abs(want))++bad;}
 for(size_t i=count;i<y.size();++i)if(y[i]!=-12345.f)++bad;if(bad)throw std::runtime_error("native Conv2D oracle/guard failed");errors[after]=err;}
 for(int i=0;i<30;++i){launch(false);launch(true);}ck(cudaDeviceSynchronize());cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));std::vector<double>times[2];
 for(int round=0;round<11;++round)for(int order=0;order<2;++order){int a=(round+order)%2;ck(cudaEventRecord(start));for(int i=0;i<30;++i)launch(a);ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));times[a].push_back(ms*1000/30);}
 ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));if(!first)std::cout<<',';first=false;std::cout<<"{\"batch\":"<<N<<",\"height\":"<<H<<",\"width\":"<<W<<",\"output_channels\":"<<O<<",\"passed\":true,\"baseline_us\":"<<median(times[0])<<",\"after_us\":"<<median(times[1])<<",\"max_error\":["<<errors[0]<<','<<errors[1]<<"],\"rounds_us\":[";
 for(int a=0;a<2;++a){if(a)std::cout<<',';std::cout<<'[';for(size_t j=0;j<times[a].size();++j){if(j)std::cout<<',';std::cout<<times[a][j];}std::cout<<']';}std::cout<<"]}"<<std::flush;
 }
 std::cout<<"]}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
