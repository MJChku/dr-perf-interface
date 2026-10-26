#include "common_extracted.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>
#ifndef CASE_RMS
#define CASE_RMS 1
#endif
namespace baseline {
#include "before.cu"
}
namespace optimized {
#include "after.cu"
}
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Device{float*p;size_t n;Device(size_t n):n(n){ck(cudaMalloc(&p,n*4));}~Device(){cudaFree(p);}void put(const std::vector<float>&v){ck(cudaMemcpy(p,v.data(),n*4,cudaMemcpyHostToDevice));}std::vector<float>get(){std::vector<float>v(n);ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost));return v;}};
static float value(size_t i,unsigned seed){uint32_t z=uint32_t(i)^seed;z^=z>>16;z*=0x7feb352d;z^=z>>15;z*=0x846ca68b;z^=z>>16;return float(z&65535)/16384.f-2.f;}
static double median(std::vector<double>v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(){try{std::cout<<std::setprecision(12)<<"{\"cases\":[";bool first=true;
 struct Shape{int rows,width;};
 for(Shape shape:std::vector<Shape>{{4096,256},{4096,4096},{4096,16384},{1024,65536}}){int rows=shape.rows,width=shape.width,count=rows*width,threads=width<1024?32:1024;float eps=1e-5f;
 std::vector<float>x(count),g(CASE_RMS?count:1),sent(count+16,-12345.f);
 #pragma omp parallel for
 for(int i=0;i<count;++i)x[i]=value(i,1234);
 if(CASE_RMS){
 #pragma omp parallel for
 for(int i=0;i<count;++i)g[i]=value(i,5678);
 }
 Device dx(count),dg(g.size()),dy(count+16);dx.put(x);dg.put(g);
 auto launch=[&](bool after){
 #if CASE_RMS
 if(after){if(threads==32)optimized::rms_norm_back_f32<32><<<rows,32>>>(dg.p,dx.p,dy.p,width,eps);else optimized::rms_norm_back_f32<1024><<<rows,1024>>>(dg.p,dx.p,dy.p,width,eps);}
 else {if(threads==32)baseline::rms_norm_back_f32<32><<<rows,32>>>(dg.p,dx.p,dy.p,width,eps);else baseline::rms_norm_back_f32<1024><<<rows,1024>>>(dg.p,dx.p,dy.p,width,eps);}
 #else
 if(after){if(threads==32)optimized::group_norm_f32<32><<<rows,32>>>(dx.p,dy.p,width,count,eps);else optimized::group_norm_f32<1024><<<rows,1024,256>>>(dx.p,dy.p,width,count,eps);}
 else {if(threads==32)baseline::group_norm_f32<32><<<rows,32>>>(dx.p,dy.p,width,count,eps);else baseline::group_norm_f32<1024><<<rows,1024,256>>>(dx.p,dy.p,width,count,eps);}
 #endif
 ck(cudaGetLastError());};
 std::vector<double>means(rows),scales(rows),coeffs(rows);
 #pragma omp parallel for
 for(int row=0;row<rows;++row){double m=0,xx=0,xg=0,v=0;for(int j=0;j<width;++j){size_t i=size_t(row)*width+j;double z=x[i];m+=z;xx+=z*z;if(CASE_RMS)xg+=z*g[i];}m/=width;
 if(!CASE_RMS)for(int j=0;j<width;++j){double z=x[size_t(row)*width+j]-m;v+=z*z;}
 means[row]=m;scales[row]=1/std::sqrt((CASE_RMS?xx:v)/width+eps);coeffs[row]=-scales[row]*xg/(xx+width*eps);}
 double errors[2]={};
 for(int a=0;a<2;++a){dy.put(sent);launch(a);ck(cudaDeviceSynchronize());auto y=dy.get();double err=0;int bad=0;
 #pragma omp parallel for reduction(max:err) reduction(+:bad)
 for(int i=0;i<count;++i){int row=i/width;double want=CASE_RMS?scales[row]*g[i]+coeffs[row]*x[i]:(x[i]-means[row])*scales[row];double e=std::abs(double(y[i])-want);err=std::max(err,e);if(!std::isfinite(y[i])||e>4e-5+4e-5*std::abs(want))++bad;}
 for(size_t i=count;i<y.size();++i)if(y[i]!=-12345.f)++bad;if(bad)throw std::runtime_error("native independent oracle/guard failed");errors[a]=err;}
 for(int i=0;i<30;++i){launch(false);launch(true);}ck(cudaDeviceSynchronize());cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));std::vector<double>times[2];
 for(int round=0;round<11;++round)for(int order=0;order<2;++order){int a=(round+order)%2;ck(cudaEventRecord(start));for(int i=0;i<30;++i)launch(a);ck(cudaEventRecord(stop));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));times[a].push_back(ms*1000/30);}
 ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));if(!first)std::cout<<',';first=false;std::cout<<"{\"rows\":"<<rows<<",\"columns\":"<<width<<",\"threads\":"<<threads<<",\"passed\":true,\"baseline_us\":"<<median(times[0])<<",\"after_us\":"<<median(times[1])<<",\"max_error\":["<<errors[0]<<','<<errors[1]<<"],\"rounds_us\":[";
 for(int a=0;a<2;++a){if(a)std::cout<<',';std::cout<<'[';for(size_t i=0;i<times[a].size();++i){if(i)std::cout<<',';std::cout<<times[a][i];}std::cout<<']';}std::cout<<"]}"<<std::flush;
 }
 std::cout<<"]}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
