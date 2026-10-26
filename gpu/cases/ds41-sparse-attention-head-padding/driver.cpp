#include "drperf_gpu.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
static uint16_t bf(float x){auto v=std::bit_cast<uint32_t>(x);return (v+32767+((v>>16)&1))>>16;}
static float f(uint16_t x){return std::bit_cast<float>(uint32_t(x)<<16);}
int main(int argc,char** argv){try{if(argc!=2)return 2;drperf::gpu::Module module(argv[1]);
for(int heads:{4,8,16})for(int selected:{64,128}){
 int b=1,m=1,n=128,topk=selected;std::vector<float> sink(16,0);std::vector<uint16_t> kv(n*512),q(16*512,0),out(16*512+32,0x4abc);std::vector<int> ids(selected);
 for(int j=0;j<n;++j)for(int d=0;d<512;++d)kv[j*512+d]=bf(float((j*7+d*3)%19-9)/32);
 for(int h=0;h<heads;++h)for(int d=0;d<512;++d)q[h*512+d]=bf(float((h*11+d*5)%17-8)/32);
 for(int i=0;i<selected;++i)ids[i]=i;
 uintptr_t sp=(uintptr_t)sink.data(),kp=(uintptr_t)kv.data(),op=(uintptr_t)out.data(),qp=(uintptr_t)q.data(),ip=(uintptr_t)ids.data();void* args[]={&sp,&kp,&op,&qp,&ip,&b,&m,&n,&topk};
 {drperf::gpu::Region region("sparse_head_padding",{{"real_heads",heads},{"padded_heads",16},{"selected_keys",selected},{"dim",512},{"useful_head_keys",heads*selected}});module.launch("sparse_attn_kernel__kernel",{1,1},{256},args,86016);}
 for(int h=0;h<16;++h){std::vector<float> scores(selected);float mx=0;for(int j=0;j<selected;++j){float dot=0;for(int d=0;d<512;++d)dot+=f(q[h*512+d])*f(kv[j*512+d]);scores[j]=dot/std::sqrt(512.f);mx=std::max(mx,scores[j]);}float denom=std::exp(-mx);for(float s:scores)denom+=std::exp(s-mx);
 for(int d=0;d<512;++d){float num=0;float prefix=-INFINITY;for(int base=0;base<selected;base+=64){for(int j=base;j<base+64;++j)prefix=std::max(prefix,scores[j]);float tile=0;for(int j=base;j<base+64;++j)tile+=f(bf(std::exp(scores[j]-prefix)))*f(kv[j*512+d]);num+=tile*std::exp(prefix-mx);}float want=f(bf(num/denom)),got=f(out[h*512+d]);if(!std::isfinite(got)||std::abs(got-want)>.001f+.015625f*std::abs(want))throw std::runtime_error("independent padded sparse attention scalar oracle");}}
 for(size_t i=16*512;i<out.size();++i)if(out[i]!=0x4abc)throw std::runtime_error("sparse attention output guard");
}std::cout<<"{\"numerical_status\":\"pass\",\"states\":6,\"oracle\":\"scalar scores, BF16 probabilities per64 tile, sink softmax\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
