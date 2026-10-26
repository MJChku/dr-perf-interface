#include "drperf_gpu.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <string>
struct Tie{unsigned idx;float score;};struct Params{const float* input;int*output;const int*lengths;Tie*tie_ws;unsigned num_rows,stride;};
static void ck(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(int argc,char**argv){try{ck(argc==3,"topk driver module [trivial|rf|large]");drperf::gpu::Module m(argv[1]);std::string mode=argv[2];int sizes[2]={};int*sp=sizes;void*sa[]={&sp};{drperf::gpu::Region r("topk2048.shared_size");m.launch("topk_shared",dim3(1),dim3(1),sa);}std::cout<<"smem="<<sizes[0]<<" layout="<<sizes[1]<<std::endl;
 // Full original K2048 launcher. T<=512 trivial, T=2304/4096 RF,
 // T=16400/16640 minimally crosses the actual RF threshold; w=1 only there.
 bool large=mode=="large"||mode=="large_ties";bool tie=mode=="large_ties";std::vector<int>ts=large?std::vector<int>{16400,16640}:mode=="rf"?std::vector<int>{2304,4096}:std::vector<int>{32,128,512};
 for(int t:ts)for(int w:large?std::vector<int>{1}:std::vector<int>{1,2}){
 constexpr int K=2048;int stride=(t+255)/256*256;std::vector<float>scores(w*stride,-1);std::vector<int>len(w,t),out(w*K+2,-999);std::vector<Tie>ties(w*K);for(int r=0;r<w;r++)for(int i=0;i<t;i++)scores[r*stride+i]=tie?1.f:float(i+1);
 Params p{scores.data(),out.data()+1,len.data(),ties.data(),unsigned(w),unsigned(stride)};void*args[]={&p};
 {drperf::gpu::Region r(("topk2048.cs16."+mode).c_str(),{{"T",t},{"w",w},{"K",K},{"wT",w*t},{"wK",w*K},{"stride",stride},{"blocks",w*16}});m.launch_cluster("_ZN4vllm11cooperative21cooperative_topk_cs16ILj2048EEEvNS0_21CooperativeTopKParamsIXT_EEE",dim3(w,16),dim3(1024),dim3(1,16),args,sizes[0]);}
 for(int r=0;r<w;r++){std::vector<bool>seen(t);for(int j=0;j<K;j++){int id=out[1+r*K+j];if(t<K){ck(id==(j<t?j:-1),"trivial oracle");}else{ck(id>=(tie?0:t-K)&&id<t&&!seen[id],"topk set oracle");seen[id]=true;}}}ck(out.front()==-999&&out.back()==-999,"topk guards");std::cout<<mode<<" T="<<t<<" w="<<w<<" PASS"<<std::endl;
 }
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
