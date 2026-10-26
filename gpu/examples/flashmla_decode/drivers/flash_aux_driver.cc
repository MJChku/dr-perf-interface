#include "drperf_gpu.h"
#include "flash_params.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <string>
static void ck(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static std::string kernel(const char*ptx,const std::string&part){std::ifstream f(ptx);std::string s;while(std::getline(f,s)){auto p=s.find(".visible .entry ");if(p!=std::string::npos&&s.find(part)!=std::string::npos){s=s.substr(p+16);return s.substr(0,s.find('('));}}throw std::runtime_error("kernel name");}
int main(int argc,char**argv){try{ck(argc==4,"aux driver module ptx [meta|combine]");drperf::gpu::Module m(argv[1]);bool meta=std::string(argv[3])=="meta";
 if(meta){auto kn=kernel(argv[2],"get_mla_metadata_kernel");std::ofstream save("analysis2/flash_small_schedules.jsonl");
 for(int w:{1,2,3,4})for(int k:{64,128,256,512})for(int parts:{1,2,3,4,8})for(int overhead:{1,5}){
 std::vector<DecodingSchedMeta> sched(parts);std::vector<int>splits(w+3,-999);GetDecodeSchedMetaParams p{};p.b=w;p.s_q=1;p.block_size_n=64;p.fixed_overhead_num_blocks=overhead;p.topk=k;p.tile_scheduler_metadata_ptr=sched.data();p.num_splits_ptr=splits.data()+1;p.num_sm_parts=parts;void*args[]={&p};
 {drperf::gpu::Region r("flashmla.metadata.small",{{"w",w},{"K",k},{"parts",parts},{"overhead",overhead},{"wK",w*k}});m.launch(kn.c_str(),dim3(1),dim3(32),args,4*(5*w+1));}
 ck(splits[0]==-999&&splits.back()==-999,"schedule split guards");ck(splits[1]==0,"schedule starts zero");std::vector<int>visited(w*(k/64));int total=0,tasks=0;
 for(auto&s:sched){if(s.begin_req_idx>=w)continue;ck(s.begin_req_idx>=0&&s.end_req_idx<w,"schedule req bounds");for(int r=s.begin_req_idx;r<=s.end_req_idx;r++){int lo=r==s.begin_req_idx?s.begin_block_idx:0,hi=r==s.end_req_idx?s.end_block_idx:k/64;for(int j=lo;j<hi;j++){ck(j>=0&&j<k/64,"schedule block bounds");visited[r*(k/64)+j]++;}if(lo<hi){tasks++;total++;}}}
 for(int v:visited)ck(v==1,"schedule partitions every block once");ck(splits[w+1]==total,"schedule total splits");save<<"{\"w\":"<<w<<",\"K\":"<<k<<",\"parts\":"<<parts<<",\"overhead\":"<<overhead<<",\"tasks\":"<<tasks<<",\"splits\":[";for(int r=0;r<=w;r++)save<<(r?",":"")<<splits[r+1];save<<"]}\n";
 }std::cout<<"metadata 160 small states PASS\n";
 }else{
 for(int h:{8,16,32,64})for(int w:{1,2,3,4})for(int n:{1,2,3,4,8})for(int cap:{32,64}){
 auto kn=kernel(argv[2],cap==32?"Li32ELi256":"Li64ELi256");int S=w*n;std::vector<int>splits(w+1);for(int r=0;r<=w;r++)splits[r]=r*n;
 std::vector<float>la(S*h,0),oa(S*h*512,2),lse(w*h+2,-999);std::vector<uint16_t>out(w*h*512+2,0xdead);
 CombineParams p{};p.b=w;p.s_q=1;p.h_q=h;p.d_v=512;p.lse=lse.data()+1;p.out=out.data()+1;p.stride_lse_b=h;p.stride_lse_s_q=h;p.stride_o_b=h*512;p.stride_o_s_q=h*512;p.stride_o_h_q=512;p.lse_accum=la.data();p.o_accum=oa.data();p.stride_lse_accum_split=h;p.stride_lse_accum_s_q=h;p.stride_o_accum_split=h*512;p.stride_o_accum_s_q=h*512;p.stride_o_accum_h_q=512;p.num_splits_ptr=splits.data();p.num_sm_parts=cap;void*args[]={&p};
 {drperf::gpu::Region r("flashmla.combine.small",{{"w",w},{"heads",h},{"splits_per_req",n},{"split_head_rows",n>1?S*h:0},{"active_heads",n>1?w*h:0},{"head_rows",w*h},{"max_splits",cap}});m.launch(kn.c_str(),dim3(w,1,h/8),dim3(256),args);}
 ck(out.front()==0xdead&&out.back()==0xdead&&lse.front()==-999&&lse.back()==-999,"combine guards");for(int i=0;i<w*h*512;i++)ck(out[i+1]==(n>1?0x4000:0xdead),"combine value oracle");for(int i=0;i<w*h;i++)ck(n==1?lse[i+1]==-999:std::abs(lse[i+1]-std::log(float(n)))<1e-5,"combine lse oracle");
 }std::cout<<"combine 160 small states PASS\n";
 }
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
