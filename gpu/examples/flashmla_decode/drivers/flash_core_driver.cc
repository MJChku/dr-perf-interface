#include "drperf_gpu.h"
#include "drperf_gpu_tensor_map.h"
#include "flash_params.h"
#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <string>
#include <cstring>
using namespace drperf_gpu_tensor_abi;
static void ck(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static std::string kernel(const char*ptx,const std::string&part){std::ifstream f(ptx);std::string s;while(std::getline(f,s)){auto p=s.find(".visible .entry ");if(p!=std::string::npos&&s.find(part)!=std::string::npos){s=s.substr(p+16);return s.substr(0,s.find('('));}}throw std::runtime_error("kernel name");}
static CUtensorMap desc(void*base,CUtensorMapDataType type,std::vector<uint64_t>dims,std::vector<uint64_t>strides,std::vector<uint32_t>box,CUtensorMapSwizzle swizzle){static int dn=0; ++dn; std::cout<<"desc="<<dn<<std::endl;CUtensorMap x{};uint32_t step[5]={1,1,1,1,1};ck(gx_tensor_map_encode_tiled(&x,type,dims.size(),base,dims.data(),strides.data(),box.data(),step,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==0,"TMA descriptor");return x;}
int main(int argc,char**argv){try{ck(argc==6,"core driver core.so core.ptx aux.so aux.ptx smoke|grid");drperf::gpu::Module m(argv[1]),aux(argv[3]);auto core=kernel(argv[2],"flash_fwd_splitkv_mla_fp8_sparse_kernel");auto meta=kernel(argv[4],"get_mla_metadata_kernel");auto comb=kernel(argv[4],"Li32ELi256");int shared=0;int*ps=&shared;void*as[]={&ps};{drperf::gpu::Region r("flashmla.core.size_probe");m.launch("flash_shared",dim3(1),dim3(1),as);}ck(shared>100000,"core shared size");std::cout<<"shared="<<shared<<std::endl;
 bool smoke=std::string(argv[5])=="smoke";
 for(int w:smoke?std::vector<int>{1}:std::vector<int>{1,2,3,4})for(int k:smoke?std::vector<int>{64}:std::vector<int>{64,128,256})for(int parts:smoke?std::vector<int>{1}:std::vector<int>{1,2,4,8}){
 int h=64,sq=1,nb=w*k/64;std::vector<uint16_t>q(w*h*576,0),out(w*h*512+16,0xdead);std::vector<uint8_t>kv(w*k*656,0);for(int r=0;r<w*k;r++){std::memset(kv.data()+r*656,0x38,512);for(int s=0;s<4;s++){float x=1;std::memcpy(kv.data()+r*656+512+s*4,&x,4);}for(int d=0;d<64;d++){uint16_t x=0x3f80;std::memcpy(kv.data()+r*656+528+d*2,&x,2);}}
 std::vector<int>ids(w*k),splits(w+1);for(int i=0;i<w*k;i++)ids[i]=i;std::vector<DecodingSchedMeta>sched(parts);std::vector<float>lse(w*h+2,-999),la((w+parts)*h,-999),oa((w+parts)*h*512,-999);
 GetDecodeSchedMetaParams mp{};mp.b=w;mp.s_q=1;mp.block_size_n=64;mp.fixed_overhead_num_blocks=1;mp.topk=k;mp.tile_scheduler_metadata_ptr=sched.data();mp.num_splits_ptr=splits.data();mp.num_sm_parts=parts;void*ma[]={&mp};{drperf::gpu::Region r("flashmla.core.fixture_schedule",{{"w",w},{"K",k},{"parts",parts}});aux.launch(meta.c_str(),dim3(1),dim3(32),ma,4*(5*w+1));}
 CUtensorMap maps[5];maps[0]=desc(q.data(),CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,{576,64,1,(uint64_t)w},{576*2,64*576*2,64*576*2},{64,64,1,1},CU_TENSOR_MAP_SWIZZLE_128B);
 maps[1]=desc(out.data()+8,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,{512,64,1,(uint64_t)w},{512*2,64*512*2,64*512*2},{64,64,1,1},CU_TENSOR_MAP_SWIZZLE_128B);
 maps[2]=desc(q.data()+512,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,{64,64,2,1,(uint64_t)w},{576*2,32*2,64*576*2,64*576*2},{32,64,2,1,1},CU_TENSOR_MAP_SWIZZLE_64B);
 maps[3]=desc(kv.data(),CU_TENSOR_MAP_DATA_TYPE_INT64,{64,(uint64_t)w*k},{656},{64,1},CU_TENSOR_MAP_SWIZZLE_NONE);
 maps[4]=desc(kv.data()+528,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,{64,(uint64_t)w*k},{656},{32,1},CU_TENSOR_MAP_SWIZZLE_64B);
 std::vector<unsigned char>tb(1408);auto*tbp=tb.data();auto*maps_ptr=maps;void*ta[]={&tbp,&maps_ptr,&w,&sq};{drperf::gpu::Region r("flashmla.core.fixture_tma",{{"w",w}});m.launch("flash_tma_init",dim3(1),dim3(1),ta);}
 SparseAttnDecodeParams p{};p.b=w;p.s_q=1;p.h_q=64;p.h_kv=1;p.d_qk=576;p.d_v=512;p.sm_scale=1;p.sm_scale_div_log2=1.4426950408889634;p.num_blocks=nb;p.page_block_size=64;p.topk=k;p.q=q.data();p.kv=(uint16_t*)kv.data();p.indices=ids.data();p.lse=lse.data()+1;p.out=out.data()+8;p.stride_q_b=64*576;p.stride_q_s_q=64*576;p.stride_q_h_q=576;p.stride_kv_block=64*656;p.stride_kv_row=656;p.stride_indices_b=k;p.stride_indices_s_q=k;p.stride_lse_b=64;p.stride_lse_s_q=64;p.stride_o_b=64*512;p.stride_o_s_q=64*512;p.stride_o_h_q=512;p.lse_accum=la.data();p.o_accum=oa.data();p.stride_lse_accum_split=64;p.stride_lse_accum_s_q=64;p.stride_o_accum_split=64*512;p.stride_o_accum_s_q=64*512;p.stride_o_accum_h_q=512;p.tile_scheduler_metadata_ptr=sched.data();p.num_splits_ptr=splits.data();p.num_sm_parts=parts;
 int tasks=0,split_tasks=0,unsplit=0;for(auto&s:sched){if(s.begin_req_idx>=w)continue;for(int r=s.begin_req_idx;r<=s.end_req_idx;r++){tasks++;if(splits[r+1]-splits[r]>1)split_tasks++;else unsplit++;}}
 void*args[]={&p,tb.data()};{drperf::gpu::Region r("flashmla.core.small",{{"w",w},{"K",k},{"heads",64},{"parts",parts},{"wK",w*k},{"blocks64",w*k/64},{"tasks",tasks},{"split_tasks",split_tasks},{"unsplit_tasks",unsplit}});m.launch(core.c_str(),dim3(1,parts),dim3(384),args,shared);}
 if(parts>1){CombineParams c{};c.b=w;c.s_q=1;c.h_q=64;c.d_v=512;c.lse=p.lse;c.out=p.out;c.stride_lse_b=64;c.stride_lse_s_q=64;c.stride_o_b=64*512;c.stride_o_s_q=64*512;c.stride_o_h_q=512;c.lse_accum=la.data();c.o_accum=oa.data();c.stride_lse_accum_split=64;c.stride_lse_accum_s_q=64;c.stride_o_accum_split=64*512;c.stride_o_accum_s_q=64*512;c.stride_o_accum_h_q=512;c.num_splits_ptr=splits.data();c.num_sm_parts=parts;void*ca[]={&c};{drperf::gpu::Region r("flashmla.core.fixture_combine",{{"w",w},{"K",k},{"parts",parts}});aux.launch(comb.c_str(),dim3(w,1,8),dim3(256),ca);}}
 ck(out.front()==0xdead&&out.back()==0xdead,"core output guards");for(int i=0;i<w*h*512;i++)ck(out[i+8]==0x3f80,"core uniform value oracle");for(int i=0;i<w*h;i++)ck(std::abs(lse[i+1]-std::log(float(k)))<1e-4,"core lse oracle");std::cout<<"core w="<<w<<" K="<<k<<" parts="<<parts<<" tasks="<<tasks<<" PASS"<<std::endl;
 }
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
