#include "drperf_gpu.h"
#include "drperf_gpu_tensor_map.h"
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdint>
using namespace drperf_gpu_tensor_abi;
static void ck(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static CUtensorMap desc(void*base,CUtensorMapDataType type,std::vector<uint64_t>dims,std::vector<uint64_t>strides,std::vector<uint32_t>box,bool swizzle){CUtensorMap x{};uint32_t step[3]={1,1,1};ck(gx_tensor_map_encode_tiled(&x,type,dims.size(),base,dims.data(),strides.data(),box.data(),step,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==0,"TMA descriptor");return x;}
int main(int argc,char**argv){try{ck(argc==2,"DG driver module");drperf::gpu::Module m(argv[1]);int shared=0;int*sizeptr=&shared;void*sa[]={&sizeptr};{drperf::gpu::Region r("dg_shared_size_probe",{{"T",32},{"w",1}});m.launch("dg_smem_size",dim3(1),dim3(1),sa);}ck(shared>150000&&shared<300000,"DG smem sizeof");std::cout<<"Shared bytes="<<shared<<std::endl;
 for(uint32_t T:{32u,64u,128u,256u,512u})for(uint32_t w:{1u,2u,3u,4u})for(uint32_t sms:{1u,2u}){
  uint32_t per=(T+255)/256,pages=per*w,blocks=pages*2,stride=per*256,bts=(T+127)/128;
  std::vector<uint8_t>q(w*32*128,0x38),kv(pages*256*128,0x38);std::vector<float>scale(pages*256,1),weights(w*32,1),out(w*stride+2,-999);std::vector<uint32_t>lens(w,T),bt(w*bts),schedule((sms+1)*2);for(uint32_t r=0;r<w;r++)for(uint32_t j=0;j<bts;j++)bt[r*bts+j]=r*per*2+j;
  for(uint32_t i=0;i<=sms;i++){uint32_t n=w*per,work=i*(n/sms)+std::min(i,n%sms);schedule[2*i]=work/per;schedule[2*i+1]=work==n?0:work%per;}
  for(uint32_t r=0;r<w;r++){uint8_t codes[4]={0x38,0x40,0x30,0x48};std::fill(q.begin()+r*4096,q.begin()+(r+1)*4096,codes[r]);std::fill(kv.begin()+r*per*32768,kv.begin()+(r+1)*per*32768,codes[(r+1)%4]);}
  uint32_t tasks=0,qrows=0;for(uint32_t sm=0;sm<sms;sm++){uint32_t n=w*per,lo=sm*(n/sms)+std::min(sm,n%sms),hi=(sm+1)*(n/sms)+std::min(sm+1,n%sms);for(uint32_t r=0;r<w;r++)if(std::max(lo,r*per)<std::min(hi,(r+1)*per)){++tasks;qrows+=std::min(4u,w-r);}}
  CUtensorMap tq=desc(q.data(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,w*32},{128},{128,128},true);
  CUtensorMap tk=desc(kv.data(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,128,blocks},{128,128*128},{128,128,1},true);
  CUtensorMap ts=desc(scale.data(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{128,blocks},{128*4},{128,1},false);
  CUtensorMap tw=desc(weights.data(),CU_TENSOR_MAP_DATA_TYPE_FLOAT32,{32,w},{32*4},{32,4},false);
  auto*lp=lens.data();auto*op=out.data()+1;auto*bp=bt.data();uint32_t*indices=nullptr;auto*mp=schedule.data();void*args[]={&w,&stride,&bts,&lp,&op,&bp,&indices,&mp,&tq,&ts,&tk,&ts,&tw};
  {drperf::gpu::Region r("deepgemm_paged_scores.bq4_control",{{"T",T},{"w",w},{"K",8},{"heads",32},{"dim",128},{"pages",pages},{"splits",w*per},{"blocks",sms},{"logits",w*stride},{"tasks",tasks},{"q_inbounds_rows",qrows},{"active_sms",std::min(sms,w*per)},{"block_table_fetches",tasks*bts}});m.launch("_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj32ELj128ELj128ELb0ELb1ELb0ELj3ELj5ELj256ELj16ELj128ELj256EN7cutlass12float_e4m3_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_",dim3(sms),dim3(384),args,shared);}
  for(unsigned r=0;r<w;r++)for(unsigned j=0;j<T;j++)ck(op[r*stride+j]==4096.f*std::vector<float>{1,2,0.5,4}[r]*std::vector<float>{2,0.5,4,1}[r],"DG distinct-request FP8 score oracle");ck(out.front()==-999&&out.back()==-999,"DG output guards");std::cout<<"DG T="<<T<<" w="<<w<<" grid="<<sms<<" PASS"<<std::endl;
 }
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
