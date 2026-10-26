#include "drperf_gpu.h"
#include <vector>
#include <cstdint>
#include <string>
#include <iostream>
#include <stdexcept>
static void ck(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(int argc,char**argv){try{
 ck(argc==2,"metadata driver module");drperf::gpu::Module m(argv[1]);
 for(int sms:{148,160})for(int split:{1,256})for(int t:{32,64,128,256,512})for(unsigned w:{1,2,3,4}){
  std::vector<uint32_t>lens(w,t),schedule((sms+1)*2+2,0xdeadbeef);uint32_t*lp=lens.data(),*ip=nullptr,*op=schedule.data()+1;int per=(t+split-1)/split,total=w*per,active=std::min(total,sms),probes=0,nonfirst=0;
  std::vector<uint32_t>expected((sms+1)*2);for(int sm=0;sm<=sms;sm++){int work=sm*(total/sms)+std::min(sm,total%sms);int lo=0,hi=w;while(lo<hi){int mid=(lo+hi)/2;++probes;if((mid+1)*per<=work)lo=mid+1;else hi=mid;}expected[sm*2]=lo;expected[sm*2+1]=lo<int(w)?work-lo*per:0;nonfirst+=lo>0&&lo<int(w);}
  std::string kernel="_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj1ELb1ELb0ELj1ELj"+std::to_string(split)+"ELj"+std::to_string(sms)+"EEEvjjPKjS3_Pj";
  void*args[]={&w,&w,&lp,&ip,&op};
  {drperf::gpu::Region r("deepgemm_paged_metadata",{{"T",t},{"w",w},{"split_size",split},{"splits_per_request",per},{"total_splits",total},{"active_sms",active},{"sms",sms},{"binary_search_probes",probes},{"nonfirst_request_boundaries",nonfirst},{"blocks",1}});m.launch(kernel.c_str(),dim3(1),dim3(256),args,2*w*sizeof(uint32_t));}
  ck(schedule.front()==0xdeadbeef&&schedule.back()==0xdeadbeef,"metadata guards");for(int i=0;i<(sms+1)*2;i++)ck(op[i]==expected[i],"metadata schedule oracle");
 }
 std::cout<<"metadata 80 launches PASS; histories <=512; split sizes {1,256}; SMs {148,160}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
