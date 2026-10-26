#include "drperf_gpu.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
static void pack(drperf::gpu::Module&m,int t,bool refined){
 int batches=1,heads=1,nope=1,rope=1,value=1;
 std::vector<float>x(2*t),rot(t),k(2*t+2,-999),v(t+2,-999);
 for(int i=0;i<t;i++){x[2*i]=float(i%97);x[2*i+1]=float(i%53);rot[i]=float(i%31);}
 auto*xp=x.data();auto*rp=rot.data();float*kp=k.data()+1,*vp=v.data()+1;
 void*args[]={&xp,&rp,&kp,&vp,&batches,&t,&heads,&nope,&rope,&value};
 auto launch=[&]{m.launch("materialize_kv_head_major",dim3((2*t+127)/128),dim3(128),args);};
 if(refined){drperf::gpu::Region r("history-pack-refined",{{"queries",1},{"history",t}});launch();}
 else{drperf::gpu::Region r("history-pack-initial",{{"queries",1}});launch();}
 for(int i=0;i<t;i++)if(kp[2*i]!=x[2*i]||kp[2*i+1]!=rot[i]||vp[i]!=x[2*i+1])throw std::runtime_error("materialize oracle");
 if(k.front()!=-999||k.back()!=-999||v.front()!=-999||v.back()!=-999)throw std::runtime_error("pack guards");
}
static void softmax(drperf::gpu::Module&m,int t,int count,bool refined,bool compact=false){
 int batches=1,heads=1,queries=1,offset=t-1,length=t;float scale=.5f;
 int outputs=compact?count:t;
 std::vector<uint16_t>scores(t,0),p(outputs+2,0xbeef);std::vector<int>ids(count);
 for(int i=0;i<count;i++)ids[i]=int(int64_t(i)*(t-1)/(count-1));
 auto*sp=scores.data();uint16_t*pp=p.data()+1;auto*ip=ids.data();int*op=&offset,*lp=&length;
 void*args[]={&sp,&pp,&ip,&op,&lp,&batches,&heads,&queries,&t,&count,&scale};
 auto launch=[&]{m.launch(compact?"compact_selected_softmax_bf16":"sparse_selected_softmax_bf16",dim3(1),dim3(256),args,count*sizeof(float));};
 if(compact){drperf::gpu::Region r("selected-softmax-compact",{{"selected",count}});launch();}
 else if(refined){drperf::gpu::Region r("selected-softmax-refined",{{"selected",count},{"history",t}});launch();}
 else{drperf::gpu::Region r("selected-softmax-initial",{{"selected",count}});launch();}
 // All selected scores are zero; counts are powers of two, so 1/count is
 // exactly representable in both FP32 and BF16. No tolerance/oracle ambiguity.
 int log=0;for(int n=count;n>1;n/=2)log++;
 std::vector<uint16_t>expected(outputs,0);
 if(compact)for(auto&x:expected)x=uint16_t((127-log)<<7);
 else for(int id:ids)expected[id]=uint16_t((127-log)<<7);
 for(int i=0;i<outputs;i++)if(pp[i]!=expected[i])throw std::runtime_error("sparse softmax uniform-score oracle");
 if(p.front()!=0xbeef||p.back()!=0xbeef)throw std::runtime_error("softmax guards");
}
int main(int argc,char**argv){try{
 if(argc!=2)return 2;drperf::gpu::Module m(argv[1]);
 for(auto pair:std::vector<std::pair<int,int>>{{32,16},{64,16},{96,32},{128,16},{128,32},{256,32},{512,64},{2048,2048},{100000,2048}}){
  for(bool refined:{false,true}){pack(m,pair.first,refined);softmax(m,pair.first,pair.second,refined);}
  softmax(m,pair.first,pair.second,false,true);
  std::cout<<"history="<<pair.first<<" selected="<<pair.second<<" numerical/guard checks PASS"<<std::endl;
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
