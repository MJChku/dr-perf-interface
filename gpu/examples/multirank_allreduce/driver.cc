// CPU-only, communicating original Lamport kernels: one independent host thread/rank.
#include "drperf_gpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Exact host ABI of frozen AllReduceFusionParams<BF16>; no CUDA API calls.
struct Params {
 int nranks=0,rank=0,size=0,hidden_dim=0;
 void** workspace=nullptr;
 void *allreduce_in=nullptr,*allreduce_out=nullptr,*residual_in=nullptr,
      *residual_out=nullptr,*norm_out=nullptr,*quant_out=nullptr,*scale_out=nullptr,*rms_gamma=nullptr;
 float rms_eps=1e-5f,weight_bias=0;
 float* scale_factor=nullptr;
 bool use_oneshot=true;
 int layout=0;
 void* stream=nullptr;
 int pattern=0;
 bool trigger_completion_at_end=true;
 int block_quant_group_size=0,tma_aligned_mn=0;
};
static_assert(sizeof(Params)==136 && offsetof(Params,rms_gamma)==80 && offsetof(Params,rms_eps)==88);
static float f32(uint16_t b){uint32_t u=uint32_t(b)<<16;float f;std::memcpy(&f,&u,4);return f;}
static uint16_t bf(float f){uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
static float rb(float f){return f32(bf(f));}
class Barrier {
 std::mutex m;std::condition_variable cv;int left,n,generation=0;bool broken=false;
 public: explicit Barrier(int n):left(n),n(n){}
 void wait(){std::unique_lock<std::mutex> lock(m);int g=generation;if(broken)throw std::runtime_error("peer failed");
  if(--left==0){left=n;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g||broken;});
  if(broken)throw std::runtime_error("peer failed");}
 void cancel(){std::lock_guard<std::mutex> lock(m);broken=true;cv.notify_all();}
};
struct Rank {
 std::vector<uint16_t> storage,input,residual,gamma,ar,norm;
 std::vector<float> residual32,out32;
 int flags[5]={};std::vector<void*> workspace;
 Rank(int hidden,int ranks):storage(3*ranks*4*hidden+16,0x8000),input(4*hidden+16,0x7fc1),
 residual(4*hidden+16,0x7fc1),gamma(hidden+16,0x7fc1),ar(4*hidden+16,0x7fc1),norm(4*hidden+16,0x7fc1),
 residual32(4*hidden+8,-9876),out32(4*hidden+8,-9876),workspace(3*ranks+1){flags[3]=ranks*4*hidden*2;}
 uint16_t* data(){return storage.data()+8;}
};
static std::string kernel(const std::string& mode,int ranks){
 if(mode=="residual")return "_ZN14glm53_ar_fused19ar_residual_oneshotI13__nv_bfloat16Lb1EEEvN10flashinfer23trtllm_allreduce_fusion21AllReduceFusionParamsIT_EEPKfPf";
 return "_ZN10flashinfer23trtllm_allreduce_fusion39allreduce_fusion_kernel_oneshot_lamportILNS0_22AllReduceFusionPatternE"+
 std::string(mode=="norm"?"1":"0")+"E13__nv_bfloat16Li"+std::to_string(ranks)+"ELb1ELb1EEEvNS0_21AllReduceFusionParamsIT0_EE";
}
static void require(bool value,const char* what){if(!value)throw std::runtime_error(what);}
static void scenario(const char* module,const std::string& mode,int hidden,int ranks,int forced_cluster,bool smoke,int skew_us){
 require(mode!="residual"||ranks==2,"project residual only supports 2 ranks");
 int cluster=8,threads=hidden/8;
 while(threads%cluster && cluster>1)cluster/=2;
 threads/=cluster;
 while(threads<128&&cluster>=2){threads*=2;cluster/=2;}
 if(forced_cluster){cluster=forced_cluster;threads=hidden/8/cluster;}
 require(threads*cluster==hidden/8 && threads>0,"invalid geometry");
 // Avoid a fake dependency on pre-filled peer data: all 3 slots begin at -0.
 std::vector<std::unique_ptr<Rank>> state;
 for(int r=0;r<ranks;++r)state.emplace_back(new Rank(hidden,ranks));
 for(int r=0;r<ranks;++r){for(int j=0;j<3*ranks;++j)state[r]->workspace[j]=state[j%ranks]->data();state[r]->workspace[3*ranks]=state[r]->flags;}
 std::vector<int> rows=smoke?std::vector<int>{1,4,2,1}:std::vector<int>{1,4,2,1,3,4,1,2,4,3,1,4};
 Barrier sync(ranks);std::mutex error_mutex;std::exception_ptr error;std::atomic<uint64_t> checked{0};
 std::atomic<uint64_t> norm_ulp1{0};
 std::vector<std::thread> workers;
 for(int r=0;r<ranks;++r)workers.emplace_back([&,r]{try{
  drperf::gpu::Module m(module);auto& s=*state[r];int prev=0;
  for(unsigned call=0;call<rows.size();++call){int w=rows[call],n=w*hidden;
   std::fill(s.ar.begin(),s.ar.end(),0x7fc1);std::fill(s.norm.begin(),s.norm.end(),0x7fc1);std::fill(s.out32.begin(),s.out32.end(),-9876);
   for(int i=0;i<n;++i){
    float x=float(((i*7+int(call)*11+r*5)%31)-15)/32;
    s.input[i+8]=(i%19==0)?uint16_t(0x8000):bf(x);
    s.residual[i+8]=bf(float((i*3+r*2+int(call))%23-11)/16);
    s.residual32[i+4]=float((i*3+r*2+int(call))%23-11)/16+.00017f;
    s.out32[i+4]=-9876;s.ar[i+8]=s.norm[i+8]=0x7fc1;
   }
   for(int i=0;i<hidden;++i)s.gamma[i+8]=bf(.75f+float(i%5)/8);
   Params p;p.nranks=ranks;p.rank=r;p.size=n;p.hidden_dim=hidden;p.workspace=s.workspace.data();
   p.allreduce_in=s.input.data()+8;p.pattern=mode=="norm"?1:0;
   if(mode=="allreduce")p.allreduce_out=s.ar.data()+8;
   if(mode=="norm"){p.residual_in=p.residual_out=s.residual.data()+8;p.norm_out=s.norm.data()+8;p.rms_gamma=s.gamma.data()+8;}
   const float* res=s.residual32.data()+4;float* out=s.out32.data()+4;void* args[]={&p,&res,&out};
   sync.wait();
   // Deliberate host arrival skew changes retries, never the peer values.
   if(skew_us && r==int(call%ranks))std::this_thread::sleep_for(std::chrono::microseconds(skew_us));
   std::string region=mode+(forced_cluster?".cluster":".launcher")+".r"+std::to_string(ranks);
   {drperf::gpu::Region region_marker(region.c_str(),{{"hidden",hidden},{"w",w},{"ranks",ranks},
     {"N",n},{"C",prev*hidden*ranks},{"previous_w",prev},{"blocks",w*cluster},{"threads",threads},{"cluster",cluster},
     {"rank_N",ranks*n}});
    m.launch_cluster(kernel(mode,ranks).c_str(),dim3(w*cluster),dim3(threads),dim3(cluster),args);}
   sync.wait();
   require(s.flags[0]==0 && s.flags[2]==int((call+1)%3) && s.flags[3]==ranks*4*hidden*2 && s.flags[4]==ranks*n,"Lamport flag/counter update");
   // All ranks have completed this call before checking protocol memory.
   const int data_slot=int(call%3),clear_slot=int((call+2)%3),capacity=ranks*4*hidden;
   for(int i=0;i<ranks*n;++i){int peer=i/n,j=i%n;uint16_t expected=state[peer]->input[j+8];if(expected==0x8000)expected=0;
    require(s.data()[data_slot*capacity+i]==expected,"cross-wired peer payload mismatch");}
   for(int i=0;i<prev*hidden*ranks;++i)require(s.data()[clear_slot*capacity+i]==0x8000,"previous-call clear mismatch");
   for(int i=0;i<n;++i){uint16_t original=(i%19==0)?uint16_t(0x8000):bf(float(((i*7+int(call)*11+r*5)%31)-15)/32);
    require(s.input[i+8]==original,"read-only input was modified");
    require(s.residual32[i+4]==float((i*3+r*2+int(call))%23-11)/16+.00017f,"read-only FP32 residual was modified");}
   for(int i=0;i<hidden;++i)require(s.gamma[i+8]==bf(.75f+float(i%5)/8),"read-only gamma was modified");
   std::vector<float> expected(n);
   for(int i=0;i<n;++i){float sum=0;for(int peer=0;peer<ranks;++peer){uint16_t v=state[peer]->input[i+8];sum+=f32(v==0x8000?0:v);}float reduced=rb(sum);
    if(mode=="allreduce")require(f32(s.ar[i+8])==reduced,"allreduce numerical oracle");
    if(mode=="residual")require(s.out32[i+4]==rb(s.residual32[i+4]+reduced),"fused residual numerical oracle");
    if(mode=="norm"){expected[i]=rb(float((i*3+r*2+int(call))%23-11)/16+reduced);require(f32(s.residual[i+8])==expected[i],"norm residual numerical oracle");}
   }
   if(mode=="norm")for(int row=0;row<w;++row){float ss=0;for(int i=0;i<hidden;++i)ss+=expected[row*hidden+i]*expected[row*hidden+i];float inv=1/std::sqrt(ss/hidden+p.rms_eps);
    for(int i=0;i<hidden;++i){uint16_t want=bf((expected[row*hidden+i]*inv)*f32(s.gamma[i+8])),got=s.norm[row*hidden+i+8];int ulp=std::abs(int(want)-int(got));
     require(std::isfinite(f32(got))&&ulp<=1,"RMSNorm numerical oracle (>1 BF16 ULP)");if(ulp)++norm_ulp1;}}
   for(int i=0;i<8;++i){require(s.storage[i]==0x8000&&s.storage[s.storage.size()-1-i]==0x8000,"workspace guards");
    require(s.input[i]==0x7fc1&&s.gamma[i]==0x7fc1&&s.ar[i]==0x7fc1&&s.norm[i]==0x7fc1&&s.residual[i]==0x7fc1,"input/output prefix guards");}
   for(int i=n+8;i<int(s.ar.size());++i)require(s.ar[i]==0x7fc1&&s.norm[i]==0x7fc1,"output tail guards");
   for(int i=0;i<4;++i)require(s.out32[i]==-9876,"FP32 prefix guard");
   for(int i=n+4;i<int(s.out32.size());++i)require(s.out32[i]==-9876,"FP32 tail guard");
   checked+=n;prev=w;
   sync.wait(); // Do not overwrite an input until every rank has verified it.
  }
 }catch(...){sync.cancel();std::lock_guard<std::mutex> lock(error_mutex);if(!error)error=std::current_exception();}});
 for(auto& t:workers)t.join();if(error)std::rethrow_exception(error);
 std::cout<<"{\"mode\":\""<<mode<<"\",\"hidden\":"<<hidden<<",\"ranks\":"<<ranks<<",\"cluster\":"<<cluster
  <<",\"threads\":"<<threads<<",\"calls\":"<<rows.size()<<",\"checked_elements\":"<<checked<<",\"norm_ulp1\":"<<norm_ulp1<<",\"status\":\"pass\"}"<<std::endl;
}
int main(int argc,char** argv){try{
 if(argc<2)throw std::runtime_error("usage: driver module.so [mode=all] [ranks=2] [smoke=0] [cluster=0] [skew_us=0] [hidden_override=0]");
 std::string mode=argc>2?argv[2]:"all";int ranks=argc>3?std::stoi(argv[3]):2;bool smoke=argc>4?std::stoi(argv[4]):0;
 int cluster=argc>5?std::stoi(argv[5]):0,skew=argc>6?std::stoi(argv[6]):0;
 int hidden_override=argc>7?std::stoi(argv[7]):0;
 for(const std::string& kind:std::vector<std::string>{"residual","allreduce","norm"})if(mode=="all"||mode==kind){if(kind=="residual"&&ranks!=2)continue;
  for(int hidden:hidden_override?std::vector<int>{hidden_override}:(smoke?std::vector<int>{256}:std::vector<int>{32,64,128,256,512}))scenario(argv[1],kind,hidden,ranks,cluster,smoke,skew);}
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
