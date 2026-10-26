#include "drperf_gpu.h"
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <thread>
int main(int argc,char**argv){try{
 if(argc!=2)return 2;alignas(128) unsigned slots[64]={},results[2]={};int rounds=64;std::exception_ptr errors[2];
 std::thread workers[2];
 for(int r=0;r<2;++r)workers[r]=std::thread([&,r]{try{
  drperf::gpu::Module m(argv[1]);unsigned* p=slots,*q=results;int rank=r;void* args[]={&p,&rank,&rounds,&q};
  if(r)std::this_thread::sleep_for(std::chrono::milliseconds(2));
  drperf::gpu::Region region("visibility",{{"rank",r},{"rounds",rounds}});m.launch("visibility_probe",1,1,args);
 }catch(...){errors[r]=std::current_exception();}});
 for(auto& t:workers)t.join();for(auto& e:errors)if(e)std::rethrow_exception(e);
 for(int r=0;r<2;++r)if(results[r]!=26u*rounds*(rounds+1)/2+4u*(1-r)*rounds)return 1;
 std::cout<<"{\"probe\":\"volatile vector, acquire/release, membar.sys\",\"rounds\":64,\"status\":\"pass\"}\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
