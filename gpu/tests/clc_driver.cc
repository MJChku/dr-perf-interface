// GX persistent scheduling oracle adapted to standalone CPU launches.
#include "drperf_gpu.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <stdexcept>
static int run(drperf::gpu::Module& module,unsigned blocks,unsigned cluster) {
  std::vector<unsigned> h(blocks);unsigned s=0;auto visits=h.data(),steals=&s;void* args[]={&visits,&steals};
  {drperf::gpu::Region region("clc",{{"blocks",blocks},{"cluster",cluster}});
   module.launch_cluster("persistent",blocks,64,cluster,args);}
  int bad = 0;
  for (unsigned i = 0; i < blocks; ++i) bad += h[i] != 1;
  printf("{\"blocks\":%u,\"cluster\":%u,\"steals_cta_sum\":%u,\"status\":\"%s\"}\n", blocks, cluster, s,
         bad == 0 && s > 0 ? "pass" : "fail");

  return bad != 0 || s == 0;
}

int main(int argc,char** argv) {try{
 if(argc!=2)return 2;drperf::gpu::Module module(argv[1]);
  // Under GX, keep only a few clusters resident so pending clusters exist to
  // cancel (hardware likewise only cancels clusters beyond residency).
  setenv("GX_CPU_KERNEL_BATCH_THREADS", "256", 1);
  int failed = 0;
  failed |= run(module,64, 2);
  failed |= run(module,48, 4);
  failed |= run(module,40, 1);
  return failed;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
