// Independent GX cluster fixture adapted to standalone launches.
#include "drperf_gpu.h"
#include <vector>
#include <cstdio>
#include <iostream>
#include <stdexcept>
struct Record { unsigned rank,nrank,cx,cy,cz,peer_cluster,peer_generic,barrier_ok,address_rank; };
static int run(drperf::gpu::Module& module,dim3 grid, dim3 cluster_dims, unsigned threads) {
  const unsigned blocks = grid.x * grid.y * grid.z;
  std::vector<Record> h(blocks);auto* d=h.data();void* args[]={&d};
  {drperf::gpu::Region r("cluster",{{"blocks",blocks},{"threads",threads}});
   module.launch_cluster("cluster_kernel",grid,threads,cluster_dims,args);}
  const unsigned n = cluster_dims.x * cluster_dims.y * cluster_dims.z;
  int bad = 0;
  for (unsigned z = 0; z < grid.z; ++z) for (unsigned y = 0; y < grid.y; ++y) for (unsigned x = 0; x < grid.x; ++x) {
    const unsigned linear = x + grid.x * (y + grid.y * z);
    const dim3 c(x % cluster_dims.x, y % cluster_dims.y, z % cluster_dims.z);
    const unsigned rank = c.x + cluster_dims.x * (c.y + cluster_dims.y * c.z);
    const unsigned pr = (rank + 1) % n;
    const unsigned px = x - c.x + pr % cluster_dims.x, py = y - c.y + pr / cluster_dims.x % cluster_dims.y,
                   pz = z - c.z + pr / (cluster_dims.x * cluster_dims.y);
    const unsigned peer_word = 1000 + px + grid.x * (py + grid.y * pz);
    const Record& r = h[linear];
    if (r.rank != rank || r.nrank != n || r.cx != c.x || r.cy != c.y || r.cz != c.z ||
        r.peer_cluster != peer_word || r.peer_generic != peer_word || r.barrier_ok != 1 ||
        r.address_rank != rank) {
      if (bad++ < 4) printf("mismatch block (%u,%u,%u): rank %u/%u cta (%u,%u,%u) peer %u/%u want %u barrier %u\n",
                            x, y, z, r.rank, r.nrank, r.cx, r.cy, r.cz, r.peer_cluster, r.peer_generic, peer_word, r.barrier_ok);
    }
  }
  printf("{\"grid\":[%u,%u,%u],\"cluster\":[%u,%u,%u],\"threads\":%u,\"blocks\":%u,\"status\":\"%s\"}\n",
         grid.x, grid.y, grid.z, cluster_dims.x, cluster_dims.y, cluster_dims.z, threads, blocks, bad ? "fail" : "pass");
  return bad != 0;
}

int main(int argc,char** argv) {try{
  if(argc!=2)return 2;drperf::gpu::Module module(argv[1]);
  int failed = 0;
  failed |= run(module,dim3(8, 1, 1), dim3(2, 1, 1), 64);
  failed |= run(module,dim3(8, 2, 1), dim3(4, 1, 1), 32);
  failed |= run(module,dim3(4, 4, 2), dim3(2, 2, 2), 96);
  failed |= run(module,dim3(6, 1, 1), dim3(1, 1, 1), 32);
  std::vector<unsigned> scratch(96),out(96);auto p=scratch.data(),q=out.data();void* args[]={&p,&q};
  {drperf::gpu::Region r("grid");module.launch("grid_kernel",3,32,args);}
  for(unsigned i=0;i<96;++i)if(out[i]!=((i+48)%96)*3+7)throw std::runtime_error("grid synchronization oracle");
  void* votes[]={&q};{drperf::gpu::Region r("votes");module.launch("vote_kernel",3,32,votes);}
  for(auto v:out)if(v!=67)throw std::runtime_error("barrier vote oracle");
  std::cout<<"grid and barrier votes PASS\n";
  return failed;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
