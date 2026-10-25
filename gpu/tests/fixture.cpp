#include "drperf_gpu.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using drperf::gpu::Region;
static void check(bool p,const char* what){if(!p)throw std::runtime_error(what);}
int main(int argc,char** argv) {
    if(argc<2)return 2;
    drperf::gpu::Module m(argv[1]);
    std::string mode=argc>2?argv[2]:"all";
    if(mode=="unmarked") {
        std::array<float,32> out{};auto p=out.data();void* args[]={&p};
        m.launch("shuffle",1,32,args);return 0;
    }
    if(mode=="unclosed") {
        drperf_gpu_region_begin("unclosed",0,nullptr,nullptr);return 0;
    }
    if(mode=="truncated") {
        Region r("truncated");std::_Exit(0);
    }
    if(mode=="constant") {
        for(int i=0;i<3;++i) {
            std::array<float,32> out{};auto p=out.data();void* args[]={&p};
            Region r("constant");m.launch("shuffle",1,32,args);
        }
        return 0;
    }
    if(mode=="threads") {
        std::vector<std::thread> workers;
        for(int t=0;t<4;++t)workers.emplace_back([&] {
            for(int i=0;i<3;++i) {
                std::array<float,64> out{};auto p=out.data();void* args[]={&p};
                Region r("threaded",{{"warps",2}});
                m.launch("shuffle",1,64,args);
                for(float v:out)check(v==528,"threaded warp state leaked");
            }
        });
        for(auto& worker:workers)worker.join();
        return 0;
    }
    if(mode=="unmodeled") {
        alignas(16) std::array<uint32_t,4> in{7,8,9,10},out{};
        auto p=in.data(),q=out.data();void* args[]={&p,&q};
        Region r("async-copy",{{"bytes",16}});
        m.launch("async_copy",1,1,args);
        check(out[0]==7,"cp.async numerical copy failed");
        return 0;
    }
    for(unsigned n:{1u,3u,63u,64u,65u,127u,128u,129u,257u}) {
        unsigned blocks=(n+63)/64;
        std::vector<float> input(n+2),output(blocks+2,-777);
        for(unsigned i=0;i<n;++i)input[i]=float(i%11)-5;
        auto p=input.data(),q=output.data();void* args[]={&p,&q,&n};
        {
            Region region("reduce",{{"elements",n},{"blocks",blocks}});
            m.launch("reduce",blocks,64,args);
        }
        for(unsigned b=0;b<blocks;++b) {
            double sum=0;for(unsigned i=b*64;i<std::min(n,(b+1)*64);++i)sum+=input[i];
            check(output[b]==sum,"shared reduction mismatch");
        }
        check(output[blocks]==-777&&output[blocks+1]==-777,"reduction guard overwritten");
    }
    for(unsigned warps:{1u,2u,3u,4u,6u}) {
        std::vector<float> out(32*warps+2,-777);auto p=out.data();void* args[]={&p};
        {
            Region region("shuffle",{{"warps",warps}});
            m.launch("shuffle",1,32*warps,args);
        }
        for(unsigned i=0;i<32*warps;++i)check(out[i]==528,"warp shuffle mismatch");
        check(out[32*warps]==-777,"shuffle guard overwritten");
    }
    for(unsigned n:{1u,2u,3u,5u,8u,13u,20u,27u}) {
        std::vector<float> out(34,-777);auto p=out.data();void* args[]={&p,&n};
        {
            Region region("missing-pcv",{{"elements",n}});
            m.launch("triangular",1,32,args);
        }
        {
            Region region("triangular",{{"elements",n},{"pairs",n*(n-1)/2}});
            m.launch("triangular",1,32,args);
        }
        for(unsigned i=0;i<n;++i)check(out[i]==float(i),"triangular numerical mismatch");
        check(out[n]==-777,"triangular guard overwritten");
    }
    for(int sign:{-1,1,-1,1}) {
        std::array<float,32> input,out;input.fill(float(sign));out.fill(-777);
        auto p=input.data(),q=out.data();void* args[]={&p,&q};
        {
            Region region("same-state-variation",{{"elements",32}});
            m.launch("conditional",1,32,args);
        }
        for(float value:out)check(value==(sign>0?2.f:-1.f),"conditional mismatch");
    }
    alignas(32) std::array<uint16_t,256> a,b;
    alignas(32) std::array<float,256> c{};
    alignas(32) std::array<float,256*5+8> d;
    a.fill(0x3f80);b.fill(0x3f80);
    for(unsigned tiles:{1u,2u,3u,4u,5u}) {
        d.fill(-777);auto ap=a.data(),bp=b.data();auto cp=c.data(),dp=d.data();
        void* args[]={&ap,&bp,&cp,&dp};
        {
            Region region("bf16-matrix",{{"tiles",tiles}});
            m.launch("matrix",tiles,32,args);
        }
        for(unsigned i=0;i<256*tiles;++i)check(d[i]==16,"BF16 WMMA numerical mismatch");
        check(d[256*tiles]==-777,"WMMA guard overwritten");
    }
    {
        Region outer("parent",{{"work",1}});
        Region inner("child",{{"work",1}});
        std::array<float,32> out{};auto p=out.data();void* args[]={&p};
        m.launch("shuffle",1,32,args);
    }
    puts("GPU reduction, shuffle, predication, missing-PCV, WMMA and nested-marker checks pass");
}
