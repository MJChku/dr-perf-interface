#include "drperf_gpu.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) { try {
    if(argc!=2)return 2;
    drperf::gpu::Module module(argv[1]);
    alignas(128) uint32_t input[4096], output[4096];
    for(unsigned i=0;i<4096;++i)input[i]=i+17;
    auto* in=input;auto* out=output;
    for(unsigned n:{32u,64u,96u})for(unsigned mode=0;mode<4;++mode) {
        unsigned stride=mode==0?0:mode==1?1:mode==2?8:32;
        unsigned store_stride=mode==0?0:1,offset=mode==2?7:0;
        const char* name=mode==0?"broadcast":mode==1?"streaming":mode==2?"sector_partial":"line_stride";
        std::fill(std::begin(output),std::end(output),0xdeadbeef);
        void* args[]={&in,&out,&stride,&store_stride,&offset};
        {drperf::gpu::Region region(name,{{"lanes",n}});module.launch("footprint",1,n,args);}
        unsigned written=store_stride?n:1;
        for(unsigned i=0;i<written;++i)if(output[i]!=input[i*stride+offset])throw std::runtime_error("copy value oracle");
        if(output[written]!=0xdeadbeef)throw std::runtime_error("copy guard oracle");
    }
    {
        unsigned stride=0,store_stride=1,offset=0;
        void* args[]={&in,&out,&stride,&store_stride,&offset};
        drperf::gpu::Region region("two_launches");
        module.launch("footprint",1,32,args);module.launch("footprint",1,32,args);
    }
    {
        void* args[]={&in,&out};
        drperf::gpu::Region region("generic");module.launch("generic_footprint",1,32,args);
        if(output[0]!=input[0])throw std::runtime_error("generic value oracle");
    }
    {
        input[0]=0;void* args[]={&in};
        {drperf::gpu::Region region("atomic");module.launch("atomic_footprint",1,32,args);}
        if(input[0]!=32)throw std::runtime_error("atomic value oracle");
    }
    for(unsigned scope=0;scope<3;++scope) {
        void* args[]={&in,&scope};
        drperf::gpu::Region region(scope==0?"scoped_lane":scope==1?"scoped_warp":"scoped_block");
        module.launch("scoped_footprint",1,64,args);
    }
    {
        std::fill(std::begin(output),std::end(output),0xdeadbeef);
        void* args[]={&in,&out};
        {drperf::gpu::Region region("vector");module.launch("vector_footprint",1,32,args);}
        for(unsigned i=0;i<128;++i)if(output[i]!=input[i])throw std::runtime_error("vector value oracle");
        if(output[128]!=0xdeadbeef)throw std::runtime_error("vector guard oracle");
    }
    {
        std::fill(std::begin(output),std::end(output),0xdeadbeef);
        void* args[]={&in,&out};
        {drperf::gpu::Region region("wmma_stride");module.launch("wmma_footprint",1,32,args);}
        for(unsigned row=0;row<16;++row)for(unsigned col=0;col<32;++col) {
            uint32_t want=col<16?input[row*32+col]:0xdeadbeef;
            if(output[row*32+col]!=want)throw std::runtime_error("strided WMMA value/hole oracle");
        }
        if(output[512]!=0xdeadbeef)throw std::runtime_error("strided WMMA guard oracle");
    }
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
