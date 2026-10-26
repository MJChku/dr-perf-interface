#include "ds41/gx_fixture.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
using namespace ds41::gx_fixture;
static unsigned cases=0;
static void fixture(Driver& api,Function local,Function global,unsigned rows,unsigned width,unsigned world,unsigned pattern){
    unsigned stride=width+7,pair_stride=5;std::vector<float> all(world*rows*stride,-9876),gathered(world*rows*pair_stride,-7777);std::vector<int> want(rows,0);
    for(unsigned rank=0;rank<world;++rank)for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<width;++c){
        float value=float(int((rank*97+r*51+c*73)%257)-128)/16;
        if(pattern==1)value=-1-float(c%11);
        if(pattern==2)value=0.0f;
        if(pattern==3&&rank==world-1&&c==width-1)value=std::numeric_limits<float>::quiet_NaN();
        if(pattern==4&&rank==world-1&&c==width-1)value=std::numeric_limits<float>::infinity();
        if(pattern==5&&rank==world-1&&c==width-1)value=-std::numeric_limits<float>::infinity();
        all[(rank*rows+r)*stride+c]=value;
    }
    for(unsigned r=0;r<rows;++r){float best=-std::numeric_limits<float>::infinity();bool bad=false;
        for(unsigned rank=0;rank<world;++rank)for(unsigned c=0;c<width;++c){float value=all[(rank*rows+r)*stride+c];bad|=!std::isfinite(value);if(value>best){best=value;want[r]=rank*width+c;}}
        if(bad)want[r]=-1;
    }
    for(unsigned rank=0;rank<world;++rank){std::vector<float> input(all.begin()+rank*rows*stride,all.begin()+(rank+1)*rows*stride);
        Buffer<float> x(api,input),pair(api,std::vector<float>(rows*pair_stride,-7777));auto xp=x.ptr,pp=pair.ptr;unsigned first=rank*width;
        void* args[]={&xp,&pp,&rows,&width,&stride,&pair_stride,&first};check(api.launch(local,rows,1,1,256,1,1,0,nullptr,args,nullptr));
        auto values=pair.read();if(x.read()!=input&&pattern<3)throw std::runtime_error("head wrote input");
        for(unsigned r=0;r<rows;++r){if(values[r*pair_stride+3]!=-7777||values[r*pair_stride+4]!=-7777)throw std::runtime_error("pair guard overwritten");}
        std::copy(values.begin(),values.end(),gathered.begin()+rank*rows*pair_stride);
    }
    Buffer<float> pairs(api,gathered);Buffer<int> tokens(api,std::vector<int>(rows+4,-999));auto p=pairs.ptr,t=tokens.ptr;
    void* args[]={&p,&t,&rows,&world,&pair_stride};check(api.launch(global,(rows+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));auto got=tokens.read();
    if(!std::equal(want.begin(),want.end(),got.begin())||!std::all_of(got.begin()+rows,got.end(),[](int v){return v==-999;}))throw std::runtime_error("independent greedy oracle/guard mismatch");++cases;
}
int main(int argc,char** argv){try{
#ifdef DS41_DRPERF_GPU
    prepare_execution(argc,argv);
#else
    if(argc!=2||!std::getenv("GX_KERNEL_CHECK"))throw std::runtime_error("GX numerical execution required");
#endif
    Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));std::ifstream input(argv[1]);std::string text{std::istreambuf_iterator<char>(input),{}};Module module;check(api.module(&module,text.c_str()));Function local,global;check(api.function(&local,module,"ds41_head_local"));check(api.function(&global,module,"ds41_head_global"));
    for(unsigned width:{1u,17u,257u})for(unsigned world:{1u,8u})for(unsigned pattern=0;pattern<6;++pattern)fixture(api,local,global,3,width,world,pattern);
    fixture(api,local,global,1,16160,8,0);fixture(api,local,global,1,16160,8,1);fixture(api,local,global,48,17,8,0);fixture(api,local,global,4,513,8,2);
    check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"cases\":"<<cases<<",\"scope\":\"independent global greedy oracle, shards, first-index ties, negative/zero values, NaN/Inf rejection, strides/guards, production vocabulary and batch48\",\"physical_gpu_used\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
