#include "ds41/gx_fixture.hpp"
#include "ds41/decode_metadata.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <memory>
using namespace ds41::gx_fixture;
void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
std::uint16_t bf(float x){auto v=std::bit_cast<std::uint32_t>(x);return (v+32767+((v>>16)&1))>>16;}
float fp8(unsigned x){auto m=x&127;float v=m<8?std::ldexp(float(m),-9):std::ldexp(1+float(m%8)/8,int(m/8)-7);return x&128?-v:v;}
float fp4(unsigned x){const float values[]={0,.5,1,1.5,2,3,4,6};return x&8?-values[x&7]:values[x&7];}
static Function library_fn,unpack_fn;
void fixture(Driver& api,Function fn,unsigned rows,unsigned dim,bool paged,bool compressed){
    const unsigned guard=32;unsigned heads=rows%2?3:8,window=4,selected=compressed?8:4,qs=heads*dim+3,gs=6,ws=dim+3,wss=dim/32+2,cs=dim/2+3,css=dim/16+2,page_rows=2,physical=8;
    std::vector<std::uint16_t> query(rows*qs+2*guard,0x4abc),pq(rows*16*dim+2*guard,0x4abc),wantq=pq,kv(rows*selected*dim+2*guard,0x4abc),wantkv=kv;
    std::vector<int> ids(rows*gs+2*guard,-775),indices(rows*selected+2*guard,-775),wanti=indices,errors(rows+2*guard,-775),wante=errors;
    std::vector<float> sink(16+2*guard,-779),ps(16+2*guard,-779),wants=ps;
    for(unsigned h=0;h<heads;++h)sink[guard+h]=float(h)*.3125f;
    for(unsigned h=0;h<16;++h)wants[guard+h]=h<heads?sink[guard+h]:0;
    for(unsigned r=0;r<rows;++r){errors[guard+r]=wante[guard+r]=0;for(unsigned h=0;h<16;++h)for(unsigned d=0;d<dim;++d){if(h<heads)query[guard+r*qs+h*dim+d]=bf(float(int((r*17+h*11+d*3)%97)-48)*.0625f);wantq[guard+(r*16+h)*dim+d]=h<heads?query[guard+r*qs+h*dim+d]:0;}}
    std::vector<std::unique_ptr<Buffer<std::uint8_t>>> windows,scales,values,vscales;std::vector<std::unique_ptr<Buffer<unsigned>>> pages;
    std::vector<ds41::DecodeCacheRow> metadata;
    for(unsigned r=0;r<rows;++r){
        std::vector<std::uint8_t> w(window*ws+2*guard,0xa7),s(window*wss+2*guard,0xa7),v(physical*cs+2*guard,0xa7),vs(physical*css+2*guard,0xa7);std::vector<unsigned> p={3,0,2,1};
        for(unsigned j=0;j<window;++j){for(unsigned d=0;d<dim;++d){unsigned code=(r*13+j*17+d)%127;w[guard+j*ws+d]=code|((j+d)%2?128:0);}for(unsigned d=0;d<dim/32;++d)s[guard+j*wss+d]=(j+d)%5?123+(r+j+d)%9:0;}
        for(unsigned j=0;j<physical;++j){for(unsigned d=0;d<dim/2;++d)v[guard+j*cs+d]=(r*31+j*17+d*7)%256;for(unsigned d=0;d<dim/16;++d)vs[guard+j*css+d]=(r+j*7+d*3)%127;}
        windows.push_back(std::make_unique<Buffer<std::uint8_t>>(api,w));scales.push_back(std::make_unique<Buffer<std::uint8_t>>(api,s));values.push_back(std::make_unique<Buffer<std::uint8_t>>(api,v));vscales.push_back(std::make_unique<Buffer<std::uint8_t>>(api,vs));pages.push_back(std::make_unique<Buffer<unsigned>>(api,p));
        unsigned position=r%3==0?0:r%3==1?3:10,cr=2+r%4;metadata.push_back({windows.back()->ptr+guard,scales.back()->ptr+guard,values.back()->ptr+guard,vscales.back()->ptr+guard,paged?pages.back()->ptr:0,position,cr,page_rows,ws,wss,cs,css,0});
        const int choices[]={0,int(cr)-1,-1,r%2?int(cr):-2};for(unsigned j=0;j<4;++j)ids[guard+r*gs+j]=choices[j];
        for(unsigned col=0;col<selected;++col){bool valid;unsigned pr=0;int id=col<window?int(position)-int(window)+1+col:choices[col-window];
            if(col<window){valid=id>=0;if(valid)pr=id%window;}else{valid=id>=0&&unsigned(id)<cr;if(valid)pr=paged?p[id/page_rows]*page_rows+id%page_rows:id;else if(id!=-1)wante[guard+r]=1;}
            wanti[guard+r*selected+col]=valid?r*selected+col:-1;
            for(unsigned d=0;d<dim;++d){float value=0;if(valid){if(col<window)value=fp8(w[guard+pr*ws+d])*std::ldexp(1.0f,int(s[guard+pr*wss+d/32])-127);else value=fp4((v[guard+pr*cs+d/2]>>((d%2)*4))&15)*fp8(vs[guard+pr*css+d/16]);}wantkv[guard+(r*selected+col)*dim+d]=bf(value);}
        }
    }
    Buffer<ds41::DecodeCacheRow> dm(api,metadata);Buffer<std::uint16_t> dq(api,query),dpq(api,pq),dkv(api,kv);Buffer<int> di(api,ids),dpi(api,indices),de(api,errors);Buffer<float> ds(api,sink),dps(api,ps);
    auto mp=dm.ptr,qp=dq.ptr+guard*2,ip=di.ptr+guard*4,sp=ds.ptr+guard*4,pqp=dpq.ptr+guard*2,kvp=dkv.ptr+guard*2,pip=dpi.ptr+guard*4,psp=dps.ptr+guard*4,ep=de.ptr+guard*4;
    void* args[]={&mp,&qp,&ip,&sp,&pqp,&kvp,&pip,&psp,&ep,&rows,&heads,&dim,&window,&selected,&qs,&gs};
    const auto work=std::max(std::uint64_t(rows)*selected*dim,std::uint64_t(rows)*16*dim)/8;
    check(api.launch(fn,(work+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));check(api.sync());
    require(dpq.read()==wantq,"decode padded query mismatch/guards");require(dkv.read()==wantkv,"decode selected packed KV mismatch/guards");require(dpi.read()==wanti&&de.read()==wante,"decode causal/invalid index mismatch/guards");require(dps.read()==wants,"decode padded sinks mismatch/guards");require(dq.read()==query&&di.read()==ids&&ds.read()==sink,"decode preparation mutated inputs");
#ifdef DS41_DECODE_LIBRARY
    require(dim==512,"library fixture dimension");
    Buffer<std::uint16_t> po(api,std::vector<std::uint16_t>(rows*16*dim+2*guard,0x4abc)),out(api,std::vector<std::uint16_t>(rows*qs+2*guard,0x4abc));
    auto pop=po.ptr+guard*2,op=out.ptr+guard*2;unsigned batch=1,cache_rows=rows*selected;
    check(load<int(*)(Function,int,int)>("cuFuncSetAttribute")(library_fn,8,86016));
    void* lib[]={&psp,&kvp,&pop,&pqp,&pip,&batch,&rows,&cache_rows,&selected};check(api.launch(library_fn,rows,1,1,256,1,1,86016,nullptr,lib,nullptr));
    void* unpack[]={&pop,&op,&rows,&heads,&dim,&qs};check(api.launch(unpack_fn,(std::uint64_t(rows)*heads*dim+2047)/2048,1,1,256,1,1,0,nullptr,unpack,nullptr));check(api.sync());
    auto actual=out.read(),expected=out.host;
    for(unsigned r=0;r<rows;++r)for(unsigned h=0;h<heads;++h){
        std::vector<float> logits(selected,-INFINITY);float maximum=-INFINITY;
        for(unsigned j=0;j<selected;++j)if(wanti[guard+r*selected+j]>=0){double dot=0;for(unsigned d=0;d<dim;++d){const auto q=std::bit_cast<float>(std::uint32_t(query[guard+r*qs+h*dim+d])<<16),v=std::bit_cast<float>(std::uint32_t(wantkv[guard+(r*selected+j)*dim+d])<<16);dot+=double(q)*v;}logits[j]=float(dot)*(1/std::sqrt(float(dim)));maximum=std::max(maximum,logits[j]);}
        double denominator=std::exp(double(sink[guard+h])-maximum);std::vector<float> probability(selected);
        for(unsigned j=0;j<selected;++j)if(wanti[guard+r*selected+j]>=0){const float e=std::exp(logits[j]-maximum);denominator+=e;probability[j]=std::bit_cast<float>(std::uint32_t(bf(e))<<16);}
        for(unsigned d=0;d<dim;++d){double numerator=0;for(unsigned j=0;j<selected;++j)numerator+=double(probability[j])*std::bit_cast<float>(std::uint32_t(wantkv[guard+(r*selected+j)*dim+d])<<16);auto i=guard+r*qs+h*dim+d;expected[i]=bf(float(numerator/denominator));float got=std::bit_cast<float>(std::uint32_t(actual[i])<<16),want=std::bit_cast<float>(std::uint32_t(expected[i])<<16);require(std::isfinite(got)&&std::abs(got-want)<=.0005+.0078125*std::abs(want),"selected-only library attention independent oracle mismatch");actual[i]=expected[i];}
    }
    require(actual==expected,"selected-only library output guards/strides modified");
#endif
    for(unsigned r=0;r<rows;++r)require(windows[r]->read()==windows[r]->host&&scales[r]->read()==scales[r]->host&&values[r]->read()==values[r]->host&&vscales[r]->read()==vscales[r]->host&&pages[r]->read()==pages[r]->host,"decode preparation mutated session cache");
}
int main(int argc,char** argv){try{const bool native=prepare_execution(argc,argv);std::ifstream file(argv[1]);require(file.good(),"PTX missing");std::string ptx{std::istreambuf_iterator<char>(file),{}};Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));Module module;check(api.module(&module,ptx.c_str()));Function fn;check(api.function(&fn,module,"ds41_prepare_decode_bf16"));unsigned cases=0;
#ifdef DS41_DECODE_LIBRARY
    auto directory=std::filesystem::path(argv[1]).parent_path();
    auto load_module=[&](const char* name){std::ifstream f(directory/name);require(f.good(),"library module missing");std::string source{std::istreambuf_iterator<char>(f),{}};Module m;check(api.module(&m,source.c_str()));return m;};
    auto lm=load_module("sparse_attention.cubin"),um=load_module("attention_prepare.cubin");check(api.function(&library_fn,lm,"sparse_attn_kernel__kernel"));check(api.function(&unpack_fn,um,"ds41_unpack_sparse_bf16"));
    const std::vector<unsigned> dims={512};
#else
    const std::vector<unsigned> dims={32,512};
#endif
    for(auto rows:{1u,3u,9u})for(auto dim:dims)for(bool paged:{false,true})for(bool compressed:{false,true}){fixture(api,fn,rows,dim,paged,compressed);++cases;}
#ifdef DS41_DECODE_LIBRARY
    check(api.unload(lm));check(api.unload(um));
#endif
    check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"cases\":"<<cases<<",\"scope\":\"Batched selected-only FP8/FP4 decode preparation, causal rings, heterogeneous contexts, strides, paged sources, invalid selections and guards\",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
