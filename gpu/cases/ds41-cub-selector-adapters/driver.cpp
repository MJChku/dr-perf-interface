#include "ds41/gx_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <unistd.h>
using namespace ds41::gx_fixture;
#if defined(DS41_CUB_ADAPTERS)
static Function cub_pack,cub_finish512,cub_finish2048;
#elif defined(DS41_CUB_LIBRARY)
static int(*cub_workspace)(unsigned,unsigned,unsigned,std::size_t*);
static int(*cub_select)(const void*,const void*,const void*,void*,void*,void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,std::size_t,void*);
#endif
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static std::uint16_t bf(float v){auto b=std::bit_cast<std::uint32_t>(v);return (b+32767+((b>>16)&1))>>16;}
static float bf(std::uint16_t v){return std::bit_cast<float>(std::uint32_t(v)<<16);}
static void reduction(Driver& api,Function fn){
    unsigned rows=3,heads=4,keys=67,ds=keys+3,ws=heads+2,ss=keys+5;
    float scale=1.0f/64;constexpr unsigned guard=32;
    std::vector<std::uint16_t> dots(rows*heads*ds+guard*2,0x4abc),weights(rows*ws+guard*2,0x4abc),output(rows*ss+guard*2,0x4abc);
    for(unsigned r=0;r<rows;++r)for(unsigned h=0;h<heads;++h){
        weights[guard+r*ws+h]=bf((int(r*heads+h)-5)*1.03125f);
        for(unsigned c=0;c<keys;++c)dots[guard+(r*heads+h)*ds+c]=bf((int((c*7+r*11+h*31)%77)-33)*.140625f);
    }
    auto expected=output;
    for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<keys;++c){
        double sum=0;
        for(unsigned h=0;h<heads;++h){float w=bf(bf(bf(weights[guard+r*ws+h])*scale));sum+=bf(bf(std::max(0.0f,bf(dots[guard+(r*heads+h)*ds+c]))*w));}
        expected[guard+r*ss+c]=bf(float(sum));
    }
    Buffer<std::uint16_t> d(api,dots),w(api,weights),o(api,output);
    auto dp=d.ptr+guard*2,wp=w.ptr+guard*2,op=o.ptr+guard*2;
    void* args[]={&dp,&wp,&op,&rows,&heads,&keys,&ds,&ws,&ss,&scale};
    check(api.launch(fn,(keys+127)/128,rows,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());
    require(o.read()==expected,"index BF16 weighted head reduction/guards mismatch");
    require(d.read()==dots&&w.read()==weights,"index reduction modified inputs");
}
static void selection(Driver& api,Function fn,unsigned keys,unsigned topk,unsigned mode,bool masked,unsigned pattern){
    unsigned rows=5,block=8,units=mode ? (keys+7)/8:keys;
    unsigned ss=keys+7,cs=((keys+7)/8+31)/32+2,is=topk+3,ms=(units+31)/32+3;
    constexpr unsigned guard=32;
    std::vector<std::uint16_t> scores(rows*ss+guard*2,0x4abc);
    std::vector<std::int32_t> lengths={0,1,int(keys/3),int(keys),int(keys+17)};
    std::vector<unsigned> candidates(rows*cs+guard*2,0x1234abcd),masks(rows*ms+guard*2,0x1234abcd);
    std::vector<std::int32_t> initial(rows*is+guard*2,-731),errors(rows,0);
    for(unsigned r=0;r<rows;++r){
        for(unsigned c=0;c<keys;++c){
            float v=(int((c*37+r*19)%103)-62)*.15625f;
            if(pattern==1)v=(c%3 ? -0.0f:0.0f);
            if(pattern==2)v=c%7 ? -INFINITY:float(int(c%23)-10);
            if(pattern==3)v=c%13 ? v:INFINITY;
            if(pattern==4&&c%37==0)v=std::numeric_limits<float>::quiet_NaN();
            scores[guard+r*ss+c]=bf(v);
        }
        for(unsigned i=0;i<cs-2;++i)candidates[guard+r*cs+i]=0;
        for(unsigned b=0;b<(keys+7)/8;++b)if((b+r)%3)candidates[guard+r*cs+b/32]|=1u<<(b%32);
    }
    auto expected=initial;
    auto expected_masks=masks;
    auto expected_errors=errors;
    std::vector<std::uint16_t> pack_expected(static_cast<std::size_t>(rows)*units,0);
    for(unsigned r=0;r<rows;++r){
        for(unsigned i=0;i<topk;++i)expected[guard+r*is+i]=-1;
        if(mode)for(unsigned i=0;i<(units+31)/32;++i)expected_masks[guard+r*ms+i]=0;
        std::vector<std::pair<float,unsigned>> ranked;
        auto reachable=std::min(keys,unsigned(lengths[r]));
        for(unsigned i=0;i<units;++i){
            auto first=mode ? i*8:i;
            if(first>=reachable)continue;
            if(!mode&&masked&&!(candidates[guard+r*cs+(i/8)/32]&(1u<<((i/8)%32))))continue;
            float score=-INFINITY;
            for(unsigned p=first;p<std::min(reachable,first+(mode?8:1));++p){const auto value=bf(scores[guard+r*ss+p]);if(std::isnan(value))expected_errors[r]=1;else score=std::max(score,value);}
            if(mode&&i==(reachable-1)/8)score=INFINITY;
            if(score>-INFINITY)ranked.emplace_back(score,i);
            if(score>-INFINITY){unsigned bits=bf(score);if(!(bits&0x7fff))bits=0;pack_expected[static_cast<std::size_t>(r)*units+i]=bits&0x8000?(~bits&0xffff):(bits^0x8000);}
        }
        std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first!=b.first ? a.first>b.first:a.second<b.second;});
        ranked.resize(std::min<std::size_t>(topk,ranked.size()));
        std::vector<unsigned> selected;for(auto [score,id]:ranked)selected.push_back(id);
        std::sort(selected.begin(),selected.end());
        for(unsigned j=0;j<selected.size();++j){auto id=selected[j];expected[guard+r*is+j]=id;if(mode)expected_masks[guard+r*ms+id/32]|=1u<<(id%32);}
    }
    Buffer<std::uint16_t> ds(api,scores);Buffer<std::int32_t> dl(api,lengths),di(api,initial),de(api,errors);
    Buffer<unsigned> dc(api,candidates),dm(api,masks);
    auto sp=ds.ptr+guard*2,lp=dl.ptr,cp=masked ? dc.ptr+guard*4:DevicePtr{0},ip=di.ptr+guard*4,mp=dm.ptr+guard*4,ep=de.ptr;
    void* args[]={&sp,&lp,&cp,&ip,&mp,&ep,&rows,&keys,&topk,&mode,&block,&ss,&cs,&is,&ms};
#ifdef DS41_CUB_ADAPTERS
    for(unsigned row=0;row<rows;++row){
        std::vector<std::uint16_t> packed(units+2*guard,0xcdef),sorted_keys(units+2*guard,0xcdef);
        std::vector<unsigned> values(units+2*guard,0xdeadbeef),sorted_values(units+2*guard,0xdeadbeef),order(units);
        for(unsigned i=0;i<units;++i)order[i]=i;
        std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return pack_expected[static_cast<std::size_t>(row)*units+a]>pack_expected[static_cast<std::size_t>(row)*units+b];});
        for(unsigned i=0;i<units;++i){sorted_values[guard+i]=order[i];sorted_keys[guard+i]=pack_expected[static_cast<std::size_t>(row)*units+order[i]];}
        Buffer<std::uint16_t> pk(api,packed),sk(api,sorted_keys);Buffer<unsigned> pv(api,values),sv(api,sorted_values);
        auto kp=pk.ptr+guard*2,vp=pv.ptr+guard*4,skp=sk.ptr+guard*2,svp=sv.ptr+guard*4;
        void* pack_args[]={&sp,&lp,&cp,&kp,&vp,&ep,&row,&keys,&mode,&block,&ss,&cs};
        if(units)check(api.launch(cub_pack,(units+255)/256,1,1,256,1,1,0,nullptr,pack_args,nullptr));check(api.sync());
        for(unsigned i=0;i<units;++i){packed[guard+i]=pack_expected[static_cast<std::size_t>(row)*units+i];values[guard+i]=i;}
        require(pk.read()==packed&&pv.read()==values,"CUB input packing oracle or guards mismatch");
#ifndef DS41_CUB_PACK_ONLY
        void* finish_args[]={&skp,&svp,&ip,&mp,&row,&units,&topk,&mode,&is,&ms};
        check(api.launch(topk<=512?cub_finish512:cub_finish2048,1,1,1,256,1,1,0,nullptr,finish_args,nullptr));check(api.sync());
        require(sk.read()==sorted_keys&&sv.read()==sorted_values,"CUB finish modified input pairs");
#endif
    }
#ifdef DS41_CUB_PACK_ONLY
    require(de.read()==expected_errors,"CUB pack nonfinite-score flags mismatch");require(ds.read()==scores&&dl.read()==lengths&&dc.read()==candidates,"CUB packing modified inputs");return;
#endif
#elif defined(DS41_CUB_LIBRARY)
    std::size_t bytes=0;check(cub_workspace(keys,mode,block,&bytes));Buffer<std::uint8_t> temp(api,std::vector<std::uint8_t>(std::max(bytes,std::size_t(1))));
    check(cub_select(reinterpret_cast<void*>(sp),reinterpret_cast<void*>(lp),reinterpret_cast<void*>(cp),reinterpret_cast<void*>(ip),reinterpret_cast<void*>(mp),reinterpret_cast<void*>(ep),rows,keys,topk,mode,block,ss,cs,is,ms,reinterpret_cast<void*>(temp.ptr),bytes,nullptr));check(api.sync());
#else
    check(api.launch(fn,rows,1,1,256,1,1,0,nullptr,args,nullptr));check(api.sync());
#endif
    auto actual=di.read();
    if(actual!=expected){for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<topk;++j)if(actual[guard+r*is+j]!=expected[guard+r*is+j]){std::cerr<<"keys="<<keys<<" topk="<<topk<<" mode="<<mode<<" pattern="<<pattern<<" row="<<r<<" j="<<j<<" got="<<actual[guard+r*is+j]<<" want="<<expected[guard+r*is+j]<<'\n';goto done;}done:throw std::runtime_error("radix selection mismatch");}
    require(dm.read()==expected_masks,"candidate bitset or padding mismatch");
    require(de.read()==expected_errors,"unexpected nonfinite-score error");
    require(ds.read()==scores&&dl.read()==lengths&&dc.read()==candidates,"selection modified inputs");
}
int main(int argc,char** argv){
    try{
        const auto native=prepare_execution(argc,argv);std::ifstream f(argv[1]);require(f.good(),"PTX missing");std::string ptx{std::istreambuf_iterator<char>(f),{}};
        Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));Module module=nullptr;
#ifdef DS41_CUB_LIBRARY
        void* library=dlopen(argv[1],RTLD_NOW|RTLD_GLOBAL);require(library,"CUB library unavailable");
        cub_workspace=reinterpret_cast<decltype(cub_workspace)>(dlsym(library,"ds41_cub_workspace"));cub_select=reinterpret_cast<decltype(cub_select)>(dlsym(library,"ds41_cub_select"));require(cub_workspace&&cub_select,"CUB adapter ABI unavailable");
        Function reduce=nullptr,select=nullptr;unsigned cases=0;
#elif defined(DS41_CUB_ADAPTERS)
        check(api.module(&module,ptx.c_str()));check(api.function(&cub_pack,module,"ds41_cub_pack"));check(api.function(&cub_finish512,module,"ds41_cub_finish512"));check(api.function(&cub_finish2048,module,"ds41_cub_finish2048"));Function reduce=nullptr,select=nullptr;unsigned cases=0;
#else
        check(api.module(&module,ptx.c_str()));
        Function reduce,select;check(api.function(&reduce,module,"ds41_index_reduce"));check(api.function(&select,module,"ds41_index_select"));
        reduction(api,reduce);unsigned cases=1;
#endif
        for(unsigned pattern=0;pattern<5;++pattern){
            selection(api,select,67,17,0,false,pattern);++cases;
            selection(api,select,521,31,0,true,pattern);++cases;
            selection(api,select,521,13,1,false,pattern);++cases;
        }
        selection(api,select,517,512,0,false,0);++cases;
        selection(api,select,32771,2048,1,false,1);++cases;
        for(unsigned pattern=0;pattern<4;++pattern){
            selection(api,select,2049,512,0,false,pattern);++cases;
            selection(api,select,4099,512,0,true,pattern);++cases;
            selection(api,select,8199,257,1,false,pattern);++cases;
        }
        selection(api,select,1,7,0,false,0);++cases;
        selection(api,select,0,7,0,false,0);++cases;
        selection(api,select,257,0,0,true,0);++cases;
        if(module)check(api.unload(module));check(api.destroy(context));
        std::cout<<"{\"numerical_status\":\"pass\",\"kernels\":\""
#ifdef DS41_CUB_PACK_ONLY
                 <<"CUB custom input packing only; library finish excluded"
#elif defined(DS41_CUB_LIBRARY)
                 <<"CUB full sort and packing/output-order integration"
#elif defined(DS41_CUB_ADAPTERS)
                 <<"CUB packing and output-order adapters"
#else
                 <<"index reduction and radix selection"
#endif
                 <<"\",\"cases\":"<<cases<<",\"pid\":"<<getpid()<<",\"rank\":0,\"device\":0,\"comparison\":\"independent scalar BF16 score keys and full-sort selection\",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
