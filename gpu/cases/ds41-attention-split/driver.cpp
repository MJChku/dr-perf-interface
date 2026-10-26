#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <filesystem>
using namespace ds41::gx_fixture;
static void require(bool v,const char* s) { if(!v) throw std::runtime_error(s); }
static std::uint16_t bf(float v) { auto b=std::bit_cast<std::uint32_t>(v);return (b+32767+((b>>16)&1))>>16; }
static float bf(std::uint16_t v) { return std::bit_cast<float>(std::uint32_t(v)<<16); }
static float fp8(unsigned v) {
    const unsigned mag=v&127;float x=mag/8 ? std::ldexp(1+(mag%8)/8.0f,int(mag/8)-7):(mag%8)/512.0f;
    return v&128 ? -x:x;
}
static float fp4(unsigned v) { const float t[]={0,.5f,1,1.5f,2,3,4,6};return (v&8 ? -1:1)*t[v&7]; }
static double maximum_error=0;
#ifdef DS41_ATTENTION_SPLIT
static Function split_functions[4];
#endif
#if defined(DS41_ATTENTION_PREPARE) || defined(DS41_ATTENTION_LIBRARY)
static Function prepare_fn,unpack_fn,library_fn;
#endif
static void fixture(Driver& api,Function fn,unsigned dim,unsigned selected,unsigned variant,bool paged=false,bool production_decode=false) {
    unsigned rows=production_decode?1:2,heads=production_decode?8:2,wr=production_decode?128:9,cr=production_decode?1024:11,qs=heads*dim+7,ws=dim+5,wss=dim/32+3;
#ifdef DS41_ATTENTION_PAIR
    rows=production_decode?3:variant==3?1:3;
#endif
    unsigned cs=dim/2+5,css=dim/16+3,is=selected+3,os=heads*dim+11;
    const unsigned guard=32;
    unsigned page_rows=4;
    std::vector<unsigned> pages={3,1,2};
    const unsigned physical_rows=paged ? 16:cr;
    std::vector<std::uint16_t> q(rows*qs+guard*2,0x4abc),expected(rows*os+guard*2,0x4abc);
    std::vector<std::uint8_t> w(wr*ws+guard*2,0xA7),wscale(wr*wss+guard*2,0xA7),c(physical_rows*cs+guard*2,0xA7),cscale(physical_rows*css+guard*2,0xA7);
    std::vector<std::int32_t> idx(rows*is+guard*2,-778),expected_error(rows+guard*2,-778);
    std::vector<float> sink(heads+guard*2,-12345);
    std::fill(expected_error.begin()+guard,expected_error.end()-guard,0);
    auto original_error=expected_error;auto original_output=expected;
    for(unsigned r=0;r<rows;++r) for(unsigned d=0;d<heads*dim;++d) q[guard+r*qs+d]=bf(float(int((r*59+d*17)%101)-50)/127);
    for(unsigned r=0;r<wr;++r) {
        for(unsigned d=0;d<dim;++d) w[guard+r*ws+d]=ds41::synthetic::fp8(99,r*dim+d);
        for(unsigned g=0;g<dim/32;++g) wscale[guard+r*wss+g]=124+(r+g)%3;
    }
    for(unsigned r=0;r<physical_rows;++r) {
        for(unsigned d=0;d<dim/2;++d) c[guard+r*cs+d]=ds41::synthetic::fp4(109,r*dim/2+d);
        for(unsigned g=0;g<dim/16;++g) cscale[guard+r*css+g]=0x28+(r+g)%9;
    }
    for(unsigned h=0;h<heads;++h) sink[guard+h]=variant==3 ? 30.0f:float(h)-.3f;
    auto value=[&](int index,unsigned d) {
        if (unsigned(index)<wr) return fp8(w[guard+index*ws+d])*std::ldexp(1.0f,int(wscale[guard+index*wss+d/32])-127);
        const unsigned logical=index-wr,r=paged ? pages[logical/page_rows]*page_rows+logical%page_rows:logical;
        const unsigned code=(c[guard+r*cs+d/2]>>((d%2)*4))&15;
        return fp4(code)*fp8(cscale[guard+r*css+d/16]);
    };
    const float scale=1/std::sqrt(float(dim));
    for(unsigned r=0;r<rows;++r) {
        for(unsigned j=0;j<selected;++j) {
            int pick=(j*7+r*3)%(wr+cr);
            if (j%11==0 || variant==1) pick=-1;
            if (variant==2 && j==2) pick=wr+cr;
            if (variant==2 && j==3) pick=-2;
            idx[guard+r*is+j]=pick;
            if (pick!=-1 && (pick<0 || unsigned(pick)>=wr+cr)) expected_error[guard+r]=1;
        }
        for(unsigned h=0;h<heads;++h) {
            std::vector<double> scores(selected,-std::numeric_limits<double>::infinity()),prefix(selected);
            double current=-1e30;
            for(unsigned block=0;block<(selected+63)/64;++block) {
                for(unsigned j=block*64;j<std::min(selected,(block+1)*64);++j) {
                    const int pick=idx[guard+r*is+j];
                    if (pick<0 || unsigned(pick)>=wr+cr) continue;
                    double dot=0;
                    for(unsigned d=0;d<dim;++d) dot+=double(bf(q[guard+r*qs+h*dim+d]))*value(pick,d);
                    scores[j]=float(dot)*scale;current=std::max(current,scores[j]);
                }
                for(unsigned j=block*64;j<std::min(selected,(block+1)*64);++j) prefix[j]=current;
            }
            // Independent closed-form oracle: rescale each tile's BF16-rounded
            // probabilities directly into the final max domain. No online
            // accumulator/warp reduction from the device is reused here.
            double denominator=std::exp(double(sink[guard+h])-current);
            for(auto score:scores) denominator+=std::exp(score-current);
            for(unsigned d=0;d<dim;++d) {
                double numerator=0;
                for(unsigned j=0;j<selected;++j) {
                    const int pick=idx[guard+r*is+j];
                    if (pick<0 || unsigned(pick)>=wr+cr) continue;
                    const auto probability=bf(bf(float(std::exp(scores[j]-prefix[j]))));
                    numerator+=double(probability)*std::exp(prefix[j]-current)*value(pick,d);
                }
                expected[guard+r*os+h*dim+d]=bf(float(numerator/denominator));
            }
        }
    }
    Buffer<std::uint16_t> dq(api,q),output(api,original_output);
    Buffer<std::uint8_t> dw(api,w),dws(api,wscale),dc(api,c),dcs(api,cscale);
    Buffer<std::int32_t> di(api,idx),errors(api,original_error);Buffer<float> ds(api,sink);
    auto qp=dq.ptr+guard*2,wp=dw.ptr+guard,wsp=dws.ptr+guard,cp=dc.ptr+guard,csp=dcs.ptr+guard;
    auto ip=di.ptr+guard*4,sp=ds.ptr+guard*4,op=output.ptr+guard*2,ep=errors.ptr+guard*4;
    auto s=scale;
    Buffer<unsigned> page_buffer(api,pages);auto page_pointer=paged ? page_buffer.ptr:DevicePtr{0};
    void* args[]={&qp,&wp,&wsp,&cp,&csp,&ip,&sp,&op,&ep,&rows,&heads,&dim,&selected,&wr,&cr,&qs,&ws,&wss,&cs,&css,&is,&os,&s,&page_pointer,&page_rows};
#ifdef DS41_ATTENTION_PAIR_EXPANDED
    std::vector<std::uint16_t> expanded_values((wr+cr)*dim);
    for(unsigned r=0;r<wr+cr;++r)for(unsigned d=0;d<dim;++d)expanded_values[r*dim+d]=bf(value(r,d));
    Buffer<std::uint16_t> expanded_cache(api,expanded_values);auto expanded_pointer=expanded_cache.ptr;args[1]=&expanded_pointer;
#endif
#if defined(DS41_ATTENTION_PREPARE) || defined(DS41_ATTENTION_LIBRARY)
    const unsigned count=wr+cr,padded_heads=16;unsigned batch=1,cache_rows=count;
    Buffer<std::uint16_t> expanded(api,std::vector<std::uint16_t>(static_cast<std::size_t>(count)*dim+guard*2,0x4abc)),padded_query(api,std::vector<std::uint16_t>(static_cast<std::size_t>(rows)*16*dim+guard*2,0x4abc)),padded_output(api,std::vector<std::uint16_t>(static_cast<std::size_t>(rows)*16*dim+guard*2,0x4abc));
    Buffer<int> packed(api,std::vector<int>(rows*selected+guard*2,-778));Buffer<float> padded_sink(api,std::vector<float>(16+guard*2,-12345));
    auto pq=padded_query.ptr+guard*2,kv=expanded.ptr+guard*2,pi=packed.ptr+guard*4,ps=padded_sink.ptr+guard*4,po=padded_output.ptr+guard*2;
    void* prepare_args[]={&qp,&wp,&wsp,&cp,&csp,&ip,&sp,&page_pointer,&pq,&kv,&pi,&ps,&ep,&rows,&heads,&dim,&selected,&wr,&cr,&qs,&ws,&wss,&cs,&css,&is,&page_rows};
    const auto work=std::max<std::uint64_t>({(static_cast<std::uint64_t>(count)*dim+7)/8,(static_cast<std::uint64_t>(rows)*16*dim+7)/8,(static_cast<std::uint64_t>(rows)*selected+3)/4,16});
    check(api.launch(prepare_fn,(work+255)/256,1,1,256,1,1,0,nullptr,prepare_args,nullptr));check(api.sync());
    auto want_kv=expanded.host,want_q=padded_query.host;auto want_i=packed.host;auto want_s=padded_sink.host;
    for(unsigned r=0;r<count;++r)for(unsigned d=0;d<dim;++d)want_kv[guard+r*dim+d]=bf(value(r,d));
    for(unsigned r=0;r<rows;++r)for(unsigned h=0;h<16;++h)for(unsigned d=0;d<dim;++d)want_q[guard+(r*16+h)*dim+d]=h<heads?q[guard+r*qs+h*dim+d]:0;
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<selected;++j){const auto v=idx[guard+r*is+j];want_i[guard+r*selected+j]=v>=0&&unsigned(v)<count?v:-1;}
    for(unsigned h=0;h<16;++h)want_s[guard+h]=h<heads?sink[guard+h]:0;
    require(expanded.read()==want_kv&&padded_query.read()==want_q&&packed.read()==want_i&&padded_sink.read()==want_s,"attention BF16/head/index preparation oracle or guards mismatch");
#ifdef DS41_ATTENTION_LIBRARY
    require(dim==512,"library fixture dimension");auto set_attribute=load<int(*)(Function,int,int)>("cuFuncSetAttribute");check(set_attribute(library_fn,8,86016));
    void* library_args[]={&ps,&kv,&po,&pq,&pi,&batch,&rows,&cache_rows,&selected};check(api.launch(library_fn,rows,1,1,256,1,1,86016,nullptr,library_args,nullptr));
#else
    po=pq;for(unsigned r=0;r<rows;++r)for(unsigned d=0;d<heads*dim;++d)expected[guard+r*os+d]=q[guard+r*qs+d];
#endif
    void* unpack_args[]={&po,&op,&rows,&heads,&dim,&os};check(api.launch(unpack_fn,(static_cast<std::uint64_t>(rows)*heads*dim+2047)/2048,1,1,256,1,1,0,nullptr,unpack_args,nullptr));check(api.sync());
#elif defined(DS41_ATTENTION_SPLIT)
    const unsigned tiles=(selected+63)/64;
    Buffer<float> score_buffer(api,std::vector<float>(rows*heads*selected+guard*2,-12345)),alpha_buffer(api,std::vector<float>(rows*heads*(tiles+1)+guard*2,-12345)),partial_buffer(api,std::vector<float>(rows*heads*tiles*dim+guard*2,-12345));
    Buffer<std::uint16_t> probability_buffer(api,std::vector<std::uint16_t>(rows*heads*selected+guard*2,0x4abc));
    auto score_pointer=score_buffer.ptr+guard*4,probability_pointer=probability_buffer.ptr+guard*2,alpha_pointer=alpha_buffer.ptr+guard*4,partial_pointer=partial_buffer.ptr+guard*4;
    std::vector<void*> split_args(std::begin(args),std::end(args));for(auto* p:{&score_pointer,&probability_pointer,&alpha_pointer,&partial_pointer})split_args.push_back(p);
    if(selected)check(api.launch(split_functions[0],rows,heads,(selected+3)/4,128,1,1,0,nullptr,split_args.data(),nullptr));
    check(api.launch(split_functions[1],rows,heads,1,32,1,1,0,nullptr,split_args.data(),nullptr));
    if(selected)check(api.launch(split_functions[2],rows,heads,tiles*((dim+127)/128),128,1,1,0,nullptr,split_args.data(),nullptr));
    check(api.launch(split_functions[3],rows,heads,(dim+127)/128,128,1,1,0,nullptr,split_args.data(),nullptr));check(api.sync());
    for(auto* buffer:{&score_buffer,&alpha_buffer,&partial_buffer}){const auto values=buffer->read();for(unsigned i=0;i<guard;++i)require(values[i]==-12345&&values[values.size()-1-i]==-12345,"split attention float workspace guard modified");}
    const auto pv=probability_buffer.read();for(unsigned i=0;i<guard;++i)require(pv[i]==0x4abc&&pv[pv.size()-1-i]==0x4abc,"split attention probability guard modified");
#elif defined(DS41_ATTENTION_PAIR)
    const unsigned shared=288*dim+18432;
    check(load<int(*)(Function,int,int)>("cuFuncSetAttribute")(fn,8,shared));
    check(api.launch(fn,(rows+1)/2,1,1,256,1,1,shared,nullptr,args,nullptr));check(api.sync());
#elif defined(DS41_ATTENTION_TC)
    const unsigned shared=160*dim+7872;
    auto set_attribute=load<int(*)(Function,int,int)>("cuFuncSetAttribute");check(set_attribute(fn,8,shared));
    check(api.launch(fn,rows,(heads+15)/16,1,128,1,1,shared,nullptr,args,nullptr));check(api.sync());
#else
    check(api.launch(fn,rows,heads,1,32,1,1,0,nullptr,args,nullptr));check(api.sync());
#endif
    auto actual=output.read();
    for(unsigned r=0;r<rows;++r) for(unsigned d=0;d<heads*dim;++d) {
        const auto index=guard+r*os+d;const auto a=bf(actual[index]),e=bf(expected[index]);
        const auto error=std::abs(double(a)-e);maximum_error=std::max(maximum_error,error);
        require(std::isfinite(a) && error<=.0005+.0078125*std::abs(e),"compact attention numerical tolerance exceeded");
        actual[index]=expected[index];
    }
    require(actual==expected,"attention output padding/guards modified");
    require(errors.read()==expected_error,"attention invalid-index error handling mismatch");
    require(dq.read()==q && dw.read()==w && dws.read()==wscale && dc.read()==c && dcs.read()==cscale && di.read()==idx && ds.read()==sink,"attention input mutated");
}
int main(int argc,char** argv) {
    try {
        const auto native=prepare_execution(argc,argv);std::ifstream file(argv[1]);require(file.good(),"PTX missing");
        std::string ptx{std::istreambuf_iterator<char>(file),{}};
        Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));
        Module module;check(api.module(&module,ptx.c_str()));Function fn,f64,f512;
#if defined(DS41_ATTENTION_PREPARE)
        check(api.function(&prepare_fn,module,"ds41_prepare_sparse_bf16"));check(api.function(&unpack_fn,module,"ds41_unpack_sparse_bf16"));fn=f64=f512=prepare_fn;
#elif defined(DS41_ATTENTION_LIBRARY)
        check(api.function(&library_fn,module,"sparse_attn_kernel__kernel"));
        const auto helper_path=std::filesystem::path(argv[1]).parent_path()/"attention_prepare.cubin";std::ifstream helper_file(helper_path);std::string helper{std::istreambuf_iterator<char>(helper_file),{}};Module helper_module;check(api.module(&helper_module,helper.c_str()));check(api.function(&prepare_fn,helper_module,"ds41_prepare_sparse_bf16"));check(api.function(&unpack_fn,helper_module,"ds41_unpack_sparse_bf16"));fn=f64=f512=prepare_fn;
#elif defined(DS41_ATTENTION_SPLIT)
        const char* names[]={"ds41_attention_split_scores","ds41_attention_split_probabilities","ds41_attention_split_values","ds41_attention_split_finish"};for(unsigned i=0;i<4;++i)check(api.function(&split_functions[i],module,names[i]));fn=f64=f512=split_functions[0];
#elif defined(DS41_ATTENTION_PAIR)
#ifdef DS41_ATTENTION_PAIR_EXPANDED
        check(api.function(&fn,module,"ds41_sparse_attention_pair_expanded32"));check(api.function(&f64,module,"ds41_sparse_attention_pair_expanded64"));check(api.function(&f512,module,"ds41_sparse_attention_pair_expanded512"));
#else
        check(api.function(&fn,module,"ds41_sparse_attention_pair32"));check(api.function(&f64,module,"ds41_sparse_attention_pair64"));check(api.function(&f512,module,"ds41_sparse_attention_pair512"));
#endif
#elif defined(DS41_ATTENTION_TC)
        check(api.function(&fn,module,"ds41_sparse_attention_tc32"));check(api.function(&f64,module,"ds41_sparse_attention_tc64"));check(api.function(&f512,module,"ds41_sparse_attention_tc512"));
#else
        check(api.function(&fn,module,"ds41_sparse_attention"));f64=f512=fn;
#endif
#ifdef DS41_DRPERF_ATTENTION_SLICE
        fixture(api,f512,512,64,0,false,true);fixture(api,f512,512,128,0,false,true);
#elif defined(DS41_ATTENTION_LIBRARY)
        fixture(api,fn,512,9,0);fixture(api,fn,512,67,0,true);fixture(api,fn,512,64,1);fixture(api,fn,512,7,2);fixture(api,fn,512,7,3);fixture(api,fn,512,0,0);fixture(api,fn,512,640,0,false,true);
#else
        fixture(api,fn,32,7,0);fixture(api,f64,64,67,0);fixture(api,f512,512,9,0);
        fixture(api,fn,32,64,1);fixture(api,fn,32,7,2);fixture(api,fn,32,7,3);fixture(api,fn,32,0,0);
        fixture(api,f512,512,67,0,true);fixture(api,fn,32,7,2,true);
        fixture(api,f512,512,640,0,false,true);
#endif
        check(api.unload(module));check(api.destroy(context));
        std::cout<<"{\"numerical_status\":\"pass\",\"kernel\":\""
#ifdef DS41_ATTENTION_PREPARE
                 <<"ds41_prepare_and_unpack_sparse_bf16"
#elif defined(DS41_ATTENTION_LIBRARY)
                 <<"primary_reference_sparse_attention_with_custom_prepare_unpack"
#else
                 <<"ds41_sparse_attention"
#endif
                 <<"\",\"cases\":"
#ifdef DS41_DRPERF_ATTENTION_SLICE
                 <<2
#elif defined(DS41_ATTENTION_LIBRARY)
                 <<7
#else
                 <<10
#endif
                 <<",\"max_absolute_error\":"<<maximum_error
                 <<",\"tolerance\":\"0.0005 + 0.0078125*abs(reference BF16)\",\"physical_gpu_used\":"<<(native ? "true":"false")<<"}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
