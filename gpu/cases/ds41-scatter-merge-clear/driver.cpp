#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace ds41::gx_fixture;
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static std::uint16_t bf(float v){auto b=std::bit_cast<std::uint32_t>(v);return (b+32767+((b>>16)&1))>>16;}
static float bf(std::uint16_t v){return std::bit_cast<float>(std::uint32_t(v)<<16);}
static float fp8(unsigned v){unsigned m=v&127;float x=m/8?std::ldexp(1+(m%8)/8.0f,int(m/8)-7):(m%8)/512.0f;return v&128?-x:x;}
static void lookup(Driver& api,Function fn,unsigned mode){
    unsigned lookups=7,dim=64,ts=dim+3,ss=dim/32+2,os=dim+5;std::uint64_t first=5,local=7;constexpr unsigned g=32;
    std::vector<std::int64_t> ids={-1,4,5,11,12,6,10000000001ll};
    std::vector<std::uint16_t> table(local*ts+2*g,0x4abc),initial(lookups*os+2*g,0x4abc),expected=initial;
    std::vector<std::uint8_t> packed(local*ts+2*g,0xa7),scales(local*ss+2*g,0xa7);
    for(unsigned r=0;r<local;++r){for(unsigned d=0;d<dim;++d){table[g+r*ts+d]=bf((int((r*11+d*17)%101)-50)*.0625f);packed[g+r*ts+d]=ds41::synthetic::fp8(177,r*dim+d);}for(unsigned d=0;d<dim/32;++d)scales[g+r*ss+d]=123+(r+d)%5;}
    for(unsigned r=0;r<lookups;++r)for(unsigned d=0;d<dim;++d){float value=0;auto id=ids[r];if(id>=int(first)&&id<int(first+local)){auto row=id-first;value=mode ? fp8(packed[g+row*ts+d])*std::ldexp(1.0f,int(scales[g+row*ss+d/32])-127):bf(table[g+row*ts+d]);}expected[g+r*os+d]=bf(value);}
    Buffer<std::uint16_t> dt(api,table),out(api,initial);Buffer<std::uint8_t> dp(api,packed),ds(api,scales);Buffer<std::int64_t> di(api,ids);
    auto tp=mode?dp.ptr+g:dt.ptr+g*2,sp=ds.ptr+g,ip=di.ptr,op=out.ptr+g*2;
    void* args[]={&tp,&sp,&ip,&op,&lookups,&dim,&first,&local,&mode,&ts,&ss,&os};
    check(api.launch(fn,(lookups*dim+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());
    require(out.read()==expected,"sharded embedding/Engram lookup mismatch");require(dt.read()==table&&dp.read()==packed&&ds.read()==scales&&di.read()==ids,"lookup mutated inputs");
}
static void engram(Driver& api,Function fn,unsigned dim,bool inplace){
    unsigned rows=3,hs=4*dim+7,ks=5*dim+5,os=inplace?hs:4*dim+3;constexpr unsigned g=32;
    std::vector<std::uint16_t> h(rows*hs+2*g,0x4abc),kv(rows*ks+2*g,0x4abc),qw(4*dim+2*g,0x4abc),kw=qw;
    for(unsigned r=0;r<rows;++r){for(unsigned d=0;d<4*dim;++d)h[g+r*hs+d]=bf(r==1?0:float(int((r*11+d*7)%83)-41)*.09375f);for(unsigned d=0;d<5*dim;++d)kv[g+r*ks+d]=bf(r==2&&d<4*dim?0:float(int((r*19+d*11)%71)-35)*.0625f);}
    for(unsigned d=0;d<4*dim;++d){qw[g+d]=bf(1+(int(d%7)-3)*.0625f);kw[g+d]=bf(.5f+(int(d%5)-2)*.03125f);}
    std::vector<std::uint16_t> initial=inplace?h:std::vector<std::uint16_t>(rows*os+2*g,0x4abc),expected=initial;
    for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<4;++c){
        double h2=0,k2=0,dot=0;for(unsigned d=0;d<dim;++d){double a=bf(h[g+r*hs+c*dim+d]),b=bf(kv[g+r*ks+c*dim+d]);h2+=a*a;k2+=b*b;dot+=(a*bf(qw[g+c*dim+d])*bf(kw[g+c*dim+d]))*b;}
        double normalized=dot/std::sqrt((h2/dim+1e-20)*(k2/dim+1e-20)*dim),root=std::copysign(std::sqrt(std::max(std::abs(normalized),1e-6)),normalized),gate=1/(1+std::exp(-root));
        for(unsigned d=0;d<dim;++d)expected[g+r*os+c*dim+d]=bf(float(bf(h[g+r*hs+c*dim+d])+gate*bf(kv[g+r*ks+4*dim+d])));
    }
    Buffer<std::uint16_t> dh(api,h),dk(api,kv),dq(api,qw),dkw(api,kw),out(api,initial);
    auto hp=dh.ptr+g*2,kp=dk.ptr+g*2,qp=dq.ptr+g*2,kwp=dkw.ptr+g*2,op=inplace?hp:out.ptr+g*2;
    void* args[]={&hp,&kp,&qp,&kwp,&op,&rows,&dim,&hs,&ks,&os};
    check(api.launch(fn,rows,4,1,32,1,1,0,nullptr,args,nullptr));check(api.sync());
    auto actual=inplace?dh.read():out.read();
    for(unsigned r=0;r<rows;++r)for(unsigned d=0;d<4*dim;++d){unsigned i=g+r*os+d;float a=bf(actual[i]),e=bf(expected[i]);require(std::isfinite(a)&&std::abs(a-e)<=.0005+.0078125*std::abs(e),"Engram normalized signed-sqrt gate mismatch");actual[i]=expected[i];}
    require(actual==expected,"Engram output padding/guard modified");require(dk.read()==kv&&dq.read()==qw&&dkw.read()==kw,"Engram weights/keys mutated");if(!inplace)require(dh.read()==h,"Engram residual mutated");
}
static void moe(Driver& api,Function scatter,Function finish,Function cast,bool padding=false){
    unsigned rows=5,dim=33,tokens=3,is=dim+3,os=dim+5,ss=dim+7,fs=dim+9;constexpr unsigned g=32;
    std::vector<int> ids={2,0,2,1,padding?-1:-2},error(1,0);
    std::vector<std::uint16_t> x(rows*is+2*g,0x4abc),out(tokens*fs+2*g,0x4abc);
    std::vector<float> sum(tokens*os+2*g,-779),shared(tokens*ss+2*g,-779);
    for(unsigned r=0;r<rows;++r)for(unsigned d=0;d<dim;++d)x[g+r*is+d]=bf((int((r*11+d*7)%61)-30)*.03125f);
    for(unsigned r=0;r<tokens;++r)for(unsigned d=0;d<dim;++d){sum[g+r*os+d]=.125f;shared[g+r*ss+d]=(int((r*19+d*31)%67)-33)*.031123f;}
    auto expected=sum;for(unsigned r=0;r<rows;++r)if(ids[r]>=0)for(unsigned d=0;d<dim;++d)expected[g+ids[r]*os+d]+=bf(x[g+r*is+d]);
    Buffer<std::uint16_t> dx(api,x),doo(api,out);Buffer<int> di(api,ids),de(api,error);Buffer<float> da(api,sum),ds(api,shared);
    auto xp=dx.ptr+g*2,ip=di.ptr,ap=da.ptr+g*4,ep=de.ptr,sp=ds.ptr+g*4,op=doo.ptr+g*2;
    void* args[]={&xp,&ip,&ap,&ep,&rows,&dim,&tokens,&is,&os};check(api.launch(scatter,(rows*dim+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());
    require(da.read()==expected&&de.read()[0]==(padding?0:1),"expert scatter duplicate/invalid/padding handling mismatch");
    auto expected_out=out;for(unsigned r=0;r<tokens;++r)for(unsigned d=0;d<dim;++d)expected_out[g+r*fs+d]=bf(expected[g+r*os+d]+bf(bf(shared[g+r*ss+d])));
    void* fa[]={&ap,&sp,&op,&tokens,&dim,&os,&ss,&fs};check(api.launch(finish,(tokens*dim+127)/128,1,1,128,1,1,0,nullptr,fa,nullptr));check(api.sync());require(doo.read()==expected_out,"shared BF16 boundary plus routed accumulator mismatch");
    unsigned mode=0;Buffer<float> expanded(api,std::vector<float>(rows*os+2*g,-779));auto fp=expanded.ptr+g*4;void* ca[]={&xp,&fp,&rows,&dim,&mode,&is,&os};check(api.launch(cast,(rows*dim+127)/128,1,1,128,1,1,0,nullptr,ca,nullptr));check(api.sync());
    auto floats=expanded.read();for(unsigned r=0;r<rows;++r)for(unsigned d=0;d<dim;++d)require(floats[g+r*os+d]==bf(x[g+r*is+d]),"BF16-to-FP32 conversion mismatch");
    Buffer<std::uint16_t> back(api,std::vector<std::uint16_t>(rows*is+2*g,0x4abc));auto bp=back.ptr+g*2;mode=1;void* cb[]={&fp,&bp,&rows,&dim,&mode,&os,&is};check(api.launch(cast,(rows*dim+127)/128,1,1,128,1,1,0,nullptr,cb,nullptr));check(api.sync());require(back.read()==x,"FP32-to-BF16 roundtrip or padding mismatch");
    require(dx.read()==x&&di.read()==ids&&ds.read()==shared,"MoE data kernel mutated source");
}
static void expand(Driver& api,Function fn){
    unsigned rows=3,dim=33;std::vector<std::uint16_t> input(rows*dim),out(rows*4*dim+64,0x4abc),expected=out;std::vector<float> pre(rows*4+64,-779),want=pre;
    for(unsigned i=0;i<input.size();++i)input[i]=bf((int(i%31)-15)*.0625f);
    for(unsigned r=0;r<rows;++r)for(unsigned c=0;c<4;++c){want[32+r*4+c]=c==0?1:0;for(unsigned d=0;d<dim;++d)expected[32+(r*4+c)*dim+d]=input[r*dim+d];}
    Buffer<std::uint16_t> di(api,input),doo(api,out);Buffer<float> dp(api,pre);auto ip=di.ptr,op=doo.ptr+64,pp=dp.ptr+128;void* args[]={&ip,&op,&pp,&rows,&dim};check(api.launch(fn,(rows*4*dim+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());require(doo.read()==expected&&dp.read()==want&&di.read()==input,"embedding HC expansion or exact initial premix mismatch");
}
static void aligned_cast(Driver& api,Function fn,unsigned dim,unsigned mode){
    unsigned rows=3,is=dim,os=dim;constexpr unsigned g=32;
    std::vector<float> fp(rows*dim+2*g,-773),want=fp;
    std::vector<std::uint16_t> bp(rows*dim+2*g,0x4abc),bw=bp;
    for(unsigned i=0;i<rows*dim;++i){fp[g+i]=(int(i*37%513)-256)*.001237f;bp[g+i]=bf(fp[g+i]);want[g+i]=bf(bp[g+i]);bw[g+i]=bf(fp[g+i]);}
    auto fi=mode?fp:std::vector<float>(fp.size(),-773);auto bi=mode?std::vector<std::uint16_t>(bp.size(),0x4abc):bp;
    Buffer<float> df(api,fi);Buffer<std::uint16_t> db(api,bi);
    auto ip=mode?df.ptr+g*4:db.ptr+g*2,op=mode?db.ptr+g*2:df.ptr+g*4;
    void* args[]={&ip,&op,&rows,&dim,&mode,&is,&os};
    #ifdef DS41_SCALAR_CAST
    check(api.launch(fn,(rows*dim+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));
#else
    check(api.launch(fn,(rows*dim+2047)/2048,1,1,256,1,1,0,nullptr,args,nullptr));
#endif
    check(api.sync());
    require(mode?db.read()==bw:df.read()==want,"aligned vector conversion/guard mismatch");require(mode?df.read()==fi:db.read()==bi,"aligned cast input modified");
}
int main(int argc,char** argv){try{
    const auto native=prepare_execution(argc,argv);std::ifstream f(argv[1]);require(f.good(),"PTX missing");std::string ptx{std::istreambuf_iterator<char>(f),{}};Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));Module module;check(api.module(&module,ptx.c_str()));
    Function lookup_fn,gate,scatter,finish,cast,expand_fn;check(api.function(&lookup_fn,module,"ds41_table_lookup"));check(api.function(&gate,module,"ds41_engram_gate"));check(api.function(&scatter,module,"ds41_scatter_add"));check(api.function(&finish,module,"ds41_finish_moe"));check(api.function(&cast,module,"ds41_cast_rows"));check(api.function(&expand_fn,module,"ds41_expand_hc"));
    lookup(api,lookup_fn,0);lookup(api,lookup_fn,1);for(unsigned dim:{33u,64u,512u}){engram(api,gate,dim,false);engram(api,gate,dim,true);}moe(api,scatter,finish,cast);moe(api,scatter,finish,cast,true);expand(api,expand_fn);
    for(unsigned dim:{8u,5120u,20480u})for(unsigned mode:{0u,1u})aligned_cast(api,cast,dim,mode);
    check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"kernels\":\"embedding, Engram gate, expert scatter/finish, casts, HC expansion\",\"cases\":17,\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
