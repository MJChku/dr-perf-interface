#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
using namespace ds41::gx_fixture;
static void require(bool v,const char* s) { if(!v) throw std::runtime_error(s); }
static float decode(unsigned c,bool fp4) {
    float value;
    if (fp4) { const float t[]={0,.5f,1,1.5f,2,3,4,6};value=t[c&7];return c&8 ? -value:value; }
    const auto mag=c&127; const int exponent=mag/8;
    value=exponent ? std::ldexp(1+float(mag%8)/8,exponent-7):float(mag%8)/512;
    return c&128 ? -value:value;
}
static std::uint16_t bf16(float x) {
    const auto bits=std::bit_cast<std::uint32_t>(x);return (bits+32767+((bits>>16)&1))>>16;
}
static float bf16(std::uint16_t x) { return std::bit_cast<float>(std::uint32_t(x)<<16); }
static Function finish;
static void fixture(Driver& api,Function function,unsigned m,unsigned n,unsigned k,unsigned formats,bool paged=false,unsigned pattern=0) {
    const bool fp4=formats&1,afp4=formats&2;
    unsigned sa=(afp4 ? k/2:k)+8,sb=(fp4 ? k/2:k)+8,sas=k/32+3,sbs=k/32+5,so=n+7;
    unsigned page_rows=8;
    std::vector<unsigned> pages((n+page_rows-1)/page_rows);
    for(unsigned p=0;p<pages.size();++p)pages[p]=pages.size()-p;
    const unsigned physical_rows=paged ? (pages.size()+1)*page_rows:n;
    const unsigned bsr=fp4 ? physical_rows:(n+31)/32,guard=32;
    std::vector<std::uint8_t> a(m*sa+guard*2,0xA7),b(physical_rows*sb+guard*2,0xA7);
    std::vector<std::uint8_t> as(m*sas+guard*2,0xA7),bs(bsr*sbs+guard*2,0xA7);
    std::vector<std::uint16_t> expected(m*so+guard*2,0x4abc);
    auto initial=expected;
    for(unsigned r=0;r<m;++r) {
        for(unsigned c=0;c<(afp4 ? k/2:k);++c) a[guard+r*sa+c]=afp4 ? ds41::synthetic::fp4(134,r*k/2+c):ds41::synthetic::fp8(134,r*k+c);
        for(unsigned c=0;c<k/32;++c) as[guard+r*sas+c]=123+(r+c)%5;
    }
    for(unsigned r=0;r<physical_rows;++r)
        for(unsigned c=0;c<(fp4 ? k/2:k);++c)
            b[guard+r*sb+c]=fp4 ? ds41::synthetic::fp4(197,r*k/2+c):ds41::synthetic::fp8(197,r*k+c);
    for(unsigned r=0;r<bsr;++r)
        for(unsigned c=0;c<k/32;++c) bs[guard+r*sbs+c]=122+(r*2+c)%6;
    if(pattern){
        auto code=[&](unsigned i,bool four){
            if(pattern==1){unsigned c=i%(four?16:256);if(!four&&(c&127)==127)--c;return c;}
            if(pattern==2)return four?7u:126u;
            if(pattern==3)return (i%32==31)?1u:(four?7u:126u);
            return four?15u:126u;
        };
        for(unsigned r=0;r<m;++r)for(unsigned c=0;c<(afp4?k/2:k);++c)a[guard+r*sa+c]=afp4?code(r*k+c*2,true)|(code(r*k+c*2+1,true)<<4):code(r*k+c,false);
        for(unsigned r=0;r<physical_rows;++r)for(unsigned c=0;c<(fp4?k/2:k);++c)b[guard+r*sb+c]=fp4?code(r*k+c*2,true)|(code(r*k+c*2+1,true)<<4):code(r*k+c,false);
    }
    if(pattern==4&&!fp4)for(unsigned r=0;r<physical_rows;++r)for(unsigned c=0;c<k;++c)b[guard+r*sb+c]^=128;
    if(pattern>=5){
        auto maximum=[](bool four){return four?7u:126u;};
        for(unsigned r=0;r<m;++r)for(unsigned c=0;c<k;++c){const unsigned v=maximum(afp4);auto& byte=a[guard+r*sa+(afp4?c/2:c)];if(afp4)byte=(byte&~(15u<<((c%2)*4)))|(v<<((c%2)*4));else byte=v;}
        for(unsigned r=0;r<physical_rows;++r)for(unsigned c=0;c<k;++c){unsigned v=c%32<30?maximum(fp4)|((pattern==5?c%2:c%32>=15)?(fp4?8u:128u):0u):1u;auto& byte=b[guard+r*sb+(fp4?c/2:c)];if(fp4)byte=(byte&~(15u<<((c%2)*4)))|(v<<((c%2)*4));else byte=v;}
    }
    // Scalar double products, scale each K32 contribution independently, then
    // FP32 accumulation and final BF16 rounding. No device fragment layout is
    // reused by this oracle.
    for(unsigned r=0;r<m;++r) for(unsigned c=0;c<n;++c) {
        const auto physical=paged ? pages[c/page_rows]*page_rows+c%page_rows:c;
        float total=0;
        for(unsigned chunk=0;chunk<k/32;++chunk) {
            double dot=0;
            for(unsigned p=0;p<32;++p) {
                const auto offset=chunk*32+p;
                const auto bc=fp4 ? ((b[guard+physical*sb+offset/2]>>((offset%2)*4))&15):b[guard+physical*sb+offset];
                const auto ac=afp4 ? ((a[guard+r*sa+offset/2]>>((offset%2)*4))&15):a[guard+r*sa+offset];
                dot+=double(decode(ac,afp4))*decode(bc,fp4);
            }
            const float scale_a=std::ldexp(1.0f,int(as[guard+r*sas+chunk])-127);
            const float scale_b=std::ldexp(1.0f,int(bs[guard+(fp4 ? physical:physical/32)*sbs+chunk])-127);
            total += (float(dot)*scale_a)*scale_b;
        }
        expected[guard+r*so+c]=bf16(total);
    }
    Buffer<std::uint8_t> da(api,a),db(api,b),das(api,as),dbs(api,bs);
    Buffer<std::uint16_t> output(api,initial);
    Buffer<unsigned> page_buffer(api,pages);
    auto page_pointer=paged ? page_buffer.ptr:DevicePtr{0};
    auto ap=da.ptr+guard,bp=db.ptr+guard,asp=das.ptr+guard,bsp=dbs.ptr+guard,op=output.ptr+guard*2;
    void* args[]={&ap,&bp,&asp,&bsp,&op,&m,&n,&k,&formats,&sa,&sb,&sas,&sbs,&so,&page_pointer,&page_rows};
    if(m && n) {
#ifdef DS41_MATVEC_FUSED
        check(api.launch(function,(n+7)/8,m,1,256,1,1,(k/32)*8*sizeof(float),nullptr,args,nullptr));
#elif defined(DS41_MATVEC_TC)
        Buffer<float> partial(api,std::vector<float>(std::uint64_t(m)*n*(k/32)+2*guard,-779));auto pp=partial.ptr+guard*4;unsigned groups=k/32;
        args[4]=&pp;check(api.launch(function,(n+7)/8,m,(groups+3)/4,32,1,1,0,nullptr,args,nullptr));
        void* fa[]={&pp,&op,&m,&n,&groups,&so};check(api.launch(finish,(m*n+127)/128,1,1,128,1,1,0,nullptr,fa,nullptr));check(api.sync());
        auto workspace=partial.read();for(unsigned i=0;i<guard;++i)require(workspace[i]==-779&&workspace[workspace.size()-1-i]==-779,"TC matvec partial guard changed");
#elif defined(DS41_MATVEC)
        check(api.launch(function,(n+3)/4,m,1,128,1,1,0,nullptr,args,nullptr));
#else
        check(api.launch(function,(n+63)/64,(m+31)/32,1,128,1,1,0,nullptr,args,nullptr));
#endif
        check(api.sync());
    }
    const auto actual=output.read();
    for(unsigned r=0;r<m;++r) for(unsigned c=0;c<n;++c) {
        const auto index=guard+r*so+c;
        if(actual[index]!=expected[index]) {
            std::cerr<<"m="<<m<<" n="<<n<<" k="<<k<<" fp4="<<fp4<<" row="<<r<<" col="<<c
                     <<" actual="<<bf16(actual[index])<<" expected="<<bf16(expected[index])<<'\n';
            throw std::runtime_error("scaled matrix output mismatch");
        }
    }
    require(actual==expected,"GEMM padding/guard changed");
    require(da.read()==a && db.read()==b && das.read()==as && dbs.read()==bs,"GEMM input mutated");
}
int main(int argc,char** argv) {
    try {
        const auto native=prepare_execution(argc,argv);
        std::ifstream file(argv[1]);require(file.good(),"PTX missing");
        std::string ptx{std::istreambuf_iterator<char>(file),{}};
        Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));
        Module module;check(api.module(&module,ptx.c_str()));Function fn;
#ifdef DS41_MATVEC_FUSED
        check(api.function(&fn,module,"ds41_matvec_tc_fused"));
#elif defined(DS41_MATVEC_TC)
        check(api.function(&fn,module,"ds41_matvec_tc_partials"));check(api.function(&finish,module,"ds41_matvec_tc_finish"));
#elif defined(DS41_MATVEC)
        check(api.function(&fn,module,"ds41_scaled_matvec"));
#else
        check(api.function(&fn,module,"ds41_scaled_gemm"));
#endif
        unsigned cases=0;
        for(unsigned kind=0;kind<4;++kind) {
            fixture(api,fn,1,8,32,kind);++cases;
            fixture(api,fn,3,17,64,kind);++cases;
            fixture(api,fn,17,65,96,kind);++cases;
            fixture(api,fn,33,9,32,kind);++cases;
            fixture(api,fn,0,17,32,kind);++cases;
            if(kind&1){fixture(api,fn,3,65,64,kind,true);++cases;}
        }
#if defined(DS41_MATVEC) || defined(DS41_MATVEC_TC) || defined(DS41_MATVEC_FUSED)
        for(unsigned kind=0;kind<4;++kind)for(unsigned pattern=1;pattern<=6;++pattern){fixture(api,fn,3,17,512,kind,false,pattern);++cases;}
#endif
#if defined(DS41_MATVEC_TC) || defined(DS41_MATVEC_FUSED)
        for(unsigned kind=0;kind<2;++kind){fixture(api,fn,1,17,5120,kind);++cases;}
#endif
        check(api.unload(module));check(api.destroy(context));
        std::cout<<"{\"numerical_status\":\"pass\",\"kernel\":\"ds41_scaled_gemm\",\"cases\":"<<cases
                 <<",\"comparison\":\"bitwise BF16 vs scalar scaled oracle\",\"physical_gpu_used\":"<<(native ? "true":"false")<<"}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
