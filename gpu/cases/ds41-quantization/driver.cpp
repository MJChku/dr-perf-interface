#include "ds41/gx_fixture.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
using namespace ds41::gx_fixture;
static void require(bool value,const char* error) { if (!value) throw std::runtime_error(error); }
static float bf(std::uint16_t v) { return std::bit_cast<float>(std::uint32_t(v)<<16); }
static std::uint16_t bf(float v) {
    auto bits=std::bit_cast<std::uint32_t>(v);
    return (bits+32767+((bits>>16)&1))>>16;
}
static float decode(unsigned code,bool fp8) {
    unsigned sign = code & (fp8 ? 128 : 8), magnitude=code & (fp8 ? 127 : 7);
    float value;
    if (fp8) {
        const int e=magnitude/8, m=magnitude%8;
        value=e ? std::ldexp(1+m/8.0f,e-7) : m/512.0f;
    } else {
        const float table[]={0,.5f,1,1.5f,2,3,4,6};value=table[magnitude];
    }
    return sign ? -value : value;
}
// Independent exhaustive nearest-neighbour oracle rather than the device's
// threshold/bit encoding. Includes signed zero and significand parity ties.
static unsigned encode(float v,bool fp8) {
    const unsigned sign=std::signbit(v) ? (fp8 ? 128 : 8) : 0;
    const auto magnitude=std::abs(v);
    unsigned result=0;
    double distance=std::numeric_limits<double>::infinity();
    for (unsigned code=0;code<(fp8 ? 127u : 8u);++code) {
        const double delta=std::abs(double(magnitude)-decode(code,fp8));
        if (delta<distance || (delta==distance && !(code&1))) { result=code;distance=delta; }
    }
    return sign | result;
}
static void fixture(Driver& api,Function kernel,unsigned mode,unsigned rows,unsigned width,unsigned variant) {
    const unsigned group=mode==2 ? 16:32, groups=width/group;
    unsigned si=width+7,sp=(mode==0 ? width:width/2)+5,ss=groups+3,sr=width+11;
    constexpr unsigned guard=32;
    std::vector<std::uint16_t> input(rows*si+guard*2,0x4abc), expected_dense(rows*sr+guard*2,0x4abc);
    std::vector<std::uint8_t> expected_packed(rows*sp+guard*2,0xA7), expected_scales(rows*ss+guard*2,0xA7);
    const auto dense_initial=expected_dense;
    const auto packed_initial=expected_packed, scale_initial=expected_scales;
    std::mt19937 rng(4134+variant);
    for (unsigned r=0;r<rows;++r) {
        for (unsigned d=0;d<width;++d) {
            float x=(int(rng()%10001)-5000)/317.0f;
            if (variant==1) x=0;
            if (variant==2) { const float boundary[]={0,-0.0f,.25f,.75f,1.25f,1.75f,2.5f,3.5f,5,6,-6};x=boundary[d%11]; }
            if (variant==3) x=(d%3==0 ? 1:-1)*std::ldexp(float(d%7+1),int(r)-20);
            if (variant==4) x=(d%2 ? -1:1)*448.0f;
            input[guard+r*si+d]=bf(x);
        }
        for (unsigned g=0;g<groups;++g) {
            float maximum=0;
            for (unsigned d=0;d<group;++d) maximum=std::max(maximum,std::abs(bf(input[guard+r*si+g*group+d])));
            unsigned scale_byte;
            float scale;
            if (mode==2) {
                maximum=std::max(maximum,6*std::ldexp(1.0f,-9));
                scale_byte=encode(maximum/6.0f,true); scale=decode(scale_byte,true);
            } else {
                const float minimum=mode==0 ? 1e-4f:6*std::ldexp(1.0f,-126);
                const float reciprocal=mode==0 ? 1.0f/448:1.0f/6;
                float target=std::max(maximum,minimum)*reciprocal;
                int exponent;
                const auto fraction=std::frexp(target,&exponent);
                if (fraction==.5f) --exponent;
                scale=std::ldexp(1.0f,exponent);scale_byte=exponent+127;
            }
            expected_scales[guard+r*ss+g]=scale_byte;
            for (unsigned d=0;d<group;++d) {
                const unsigned column=g*group+d;
                const float value=bf(input[guard+r*si+column]);
                const auto code=encode(value/scale,mode==0);
                if (mode==0) expected_packed[guard+r*sp+column]=code;
                else if (d%2==0) expected_packed[guard+r*sp+column/2]=code;
                else expected_packed[guard+r*sp+column/2] |= code<<4;
                expected_dense[guard+r*sr+column]=bf(decode(code,mode==0)*scale);
            }
        }
    }
    Buffer<std::uint16_t> di(api,input),dense(api,dense_initial);
    Buffer<std::uint8_t> packed(api,packed_initial),scales(api,scale_initial);
    auto ip=di.ptr+guard*2,op=packed.ptr+guard,sc=scales.ptr+guard,dp=dense.ptr+guard*2;
    void* args[]={&ip,&op,&sc,&dp,&rows,&width,&mode,&si,&sp,&ss,&sr};
    if (rows) { check(api.launch(kernel,(rows*groups+3)/4,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync()); }
    require(packed.read()==expected_packed,"packed quantization bytes/strides/guards mismatch");
    require(scales.read()==expected_scales,"quantization scales/strides/guards mismatch");
    require(dense.read()==expected_dense,"BF16 quantization roundtrip mismatch");
    require(di.read()==input,"quantization input mutated");
    // Production stores packed bytes without expanding a second BF16 copy.
    dp=0;
    if(rows){check(api.launch(kernel,(rows*groups+3)/4,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());}
    require(packed.read()==expected_packed&&scales.read()==expected_scales,"packed-only quantization mismatch");
}
int main(int argc,char** argv) {
    try {
        const bool native=prepare_execution(argc,argv);
        std::ifstream file(argv[1]);require(file.good(),"PTX missing");
        std::string ptx{std::istreambuf_iterator<char>(file),{}};
        Driver api;check(api.init(0));Context ctx;check(api.create(&ctx,0,0));
        Module mod;check(api.module(&mod,ptx.c_str()));Function kernel;check(api.function(&kernel,mod,"ds41_quantize_rows"));
        unsigned cases=0;
        for (unsigned mode=0;mode<3;++mode) {
            for (unsigned variant=0;variant<5;++variant) { fixture(api,kernel,mode,3,64,variant);++cases; }
            fixture(api,kernel,mode,1,512,0);++cases;
            fixture(api,kernel,mode,0,32,0);++cases;
        }
        check(api.unload(mod));check(api.destroy(ctx));
        std::cout<<"{\"numerical_status\":\"pass\",\"kernel\":\"ds41_quantize_rows\",\"cases\":"<<cases
                 <<",\"oracle\":\"exhaustive nearest representable codes and independent power scale\",\"physical_gpu_used\":"<<(native ? "true":"false")<<"}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
