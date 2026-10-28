// Independent numerical oracles for the serving-kernel instruction gaps.
#include "drperf_gpu.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using drperf::gpu::Region;
static void check(bool v, const std::string& why) { if (!v) throw std::runtime_error(why); }
static int64_t component(uint32_t bits, unsigned index, unsigned width, bool sign) {
    const uint64_t radix = uint64_t(1) << width;
    int64_t value = (uint64_t(bits) / (uint64_t(1) << (index*width))) % radix;
    return sign && value >= int64_t(radix/2) ? value-int64_t(radix) : value;
}
// Independent oracle: search adjacent representable magnitudes in real values,
// then choose by rounding direction/distance, rather than rounding source bits.
static double bf_magnitude(unsigned bits) {
    unsigned exponent=bits/128, fraction=bits%128;
    return exponent ? std::ldexp(1.0+fraction/128.0,int(exponent)-127) : std::ldexp(double(fraction),-133);
}
static uint16_t bf_oracle(uint32_t bits, unsigned mode) {
    float value; std::memcpy(&value,&bits,4);
    if(std::isnan(value))return 0x7fff;
    if(std::isinf(value))return bits>>16;
    const bool negative=std::signbit(value);const double magnitude=std::abs(double(value));
    unsigned low=0,high=0x7f80;
    while(low<high) {unsigned mid=(low+high+1)/2;if(bf_magnitude(mid)<=magnitude)low=mid;else high=mid-1;}
    unsigned chosen=low;
    if(bf_magnitude(low)!=magnitude) {
        if(mode==0) {
            double down=magnitude-bf_magnitude(low),up=bf_magnitude(low+1)-magnitude;
            if(up<down || (up==down && (low&1)))++chosen;
        } else if((mode==2&&negative)||(mode==3&&!negative))++chosen;
    }
    return uint16_t(chosen|(negative?0x8000:0));
}
int main(int argc, char** argv) { try {
    if (argc != 3) return 2;
    drperf::gpu::Module m(argv[1]); std::string mode=argv[2];
    if (mode=="dot") {
        std::mt19937 rng(123456);
        for (int width : {16,8}) for (int half=0; half<(width==16?2:1); ++half)
        for (bool sa : {false,true}) for (bool sb : {false,true}) {
            std::string name="dot_"+std::to_string(width)+"_"+std::to_string(half)+"_"+std::to_string(sa)+"_"+std::to_string(sb);
            for (unsigned n : {1u,32u,64u}) {
                std::vector<uint32_t> input(3*n), out(n+1,0xabcdef01);
                for (auto& x : input) x=rng();
                // All bits set, signed minima, zero, overflow, byte/half selection.
                const uint32_t edges[]={0,~0u,0x80008000u,0x80808080u,0x7fff7fffu,0x7f7f7f7fu,0x04030201u};
                for (unsigned i=0;i<n&&i<7;++i) {
                    input[3*i]=edges[i];input[3*i+1]=edges[6-i];input[3*i+2]=0xffffffffu-i;
                }
                auto p=input.data(), q=out.data();void* args[]={&p,&q,&q};
                {Region r(name.c_str(),{{"lanes",n}});m.launch(name.c_str(),1,n,args);}
                for (unsigned lane=0;lane<n;++lane) {
                    int64_t want=input[3*lane+2];
                    for(unsigned k=0;k<32/unsigned(width);++k)
                        want += component(input[3*lane],k,width,sa)*component(input[3*lane+1],k+2*half,8,sb);
                    check(out[lane]==uint32_t(want),name+": packed result lane "+std::to_string(lane));
                }
                check(out[n]==0xabcdef01,"dot output guard");
            }
        }
    } else if (mode=="matrix") {
        for(int count : {1,2,4}) for(bool trans : {false,true}) for(int spelling=0;spelling<3;++spelling)
        for(bool cta : {false,true}) {
            std::string name="matrix_"+std::to_string(count)+"_"+std::to_string(trans)+"_"+std::to_string(spelling)+"_"+std::to_string(cta);
            alignas(16) std::array<uint16_t,256> input{};
            alignas(16) std::array<uint32_t,129> fragments{};
            alignas(16) std::array<uint16_t,258> stored{};
            for(unsigned i=0;i<input.size();++i)input[i]=uint16_t(0x8001+137*i);
            fragments.fill(0xdeadbeef); stored.fill(0xabcd);
            auto p=input.data();auto q=fragments.data();auto z=stored.data();void* args[]={&p,&q,&z};
            {Region r(name.c_str(),{{"matrices",count}});m.launch(name.c_str(),1,32,args);}
            for(unsigned lane=0;lane<32;++lane)for(int matrix=0;matrix<count;++matrix) {
                unsigned row=lane/4,col=2*(lane%4);
                uint16_t lo=input[matrix*64+(trans?col*8+row:row*8+col)];
                uint16_t hi=input[matrix*64+(trans?(col+1)*8+row:row*8+col+1)];
                check(fragments[lane*4+matrix]==(uint32_t(lo)|(uint32_t(hi)<<16)),name+": load fragment");
            }
            for(unsigned i=0;i<256;++i)check(stored[i]==(i<unsigned(count*64)?input[i]:0),name+": store mapping");
            check(fragments[128]==0xdeadbeef && stored[256]==0xabcd && stored[257]==0xabcd,"matrix output guard");
        }
    } else if (mode=="bf16") {
        std::mt19937 rng(789);
        for(unsigned round=0;round<4;++round)for(unsigned n : {32u,256u}) {
            std::vector<uint32_t> input(n);std::vector<uint16_t> output(n+1,0xabcd);
            for(auto& x:input)x=rng();
            const uint32_t edges[]={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
                0x00800000u,0x00008000u,0x80008000u,0x00018000u,0x3f808000u,0xbf808000u,
                0x3f818000u,0xbf818000u,0x7f7fffffu,0xff7fffffu,0x7f7f8000u,0xff7f8000u,
                0x7f800000u,0xff800000u,0x7f800001u,0xffc00001u};
            for(unsigned i=0;i<sizeof(edges)/sizeof(edges[0]);++i)input[i]=edges[i];
            auto p=input.data();auto q=output.data();void* args[]={&p,&q,&q};
            std::string name="bf16_"+std::to_string(round);
            {Region r(name.c_str(),{{"lanes",n}});m.launch(name.c_str(),1,n,args);}
            for(unsigned i=0;i<n;++i)check(output[i]==bf_oracle(input[i],round),name+": rounded BF16 lane "+std::to_string(i));
            check(output[n]==0xabcd,"BF16 guard");
        }
    } else if (mode=="ldu") {
        for(int width : {1,2,4}) for(unsigned n : {1u,32u}) {
            alignas(16) std::array<uint32_t,4> input{0x11223344,0x89abcdef,37,99};
            std::vector<uint32_t> out(4*n+1,0xdeadbeef);auto p=input.data(),q=out.data();void* args[]={&p,&q,&q};
            std::string name="uniform_"+std::to_string(width);
            {Region r(name.c_str(),{{"lanes",n}});m.launch(name.c_str(),1,n,args);}
            for(unsigned lane=0;lane<n;++lane)for(int i=0;i<width;++i)check(out[4*lane+i]==input[i],"uniform load");
            check(out[4*n]==0xdeadbeef,"uniform guard");
        }
    } else return 2;
    std::cout<<mode<<": independent numerical oracles passed\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; } }
