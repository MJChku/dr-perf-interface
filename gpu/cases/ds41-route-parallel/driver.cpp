#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
using namespace ds41::gx_fixture;
static void require(bool v,const char* text){if(!v)throw std::runtime_error(text);}
static float fp8(unsigned x){auto m=x&127;float v=m/8?std::ldexp(1+(m%8)/8.0f,int(m/8)-7):(m%8)/512.0f;return x&128?-v:v;}
static float fp4(unsigned x){const float v[]={0,.5,1,1.5,2,3,4,6};return x&8?-v[x&7]:v[x&7];}
static std::uint16_t bf(float x){auto b=std::bit_cast<std::uint32_t>(x);return (b+32767+((b>>16)&1))>>16;}
static void fixture(Driver& api,Function part,Function finish,Function token_fn,unsigned rows,unsigned k,unsigned active,bool packed,bool invalid){
    unsigned n=17;
#ifdef DS41_DRPERF_GPU
    if(const char* width=std::getenv("DS41_TC_PROBE_N"))n=std::stoul(width);
#endif
    unsigned first=96,owned=48,as=k+8,ass=k/32+3,bs=k/2+8,bss=k/32+5,is=9,os=n+7,groups=k/32,packed_arg=packed;const unsigned g=32,input_rows=packed?rows*6:rows;
    std::vector<std::uint8_t> a(input_rows*as+2*g,0xa7),asc(input_rows*ass+2*g,0xa7),b(owned*n*bs+2*g,0xa7),bsc(owned*n*bss+2*g,0xa7);
    for(unsigned r=0;r<input_rows;++r){for(unsigned d=0;d<k;++d)a[g+r*as+d]=ds41::synthetic::fp8(901,r*k+d);for(unsigned j=0;j<groups;++j)asc[g+r*ass+j]=122+(r+j)%7;}
    for(unsigned e=0;e<owned;++e)for(unsigned r=0;r<n;++r){for(unsigned d=0;d<k/2;++d)b[g+(e*n+r)*bs+d]=ds41::synthetic::fp4(903+e,r*(k/2)+d);for(unsigned j=0;j<groups;++j)bsc[g+(e*n+r)*bss+j]=123+(e+r+j)%5;}
    std::vector<int> ids(rows*is+2*g,-775),tokens(rows*6+2*g,-775),want_tokens=tokens,errors(1,0);
    std::vector<std::uint16_t> out(rows*6*os+2*g,0x4abc),want=out;
    for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<6;++j){const unsigned route=r*6+j;int id=j<active?first+(r*7+j)%owned:17+r*5+j;if(invalid&&j==5)id=-2;ids[g+r*is+j]=id;const bool local=id>=int(first)&&id<int(first+owned);want_tokens[g+route]=local?int(r):-1;
        for(unsigned col=0;col<n;++col){float total=0;if(local){unsigned input=packed?route:r,e=id-first;for(unsigned chunk=0;chunk<groups;++chunk){double dot=0;for(unsigned p=0;p<32;++p){unsigned d=chunk*32+p,code=(b[g+(e*n+col)*bs+d/2]>>((d%2)*4))&15;dot+=double(fp8(a[g+input*as+d]))*fp4(code);}total+=(float(dot)*std::ldexp(1.0f,int(asc[g+input*ass+chunk])-127))*std::ldexp(1.0f,int(bsc[g+(e*n+col)*bss+chunk])-127);}}want[g+route*os+col]=bf(total);}
    }
    Buffer<std::uint8_t> da(api,a),das(api,asc),db(api,b),dbs(api,bsc);Buffer<int> di(api,ids),dt(api,tokens),de(api,errors);Buffer<std::uint16_t> output(api,out);Buffer<float> scratch(api,std::vector<float>(std::size_t(rows)*6*n*groups+2*g,-779));
    std::vector<std::uint64_t> table;for(unsigned e=0;e<owned;++e){table.push_back(db.ptr+g+std::uint64_t(e)*n*bs);table.push_back(dbs.ptr+g+std::uint64_t(e)*n*bss);}Buffer<std::uint64_t> pairs(api,table);
    auto ap=da.ptr+g,asp=das.ptr+g,tp=pairs.ptr,ip=di.ptr+g*4,pp=scratch.ptr+g*4,op=output.ptr+g*2,tokenp=dt.ptr+g*4,ep=de.ptr;
    #if defined(DS41_ROUTED_FUSED) || defined(DS41_ROUTED_PARALLEL)
    const unsigned route_blocks=
#ifdef DS41_ROUTED_PARALLEL
    6;
#else
    1;
#endif
    void* pa[]={&ap,&asp,&tp,&ip,&op,&rows,&n,&k,&as,&ass,&bs,&bss,&is,&first,&owned,&packed_arg,&os};check(api.launch(part,(n+7)/8,rows,route_blocks,256,1,1,groups*8*sizeof(float),nullptr,pa,nullptr));
#else
    void* pa[]={&ap,&asp,&tp,&ip,&pp,&rows,&n,&k,&as,&ass,&bs,&bss,&is,&first,&owned,&packed_arg};check(api.launch(part,(n+7)/8,rows,(groups+3)/4,32,1,1,0,nullptr,pa,nullptr));
    void* fa[]={&pp,&ip,&op,&rows,&n,&groups,&os,&is,&first,&owned};check(api.launch(finish,(rows*6*n+127)/128,1,1,128,1,1,0,nullptr,fa,nullptr));
#endif
    void* ta[]={&ip,&tokenp,&ep,&rows,&is,&first,&owned};check(api.launch(token_fn,(rows*6+127)/128,1,1,128,1,1,0,nullptr,ta,nullptr));check(api.sync());
    require(output.read()==want,"routed matvec bitwise result/guard mismatch");require(dt.read()==want_tokens&&de.read()[0]==int(invalid),"device route ownership/invalid handling mismatch");
    auto partial=scratch.read();for(unsigned i=0;i<g;++i)require(partial[i]==-779&&partial[partial.size()-1-i]==-779,"routed partial guard modified");
    require(da.read()==a&&das.read()==asc&&db.read()==b&&dbs.read()==bsc&&di.read()==ids&&pairs.read()==table,"routed matvec mutated inputs");
}
int main(int argc,char** argv){try{prepare_execution(argc,argv);std::ifstream file(argv[1]);require(file.good(),"PTX missing");std::string ptx{std::istreambuf_iterator<char>(file),{}};Driver api;check(api.init(0));Context ctx;check(api.create(&ctx,0,0));Module module;check(api.module(&module,ptx.c_str()));Function part,finish,token_fn;
#ifdef DS41_ROUTED_PARALLEL
check(api.function(&part,module,"ds41_routed_tc_parallel"));
#elif defined(DS41_ROUTED_FUSED)
check(api.function(&part,module,"ds41_routed_tc_fused"));
#else
check(api.function(&part,module,"ds41_routed_tc_partials"));
#endif
check(api.function(&finish,module,"ds41_routed_tc_finish"));check(api.function(&token_fn,module,"ds41_route_token_ids"));unsigned cases=0;
#ifdef DS41_DRPERF_FOCUSED
    for(auto rows:{1u})for(auto k:{64u,512u,1280u})
#else
    for(auto rows:{1u,3u})for(auto k:{32u,512u})
#endif
for(auto active:{0u,1u,3u,6u})for(bool packed:{false,true}){fixture(api,part,finish,token_fn,rows,k,active,packed,false);++cases;}
    fixture(api,part,finish,token_fn,3,64,3,false,true);++cases;
    check(api.unload(module));check(api.destroy(ctx));std::cout<<"{\"numerical_status\":\"pass\",\"cases\":"<<cases<<",\"scope\":\"Device-routed FP4 expert matvec, canonical/packed activation, zero/one/three/six local routes, invalid IDs, ownership and guards; bitwise scalar oracle\",\"physical_gpu_used\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
