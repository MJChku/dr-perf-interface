#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
using namespace ds41::gx_fixture;
static void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
static float fp8(unsigned x){unsigned a=x&127;float v=a/8?std::ldexp(1+(a%8)/8.f,int(a/8)-7):(a%8)/512.f;return x&128?-v:v;}
static std::uint16_t bf(float x){auto b=std::bit_cast<unsigned>(x);return (b+32767+((b>>16)&1))>>16;}
int main(int argc,char** argv){try{
 bool native=prepare_execution(argc,argv);std::string file=std::filesystem::path(argv[1]).filename();std::smatch match;require(std::regex_search(file,match,std::regex("_m([0-9]+)_n([0-9]+)")),"tile geometry missing");unsigned bm=std::stoul(match[1]),bn=std::stoul(match[2]);
 Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));std::ifstream stream(argv[1]);require(bool(stream),"module missing");std::string image{std::istreambuf_iterator<char>(stream),{}};Module module;check(api.module(&module,image.c_str()));Function fn;check(api.function(&fn,module,"fp8_gemm_kernel__kernel"));
 constexpr unsigned n=128,guard=32;const unsigned k=file.find("deep")!=std::string::npos?320:64;unsigned shared=bm==64?8192:10240;
 if(file.starts_with("k64_"))shared=20480;
 if(file.starts_with("k128_"))shared=40960;
 for(unsigned m:{1u,65u,129u}){
  std::vector<std::uint8_t> a(m*k+2*guard,0xA7),b(n*k+2*guard,0xA7),as(m*(k/32)+2*guard,0xA7),bs((n/32)*(k/32)+2*guard,0xA7);
  for(unsigned r=0;r<m;++r){for(unsigned d=0;d<k;++d)a[guard+r*k+d]=ds41::synthetic::fp8(177,r*k+d);for(unsigned g=0;g<k/32;++g)as[guard+r*(k/32)+g]=121+(r+g)%5;}
  for(unsigned r=0;r<n;++r)for(unsigned d=0;d<k;++d)b[guard+r*k+d]=ds41::synthetic::fp8(179,r*k+d);
  for(unsigned i=0;i<(n/32)*(k/32);++i)bs[guard+i]=122+i%4;
  std::vector<std::uint16_t> want(m*n+2*guard,0x4abc);
  for(unsigned r=0;r<m;++r)for(unsigned c=0;c<n;++c){float sum=0;for(unsigned g=0;g<k/32;++g){double dot=0;for(unsigned d=0;d<32;++d)dot+=double(fp8(a[guard+r*k+g*32+d]))*fp8(b[guard+c*k+g*32+d]);sum+=(float(dot)*std::ldexp(1.f,int(as[guard+r*(k/32)+g])-127))*std::ldexp(1.f,int(bs[guard+(c/32)*(k/32)+g])-127);}want[guard+r*n+c]=bf(sum);}
  Buffer<std::uint8_t> da(api,a),db(api,b),das(api,as),dbs(api,bs);Buffer<std::uint16_t> output(api,std::vector<std::uint16_t>(want.size(),0x4abc));auto ap=da.ptr+guard,bp=db.ptr+guard,cp=output.ptr+2*guard,asp=das.ptr+guard,bsp=dbs.ptr+guard;void* args[]={&ap,&bp,&cp,&asp,&bsp,&m};
  check(api.launch(fn,(n+bn-1)/bn,(m+bm-1)/bm,1,128,1,1,shared,nullptr,args,nullptr));check(api.sync());auto got=output.read();
  if(got!=want){unsigned mismatch=0,guard_changes=0,max_ulp=0;double max_abs=0;for(unsigned i=0;i<got.size();++i)if(got[i]!=want[i]){++mismatch;bool outside=i<guard||i>=guard+m*n;guard_changes+=outside;if(!outside){float g=std::bit_cast<float>(unsigned(got[i])<<16),w=std::bit_cast<float>(unsigned(want[i])<<16);max_abs=std::max(max_abs,double(std::abs(g-w)));max_ulp=std::max(max_ulp,unsigned(std::abs(int(got[i])-int(want[i]))));}if(mismatch<=8)std::cerr<<"m="<<m<<" i="<<i<<" got="<<got[i]<<" want="<<want[i]<<" guard="<<outside<<'\n';}std::cout<<"{\"numerical_status\":\"fail\",\"tile_m\":"<<bm<<",\"tile_n\":"<<bn<<",\"rows\":"<<m<<",\"mismatch_count\":"<<mismatch<<",\"guard_changes\":"<<guard_changes<<",\"max_absolute_error\":"<<max_abs<<",\"max_BF16_code_distance\":"<<max_ulp<<",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";throw std::runtime_error("FP8 tile outputs/tail/guards differ from independent scalar K32 oracle");}
  require(da.read()==a&&db.read()==b&&das.read()==as&&dbs.read()==bs,"FP8 tile modified inputs");
 }
 check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"scope\":\"Primary library FP8 tile parameters,3tail fixtures, independent scalar K32 oracle\",\"comparison\":\"bitwise BF16\",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
