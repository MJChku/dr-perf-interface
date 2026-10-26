#include "ds41/gx_fixture.hpp"
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
using namespace ds41::gx_fixture;
static std::uint16_t bf(float f){auto x=std::bit_cast<std::uint32_t>(f);return (x+32767+((x>>16)&1))>>16;}
static float fp(std::uint16_t x){return std::bit_cast<float>(std::uint32_t(x)<<16);}
static void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
int main(int argc,char** argv){try{
 const bool native=prepare_execution(argc,argv);Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));std::ifstream f(argv[1]);std::string code{std::istreambuf_iterator<char>(f),{}};Module module;check(api.module(&module,code.c_str()));Function kernel;check(api.function(&kernel,module,"ds41_hc_bf16_splitk"));unsigned cases=0;
 for(unsigned k:{128u,20480u})for(unsigned splits:{1u,8u,128u}){
  if(k==20480&&splits==1)continue;
  const unsigned guard=32;unsigned rows=1,n=24,as=k+7,bs=k+11;
  std::vector<std::uint16_t> input(rows*as,0xa55a);std::vector<float> weights(n*bs,123.25f),expected(splits*n);
  for(unsigned j=0;j<k;++j)input[j]=bf(std::sin(float(j)*.17f)*.125f);
  for(unsigned c=0;c<n;++c)for(unsigned j=0;j<k;++j)weights[c*bs+j]=std::cos(float(c*19+j)*.13f)*.0625f;
  for(unsigned part=0;part<splits;++part)for(unsigned c=0;c<n;++c){long double dot=0;for(unsigned j=(k/splits)*part;j<(k/splits)*(part+1);++j)dot+=static_cast<long double>(fp(input[j]))*weights[c*bs+j];expected[part*n+c]=float(dot);}
  Buffer<std::uint16_t> a(api,input);Buffer<float> b(api,weights),p(api,std::vector<float>(guard+expected.size()+guard,123.25f));auto ap=a.ptr,bp=b.ptr,pp=p.ptr+guard*4;void* args[]={&ap,&bp,&pp,&rows,&n,&k,&as,&bs,&splits};check(api.launch(kernel,splits,rows,1,256,1,1,0,nullptr,args,nullptr));check(api.sync());auto actual=p.read();
  for(unsigned i=0;i<actual.size();++i){if(i<guard||i>=guard+expected.size())require(actual[i]==123.25f,"HC partial output guard changed");else require(std::isfinite(actual[i])&&std::abs(actual[i]-expected[i-guard])<=1e-5f+2e-5f*std::abs(expected[i-guard]),"HC BF16 input independent long-double partial oracle mismatch");}
  require(a.read()==input&&b.read()==weights,"HC BF16 input mutated");++cases;
 }
 check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"cases\":"<<cases<<",\"oracle\":\"independent long-double dot per K partition; padded strides, full input immutability and output guards\",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
