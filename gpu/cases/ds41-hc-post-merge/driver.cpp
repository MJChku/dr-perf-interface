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
 const bool native=prepare_execution(argc,argv);Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));
 std::ifstream f(argv[1]);std::string code{std::istreambuf_iterator<char>(f),{}};Module module;check(api.module(&module,code.c_str()));Function kernel;check(api.function(&kernel,module,"ds41_hc_post_merge"));unsigned cases=0;
 for(unsigned rows:{1u,3u,8u})for(unsigned dim:{32u,288u,5120u})for(unsigned kind:{0u,1u}){
  const unsigned guard=32;std::vector<float> primary(rows*dim),shared(rows*dim),coeff(rows*24);std::vector<std::uint16_t> residual(rows*4*dim),expected(rows*4*dim);
  for(unsigned i=0;i<primary.size();++i){primary[i]=float(int(i*73%211)-105)/256+float((i%3)-1.0)*0x1p-18f;shared[i]=float(int(i*31%197)-98)/128;}
  for(unsigned i=0;i<residual.size();++i)residual[i]=bf(float(int(i*29%193)-96)/128);
  for(unsigned i=0;i<coeff.size();++i)coeff[i]=float(int(i*17%59)-29)/64;
  for(unsigned row=0;row<rows;++row)for(unsigned target=0;target<4;++target)for(unsigned d=0;d<dim;++d){
   float y=primary[row*dim+d];if(kind)y+=fp(bf(shared[row*dim+d]));y=fp(bf(y));float result=coeff[row*24+4+target]*y;
   for(unsigned source=0;source<4;++source)result=std::fma(coeff[row*24+8+source*4+target],fp(residual[(row*4+source)*dim+d]),result);
   expected[(row*4+target)*dim+d]=bf(result);
  }
  Buffer<float> p(api,primary),s(api,shared),c(api,coeff);Buffer<std::uint16_t> r(api,residual),out(api,std::vector<std::uint16_t>(guard+expected.size()+guard,0xa55a));
  auto pp=p.ptr,sp=s.ptr,rp=r.ptr,cp=c.ptr,op=out.ptr+guard*2;void* args[]={&pp,&sp,&rp,&cp,&op,&rows,&dim,&kind};check(api.launch(kernel,(rows*4*dim+127)/128,1,1,128,1,1,0,nullptr,args,nullptr));check(api.sync());auto actual=out.read();
  for(unsigned i=0;i<actual.size();++i)require(actual[i]==(i<guard||i>=guard+expected.size()?std::uint16_t(0xa55a):expected[i-guard]),"merged HC boundary/oracle/guard mismatch");
  require(p.read()==primary&&s.read()==shared&&c.read()==coeff&&r.read()==residual,"merged HC input mutation");++cases;
 }
 check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"cases\":"<<cases<<",\"comparison\":\"bitwise independent scalar BF16-boundary/FMA oracle, full guards and immutable inputs\",\"physical_gpu_used\":"<<(native?"true":"false")<<"}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
