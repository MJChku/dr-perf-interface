#include "ds41/gx_fixture.hpp"
#include "ds41/synthetic.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
using namespace ds41::gx_fixture;
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static float f8(unsigned x){unsigned m=x&127;float v=m/8?std::ldexp(1+(m%8)/8.f,int(m/8)-7):(m%8)/512.f;return x&128?-v:v;}
static float f4(unsigned x){const float v[]={0,.5,1,1.5,2,3,4,6};return x&8?-v[x&7]:v[x&7];}
static std::uint16_t bf(float x){auto v=std::bit_cast<unsigned>(x);return (v+32767+((v>>16)&1))>>16;}
int main(int argc,char** argv){try{prepare_execution(argc,argv);std::ifstream file(argv[1]);std::string text{std::istreambuf_iterator<char>(file),{}};Driver api;check(api.init(0));Context context;check(api.create(&context,0,0));Module module;check(api.module(&module,text.c_str()));Function fn;check(api.function(&fn,module,"ds41_library_grouped_fp4"));
 const unsigned tile=std::getenv("DS41_GROUP_TILE")?std::stoul(std::getenv("DS41_GROUP_TILE")):32;require(tile==16||tile==32,"row rounding must be16/32");unsigned cases=0;
 for(std::vector<int> real:std::vector<std::vector<int>>{{1,0,35,17},{1,3,7,15},{16,17,31,33},{32,64,32,64},{0,0,0,0},{15,16,17,0},{0,1,0,1},{33,49,63,2}}){
  unsigned groups=4,n=128,k=64,guard=32;std::vector<int> count(4),offset(4);unsigned padded=0,actual=0,active=0;
  for(unsigned e=0;e<4;++e){offset[e]=padded;count[e]=(real[e]+tile-1)/tile*tile;padded+=count[e];actual+=real[e];active+=real[e]>0;}
#ifdef DS41_DRPERF_GPU
  setenv("DS41_PCV_REAL_ROWS",std::to_string(actual).c_str(),1);setenv("DS41_PCV_PADDED_ROWS",std::to_string(padded).c_str(),1);setenv("DS41_PCV_ACTIVE_EXPERTS",std::to_string(active).c_str(),1);unsigned tiles16=0,tiles32=0;for(auto c:count){tiles32+=c/32;tiles16+=(c%32)!=0;}setenv("DS41_PCV_TILES16",std::to_string(tiles16).c_str(),1);setenv("DS41_PCV_TILES32",std::to_string(tiles32).c_str(),1);setenv("DS41_PCV_ACTIVE_TILES",std::to_string(tiles16+tiles32).c_str(),1);
#endif
  std::vector<std::uint8_t> a(std::max(1u,padded)*k),as(std::max(1u,padded)*k/32,124),b(groups*n*k/2),bs(groups*n*k/32,123);std::vector<std::uint16_t> want(padded*n+2*guard,0x4abc);
  for(unsigned i=0;i<b.size();++i)b[i]=ds41::synthetic::fp4(179,i);
  for(unsigned e=0;e<4;++e)for(int r=0;r<real[e];++r)for(unsigned d=0;d<k;++d)a[(offset[e]+r)*k+d]=ds41::synthetic::fp8(177+e,r*k+d);
  for(unsigned e=0;e<4;++e)for(int r=0;r<count[e];++r)for(unsigned c=0;c<n;++c){float sum=0;for(unsigned g=0;g<k/32;++g){double dot=0;for(unsigned d=0;d<32;++d){unsigned j=g*32+d;dot+=double(f8(a[(offset[e]+r)*k+j]))*f4((b[(e*n+c)*k/2+j/2]>>((j%2)*4))&15);}sum+=(float(dot)*std::ldexp(1.f,124-127))*std::ldexp(1.f,123-127);}want[guard+(offset[e]+r)*n+c]=bf(sum);}
  Buffer<std::uint8_t> da(api,a),das(api,as),db(api,b),dbs(api,bs);Buffer<int> counts(api,count),offsets(api,offset);Buffer<std::uint16_t> out(api,std::vector<std::uint16_t>(want.size(),0x4abc));std::vector<std::uint64_t> ptrs;
  for(unsigned e=0;e<4;++e){ptrs.push_back(db.ptr+e*n*k/2);ptrs.push_back(dbs.ptr+e*n*k/32);}Buffer<std::uint64_t> table(api,ptrs);
  auto ap=da.ptr,op=out.ptr+guard*2,asp=das.ptr,tp=table.ptr,osp=offsets.ptr,cp=counts.ptr;void* args[]={&ap,&op,&asp,&tp,&osp,&cp};check(api.launch(fn,1,2,groups,128,1,1,10240,nullptr,args,nullptr));check(api.sync());require(out.read()==want,"tight grouped scalar K32 oracle or guard mismatch");require(da.read()==a&&das.read()==as&&db.read()==b&&dbs.read()==bs&&counts.read()==count&&offsets.read()==offset&&table.read()==ptrs,"tight grouped inputs mutated");++cases;
 }
 check(api.unload(module));check(api.destroy(context));std::cout<<"{\"numerical_status\":\"pass\",\"cases\":"<<cases<<",\"rounding_rows\":"<<tile<<",\"oracle\":\"independent K32 scalar; fixed logical values per expert row, boundary counts, all padding and guards\",\"physical_gpu_used\":false}\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
