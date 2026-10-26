#pragma once
// Workspace-only adapter: reuse independent kernel-oracle fixtures through the
// standalone DrPerf GPU launch API. Host allocations/copies are fixture setup,
// never GPU work. This build cannot load CUDA or execute a physical GPU.
#include "drperf_gpu.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace ds41::gx_fixture {
using DevicePtr=std::uint64_t;using Context=void*;using Module=void*;using Function=void*;
inline std::string module_path;
inline bool prepare_execution(int& argc,char** argv){if(argc!=2)throw std::invalid_argument("expected translated CPU module.so");module_path=argv[1];return false;}
inline void check(int status){if(status)throw std::runtime_error("DrPerf fixture status "+std::to_string(status));}
struct FunctionBox{drperf::gpu::Module* module;std::string name;};
struct ModuleBox{drperf::gpu::Module module;std::vector<std::unique_ptr<FunctionBox>> functions;explicit ModuleBox(const char*p):module(p){}};
inline unsigned scalar(void** args,unsigned i){return *static_cast<unsigned*>(args[i]);}
inline uintptr_t pointer(void** args,unsigned i){return *static_cast<uintptr_t*>(args[i]);}
inline std::vector<std::pair<std::string,int64_t>> pcvs(const std::string& name,void** a,dim3 grid,dim3 block){
 std::vector<std::pair<std::string,int64_t>> v;auto add=[&](const char*n,int64_t x){v.emplace_back(n,x);};
#include "drperf_gpu_argument_pcvs.inc"
 if(name.starts_with("ds41_attention_split_")){unsigned rows=scalar(a,9),heads=scalar(a,10),dim=scalar(a,11),selected=scalar(a,12),wr=scalar(a,13),cr=scalar(a,14),stride=scalar(a,20);unsigned valid=0;auto* ids=reinterpret_cast<const int*>(pointer(a,5));for(unsigned r=0;r<rows;++r)for(unsigned j=0;j<selected;++j)valid+=ids[r*stride+j]>=0&&unsigned(ids[r*stride+j])<wr+cr;add("real_rows",rows);add("heads",heads);add("dim",dim);add("context_keys",wr+cr);add("selected_keys",selected);add("valid_query_keys",valid);add("valid_pairs",uint64_t(valid)*heads);add("valid_pair_dim",uint64_t(valid)*heads*dim);add("selected_pair_dim",uint64_t(rows)*heads*selected*dim);add("selected_tiles",(selected+63)/64);}
 if(name=="ds41_library_grouped_fp4"&&std::getenv("DS41_PCV_REAL_ROWS")){add("real_rows",std::stoll(std::getenv("DS41_PCV_REAL_ROWS")));add("padded_rows",std::stoll(std::getenv("DS41_PCV_PADDED_ROWS")));add("active_experts",std::stoll(std::getenv("DS41_PCV_ACTIVE_EXPERTS")));add("capacity_rows",256);add("N",128);add("K",64);for(auto pair:{std::pair{"DS41_PCV_TILES16","tiles16"},std::pair{"DS41_PCV_TILES32","tiles32"},std::pair{"DS41_PCV_ACTIVE_TILES","active_tiles"}})if(const char* value=std::getenv(pair.first))add(pair.second,std::stoll(value));}
 add("blocks",uint64_t(grid.x)*grid.y*grid.z);add("launched_lanes",uint64_t(grid.x)*grid.y*grid.z*block.x*block.y*block.z);
 unsigned m=0,n=0,k=0,active=0;bool matrix=false,routed=false;
 if((name=="ds41_matvec_tc_partials"||name=="ds41_matvec_tc_fused")){m=scalar(a,5);n=scalar(a,6);k=scalar(a,7);matrix=true;}
 else if(name=="ds41_scaled_matvec"||name=="ds41_scaled_gemm"){m=scalar(a,5);n=scalar(a,6);k=scalar(a,7);matrix=true;}
 else if(name=="ds41_matvec_tc_finish"){m=scalar(a,2);n=scalar(a,3);k=scalar(a,4)*32;matrix=true;}
 else if(name=="ds41_routed_tc_partials"||name=="ds41_routed_tc_fused"||name=="ds41_routed_tc_parallel"||name=="ds41_routed_tc_finish"){
  routed=true;unsigned first,owned,stride;const int* ids;
  if(name=="ds41_routed_tc_partials"||name=="ds41_routed_tc_fused"||name=="ds41_routed_tc_parallel"){m=scalar(a,5);n=scalar(a,6);k=scalar(a,7);stride=scalar(a,12);first=scalar(a,13);owned=scalar(a,14);ids=reinterpret_cast<const int*>(pointer(a,3));}
  else{m=scalar(a,3);n=scalar(a,4);k=scalar(a,5)*32;stride=scalar(a,7);first=scalar(a,8);owned=scalar(a,9);ids=reinterpret_cast<const int*>(pointer(a,1));}
  std::set<int> experts;unsigned max_routes=0;
  for(unsigned r=0;r<m;++r){unsigned local=0;for(unsigned j=0;j<6;++j){int id=ids[r*stride+j];if(id>=int(first)&&id<int(first+owned)){++active;++local;experts.insert(id);}}max_routes=std::max(max_routes,local);}
  add("active_experts",experts.size());add("active_routes",active);add("max_routes",max_routes);add("any_route",max_routes>0);add("all_routes",max_routes==6);add("route_groups",max_routes*(k/32));add("route_iters8",max_routes*((k/32+7)/8));add("route_iters4",max_routes*std::min(k/32,4u));add("any_groups",(max_routes>0)*(k/32));add("any_iters8",(max_routes>0)*((k/32+7)/8));add("capacity_rows",m*6);add("active_n",uint64_t(active)*n);add("active_n_k",uint64_t(active)*n*k);add("route_tile_visits",uint64_t(m)*((n+7)/8)*((k/32+3)/4)*6);
  matrix=true;
 }
 if(matrix&&(name=="ds41_matvec_tc_partials"||name=="ds41_matvec_tc_fused"||name=="ds41_scaled_matvec"||name=="ds41_scaled_gemm")){
  unsigned formats=scalar(a,8);add("weight_fp4",formats&1);add("activation_fp4",bool(formats&2));add("minimum_weight_bytes",uint64_t(n)*k/(formats&1?2:1));add("row_weight_payload_bytes",uint64_t(m)*n*k/(formats&1?2:1));add("minimum_activation_bytes",uint64_t(m)*k/(formats&2?2:1));
 }
 if(matrix){add("real_rows",m);add("N",n);add("K",k);add("outputs",uint64_t(m)*n);add("useful_fmas",uint64_t(routed?active:m)*n*k);add("tile_groups",uint64_t(routed?active:m)*((n+7)/8)*(k/32));add("k_groups",k/32);return v;}
 auto shape=[&](unsigned rows,unsigned dim){add("real_rows",rows);add("width",dim);add("elements",uint64_t(rows)*dim);add("scale_groups",uint64_t(rows)*((dim+31)/32));};
 if(name=="ds41_hc_pre_quant")shape(scalar(a,6),scalar(a,7));
 else if(name=="ds41_rms_quant")shape(scalar(a,5),scalar(a,6));
 else if(name=="ds41_swiglu_quant"){shape(scalar(a,6),scalar(a,7));add("weighted_elements",uint64_t(scalar(a,6))*scalar(a,7)*scalar(a,8));}
 else if(name=="ds41_hc_coeff_post")shape(scalar(a,7),scalar(a,8));
 else if(name=="ds41_hc_post_merge")shape(scalar(a,5),scalar(a,6));
 else if(name=="ds41_hc_coefficients")shape(scalar(a,5),scalar(a,6));
 else if(name=="ds41_hc_pre_norm")shape(scalar(a,4),scalar(a,5));
 else if(name=="ds41_hc_post")shape(scalar(a,4),scalar(a,5));
 else if(name=="ds41_rms_norm")shape(scalar(a,3),scalar(a,4));
 else if(name=="ds41_head_local")shape(scalar(a,2),scalar(a,3));
 else if(name=="ds41_head_global")shape(scalar(a,2),scalar(a,3));
 else if(name=="ds41_quantize_rows")shape(scalar(a,4),scalar(a,5));
 return v;
}
inline int ignore_cost_context(const char*,const std::uint64_t*,std::size_t){return 0;}
inline void ignore_clear_context(){}
template<class F> F load(const char* name){
 if(std::string(name)=="gx_kernel_cost_context_set")return reinterpret_cast<F>(&ignore_cost_context);
 if(std::string(name)=="gx_kernel_cost_context_clear")return reinterpret_cast<F>(&ignore_clear_context);
 throw std::runtime_error(std::string("CPU resource fixture cannot provide driver API: ")+name);
}
struct Driver{
 int init(unsigned){return 0;}int create(Context* c,unsigned,int){*c=this;return 0;}int destroy(Context){return 0;}
 int module(Module* m,const void*){if(module_path.empty()){const char*p=std::getenv("DS41_DRPERF_MODULE");if(!p)throw std::runtime_error("module path missing");module_path=p;}*m=new ModuleBox(module_path.c_str());return 0;}
 int function(Function* f,Module m,const char* name){auto* box=static_cast<ModuleBox*>(m);box->functions.push_back(std::make_unique<FunctionBox>(FunctionBox{&box->module,name}));*f=box->functions.back().get();return 0;}
 int unload(Module m){delete static_cast<ModuleBox*>(m);return 0;}
 int alloc(DevicePtr* p,size_t n){void*v=nullptr;if(posix_memalign(&v,128,std::max(size_t(1),n)))throw std::bad_alloc();*p=reinterpret_cast<DevicePtr>(v);return 0;}
 int free(DevicePtr p){std::free(reinterpret_cast<void*>(p));return 0;}
 int upload(DevicePtr p,const void* data,size_t n){std::memcpy(reinterpret_cast<void*>(p),data,n);return 0;}
 int download(void* data,DevicePtr p,size_t n){std::memcpy(data,reinterpret_cast<void*>(p),n);return 0;}
 int sync(){return 0;}
 int launch(Function f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned dynamic,void*,void** args,void**){
  auto* fn=static_cast<FunctionBox*>(f);auto vars=pcvs(fn->name,args,{gx,gy,gz},{bx,by,bz});std::vector<const char*> names;std::vector<int64_t> values;
  for(auto& [name,value]:vars){names.push_back(name.c_str());values.push_back(value);}auto token=drperf_gpu_region_begin(fn->name.c_str(),names.size(),names.data(),values.data());
  try{fn->module->launch(fn->name.c_str(),{gx,gy,gz},{bx,by,bz},args,dynamic);}catch(...){drperf_gpu_region_end(token);throw;}
  drperf_gpu_region_end(token);return 0;
 }
};
template<class T>struct Buffer{Driver& api;DevicePtr ptr=0;std::vector<T> host;Buffer(Driver& a,std::vector<T> v):api(a),host(std::move(v)){check(api.alloc(&ptr,host.size()*sizeof(T)));check(api.upload(ptr,host.data(),host.size()*sizeof(T)));}Buffer(const Buffer&)=delete;~Buffer(){if(ptr)api.free(ptr);}std::vector<T> read(){auto v=host;check(api.download(v.data(),ptr,host.size()*sizeof(T)));return v;}};
}
