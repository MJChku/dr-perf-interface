// Standalone GPU collector. Scheduler/barriers and matrix arithmetic originate
// in GX (vendor/provenance.json); no GX simulator or CPU collector is loaded.
#include "context.h"
#include <light/fiber_stack.hh>
#include "../vendor/gx/shared_arena.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

thread_local int waiting_revive_count=0;
thread_local NamedBarrierManager* current_named_barrier_manager=nullptr;
namespace {
using Counts=std::map<std::string,uint64_t>;
thread_local gx::SharedArenaPool arena_pool;
struct Site { Counts metrics; };
struct Invocation {
    uint64_t token, parent;
    std::string name;
    std::map<std::string,int64_t> pcvs;
    Counts metrics;
    std::map<std::string,Site> sites;
    std::vector<std::string> launches;
};
struct Launch {
    Invocation* region;
    Counts metrics;
    std::map<std::string,Site> sites;
    std::vector<ThreadContext> lanes;
    dim3 grid, block;
    char* shared=nullptr;
    size_t cap=0, mapping=0, dynamic=0, static_bytes=0;
    uint64_t barrier_arrivals=0;
    struct BarrierClock { uint64_t generation=0; std::vector<ThreadContext*> lanes; };
    std::map<void*,BarrierClock> barrier_lanes;
    ~Launch() { if(shared) arena_pool.put(shared,cap,mapping); }
};
thread_local Launch* active=nullptr;
thread_local std::vector<std::unique_ptr<Invocation>> regions;
std::atomic<uint64_t> next_token{1};
std::atomic<uint64_t> open_regions{0}, completed_regions{0}, launch_failures{0};
std::mutex output_mutex;
std::string quoted(const std::string& s) {
    std::ostringstream o; o << '"';
    for (unsigned char c:s) {
        if(c=='"'||c=='\\') o << '\\' << char(c);
        else if(c<32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else o << c;
    }
    return o.str()+'"';
}
template<class T> std::string object(const std::map<std::string,T>& m) {
    std::ostringstream o; o << '{'; bool comma=false;
    for(auto& [k,v]:m) { if(comma)o<<','; comma=true; o<<quoted(k)<<':'<<v; }
    return o.str()+'}';
}
std::string sites_json(const std::map<std::string,Site>& sites) {
    std::ostringstream o; o << '{'; bool comma=false;
    for(auto& [k,v]:sites) {if(comma)o<<',';comma=true;o<<quoted(k)<<':'<<object(v.metrics);}
    return o.str()+'}';
}
void add(Counts& dst,const Counts& src) { for(auto& [k,v]:src)dst[k]+=v; }
void line(const std::string& value) {
    const char* path=std::getenv("DRPERF_GPU_CAPTURE");if(!path)return;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::ofstream f(std::string(path)+"."+std::to_string(getpid())+".jsonl",std::ios::app);
    f<<value<<'\n';f.flush();
    if(!f)throw std::runtime_error("cannot write GPU capture");
}
struct Lifecycle {
    Lifecycle() {line("{\"event\":\"start\",\"schema\":1,\"backend\":\"gpu\",\"pid\":"+std::to_string(getpid())+"}");}
    ~Lifecycle() {
        try {line("{\"event\":\"finish\",\"pid\":"+std::to_string(getpid())+
                  ",\"open_regions\":"+std::to_string(open_regions.load())+
                  ",\"regions_started\":"+std::to_string(next_token.load()-1)+
                  ",\"regions_completed\":"+std::to_string(completed_regions.load())+
                  ",\"launch_failures\":"+std::to_string(launch_failures.load())+"}");}
        catch(...) {std::abort();}
    }
} lifecycle;
void emit(const Invocation& r) {
    const char* path=std::getenv("DRPERF_GPU_CAPTURE");
    if(!path)return;
    std::ostringstream o;
    o<<"{\"schema\":1,\"backend\":\"gpu\",\"pid\":"<<getpid()<<",\"token\":"<<r.token
     <<",\"parent\":"<<r.parent<<",\"region\":"<<quoted(r.name)<<",\"pcvs\":"<<object(r.pcvs)
     <<",\"metrics\":"<<object(r.metrics)<<",\"sites\":"<<sites_json(r.sites)<<",\"launches\":[";
    for(size_t i=0;i<r.launches.size();++i) {if(i)o<<',';o<<r.launches[i];}
    o<<"]}";line(o.str());
}
uint64_t volume(dim3 d) {
    uint64_t xy=uint64_t(d.x)*d.y;
    if(d.z&&xy>std::numeric_limits<uint64_t>::max()/d.z)
        throw std::invalid_argument("GPU launch dimension overflow");
    return xy*d.z;
}
dim3 coordinates(uint64_t i,dim3 d) {return dim3(i%d.x,(i/d.x)%d.y,i/(uint64_t(d.x)*d.y));}
void need_launch() {if(!active)throw std::runtime_error("GPU operation outside a launch");}
void* symbol(void* module,const std::string& name) {
    dlerror();void* p=dlsym(module,name.c_str());const char* error=dlerror();
    if(error||!p)throw std::runtime_error("GPU module missing symbol "+name);
    return p;
}
}

extern "C" uint64_t drperf_gpu_region_begin(const char* name,size_t n,
        const char* const* names,const int64_t* values) {
    if(!name||!*name||n>256||(n&&(!names||!values)))throw std::invalid_argument("invalid GPU region");
    if(active)throw std::runtime_error("GPU region markers belong on the host launch path");
    auto r=std::make_unique<Invocation>();r->token=next_token++;
    r->parent=regions.empty()?0:regions.back()->token;r->name=name;
    for(size_t i=0;i<n;++i)
        if(!names[i]||!*names[i]||!r->pcvs.emplace(names[i],values[i]).second)
            throw std::invalid_argument("duplicate or empty GPU PCV");
    auto token=r->token;regions.push_back(std::move(r));++open_regions;return token;
}
extern "C" void drperf_gpu_region_end(uint64_t token) {
    if(active||regions.empty()||regions.back()->token!=token)throw std::runtime_error("unbalanced GPU markers");
    emit(*regions.back());regions.pop_back();--open_regions;++completed_regions;
}
extern "C" void* drperf_gpu_module_load(const char* path) {
    if(!path)throw std::invalid_argument("null GPU module path");
    void* module=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!module)throw std::runtime_error(std::string("cannot load translated GPU module: ")+dlerror());
    try {
        auto abi=static_cast<const unsigned*>(symbol(module,"drperf_gpu_instrumented_abi"));
        if(*abi!=2)throw std::runtime_error("GPU instrumentation ABI mismatch");
    } catch(...) {dlclose(module);throw;}
    return module;
}
extern "C" void drperf_gpu_module_close(void* module) {if(module)dlclose(module);}
extern "C" void drperf_gpu_launch(void* module,const char* kernel,dim3 grid,dim3 block,
                                  void** args,size_t dynamic) {
    if(active||!module||!kernel||!*kernel||!volume(grid)||!volume(block)||
       volume(block)>1024||volume(grid)>(1u<<24)||dynamic>(16u<<20))
        throw std::invalid_argument("invalid or recursive GPU launch");
    auto fn=reinterpret_cast<void(*)(void**)>(symbol(module,std::string(kernel)+"__gxptx"));
    unsigned n=*static_cast<unsigned*>(symbol(module,std::string(kernel)+"__gxptx_nparams"));
    if(n&&!args)throw std::invalid_argument("GPU launch missing arguments");
    for(unsigned i=0;i<n;++i)if(!args[i])throw std::invalid_argument("null GPU launch argument");
    Launch l;l.region=regions.empty()?nullptr:regions.back().get();l.grid=grid;l.block=block;l.dynamic=dynamic;
    l.metrics={{"launches",1},{"blocks",volume(grid)},{"launched_lanes",volume(grid)*volume(block)},
               {"warps",volume(grid)*((volume(block)+31)/32)},{"dynamic_shared_bytes_per_launch",dynamic}};
    active=&l;
    try {
        gx::FiberStack stacks;
        for(uint64_t b=0;b<volume(grid);++b) {
            NamedBarrierManager barriers(volume(block));current_named_barrier_manager=&barriers;
            l.lanes.assign(volume(block),ThreadContext{});
            gx::KernelFibers fibers;
            for(unsigned t=0;t<volume(block);++t) {
                auto& lane=l.lanes[t];lane.threadIdx=coordinates(t,block);lane.blockIdx=coordinates(b,grid);
                lane.blockDim=block;lane.gridDim=grid;
                fibers.add(stacks,[&,t] {
                    gx::KernelFibers::active()->data=&l.lanes[t];fn(args);
                    barriers.lane_exited(0,t);
                });
            }
            fibers.run();
            uint64_t max_f=0,max_b=0,max_i=0;
            for(auto& lane:l.lanes) {
                max_f=std::max(max_f,lane.flops);max_b=std::max(max_b,lane.bytes);
                max_i=std::max(max_i,lane.instructions);
            }
            l.metrics["sum_block_max_lane_flops"]+=max_f;
            l.metrics["sum_block_max_lane_bytes"]+=max_b;
            l.metrics["sum_block_lane_sequence_span"]+=max_i;
            l.metrics["sum_launch_max_block_lane_sequence_span"]=std::max(l.metrics["sum_launch_max_block_lane_sequence_span"],max_i);
            if(l.shared)std::memset(l.shared,0,l.cap);
            l.barrier_lanes.clear();
            current_named_barrier_manager=nullptr;
        }
        if(!l.metrics["ptx_lane_instructions"])
            throw std::runtime_error("GPU kernel executed without instrumentation");
        if(l.region) {
            add(l.region->metrics,l.metrics);
            for(auto& [site,c]:l.sites)add(l.region->sites[site].metrics,c.metrics);
            std::ostringstream shape;
            shape<<"{\"kernel\":"<<quoted(kernel)<<",\"grid\":["<<grid.x<<','<<grid.y<<','<<grid.z
                 <<"],\"block\":["<<block.x<<','<<block.y<<','<<block.z<<"],\"metrics\":"<<object(l.metrics)<<'}';
            l.region->launches.push_back(shape.str());
        } else if(std::getenv("DRPERF_GPU_CAPTURE")) {
            throw std::runtime_error("GPU launch has no enclosing GPU region");
        }
    } catch(...) {++launch_failures;active=nullptr;current_named_barrier_manager=nullptr;throw;}
    active=nullptr;
}

extern "C" void drperf_gpu_instruction(const char* fn,const char* op,
        unsigned site,unsigned long long flops,unsigned long long reads,unsigned long long writes,
        unsigned space,unsigned kind,unsigned long long address,unsigned unmodeled) {
    need_launch();auto* lane=gx_thread_context();lane->instructions++;
    lane->flops+=flops;lane->bytes+=reads+writes;
    Counts event{{"ptx_lane_instructions",1}};
    static const char* kinds[]={"none","fp32_flops","tensor_float_flops","tensor_integer_ops","fp64_flops","fp16_flops","bf16_flops"};
    if(flops)event[kinds[std::min(kind,6u)]]=flops;
    std::string head(op);head=head.substr(0,head.find('.'));
    if(!kind&&(head=="add"||head=="sub"||head=="mul"||head=="mad"||head=="div"||head=="rem"||
               head=="min"||head=="max"||head=="and"||head=="or"||head=="xor"||
               head=="shl"||head=="shr"||head=="popc"||head=="clz"||head=="bfind"))
        event["integer_ops"]=(head=="mad"?2:1);
    if(head=="cvt")event["conversion_ops"]=1;
    if(head=="sin"||head=="cos"||head=="ex2"||head=="lg2"||head=="rcp"||head=="rsqrt")
        event["special_function_ops"]=1;
    if(head=="bar")event["block_barrier_lane_arrivals"]=1;
    if(head=="shfl"||head=="vote"||head=="redux"||head=="barrier")event["collective_lane_arrivals"]=1;
    if(!space) {
        uint64_t base=reinterpret_cast<uint64_t>(active->shared);
        if(active->shared&&address>=base&&address-base<active->cap)space=2;
    }
    static const char* spaces[]={"generic","global","shared","local","param","const"};
    std::string s=spaces[std::min(space,5u)];
    if(reads)event[s+"_read_bytes"]=reads;
    if(writes)event[s+"_write_bytes"]=writes;
    if(unmodeled)event["unmodeled_work_events"]=1;
    add(active->metrics,event);add(active->sites[std::string(fn)+":cpp"+std::to_string(site)+":"+op].metrics,event);
}

int get_block_id(){return 0;} // A fresh manager is owned by each block.
int get_lane_id(){auto* c=gx_thread_context();return (c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z))%32;}
int get_warp_id_in_block(){auto* c=gx_thread_context();return (c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z))/32;}
void record_barrier_call(){need_launch();active->metrics["emulator_rendezvous_lane_arrivals"]++;}
void gpu_barrier_wait(NamedBarrierManager::BarrierIface* barrier) {
    auto& state=active->barrier_lanes[barrier];
    const auto generation=state.generation;
    state.lanes.push_back(gx_thread_context());barrier->wait();
    // The first returning participant has the complete generation's lane
    // clocks. Merge before any of these lanes executes its following work.
    if(generation==state.generation) {
        uint64_t depth=0;for(auto* c:state.lanes)depth=std::max(depth,c->instructions);
        for(auto* c:state.lanes)c->instructions=depth;
        state.lanes.clear();++state.generation;active->metrics["emulator_barrier_generations"]++;
    }
}
void barrier_sync(int name,int count) {
    record_barrier_call();auto* b=current_named_barrier_manager->get_barrier(name,count);BARRIER_WAIT(b->barrier);
}
void barrier_sync(int name){barrier_sync(name,0);}
void __syncthreads(){barrier_sync(0,0);}
extern "C" dim3 gx_dev_threadIdx(){return gx_thread_context()->threadIdx;}
extern "C" dim3 gx_dev_blockIdx(){return gx_thread_context()->blockIdx;}
extern "C" dim3 gx_dev_blockDim(){return gx_thread_context()->blockDim;}
extern "C" dim3 gx_dev_gridDim(){return gx_thread_context()->gridDim;}
extern "C" void gx_cpu_kernel_yield(){gx::kernel_fiber_yield();}
extern "C" void* gx_ptx_shared_window(size_t statics) {
    need_launch();active->static_bytes=std::max(active->static_bytes,statics);
    if(statics>(16u<<20))throw std::invalid_argument("GPU shared declaration exceeds standalone limit");
    size_t needed=statics+active->dynamic;
    if(active->shared) {
        if(needed>active->cap)throw std::runtime_error("shared arena grew during a kernel");
        return active->shared;
    }
    const size_t page=sysconf(_SC_PAGESIZE);
    active->cap=((std::max(needed,size_t(1024))+page-1)/page)*page;
    active->mapping=active->cap+page;
    active->shared=arena_pool.take(active->cap,active->mapping);
    if(!active->shared)
        active->shared=static_cast<char*>(mmap(nullptr,active->mapping,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    if(active->shared==MAP_FAILED){active->shared=nullptr;throw std::bad_alloc();}
    if(mprotect(active->shared+active->cap,page,PROT_NONE))throw std::runtime_error("GPU shared guard failed");
    active->metrics["static_shared_bytes_per_launch"]=statics;
    return active->shared;
}
extern "C" uint32_t gx_shared_rank_bits(){return 0;}
extern "C" uint64_t gx_cluster_shared_ptr(uint32_t offset) {
    need_launch();if(!active->shared||offset>=active->cap)throw std::runtime_error("shared offset out of range");
    return reinterpret_cast<uint64_t>(active->shared+offset);
}
extern "C" int memory_mg_is_fake(void*){return 0;}
void global_memcpy(void* dst,const void* src,size_t size,int){std::memcpy(dst,src,size);}
extern "C" void gx_ptx_trap(){throw std::runtime_error("GPU PTX trap executed");}
extern "C" void gx_ptx_trap_op(const char* op){throw std::runtime_error(std::string("unsupported GPU PTX operation: ")+op);}

extern "C" void __syncwarp_mask(unsigned mask) {
    const unsigned threads=active->lanes.size(),warp=get_warp_id_in_block(),lane=get_lane_id();
    unsigned lanes=std::min(32u,threads-warp*32);
    mask&=lanes==32?~0u:(1u<<lanes)-1;
    if(mask&(1u<<lane)){record_barrier_call();current_named_barrier_manager->warp_sync(0,warp,lane,mask);}
}
extern "C" uint32_t gx_shfl_sync(uint32_t value,uint32_t offset,uint32_t clamp,
                                  uint32_t mask,int mode,int* predicate) {
    const unsigned lane=get_lane_id(),warp=get_warp_id_in_block();
    unsigned lanes=std::min(32u,unsigned(active->lanes.size())-warp*32);
    mask&=lanes==32?~0u:(1u<<lanes)-1;
    if(!(mask&(1u<<lane))){if(predicate)*predicate=0;return value;}
    if(mode<0||mode>3)throw std::runtime_error("invalid shuffle mode");
    const unsigned seg=(clamp>>8)&31,min=lane&seg,max=min|((clamp&31)&~seg),delta=offset&31;
    unsigned source=mode==3?lane+delta:mode==2?lane-delta:mode==1?lane^delta:min|(offset&~seg&31);
    bool valid=(mode==2?(delta<=lane&&source>=max):source<=max)&&source<32&&(mask&(1u<<source));
    if(predicate)*predicate=valid;
    record_barrier_call();
    return current_named_barrier_manager->get_warp_exchange(0,warp,mask)->exchange(lane,value,valid?source:lane,false);
}
extern "C" uint32_t gx_redux_sync(uint32_t value,uint32_t mask,unsigned op) {
    unsigned lane=get_lane_id(),warp=get_warp_id_in_block();
    unsigned count=std::min(32u,unsigned(active->lanes.size())-warp*32);
    mask&=count==32?~0u:(1u<<count)-1;
    if(!(mask&(1u<<lane)))return value;
    record_barrier_call();return current_named_barrier_manager->get_warp_exchange(0,warp,mask)->reduce(lane,value,op);
}
extern "C" uint32_t gx_redux_add_sync(uint32_t v,uint32_t m){return gx_redux_sync(v,m,0);}
static uint32_t vote(unsigned mask,int p) {
    unsigned lane=get_lane_id(),warp=get_warp_id_in_block();
    unsigned count=std::min(32u,unsigned(active->lanes.size())-warp*32);
    mask&=count==32?~0u:(1u<<count)-1;
    if(!(mask&(1u<<lane)))return 0;
    record_barrier_call();return current_named_barrier_manager->get_warp_exchange(0,warp,mask)->ballot(lane,p);
}
unsigned __ballot_sync(unsigned m,int p){return vote(m,p);}
int __any_sync(unsigned m,int p){return vote(m,p)!=0;}
int __all_sync(unsigned m,int p){
    unsigned count=std::min(32u,unsigned(active->lanes.size())-unsigned(get_warp_id_in_block())*32);
    return vote(m,p)==(m&(count==32?~0u:(1u<<count)-1));
}
extern "C" uint64_t gx_block_scan_u32(uint32_t v,uint32_t exclusive) {
    auto* c=gx_thread_context();unsigned t=c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z);
    record_barrier_call();return current_named_barrier_manager->get_block_scan(0,active->lanes.size())->scan(t,v,exclusive);
}
