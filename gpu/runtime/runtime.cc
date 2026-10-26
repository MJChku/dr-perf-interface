// Standalone GPU collector. Scheduler/barriers and matrix arithmetic originate
// in GX (vendor/provenance.json); no GX simulator or CPU collector is loaded.
#include "context.h"
#include <unordered_set>
#include "footprint.h"
#include "initcheck.h"
#include <light/fiber_stack.hh>
#include "../vendor/gx/shared_arena.h"
#include "../vendor/gx/tcgen05.h"
#include "../vendor/gx/mbarrier.h"
#include <algorithm>
#include <cmath>
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
#include <tuple>
#include <unistd.h>

thread_local int waiting_revive_count=0;
namespace {
using Counts=std::map<std::string,uint64_t>;
thread_local gx::SharedArenaPool arena_pool;
using drperf_footprint::Footprint;
struct Site { Counts metrics; Footprint footprint; };
struct Invocation {
    uint64_t token, parent;
    std::string name;
    std::map<std::string,int64_t> pcvs;
    Counts metrics;
    Footprint footprint;
    bool footprint_available=true;
    std::map<std::string,Site> sites;
    std::vector<std::string> launches;
    std::string scope="host", kernel;
    uint64_t linear_block=0, group=0;
};
struct DeviceInvocation {
    Invocation row;
    unsigned expected=0;
    std::set<unsigned> entered, exited;
    std::set<unsigned> ready;
    std::map<unsigned,std::map<std::string,int64_t>> member_pcvs;
    std::map<unsigned,DrperfGpuReady> latency_entry;
};
struct DeviceLane {
    std::vector<DeviceInvocation*> stack;
    std::map<std::pair<unsigned,std::string>,uint64_t> ordinals;
};
struct BlockState {
    std::vector<ThreadContext> lanes;
    std::vector<DeviceLane> device_lanes;
    std::map<std::tuple<unsigned,unsigned,uint64_t,std::string>,std::unique_ptr<DeviceInvocation>> device_regions;
    char* shared=nullptr;
    size_t cap=0, mapping=0, static_bytes=0;
    std::map<unsigned,size_t> static_offsets;
    size_t legacy_static_end=0;
    NamedBarrierManager barriers;
    GxTmemState tmem;
    // Latency model: the CTA's tensor pipe executes tcgen05.mma in issue order.
    uint64_t tensor_free=0;DrperfGpuReady tensor_ready;std::string tensor_site;
    explicit BlockState(unsigned threads):lanes(threads),device_lanes(threads),barriers(threads) {}
    ~BlockState() { if(shared){std::memset(shared,0,cap);arena_pool.put(shared,cap,mapping);} }
};
struct SplitBarrier {
    uint64_t phase=0,arrived=0;std::vector<gx::FiberRef> waiters;
    DrperfGpuReady pending;std::map<uint64_t,DrperfGpuReady> completed;
};
struct LatencyConfig {
    static uint64_t value(const char* name,uint64_t fallback,bool nonzero=false) {
        const char* s=std::getenv(name);if(!s)return fallback;
        char* end=nullptr;auto v=std::strtoull(s,&end,10);
        if(!*s||*s=='-'||*end||v>1000000000||(!v&&nonzero))
            throw std::invalid_argument(std::string("invalid latency model setting ")+name);
        return v;
    }
    uint64_t alu=value("DRPERF_GPU_LATENCY_ALU",4);
    uint64_t global=value("DRPERF_GPU_LATENCY_GLOBAL",400);
    uint64_t shared=value("DRPERF_GPU_LATENCY_SHARED",30);
    uint64_t mma=value("DRPERF_GPU_LATENCY_MMA",32);
    uint64_t sfu=value("DRPERF_GPU_LATENCY_SFU",20);
    // Dense 16-bit tensor FLOPs per SM per model unit (8-bit operands 2x,
    // 4-bit 4x, tf32 0.5x). One tcgen05.mma occupies the CTA's tensor pipe for
    // flops / rate units; its commit arrives a further `mma` units after the pipe
    // drains. B300: ~8192 dense BF16 FLOP/clock/SM.
    uint64_t tensor_rate=value("DRPERF_GPU_TENSOR_FLOPS_PER_UNIT",8192,true);
    uint64_t sms=value("DRPERF_GPU_SM_COUNT",148,true);
    uint64_t lanes=value("DRPERF_GPU_LANES_PER_SM",128,true);
    // Optional residency split: a global load whose 128-byte lines were all
    // touched earlier in the same launch uses this (cache-hit) latency.
    // Unset = same as global (no residency distinction).
    uint64_t global_hit=value("DRPERF_GPU_LATENCY_GLOBAL_HIT",global);
    // 0 = PTX-order in-order issue (default), 1 = basic-block schedule.
    uint64_t schedule=[]{const char* v=std::getenv("DRPERF_GPU_LATENCY_SCHEDULE");
        return uint64_t(v&&std::string(v)=="block");}();
    uint64_t latency(unsigned cls) const {
        switch(cls) {case 0:return 0;case 2:case 6:return global;
            case 3:return shared;case 4:return mma;case 5:return sfu;default:return alu;}
    }
    Counts counts() const {
        return {{"latency_model_version",1},{"latency_model_alu",alu},
            {"latency_model_global",global},{"latency_model_shared",shared},
            {"latency_model_mma",mma},{"latency_model_sfu",sfu},
            {"latency_model_sm_count",sms},{"latency_model_lanes_per_sm",lanes},
            {"latency_model_schedule",schedule},{"latency_model_global_hit",global_hit},
            {"latency_model_tensor_rate",tensor_rate}};
    }
};
struct Launch {
    LatencyConfig latency;
    std::unordered_set<uint64_t> touched_lines;  // residency classification only
    DrperfGpuReady latency_finish;
    std::string critical_lane;  // JSON breakdown of the lane that set latency_finish
    Invocation* region;
    std::unique_ptr<Invocation> implicit_region;
    std::string kernel;
    Counts metrics;
    Footprint footprint, host_footprint;
    bool footprint_available=false;
    std::map<std::string,Footprint> host_site_footprints;
    std::map<std::string,Site> sites;
    Counts device_metrics;
    std::map<std::string,Site> device_sites;
    dim3 grid, block, cluster;
    size_t dynamic=0;
    std::map<uint64_t,std::unique_ptr<BlockState>> blocks;
    std::vector<unsigned char> cluster_launch; // 0 pending, 1 resident, 2 cancelled
    struct BarrierClock { uint64_t generation=0; std::vector<ThreadContext*> lanes; };
    std::map<void*,BarrierClock> barrier_lanes;
    // site: the arrival (or async completion) that set the phase's ready time.
    struct MbarrierClock { DrperfGpuReady pending; std::string site;
        std::map<uint64_t,std::pair<DrperfGpuReady,std::string>> completed; };
    std::map<uint64_t,MbarrierClock> mbarrier_clocks;
    GxMbarriers mbarriers;
    std::map<uint64_t,SplitBarrier> cluster_barriers;
    std::unique_ptr<gx::LocalFiberBarrier> grid_barrier;
    alignas(8) unsigned grid_bar[16]={0};
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
    for(auto& [k,v]:sites) {if(comma)o<<',';comma=true;o<<quoted(k)<<':'<<object(v.footprint.counts(v.metrics));}
    return o.str()+'}';
}
void add(Counts& dst,const Counts& src) { for(auto& [k,v]:src)dst[k]+=v; }
void latency_metrics(Counts& dst,const DrperfGpuReady& clock) {
    const auto old=DrperfGpuReady{dst["latency_critical_path"],dst["dependent_global_loads_on_critical_path"],0};
    if(drperf_latency::later(clock,old)) {
        dst["latency_critical_path"]=clock.time;
        dst["dependent_global_loads_on_critical_path"]=clock.loads;
    }
}
void subtract(Counts& dst,const Counts& src) {
    for(auto& [k,v]:src) {
        if(dst[k]<v)throw std::runtime_error("device region accounting underflow");
        dst[k]-=v;
    }
}
void line(const std::string& value) {
    const char* path=std::getenv("DRPERF_GPU_CAPTURE");if(!path)return;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::ofstream f(std::string(path)+"."+std::to_string(getpid())+".jsonl",std::ios::app);
    f<<value<<'\n';f.flush();
    if(!f)throw std::runtime_error("cannot write GPU capture");
}
// Translated modules loaded by this process (reported in the finish record so
// reports can resolve sites through each module's <so>.sites.json map).
std::mutex modules_mutex;std::set<std::string> loaded_modules;
std::string modules_json() {
    std::lock_guard<std::mutex> lock(modules_mutex);std::string out="[";
    for(auto& m:loaded_modules){if(out.size()>1)out+=',';out+=quoted(m);}
    return out+"]";
}
struct Lifecycle {
    Lifecycle() {line("{\"event\":\"start\",\"schema\":1,\"backend\":\"gpu\",\"pid\":"+std::to_string(getpid())+"}");}
    ~Lifecycle() {
        try {line("{\"event\":\"finish\",\"pid\":"+std::to_string(getpid())+",\"modules\":"+modules_json()+
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
     <<",\"scope\":"<<quoted(r.scope)<<",\"kernel\":"<<quoted(r.kernel)
     <<",\"linear_block\":"<<r.linear_block<<",\"group\":"<<r.group
     <<",\"latencyModelVersion\":1,\"latencyMetricScope\":\"inclusive-envelope\""
     <<",\"footprintVersion\":"<<(r.footprint_available?1:0)
     <<",\"metrics\":"<<object(r.footprint.counts(r.metrics))<<",\"sites\":"<<sites_json(r.sites)<<",\"launches\":[";
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
uint64_t cluster_major_block(uint64_t ordinal,dim3 grid,dim3 cluster) {
    auto cid=coordinates(ordinal/volume(cluster),dim3(grid.x/cluster.x,grid.y/cluster.y,grid.z/cluster.z));
    auto rank=coordinates(ordinal%volume(cluster),cluster);
    return cid.x*cluster.x+rank.x+uint64_t(grid.x)*(cid.y*cluster.y+rank.y+uint64_t(grid.y)*(cid.z*cluster.z+rank.z));
}
void need_launch() {if(!active)throw std::runtime_error("GPU operation outside a launch");}
BlockState& current_block() {
    need_launch();return *active->blocks.at(gx_thread_context()->linear_block);
}
unsigned thread_index() {
    auto* c=gx_thread_context();
    return c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z);
}
bool device_registration() {
    auto& stack=current_block().device_lanes[thread_index()].stack;
    return !stack.empty() && !stack.back()->ready.count(thread_index());
}
void device_event(const Counts& event,const std::string& site,ThreadContext* lane=nullptr) {
    if(!lane)lane=gx_thread_context();
    auto& b=*active->blocks.at(lane->linear_block);
    unsigned tid=lane->threadIdx.x+lane->blockDim.x*(lane->threadIdx.y+lane->blockDim.y*lane->threadIdx.z);
    auto& stack=b.device_lanes[tid].stack;
    if(stack.empty())return;
    auto& row=stack.back()->row;
    add(row.metrics,event);add(row.sites[site].metrics,event);
    add(active->device_metrics,event);add(active->device_sites[site].metrics,event);
}
void finish_device_block(BlockState& block) {
    for(auto& [key,ptr]:block.device_regions) {
        auto& r=*ptr;
        if(r.entered.size()!=r.expected||r.exited!=r.entered)
            throw std::runtime_error("incomplete cooperative device region participation: "+r.row.name);
        for(auto& [lane,pcvs]:r.member_pcvs)
            if(pcvs!=r.row.pcvs)
                throw std::runtime_error("inconsistent cooperative device PCVs: "+r.row.name);
        r.row.metrics["region_lane_participants"]=r.expected;
        if(!r.row.footprint_available)r.row.metrics["footprint_unavailable_invocations"]=1;
        emit(r.row);--open_regions;++completed_regions;
    }
    block.device_regions.clear();
}
void* symbol(void* module,const std::string& name) {
    dlerror();void* p=dlsym(module,name.c_str());const char* error=dlerror();
    if(error||!p)throw std::runtime_error("GPU module missing symbol "+name);
    return p;
}
std::string lane_breakdown_json(const ThreadContext& lane,uint64_t block,size_t top) {
    std::vector<std::pair<uint64_t,std::string>> rows;
    uint64_t stalled=0;
    for(auto& [site,cycles]:lane.latency_stalls){rows.push_back({cycles,site});stalled+=cycles;}
    std::sort(rows.begin(),rows.end(),[](auto& a,auto& b){return a.first!=b.first?a.first>b.first:a.second<b.second;});
    std::ostringstream o;
    o<<"{\"block\":"<<block<<",\"thread\":"<<(lane.threadIdx.x+lane.blockDim.x*(lane.threadIdx.y+lane.blockDim.y*lane.threadIdx.z))
     <<",\"path\":"<<lane.latency_clock.time<<",\"issue\":"<<lane.latency_issue<<",\"stalled\":"<<stalled
     <<",\"stallSites\":"<<rows.size()<<",\"stalls\":[";
    for(size_t i=0;i<rows.size()&&i<top;++i){if(i)o<<',';o<<"["<<quoted(rows[i].second)<<","<<rows[i].first<<"]";}
    o<<"]}";
    return o.str();
}
// The critical lane plus the slowest lane of every warp in its CTA, so a wait
// on another warp's arrival (named in the stall's "<- site") can be followed.
// Threads named in "[... thread N]" cause tags of a lane's top stalls.
static std::vector<unsigned> cause_threads(const ThreadContext& lane,size_t top) {
    std::vector<std::pair<uint64_t,std::string>> rows;
    for(auto& [site,cycles]:lane.latency_stalls)rows.push_back({cycles,site});
    std::sort(rows.begin(),rows.end(),[](auto& a,auto& b){return a.first>b.first;});
    std::vector<unsigned> out;
    for(size_t i=0;i<rows.size()&&i<top;++i) {
        auto at=rows[i].second.rfind("thread ");
        if(rows[i].second.find(" <- ")==std::string::npos||at==std::string::npos)continue;
        out.push_back(unsigned(std::stoul(rows[i].second.substr(at+7))));
    }
    return out;
}
std::string critical_lane_json(const BlockState& state,const ThreadContext& lane,uint64_t block) {
    std::string out=lane_breakdown_json(lane,block,48);out.pop_back();
    // Follow cross-lane waits: breakdowns of the threads whose arrivals (or
    // tensor-pipe / async-copy issues) the critical lane waited on, recursively.
    out+=",\"chain\":[";
    std::vector<unsigned> todo=cause_threads(lane,4);std::set<unsigned> seen;
    seen.insert(lane.threadIdx.x+lane.blockDim.x*(lane.threadIdx.y+lane.blockDim.y*lane.threadIdx.z));
    bool first=true;
    for(size_t i=0;i<todo.size()&&seen.size()<10;++i) {
        if(!seen.insert(todo[i]).second||todo[i]>=state.lanes.size())continue;
        const auto& cause=state.lanes[todo[i]];
        if(!first)out+=',';first=false;out+=lane_breakdown_json(cause,block,12);
        for(auto t:cause_threads(cause,4))todo.push_back(t);
    }
    out+="],\"warps\":[";
    for(size_t w=0;w*32<state.lanes.size();++w) {
        const ThreadContext* slow=&state.lanes[w*32];
        for(size_t t=w*32;t<std::min(state.lanes.size(),w*32+32);++t)
            if(drperf_latency::later(state.lanes[t].latency_clock,slow->latency_clock))slow=&state.lanes[t];
        if(w)out+=',';out+=lane_breakdown_json(*slow,block,12);
    }
    return out+"]}";
}
void latency_record(ThreadContext* lane) {
    latency_metrics(active->sites[lane->last_site].metrics,lane->latency_clock);
    auto& b=*active->blocks.at(lane->linear_block);
    auto tid=lane->threadIdx.x+lane->blockDim.x*(lane->threadIdx.y+lane->blockDim.y*lane->threadIdx.z);
    // Timing envelopes are inclusive, as are the existing host launch spans.
    for(auto* r:b.device_lanes[tid].stack) {
        auto start=r->latency_entry.at(tid),clock=lane->latency_clock;
        latency_metrics(r->row.sites[lane->last_site].metrics,
            {clock.time-start.time,clock.loads>start.loads?clock.loads-start.loads:0,0});
    }
}
void latency_charge(ThreadContext* lane,uint64_t before,bool issue,const std::string* cause=nullptr) {
    const uint64_t delta=lane->latency_clock.time-before,own=issue&&delta?1:0;
    lane->latency_issue+=own;
    if(delta>own) {
        const std::string site=cause&&!cause->empty()?lane->last_site+" <- "+*cause:lane->last_site;
        lane->latency_stalls[site]+=delta-own;
        if(lane->latency_spin.failures)lane->latency_spin.stalls.push_back({site,delta-own});
    }
}
void latency_join(ThreadContext* lane,const DrperfGpuReady& source,const std::string* cause=nullptr) {
    const uint64_t before=lane->latency_clock.time;
    drperf_latency::join(lane->latency_clock,source);latency_charge(lane,before,false,cause);latency_record(lane);
    // A join is a control boundary for the basic-block schedule.
    lane->latency_block_floor=lane->latency_clock.time;lane->latency_block_issued=0;lane->latency_block_loads=lane->latency_clock.loads;
}
template<class Wait> void joined_wait(void* key,Wait wait) {
    auto& state=active->barrier_lanes[key];auto generation=state.generation;
    state.lanes.push_back(gx_thread_context());wait();
    if(generation==state.generation) {
        DrperfGpuReady clock;uint64_t depth=0;
        for(auto* lane:state.lanes) {
            depth=std::max(depth,lane->instructions);
            drperf_latency::join(clock,lane->latency_clock);
        }
        for(auto* lane:state.lanes) {lane->instructions=depth;latency_join(lane,clock);}
        state.lanes.clear();++state.generation;active->metrics["emulator_barrier_generations"]++;
    }
}
}

extern "C" void drperf_gpu_device_begin(const char* name,unsigned scope) {
    need_launch();
    if(device_registration())throw std::runtime_error("nested GPU region during PCV registration");
    if(!name||!*name||std::strlen(name)>256||scope>2)
        throw std::invalid_argument("invalid GPU device region name or scope");
    auto& l=*active;
    if(!l.region) {
        l.implicit_region=std::make_unique<Invocation>();l.region=l.implicit_region.get();
        l.region->token=next_token++;l.region->parent=0;
        l.region->name="kernel."+l.kernel;l.region->scope="kernel";l.region->kernel=l.kernel;
        ++open_regions;
    }
    unsigned tid=thread_index(), group=scope==0?tid:scope==1?tid/32:0;
    auto& lane=current_block().device_lanes[tid];
    if(lane.stack.size()>=128)throw std::runtime_error("GPU device region nesting exceeds 128");
    auto ordinal=lane.ordinals[{scope,name}]++;
    auto key=std::make_tuple(scope,group,ordinal,std::string(name));
    auto& ptr=current_block().device_regions[key];
    uint64_t parent=lane.stack.empty()?l.region->token:lane.stack.back()->row.token;
    if(!ptr) {
        ptr=std::make_unique<DeviceInvocation>();auto& r=ptr->row;
        r.footprint_available=l.footprint_available;
        r.token=next_token++;r.parent=parent;
        r.scope=scope==0?"lane":scope==1?"warp":"block";
        r.name="device."+r.scope+"."+name;r.kernel=l.kernel;r.linear_block=gx_thread_context()->linear_block;r.group=group;
        ptr->expected=scope==0?1:scope==1?std::min(32u,unsigned(volume(l.block))-32*group):unsigned(volume(l.block));
        ++open_regions;
    }
    if(ptr->row.parent!=parent||!ptr->entered.insert(tid).second)
        throw std::runtime_error("inconsistent cooperative device region nesting: "+ptr->row.name);
    ptr->member_pcvs[tid];ptr->latency_entry[tid]=gx_thread_context()->latency_clock;
    lane.stack.push_back(ptr.get());
}
extern "C" void drperf_gpu_device_pcv(const char* name,int64_t value) {
    need_launch();auto tid=thread_index();auto& stack=current_block().device_lanes[tid].stack;
    if(stack.empty()||!name||!*name||std::strlen(name)>256)
        throw std::invalid_argument("GPU device PCV outside a region or empty name");
    auto& r=*stack.back();auto& pcvs=r.member_pcvs[tid];
    if(r.ready.count(tid))throw std::runtime_error("GPU PCV registered after region entry");
    if(pcvs.size()>=256||!pcvs.emplace(name,value).second)
        throw std::invalid_argument("duplicate or excessive GPU device PCV");
    auto [it,inserted]=r.row.pcvs.emplace(name,value);
    if(!inserted&&it->second!=value)
        throw std::runtime_error("inconsistent cooperative device PCV value: "+r.row.name);
}
extern "C" void drperf_gpu_device_ready() {
    need_launch();auto tid=thread_index();auto& stack=current_block().device_lanes[tid].stack;
    if(stack.empty()||!stack.back()->ready.insert(tid).second)
        throw std::runtime_error("invalid GPU PCV registration completion");
    stack.back()->latency_entry[tid]=gx_thread_context()->latency_clock;
}
extern "C" void drperf_gpu_device_end() {
    need_launch();auto tid=thread_index();auto& stack=current_block().device_lanes[tid].stack;
    if(stack.empty())throw std::runtime_error("unbalanced GPU device markers");
    if(!stack.back()->ready.count(tid))throw std::runtime_error("GPU region missing entry PCV completion; regenerate device annotations");
    auto& r=*stack.back();auto clock=gx_thread_context()->latency_clock;
    auto start=r.latency_entry.at(tid);
    latency_metrics(r.row.metrics,{clock.time-start.time,clock.loads>start.loads?clock.loads-start.loads:0,0});
    r.exited.insert(tid);stack.pop_back();
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
        if(*abi!=3)throw std::runtime_error("GPU instrumentation ABI mismatch; retranslate with current drperf");
        auto latency=static_cast<const unsigned*>(dlsym(module,"drperf_gpu_latency_abi"));
        if(!latency||*latency!=1)
            throw std::runtime_error("GPU latency instrumentation ABI mismatch; retranslate with current drperf");
    } catch(...) {dlclose(module);throw;}
    {std::lock_guard<std::mutex> lock(modules_mutex);
     char* real=realpath(path,nullptr);loaded_modules.insert(real?real:path);std::free(real);}
    return module;
}
extern "C" void drperf_gpu_module_close(void* module) {if(module)dlclose(module);}
extern "C" void drperf_gpu_launch_cluster(void* module,const char* kernel,dim3 grid,dim3 block,
                                  void** args,size_t dynamic,dim3 cluster) {
    if(active||!module||!kernel||!*kernel||!volume(grid)||!volume(block)||
       volume(block)>1024||volume(grid)>(1u<<24)||dynamic>(16u<<20)||
       !volume(cluster)||volume(cluster)>16||grid.x%cluster.x||grid.y%cluster.y||grid.z%cluster.z)
        throw std::invalid_argument("invalid or recursive GPU launch");
    auto fn=reinterpret_cast<void(*)(void**)>(symbol(module,std::string(kernel)+"__gxptx"));
    auto latency_abi=static_cast<const unsigned*>(dlsym(module,(std::string(kernel)+"__gxptx_latency_abi").c_str()));
    if(!latency_abi||*latency_abi!=1)
        throw std::runtime_error("GPU kernel latency instrumentation ABI mismatch; retranslate with current drperf");
    unsigned n=*static_cast<unsigned*>(symbol(module,std::string(kernel)+"__gxptx_nparams"));
    if(n&&!args)throw std::invalid_argument("GPU launch missing arguments");
    for(unsigned i=0;i<n;++i)if(!args[i])throw std::invalid_argument("null GPU launch argument");
    Launch l;l.region=regions.empty()?nullptr:regions.back().get();l.grid=grid;l.block=block;l.dynamic=dynamic;l.kernel=kernel;l.cluster=cluster;
    l.metrics={{"launches",1},{"blocks",volume(grid)},{"launched_lanes",volume(grid)*volume(block)},
               {"warps",volume(grid)*((volume(block)+31)/32)},{"dynamic_shared_bytes_per_launch",dynamic}};
    add(l.metrics,l.latency.counts());
    // A separate capability marker preserves work-counter ABI 3. Older
    // modules execute, but never claim complete address coverage.
    auto fpabi=static_cast<const unsigned*>(dlsym(module,"drperf_gpu_footprint_abi"));
    auto kernel_fpabi=static_cast<const unsigned*>(dlsym(module,(std::string(kernel)+"__gxptx_footprint_abi").c_str()));
    l.footprint_available=fpabi&&*fpabi==1&&kernel_fpabi&&*kernel_fpabi==1;
    if(!l.footprint_available)l.metrics["footprint_unavailable_launches"]=1;
    active=&l;
    try {
        gx::FiberStack stacks;
        const auto mode=*static_cast<unsigned*>(symbol(module,std::string(kernel)+"__gxptx_syncfree"));
        const bool resident=mode==2;
        if(resident&&volume(grid)>uint64_t(INT32_MAX)/volume(block))
            throw std::invalid_argument("cooperative launch exceeds supported participant count");
        const uint64_t batch=resident?volume(grid):volume(cluster);
        l.cluster_launch.assign(volume(grid)/volume(cluster),0);
        for(uint64_t first=0;first<volume(grid);first+=batch) {
            if(!resident&&l.cluster_launch[first/volume(cluster)]==2)continue;
            for(uint64_t ordinal=first;ordinal<first+batch;++ordinal) {
                uint64_t b=cluster_major_block(ordinal,grid,cluster);
                l.cluster_launch[ordinal/volume(cluster)]=1;
                ++l.metrics["executed_blocks"];l.metrics["executed_lanes"]+=volume(block);
                auto state=std::make_unique<BlockState>(volume(block));
                for(unsigned t=0;t<volume(block);++t) {
                    auto& lane=state->lanes[t];lane.threadIdx=coordinates(t,block);lane.blockIdx=coordinates(b,grid);
                    lane.blockDim=block;lane.gridDim=grid;lane.linear_block=b;lane.barriers=&state->barriers;
                }
                l.blocks.emplace(b,std::move(state));
            }
            gx::KernelFibers fibers;
            for(uint64_t ordinal=first;ordinal<first+batch;++ordinal)for(unsigned t=0;t<volume(block);++t) {
                uint64_t b=cluster_major_block(ordinal,grid,cluster);
                fibers.add(stacks,[&,b,t] {
                    auto& state=*l.blocks.at(b);
                    gx::KernelFibers::active()->data=&state.lanes[t];fn(args);
                    if(!state.device_lanes[t].stack.empty())throw std::runtime_error("unclosed GPU device region at kernel exit");
                    state.barriers.lane_exited(0,t);
                });
            }
            fibers.run();
            for(auto& [b,state]:l.blocks) {
                finish_device_block(*state);
                uint64_t max_f=0,max_b=0,max_i=0;
                const ThreadContext* slowest=nullptr;
                for(auto& lane:state->lanes) {
                    max_f=std::max(max_f,lane.flops);max_b=std::max(max_b,lane.bytes);
                    max_i=std::max(max_i,lane.instructions);
                    if(!slowest||drperf_latency::later(lane.latency_clock,slowest->latency_clock))slowest=&lane;
                }
                if(slowest&&drperf_latency::later(slowest->latency_clock,l.latency_finish))
                    l.critical_lane=critical_lane_json(*state,*slowest,b);
                if(slowest)drperf_latency::join(l.latency_finish,slowest->latency_clock);
                l.metrics["sum_block_max_lane_flops"]+=max_f;
                l.metrics["sum_block_max_lane_bytes"]+=max_b;
                l.metrics["sum_block_lane_sequence_span"]+=max_i;
                l.metrics["sum_launch_max_block_lane_sequence_span"]=std::max(l.metrics["sum_launch_max_block_lane_sequence_span"],max_i);
            }
            l.blocks.clear();l.barrier_lanes.clear();
        }
        latency_metrics(l.metrics,l.latency_finish);
        if(!l.metrics["ptx_lane_instructions"])
            throw std::runtime_error("GPU kernel executed without instrumentation");
        if(l.region) {
            auto own=l.metrics;subtract(own,l.device_metrics);add(l.region->metrics,own);
            l.region->footprint_available&=l.footprint_available;
            l.region->footprint.merge(l.host_footprint);
            for(auto& [site,c]:l.sites) {
                auto counts=c.metrics;subtract(counts,l.device_sites[site].metrics);
                add(l.region->sites[site].metrics,counts);
                l.region->sites[site].footprint.merge(l.host_site_footprints[site]);
            }
            std::ostringstream shape;
            shape<<"{\"kernel\":"<<quoted(kernel)<<",\"grid\":["<<grid.x<<','<<grid.y<<','<<grid.z
                 <<"],\"block\":["<<block.x<<','<<block.y<<','<<block.z<<"],\"metricScope\":\"inclusive\",\"footprintVersion\":"<<(l.footprint_available?1:0)
                 <<",\"metrics\":"<<object(l.footprint.counts(l.metrics))<<",\"sites\":"<<sites_json(l.sites);
            if(!l.critical_lane.empty())shape<<",\"criticalLane\":"<<l.critical_lane;
            shape<<'}';
            l.region->launches.push_back(shape.str());
            if(l.implicit_region){emit(*l.region);--open_regions;++completed_regions;}
        } else if(std::getenv("DRPERF_GPU_CAPTURE")) {
            throw std::runtime_error("GPU launch has no enclosing GPU region");
        }
    } catch(...) {++launch_failures;active=nullptr;throw;}
    active=nullptr;
}

extern "C" void drperf_gpu_launch(void* module,const char* kernel,dim3 grid,dim3 block,void** args,size_t dynamic) {
    drperf_gpu_launch_cluster(module,kernel,grid,block,args,dynamic,dim3(1));
}

// Semantic address hooks describe logical ranges, excluding implementation
// loops except when those loops enumerate descriptor/stride-selected elements.
extern "C" void drperf_gpu_resource(const char* metric,uint64_t amount);
extern "C" void drperf_gpu_mark_uninitialized(const void* p,size_t n,const char* name) {
    drperf_initcheck::tracker().mark(reinterpret_cast<uint64_t>(p),n,false,name);
}
extern "C" void drperf_gpu_mark_initialized(const void* p,size_t n) {
    drperf_initcheck::tracker().mark(reinterpret_cast<uint64_t>(p),n,true,nullptr);
}
extern "C" void drperf_gpu_forget_buffer(const void* p) {
    drperf_initcheck::tracker().forget(reinterpret_cast<uint64_t>(p));
}

extern "C" void drperf_gpu_memory(uint64_t address,uint64_t reads,uint64_t writes,unsigned space) {
    need_launch();
    if(device_registration())return;
    // Read-before-write check on buffers the driver marked uninitialized; shared
    // addresses never match a registered global range.
    if((reads||writes)&&space<=1&&!drperf_initcheck::tracker().empty()) {
        const uint64_t bad=drperf_initcheck::tracker().access(address,reads,writes,gx_thread_context()->last_site);
        if(bad)drperf_gpu_resource("uninitialized_read_bytes",bad);
    }
    if((reads||writes)&&space<=1&&active->latency.global_hit!=active->latency.global&&address) {
        bool hit=true;
        for(uint64_t l=address/128,e=(address+std::max(reads,writes)-1)/128;l<=e;++l)
            hit&=!active->touched_lines.insert(l).second;
        gx_thread_context()->latency_load_hit=hit&&reads;
    }
    if(!active->footprint_available||(!reads&&!writes))return;
    if(space==0)for(auto& [id,state]:active->blocks) {
        auto base=reinterpret_cast<uint64_t>(state->shared);
        if(state->shared&&address>=base&&address-base<state->cap)return;
    }
    if(space!=0&&space!=1)return;
    auto* lane=gx_thread_context();
    const std::string category=space==1?"global":"generic";
    auto touch=[&](Footprint& fp) {
        fp.touch(address,reads,category,"read");fp.touch(address,writes,category,"write");
    };
    touch(active->footprint);touch(active->sites[lane->last_site].footprint);
    auto& stack=current_block().device_lanes[thread_index()].stack;
    if(stack.empty()) {
        touch(active->host_footprint);touch(active->host_site_footprints[lane->last_site]);
    } else {
        auto& row=stack.back()->row;touch(row.footprint);touch(row.sites[lane->last_site].footprint);
    }
}

extern "C" void drperf_gpu_latency(unsigned cls,const DrperfGpuReady* src,size_t nsrc,
                                    DrperfGpuReady* const* dst,size_t ndst,bool record) {
    need_launch();if(device_registration())return;auto* lane=gx_thread_context();
    const bool hit=(cls==2||cls==6)&&lane->latency_load_hit;lane->latency_load_hit=false;
    const uint64_t op_latency=hit?active->latency.global_hit:active->latency.latency(cls);
    const uint64_t before=lane->latency_clock.time;
    DrperfGpuReady ready;
    if(active->latency.schedule) {
        ready=drperf_latency::issue_block(lane->latency_clock,lane->latency_block_floor,lane->latency_block_loads,lane->latency_block_issued,
                                          src,nsrc,op_latency,cls==2);
        // The instruction call precedes this one, so last_site names its opcode.
        const auto colon=lane->last_site.rfind(':');
        const std::string op=colon==std::string::npos?lane->last_site:lane->last_site.substr(colon+1);
        auto starts=[&](const char* p){return op.rfind(p,0)==0;};
        // "*.ifconv": a branch of a short forward region ptxas would predicate
        // (tagged at translation); it is not a scheduling boundary.
        const bool ifconv=op.size()>7&&op.compare(op.size()-7,7,".ifconv")==0;
        if((starts("bra")&&!ifconv)||starts("bar")||starts("barrier")||starts("ret")||starts("exit")||starts("call")||
           starts("brx")||starts("mbarrier")||starts("griddepcontrol")||starts("fence")||starts("membar")) {
            lane->latency_block_floor=lane->latency_clock.time;lane->latency_block_issued=0;lane->latency_block_loads=lane->latency_clock.loads;
        }
    } else ready=drperf_latency::issue(lane->latency_clock,src,nsrc,op_latency,cls==2);
    latency_charge(lane,before,true);
    for(size_t i=0;i<ndst;++i)*dst[i]=ready;
    lane->latency_last_dst=dst;lane->latency_last_ndst=ndst;
    if(record)latency_record(lane);
}
extern "C" void drperf_gpu_latency_predicated_off(const char* fn,const char* op,unsigned site,
                                                   const DrperfGpuReady* src,size_t nsrc) {
    need_launch();auto* lane=gx_thread_context();
    lane->last_site=std::string(fn)+":cpp"+std::to_string(site)+":"+op;
    drperf_gpu_latency(0,src,nsrc,nullptr,0,true);
}

static DrperfGpuReady async_ready_from_now();
extern "C" void drperf_gpu_instruction(const char* fn,const char* op,
        unsigned site,unsigned long long flops,unsigned long long reads,unsigned long long writes,
        unsigned space,unsigned kind,unsigned long long address,unsigned unmodeled) {
    need_launch();if(device_registration())return;auto* lane=gx_thread_context();lane->instructions++;
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
    if(!space&&(reads||writes)) {
        uint64_t base=reinterpret_cast<uint64_t>(current_block().shared);
        if(current_block().shared&&address>=base&&address-base<current_block().cap)space=2;
        else for(auto& [id,state]:active->blocks) {
            uint64_t peer=reinterpret_cast<uint64_t>(state->shared);
            if(state->shared&&address>=peer&&address-peer<state->cap){space=2;break;}
        }
    }
    static const char* spaces[]={"generic","global","shared","local","param","const"};
    std::string s=spaces[std::min(space,5u)];
    if(head=="ld"||head=="ldu")event[s+"_load_instructions"]=1;
    if(reads)event[(space==6?"global":s)+"_read_bytes"]=reads;
    if(writes)event[(space==6?"shared":s)+"_write_bytes"]=writes;
    if(unmodeled)event["unmodeled_work_events"]=1;
    lane->last_site=std::string(fn)+":cpp"+std::to_string(site)+":"+op;
    // WMMA helpers enumerate the actual strided tile; cp.async reads global
    // source bytes and zero-fills shared destination bytes separately.
    if(head!="wmma")drperf_gpu_memory(address,reads,space==6?0:writes,space==6?1:space);
    if(space==6)drperf_latency::join(lane->latency_async,async_ready_from_now());
    add(active->metrics,event);add(active->sites[lane->last_site].metrics,event);
    device_event(event,lane->last_site);
}

// Dynamic tensor descriptors determine work at runtime. Attribute these counts
// to the original instruction, excluding the reference interpreter's loops.
extern "C" void drperf_gpu_resource_at(ThreadContext* lane,const char* metric,uint64_t amount) {
    need_launch();
    // Semantic helpers can attribute to a different lane; inspect that lane.
    auto& stack=active->blocks.at(lane->linear_block)->device_lanes[
        lane->threadIdx.x+lane->blockDim.x*(lane->threadIdx.y+lane->blockDim.y*lane->threadIdx.z)].stack;
    unsigned tid=lane->threadIdx.x+lane->blockDim.x*(lane->threadIdx.y+lane->blockDim.y*lane->threadIdx.z);
    if(!stack.empty()&&!stack.back()->ready.count(tid))return;
    active->metrics[metric]+=amount;
    active->sites[lane->last_site].metrics[metric]+=amount;
    device_event(Counts{{metric,amount}},lane->last_site,lane);
    const std::string name(metric);
    if(name.find("flops")!=std::string::npos||name=="tensor_integer_ops")lane->flops+=amount;
    if(name.find("_bytes")!=std::string::npos)lane->bytes+=amount;
}
extern "C" void drperf_gpu_resource(const char* metric,uint64_t amount) {
    drperf_gpu_resource_at(gx_thread_context(),metric,amount);
}

GxTmemState& gx_cta_tmem(long block) {
    need_launch();auto it=active->blocks.find(block);
    if(it==active->blocks.end())throw std::runtime_error("tensor operation on nonresident CTA");
    return it->second->tmem;
}
long gx_current_linear_block(){need_launch();return gx_thread_context()->linear_block;}
extern "C" void gx_mbarrier_init(uint64_t a,uint32_t n){
    need_launch();active->mbarriers.init(a,n);active->mbarrier_clocks.erase(a);
}
static std::string by_thread(const ThreadContext* lane,const char* what) {
    return std::string(" [")+what+"thread "+std::to_string(lane->threadIdx.x+lane->blockDim.x*(lane->threadIdx.y+lane->blockDim.y*lane->threadIdx.z))+"]";
}
static void mbarrier_clock_pending(uint64_t a,const DrperfGpuReady& at,const std::string& site) {
    auto& c=active->mbarrier_clocks[a];
    if(drperf_latency::later(at,c.pending)||c.site.empty())c.site=site;
    drperf_latency::join(c.pending,at);
}
static void mbarrier_clock_arrival(uint64_t a,uint64_t phase,const DrperfGpuReady* at=nullptr,const std::string* site=nullptr) {
    auto* lane=gx_thread_context();
    mbarrier_clock_pending(a,at?*at:lane->latency_clock,site?*site:lane->last_site+by_thread(lane,""));
    auto& c=active->mbarrier_clocks[a];
    if(active->mbarriers.phase(a)!=phase) {c.completed[phase]={c.pending,c.site};c.pending={};c.site.clear();}
}
// Data delivered by an asynchronous copy is ready a global latency after issue.
static DrperfGpuReady async_ready_from_now() {
    auto* lane=gx_thread_context();DrperfGpuReady r=lane->latency_clock;
    r.time+=active->latency.global;r.loads+=1;r.global=0;return r;
}
extern "C" void gx_cp_async_mbarrier_arrive(uint64_t a,uint32_t noinc) {
    need_launch();auto* lane=gx_thread_context();
    DrperfGpuReady at=lane->latency_clock;drperf_latency::join(at,lane->latency_async);
    if(noinc) {auto phase=active->mbarriers.arrive(a,1,0);const auto site=lane->last_site+by_thread(lane,"async copy, ");
        mbarrier_clock_arrival(a,phase,&at,&site);}
    else mbarrier_clock_pending(a,at,lane->last_site+by_thread(lane,"async copy, "));
}
extern "C" void gx_cp_async_wait() {
    need_launch();auto* lane=gx_thread_context();
    if(drperf_latency::later(lane->latency_async,lane->latency_clock))latency_join(lane,lane->latency_async);
}
extern "C" uint64_t gx_mbarrier_arrive(uint64_t a,uint32_t n,uint32_t b){
    need_launch();auto phase=active->mbarriers.arrive(a,n,b);mbarrier_clock_arrival(a,phase);return phase;
}
extern "C" void gx_mbarrier_complete_tx(uint64_t a,uint32_t n){
    // TMA / bulk copies: the transaction completes a global latency after issue.
    need_launch();auto phase=active->mbarriers.phase(a);active->mbarriers.complete_tx(a,n);
    auto at=async_ready_from_now();const std::string site=gx_thread_context()->last_site+by_thread(gx_thread_context(),"async copy, ");
    mbarrier_clock_arrival(a,phase,&at,&site);
}
// tcgen05.mma: occupy the CTA's tensor pipe (see LatencyConfig::tensor_rate).
extern "C" void drperf_gpu_tensor_issue(uint64_t flops,uint32_t operand_bits,uint32_t cta_group) {
    need_launch();if(device_registration())return;auto* lane=gx_thread_context();
    auto& b=current_block();const auto& cfg=active->latency;
    const long double rate=(long double)cfg.tensor_rate*16/std::max<uint32_t>(operand_bits,1)*std::max<uint32_t>(cta_group,1);
    const uint64_t cost=std::max<uint64_t>(1,(uint64_t)std::ceil((long double)flops/rate));
    b.tensor_free=std::max(b.tensor_free,lane->latency_clock.time)+cost;
    b.tensor_ready={b.tensor_free+cfg.mma,lane->latency_clock.loads,0};b.tensor_site=lane->last_site+by_thread(lane,"tensor pipe, ");
}
// tcgen05.commit: the arrival happens when the CTA's tensor pipe has drained.
extern "C" uint64_t gx_mbarrier_arrive_tensor(uint64_t a){
    need_launch();auto* lane=gx_thread_context();auto& b=current_block();
    DrperfGpuReady at=lane->latency_clock;std::string site=lane->last_site+by_thread(lane,"");
    if(drperf_latency::later(b.tensor_ready,at)){at=b.tensor_ready;site=b.tensor_site;}
    auto phase=active->mbarriers.arrive(a,1,0);mbarrier_clock_arrival(a,phase,&at,&site);return phase;
}
extern "C" void gx_mbarrier_expect_tx(uint64_t a,uint32_t n){need_launch();active->mbarriers.expect_tx(a,n);}
static void mbarrier_clock_wait(uint64_t a,uint64_t phase) {
    auto& completed=active->mbarrier_clocks[a].completed;auto it=completed.find(phase);
    if(it!=completed.end())latency_join(gx_thread_context(),it->second.first,&it->second.second);
}
extern "C" int gx_mbarrier_test(uint64_t a,uint64_t t,int p){
    need_launch();bool done=active->mbarriers.test(a,t,p);
    auto* lane=gx_thread_context();auto& spin=lane->latency_spin;
    if(!done) {
        if(!spin.failures||spin.address!=a||spin.phase!=t)
            spin={a,t,0,lane->instructions,lane->latency_issue,lane->latency_clock,{}};
        ++spin.failures;
        return done;
    }
    bool rolled=false;
    if(spin.failures&&spin.address==a&&spin.phase==t) {
        // Only a tight retry loop (a few instructions per probe) is rolled back;
        // work done between probes keeps its issue time.
        const uint64_t spun=lane->instructions-spin.instructions;
        if(spun<=8*spin.failures+8&&!device_registration()) {
            active->metrics["mbarrier_spin_lane_instructions"]+=spun;
            lane->latency_clock=spin.clock;lane->latency_issue=spin.issue;
            for(auto& [site,cycles]:spin.stalls)
                if((lane->latency_stalls[site]-=cycles)==0)lane->latency_stalls.erase(site);
            lane->latency_block_floor=lane->latency_clock.time;lane->latency_block_issued=0;lane->latency_block_loads=lane->latency_clock.loads;
            rolled=true;
        }
    }
    spin.failures=0;spin.stalls.clear();
    mbarrier_clock_wait(a,p?active->mbarriers.phase(a)-1:t);
    // The probe's predicate was timed at the last spin iteration; after a
    // rollback it is ready one ALU latency after the (joined) lane clock.
    if(rolled)for(size_t i=0;i<lane->latency_last_ndst;++i)
        *lane->latency_last_dst[i]={lane->latency_clock.time+active->latency.alu,lane->latency_clock.loads,0};
    return done;
}
extern "C" void gx_mbarrier_wait(uint64_t a,uint64_t t,uint32_t p){
    need_launch();auto phase=active->mbarriers.phase(a);
    auto target=p?(active->mbarriers.test(a,t,true)?phase-1:phase):t;
    active->mbarriers.wait(a,t,p);mbarrier_clock_wait(a,target);
}

int get_block_id(){return 0;} // A fresh manager is owned by each block.
int get_lane_id(){auto* c=gx_thread_context();return (c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z))%32;}
int get_warp_id_in_block(){auto* c=gx_thread_context();return (c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z))/32;}
void record_barrier_call(){need_launch();active->metrics["emulator_rendezvous_lane_arrivals"]++;}
void gpu_barrier_wait(NamedBarrierManager::BarrierIface* barrier) {
    joined_wait(barrier,[&]{barrier->wait();});
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
    need_launch();
    if(statics>(16u<<20)||active->dynamic>(16u<<20)-statics)
        throw std::invalid_argument("GPU shared declaration exceeds 24-bit cluster address space");
    const size_t needed=statics+active->dynamic,page=sysconf(_SC_PAGESIZE);
    // The module layout is uniform. Allocate peers before a cluster operation
    // can address a CTA whose first lane has not yet run.
    for(auto& [id,ptr]:active->blocks) {
        auto& b=*ptr;b.static_bytes=std::max(b.static_bytes,statics);
        if(b.shared) {
            if(needed>b.cap)throw std::runtime_error("shared arena grew during a kernel");
            continue;
        }
        b.cap=((std::max(needed,size_t(1024))+page-1)/page)*page;b.mapping=b.cap+page;
        b.shared=arena_pool.take(b.cap,b.mapping);
        if(!b.shared)b.shared=static_cast<char*>(mmap(nullptr,b.mapping,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
        if(b.shared==MAP_FAILED){b.shared=nullptr;throw std::bad_alloc();}
        if(mprotect(b.shared+b.cap,page,PROT_NONE))throw std::runtime_error("GPU shared guard failed");
    }
    active->metrics["static_shared_bytes_per_launch"]=std::max<uint64_t>(active->metrics["static_shared_bytes_per_launch"],statics);
    return current_block().shared;
}
extern "C" int memory_mg_is_fake(void*){return 0;}
void global_memcpy(void* dst,const void* src,size_t size,int){std::memcpy(dst,src,size);}
extern "C" void gx_ptx_trap(){throw std::runtime_error("GPU PTX trap executed");}
extern "C" void gx_ptx_trap_op(const char* op){throw std::runtime_error(std::string("unsupported GPU PTX operation: ")+op);}

extern "C" void __syncwarp_mask(unsigned mask) {
    const unsigned threads=current_block().lanes.size(),warp=get_warp_id_in_block(),lane=get_lane_id();
    unsigned lanes=std::min(32u,threads-warp*32);
    mask&=lanes==32?~0u:(1u<<lanes)-1;
    if(mask&(1u<<lane)){record_barrier_call();current_named_barrier_manager->warp_sync(0,warp,lane,mask);}
}
extern "C" uint32_t gx_shfl_sync(uint32_t value,uint32_t offset,uint32_t clamp,
                                  uint32_t mask,int mode,int* predicate) {
    const unsigned lane=get_lane_id(),warp=get_warp_id_in_block();
    unsigned lanes=std::min(32u,unsigned(current_block().lanes.size())-warp*32);
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
    unsigned count=std::min(32u,unsigned(current_block().lanes.size())-warp*32);
    mask&=count==32?~0u:(1u<<count)-1;
    if(!(mask&(1u<<lane)))return value;
    record_barrier_call();return current_named_barrier_manager->get_warp_exchange(0,warp,mask)->reduce(lane,value,op);
}
extern "C" uint32_t gx_redux_add_sync(uint32_t v,uint32_t m){return gx_redux_sync(v,m,0);}
static uint32_t vote(unsigned mask,int p) {
    unsigned lane=get_lane_id(),warp=get_warp_id_in_block();
    unsigned count=std::min(32u,unsigned(current_block().lanes.size())-warp*32);
    mask&=count==32?~0u:(1u<<count)-1;
    if(!(mask&(1u<<lane)))return 0;
    record_barrier_call();return current_named_barrier_manager->get_warp_exchange(0,warp,mask)->ballot(lane,p);
}
unsigned __ballot_sync(unsigned m,int p){return vote(m,p);}
int __any_sync(unsigned m,int p){return vote(m,p)!=0;}
int __all_sync(unsigned m,int p){
    unsigned count=std::min(32u,unsigned(current_block().lanes.size())-unsigned(get_warp_id_in_block())*32);
    return vote(m,p)==(m&(count==32?~0u:(1u<<count)-1));
}
extern "C" uint64_t gx_block_scan_u32(uint32_t v,uint32_t exclusive) {
    auto* c=gx_thread_context();unsigned t=c->threadIdx.x+c->blockDim.x*(c->threadIdx.y+c->blockDim.y*c->threadIdx.z);
    record_barrier_call();return current_named_barrier_manager->get_block_scan(0,current_block().lanes.size())->scan(t,v,exclusive);
}

// Runtime adapters shared by standard PTX and GX's explicit semantic helpers.
// Geometry and shared addresses follow GX's rank<<24 | offset representation.
extern "C" dim3 gx_dev_clusterDim(){need_launch();return active->cluster;}
extern "C" dim3 gx_dev_cluster_nctaid(){return gx_dev_clusterDim();}
extern "C" dim3 gx_dev_cluster_ctaid(){auto b=gx_thread_context()->blockIdx,c=active->cluster;return dim3(b.x%c.x,b.y%c.y,b.z%c.z);}
extern "C" dim3 gx_dev_clusterid(){auto b=gx_thread_context()->blockIdx,c=active->cluster;return dim3(b.x/c.x,b.y/c.y,b.z/c.z);}
extern "C" dim3 gx_dev_nclusterid(){auto g=active->grid,c=active->cluster;return dim3(g.x/c.x,g.y/c.y,g.z/c.z);}
extern "C" uint32_t gx_dev_cluster_ctarank(){auto b=gx_dev_cluster_ctaid(),c=active->cluster;return b.x+c.x*(b.y+c.y*b.z);}
extern "C" uint32_t gx_dev_cluster_nctarank(){need_launch();return volume(active->cluster);}
extern "C" uint32_t gx_dev_is_explicit_cluster(){return gx_dev_cluster_nctarank()>1;}
extern "C" uint32_t gx_shared_rank_bits(){return gx_dev_cluster_ctarank()<<24;}
long gx_cluster_rank_linear_block(uint32_t rank){
    need_launch();if(rank>=volume(active->cluster))throw std::runtime_error("cluster rank outside launch geometry");
    auto c=active->cluster,b=gx_thread_context()->blockIdx,r=coordinates(rank,c),g=active->grid;
    return b.x/c.x*c.x+r.x+uint64_t(g.x)*(b.y/c.y*c.y+r.y+uint64_t(g.y)*(b.z/c.z*c.z+r.z));
}
extern "C" uint32_t gx_cluster_mapa(uint32_t address,uint32_t rank){
    if(rank>=gx_dev_cluster_nctarank())throw std::runtime_error("invalid mapa rank");
    return (rank<<24)|(address&0xffffff);
}
extern "C" uint64_t gx_cluster_shared_ptr(uint32_t address){
    auto block=gx_cluster_rank_linear_block(address>>24);auto it=active->blocks.find(block);
    if(it==active->blocks.end())throw std::runtime_error("shared access to nonresident CTA");
    auto& state=*it->second;uint32_t offset=address&0xffffff;
    if(!state.shared||offset>=state.cap)throw std::runtime_error("shared offset out of range");
    return reinterpret_cast<uint64_t>(state.shared+offset);
}
extern "C" uint64_t gx_cluster_mapa_generic(uint64_t address,uint32_t rank){
    auto& b=current_block();uint64_t base=reinterpret_cast<uint64_t>(b.shared);
    if(address<base||address-base>=b.cap)throw std::runtime_error("generic mapa outside own shared window");
    return gx_cluster_shared_ptr(gx_cluster_mapa(address-base,rank));
}
static uint64_t cluster_id(){auto b=gx_dev_clusterid(),g=gx_dev_nclusterid();return b.x+uint64_t(g.x)*(b.y+uint64_t(g.y)*b.z);}
extern "C" void gx_cluster_barrier_arrive(){
    auto& barrier=active->cluster_barriers[cluster_id()];
    gx_thread_context()->cluster_barrier_token=barrier.phase;
    drperf_latency::join(barrier.pending,gx_thread_context()->latency_clock);
    if(++barrier.arrived==volume(active->cluster)*volume(active->block)){
        barrier.completed[barrier.phase]=barrier.pending;barrier.pending={};
        barrier.arrived=0;++barrier.phase;auto self=gx::FiberRef::active();
        for(auto waiter:barrier.waiters)self.schedule(waiter);
        barrier.waiters.clear();
    }
}
extern "C" void gx_cluster_barrier_wait(){
    auto& barrier=active->cluster_barriers[cluster_id()];
    auto phase=gx_thread_context()->cluster_barrier_token;
    if(barrier.phase<=phase) {
        auto self=gx::FiberRef::active();barrier.waiters.push_back(self);self.suspend();
    }
    latency_join(gx_thread_context(),barrier.completed.at(phase));
}
extern "C" void gx_grid_sync(){
    need_launch();if(!active->grid_barrier)active->grid_barrier=std::make_unique<gx::LocalFiberBarrier>(volume(active->grid)*volume(active->block));
    record_barrier_call();joined_wait(active->grid_barrier.get(),[&]{active->grid_barrier->wait();});
}
extern "C" uint64_t gx_grid_barrier_ptr(){need_launch();return reinterpret_cast<uint64_t>(active->grid_bar);}
extern "C" uint32_t gx_dev_envreg(unsigned index){
    if(index!=1&&index!=2)throw std::runtime_error("invalid cooperative grid environment register");
    auto a=gx_grid_barrier_ptr();return index==1?a>>32:a;
}
extern "C" void gx_cpu_kernel_poll(){need_launch();gx::kernel_fiber_yield();}
extern "C" void gx_nccl_abort_poll(uint64_t a){if(!a)throw std::runtime_error("null abort poll");gx_cpu_kernel_poll();}
extern "C" void gx_nccl_payload_scope(uint32_t){need_launch();}
extern "C" void gx_nccl_payload_event_v2(uint64_t,uint64_t bytes,uint32_t load,uint64_t instructions){
    drperf_gpu_resource(load?"annotated_global_read_bytes":"annotated_global_write_bytes",bytes);
    drperf_gpu_resource("annotated_memory_lane_instructions",instructions);
}
extern "C" void gx_global_memcpy_counted(void* dst,const void* src,size_t size,int load,uint64_t instructions){
    std::memcpy(dst,src,size);
    drperf_gpu_memory(reinterpret_cast<uint64_t>(load?src:dst),load?size:0,load?0:size,1);
    drperf_gpu_resource(load?"global_read_bytes":"global_write_bytes",size);
    drperf_gpu_resource("annotated_memory_lane_instructions",instructions);
}
extern "C" uint32_t gx_global_load_u32_repeated(uint64_t a,uint32_t n){
    if(!a||a%4||!n)throw std::runtime_error("invalid repeated word load");uint32_t v;std::memcpy(&v,reinterpret_cast<void*>(a),4);
    drperf_gpu_memory(a,4,0,1);
    drperf_gpu_resource("global_read_bytes",uint64_t(n)*4);drperf_gpu_resource("annotated_memory_lane_instructions",n);return v;
}
extern "C" uint32_t gx_global_load_u8_repeated(uint64_t a,uint32_t n){
    if(!a||!n)throw std::runtime_error("invalid repeated byte load");
    drperf_gpu_memory(a,1,0,1);
    drperf_gpu_resource("global_read_bytes",n);drperf_gpu_resource("annotated_memory_lane_instructions",n);return *reinterpret_cast<uint8_t*>(a);
}
extern "C" void gx_warp_ballot_columns(uint32_t predicates,uint32_t columns,uint32_t* out){
    unsigned lanes=std::min(32u,unsigned(current_block().lanes.size())-get_warp_id_in_block()*32);
    unsigned mask=lanes==32?~0u:(1u<<lanes)-1;
    record_barrier_call();current_named_barrier_manager->get_warp_exchange(0,get_warp_id_in_block(),mask)->ballot_columns(get_lane_id(),predicates,columns,out);
}
static int barrier_red(int name,int count,int pred,bool is_or){
    if(!count)count=current_block().lanes.size();
    if(count<1||count>int(current_block().lanes.size()))throw std::runtime_error("invalid reduction barrier participant count");
    auto* b=current_named_barrier_manager->get_barrier(60000000+(name&15)*1024+(count&1023),count);
    if(!b->vote_count)b->result=!is_or;
    if(is_or?pred!=0:pred==0)b->result=is_or;
    if(++b->vote_count==count)b->vote_count=0;
    record_barrier_call();BARRIER_WAIT(b->barrier);bool result=b->result;BARRIER_WAIT(b->barrier);return result;
}
extern "C" int gx_barrier_red_or(int b,int n,int p){return barrier_red(b,n,p,true);}
extern "C" int gx_barrier_red_and(int b,int n,int p){return barrier_red(b,n,p,false);}
extern "C" int gx_barrier_red_popc(int name,int count,int pred){
    if(!count)count=current_block().lanes.size();
    if(count<1||count>int(current_block().lanes.size()))throw std::runtime_error("invalid reduction barrier participant count");
    auto* b=current_named_barrier_manager->get_barrier(61000000+(name&15)*1024+(count&1023),count);
    if(!b->vote_count)b->barrier_result_uint=0;
    b->barrier_result_uint+=pred!=0;
    if(++b->vote_count==count)b->vote_count=0;
    record_barrier_call();BARRIER_WAIT(b->barrier);auto result=b->barrier_result_uint;BARRIER_WAIT(b->barrier);return result;
}
extern "C" void* gx_shared_dynamic_ptr(){gx_ptx_shared_window(current_block().static_bytes);return current_block().shared+current_block().static_bytes;}
extern "C" void* gx_shared_static_alloc(unsigned key,size_t size,size_t align){
    if(!align||(align&(align-1)))throw std::runtime_error("invalid shared alignment");
    auto& b=current_block();auto it=b.static_offsets.find(key);
    if(it!=b.static_offsets.end())return b.shared+it->second;
    size_t offset=(b.legacy_static_end+align-1)&~(align-1);
    gx_ptx_shared_window(offset+size);b.legacy_static_end=offset+size;b.static_offsets[key]=offset;return b.shared+offset;
}
extern "C" void gx_clc_try_cancel(uint32_t response,uint32_t mbar,uint32_t multicast){
    uint32_t words[4]={0,0,0,0};unsigned n=multicast?gx_dev_cluster_nctarank():1;
    for(size_t k=0;k<active->cluster_launch.size();++k){
        if(active->cluster_launch[k])continue;
        active->cluster_launch[k]=2;
        auto b=coordinates(cluster_major_block(k*volume(active->cluster),active->grid,active->cluster),active->grid);
        words[0]=1;words[1]=b.x;words[2]=b.y;words[3]=b.z;
        drperf_gpu_resource("cancelled_clusters",1);drperf_gpu_resource("cancelled_blocks",volume(active->cluster));
        break;
    }
    for(unsigned i=0;i<n;++i){unsigned r=multicast?i:gx_dev_cluster_ctarank();
        std::memcpy(reinterpret_cast<void*>(gx_cluster_shared_ptr((r<<24)|(response&0xffffff))),words,16);
        gx_mbarrier_complete_tx(gx_cluster_shared_ptr((r<<24)|(mbar&0xffffff)),16);
    }
}
extern "C" uint32_t gx_clc_query(uint64_t lo,uint64_t hi,uint32_t word){
    uint32_t words[4]={uint32_t(lo),uint32_t(lo>>32),uint32_t(hi),uint32_t(hi>>32)};
    if(word>4)throw std::runtime_error("invalid clusterlaunchcontrol query");return word==4?0:word==0?words[0]&1:words[word];
}
