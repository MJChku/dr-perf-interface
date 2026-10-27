// Collector only. GX owns CUDA dispatch, streams, fibers, memory and semantics.
#include "gx_resource_observer.h"
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
using Counts = std::map<std::string,uint64_t>;
std::string quote(const std::string& s) {
    std::ostringstream o; o << '"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\')o << '\\' << char(c);
        else if(c<32)o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else o << c;
    }
    return o.str()+'"';
}
template<class T> std::string object(const std::map<std::string,T>& m) {
    std::ostringstream o; o << '{'; bool comma=false;
    for(auto& [k,v]:m) {if(comma)o<<',';comma=true;o<<quote(k)<<':'<<v;}
    return o.str()+'}';
}
struct Region {
    uint64_t token,parent;
    std::string name;
    std::map<std::string,int64_t> pcvs;
    Counts metrics;
    std::map<std::string,Counts> sites;
    size_t pending=0, launches=0;
    bool closed=false;
};
struct State {
    std::mutex mutex;
    uint64_t next=1,completed=0,failures=0,pending=0;
    bool started=false, finalized=false;
    void line(const std::string& s) {
        const char* path=getenv("DRPERF_GPU_CAPTURE");
        if(!path||!*path)throw std::runtime_error("GX GPU collector requires DRPERF_GPU_CAPTURE");
        std::ofstream f(std::string(path)+"."+std::to_string(getpid())+".jsonl",std::ios::app);
        f<<s<<'\n';f.flush();if(!f)throw std::runtime_error("cannot write GX GPU capture");
    }
    void start() {
        if(started)return;
        line("{\"event\":\"start\",\"schema\":1,\"backend\":\"gpu\",\"pid\":"+std::to_string(getpid())+"}");
        started=true;
    }
    void emit(const std::shared_ptr<Region>& r) {
        if(!r->closed||r->pending)return;
        std::string sites="{";
        for(auto& [k,v]:r->sites){if(sites.size()>1)sites+=',';sites+=quote(k)+":"+object(v);}sites+='}';
        line("{\"schema\":1,\"backend\":\"gpu\",\"pid\":"+std::to_string(getpid())+
            ",\"token\":"+std::to_string(r->token)+",\"parent\":"+std::to_string(r->parent)+
            ",\"region\":"+quote(r->name)+",\"pcvs\":"+object(r->pcvs)+
            ",\"scope\":\"host\",\"footprintVersion\":0,\"metrics\":"+object(r->metrics)+
            ",\"sites\":"+sites+",\"launches\":[],\"gxLaunchCount\":"+std::to_string(r->launches)+"}");
        ++completed;
    }
    void finalize() {
        std::lock_guard<std::mutex> lock(mutex);
        finalized=true;
        if(started)line("{\"event\":\"finish\",\"pid\":"+std::to_string(getpid())+
            ",\"modules\":[],\"open_regions\":"+std::to_string(next-1-completed)+
            ",\"regions_started\":"+std::to_string(next-1)+",\"regions_completed\":"+std::to_string(completed)+
            ",\"launch_failures\":"+std::to_string(failures+pending)+"}");
    }
};
State& state();
State& state(){
    // GX may destroy queued stream tasks after other DSOs' exit handlers.
    // Keep storage alive; finalize once and reject unfinished captures then.
    static State* s=[] {auto* p=new State;std::atexit([]{state().finalize();});return p;}();
    return *s;
}
thread_local std::vector<std::shared_ptr<Region>> stack;
struct Launch {
    std::shared_ptr<Region> region;
    std::string kernel;
    bool translated;
    Counts metrics;
    std::map<std::string,Counts> sites;
};
void* submit(const char* kernel,int translated) {
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);s.start();
    auto r=stack.empty()?std::make_shared<Region>():stack.back();
    if(stack.empty()) {
        r->token=s.next++;r->parent=0;r->name=std::string("kernel:")+(kernel?kernel:"unknown");r->closed=true;
    }
    ++r->pending;++r->launches;++s.pending;
    return new Launch{r,kernel?kernel:"unknown",bool(translated),{}, {}};
}
void count(void* handle,const char* site,const char* metric,uint64_t amount) {
    if(!handle||!amount)return;
    auto& l=*static_cast<Launch*>(handle);
    // GX executes each launch on one stream worker; lanes are cooperative fibers.
    l.metrics[metric]+=amount;l.sites[site][metric]+=amount;
}
void finish(void* handle,int completed) {
    if(!handle)return;
    std::unique_ptr<Launch> l(static_cast<Launch*>(handle));
    if(!l->translated||!l->metrics.count("ptx_lane_instructions"))
        count(l.get(),(l->kernel+":unmeasured").c_str(),"unmodeled_work_events",1);
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    if(!completed)++s.failures;
    if(s.finalized)return;
    auto r=l->region;
    for(auto& [k,v]:l->metrics)r->metrics[k]+=v;
    for(auto& [site,metrics]:l->sites)for(auto& [k,v]:metrics)r->sites[site][k]+=v;
    --r->pending;--s.pending;s.emit(r);
}
}
extern "C" uint64_t drperf_gpu_region_begin(const char* name,size_t n,const char* const* names,const int64_t* values) {
    if(!name||!*name||n>256||(n&&(!names||!values)))throw std::invalid_argument("invalid GPU region/PCVs");
    auto r=std::make_shared<Region>();r->name=name;
    for(size_t i=0;i<n;++i)if(!names[i]||!*names[i]||!r->pcvs.emplace(names[i],values[i]).second)
        throw std::invalid_argument("invalid/duplicate GPU PCV");
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);s.start();
    r->token=s.next++;r->parent=stack.empty()?0:stack.back()->token;stack.push_back(r);return r->token;
}
extern "C" void drperf_gpu_region_end(uint64_t token) {
    if(stack.empty()||stack.back()->token!=token)throw std::invalid_argument("GPU regions must close in stack order");
    auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    auto r=stack.back();stack.pop_back();r->closed=true;s.emit(r);
}
extern "C" const gx_resource_observer_v1* drperf_gpu_gx_observer_v1() {
    static const gx_resource_observer_v1 api{GX_RESOURCE_OBSERVER_ABI,submit,count,finish};return &api;
}
// Existing CPU perfmark annotations can describe GPU launches too. These hooks
// are loaded explicitly by libperfmark only when GX GPU capture is enabled.
extern "C" void drperf_gpu_perfmark_begin(const char* name,int n,const char* const* names,const int64_t* values) {
    if(n<0)throw std::invalid_argument("negative GPU PCV count");
    drperf_gpu_region_begin(name,static_cast<size_t>(n),names,values);
}
extern "C" void drperf_gpu_perfmark_end(const char* name) {
    if(stack.empty()||!name||stack.back()->name!=name)
        throw std::invalid_argument("mismatched GPU perfmark region end");
    drperf_gpu_region_end(stack.back()->token);
}
