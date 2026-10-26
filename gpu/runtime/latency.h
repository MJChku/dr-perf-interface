// An uncalibrated max-plus issue model, not GPU time. Keep Ready's layout in
// sync with the generated-code prelude (latency instrumentation ABI 1).
#pragma once
#include <cstdint>
#include <cstddef>
#include <algorithm>

struct DrperfGpuReady {
    uint64_t time=0, loads=0, global=0;
};
namespace drperf_latency {
inline bool later(const DrperfGpuReady& a,const DrperfGpuReady& b) {
    return a.time>b.time || (a.time==b.time && a.loads>b.loads);
}
inline void join(DrperfGpuReady& dst,const DrperfGpuReady& src) {
    if(later(src,dst))dst=src;
    dst.global=0;
}
// Source arrays are copied before destination writes (including read/modify/
// write and vector instructions). Never drain unused results at lane exit.
inline DrperfGpuReady issue(DrperfGpuReady& clock,const DrperfGpuReady* src,size_t n,
                            uint64_t latency,bool global) {
    DrperfGpuReady next{clock.time+1,clock.loads,0};
    for(size_t i=0;i<n;++i) {
        auto dependency=src[i];
        if(dependency.time>clock.time+1)dependency.loads+=dependency.global;
        else continue; // Already hidden by issue distance / another wait.
        dependency.global=0;
        if(later(dependency,next))next=dependency;
    }
    clock=next;
    return {next.time+latency,next.loads,uint64_t(global)};
}
// Basic-block variant approximating ptxas list scheduling: within a block an
// instruction issues at max(floor + slots used + 1, source readiness), so an
// independent load is not held behind an earlier stalled consumer in PTX
// order. Control boundaries (branches, barriers, returns, calls) reset the
// floor to the latest issue; nothing is hoisted across them. Exposed loads
// are counted against the issue slot the instruction would otherwise get.
inline DrperfGpuReady issue_block(DrperfGpuReady& clock,uint64_t floor,uint64_t& issued,
                                  const DrperfGpuReady* src,size_t n,uint64_t latency,bool global) {
    const uint64_t slot=floor+(++issued);
    DrperfGpuReady next{slot,clock.loads,0};
    if(clock.time>=slot) next.loads=clock.loads; // path count travels with the latest issue
    for(size_t i=0;i<n;++i) {
        auto dependency=src[i];
        if(dependency.time>slot)dependency.loads+=dependency.global;
        else continue;
        dependency.global=0;
        if(later(dependency,next))next=dependency;
    }
    if(later(next,clock))clock=next;
    return {next.time+latency,next.loads,uint64_t(global)};
}
}
