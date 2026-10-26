#include "../runtime/footprint.h"
#include <cassert>
#include <random>
#include <set>

using drperf_footprint::AddressUnion;
using drperf_footprint::Footprint;
int main() {
    // Random overlapping, disjoint, adjacent, and duplicate ranges compared
    // with a deliberately independent byte-set oracle after each insertion.
    AddressUnion a,b;std::set<uint64_t> reference;
    std::mt19937 random(137);
    for(unsigned i=0;i<1000;++i) {
        uint64_t first=random()%4096,last=first+random()%129;
        (i%2?a:b).insert(first,last);
        for(auto n=first;n<=last;++n)reference.insert(n);
        AddressUnion merged=a;merged.merge(b);
        assert(merged.size==reference.size());
    }
    AddressUnion edge;edge.insert(UINT64_MAX,UINT64_MAX);
    edge.insert(UINT64_MAX-2,UINT64_MAX-1);assert(edge.size==3);
    edge.merge(edge);assert(edge.size==3);
    Footprint fp;fp.touch(31,2,"global","read");fp.touch(127,2,"global","read");
    fp.merge(fp);
    auto counts=fp.counts({{"global_read_bytes",4}});
    assert(counts.at("global_unique_read_bytes")==4);
    assert(counts.at("global_unique_read_sectors")==4);
    assert(counts.at("global_unique_read_lines")==2);
    assert(counts.at("global_unique_read_sector_bytes")==128);
    assert(counts.at("global_unique_read_line_bytes")==256);
    fp.touch(0,0,"global","write");
    assert(!fp.counts({}).count("global_unique_write_bytes"));
    fp.touch(UINT64_MAX,1,"global","write");
    bool overflow=false;
    try {fp.touch(UINT64_MAX,2,"global","read");}catch(const std::overflow_error&) {overflow=true;}
    assert(overflow);
    overflow=false;
    try {AddressUnion full;full.insert(0,UINT64_MAX);}catch(const std::overflow_error&) {overflow=true;}
    assert(overflow);
}
