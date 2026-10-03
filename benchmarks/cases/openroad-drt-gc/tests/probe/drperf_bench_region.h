// Recording stand-in for benchmarks/regions/support/drperf_bench_region.h used
// by the correctness tests: counts entries per region name and prints them to
// stderr at exit ("marker_hits NAME COUNT"), then forwards to the real empty
// marker so the same binary also runs under drperf. Declares no states.
#ifndef DRPERF_BENCH_REGION_H
#define DRPERF_BENCH_REGION_H

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "perfmark.h"

namespace drperf_bench {
struct ProbeCounts
{
  const char* names[16] = {};
  long counts[16] = {};
  ProbeCounts() { std::atexit(&ProbeCounts::report); }
  static ProbeCounts& get()
  {
    static ProbeCounts c;
    return c;
  }
  static void report()
  {
    ProbeCounts& c = get();
    for (int i = 0; i < 16 && c.names[i]; i++) {
      std::fprintf(stderr, "marker_hits %s %ld\n", c.names[i], c.counts[i]);
    }
  }
  void hit(const char* name)
  {
    for (int i = 0; i < 16; i++) {
      if (names[i] == nullptr) {
        names[i] = name;
      }
      if (std::strcmp(names[i], name) == 0) {
        counts[i]++;
        return;
      }
    }
  }
};

class Region
{
 public:
  explicit Region(const char* name) : name_(name)
  {
    ProbeCounts::get().hit(name_);
    perfmark_begin_v(name_, 0, nullptr, nullptr);
  }
  ~Region() { perfmark_end(name_); }
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

 private:
  const char* name_;
};
}  // namespace drperf_bench

#define DRPERF_BENCH_JOIN_IMPL(a, b) a##b
#define DRPERF_BENCH_JOIN(a, b) DRPERF_BENCH_JOIN_IMPL(a, b)
#define DRPERF_BENCH_REGION(name) \
  ::drperf_bench::Region DRPERF_BENCH_JOIN(drperf_bench_region_, __LINE__)(name)

#endif
