// gcbench: build DRC workers from a deterministic scenario and run the worker
// setup (FlexGCWorker::init) on each.
//
//   gcbench scenario=pgvias|pins|mix|drnets size=tiny|small|large
//           [repeat=N] [seed=S] [dump=FILE] [time=1] [alloc=default]
//
// stdout: totals of what setup built and a digest of all of it (pins with
// their polygons, edges, corners and maximal rectangles with fixed flags, and
// the per-layer shapes the nets hold), identical for any build that computes
// the same result. dump=FILE writes the digested text.
//
// Allocator: so that the cost of an allocation depends on the worker being set
// up rather than on what earlier workers left in glibc's free lists, gcbench
// re-executes itself with glibc's per-thread cache enlarged
// (GLIBC_TUNABLES=glibc.malloc.tcache_count=65535) and fills that cache before
// the first worker. alloc=default runs with the allocator as configured.
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "FlexGC.h"
#include "workload.h"

namespace {

const char* argValue(int argc, char** argv, const char* key, const char* def)
{
  const size_t kl = std::strlen(key);
  for (int i = 1; i < argc; i++) {
    if (std::strncmp(argv[i], key, kl) == 0 && argv[i][kl] == '=') {
      return argv[i] + kl + 1;
    }
  }
  return def;
}

// FNV-1a over everything written; optionally also to a file.
class Digest
{
 public:
  explicit Digest(FILE* f) : f_(f) {}
  void put(const char* fmt, long long a = 0, long long b = 0, long long c = 0,
           long long d = 0, long long e = 0, long long g = 0)
  {
    char buf[256];
    int n = std::snprintf(buf, sizeof buf, fmt, a, b, c, d, e, g);
    for (int i = 0; i < n; i++) {
      h_ = (h_ ^ static_cast<unsigned char>(buf[i])) * 0x100000001b3ULL;
    }
    if (f_) {
      std::fwrite(buf, 1, n, f_);
    }
  }
  uint64_t value() const { return h_; }

 private:
  FILE* f_;
  uint64_t h_ = 0xcbf29ce484222325ULL;
};

struct Totals
{
  long long nets = 0, pins = 0, edges = 0, fixedEdges = 0, corners = 0,
            fixedCorners = 0, maxRects = 0, fixedMaxRects = 0;
};

void dumpPolygonSet(Digest& d, const char* tag, int layer,
                    const gtl::polygon_90_set_data<drt::frCoord>& set)
{
  gtl::polygon_90_set_data<drt::frCoord> copy(set);
  copy.clean();
  if (copy.value().empty()) {
    return;
  }
  d.put(tag);
  d.put(" %lld:", layer);
  for (const auto& v : copy.value()) {
    d.put(" %lld,%lld,%lld", v.first, v.second.first, v.second.second);
  }
  d.put("\n");
}

void dumpRects(Digest& d, const char* tag, int layer,
               const std::vector<gtl::rectangle_data<drt::frCoord>>& rects)
{
  if (rects.empty()) {
    return;
  }
  d.put(tag);
  d.put(" %lld:", layer);
  for (const auto& r : rects) {
    d.put(" %lld,%lld,%lld,%lld", gtl::xl(r), gtl::yl(r), gtl::xh(r), gtl::yh(r));
  }
  d.put("\n");
}

void dumpWorker(Digest& d, const drt::FlexGCWorker& gc, Totals& t)
{
  const int numLayers = gc.getTech()->numLayers();
  for (const auto& uNet : gc.getNets()) {
    const drt::gcNet* net = uNet.get();
    t.nets++;
    d.put("net %lld owner %lld\n", net->getId(),
          net->getOwner() ? static_cast<const drt::Owner*>(net->getOwner())->id
                          : -1);
    for (int i = 0; i < numLayers; i++) {
      dumpPolygonSet(d, " routepolys", i, net->getPolygons(i, false));
      dumpPolygonSet(d, " fixedpolys", i, net->getPolygons(i, true));
      dumpRects(d, " routerects", i, net->getRectangles(i, false));
      dumpRects(d, " fixedrects", i, net->getRectangles(i, true));
      if (i >= 2) {
        const int z = i / 2 - 1;
        if (!net->getTaperedRects(z).empty()
            || !net->getNonTaperedRects(z).empty()) {
          d.put(" tapered %lld: %lld %lld\n", z, net->getTaperedRects(z).size(),
                net->getNonTaperedRects(z).size());
        }
      }
      for (const auto& pin : net->getPins(i)) {
        t.pins++;
        const drt::gcPolygon* poly = pin->getPolygon();
        d.put(" pin %lld layer %lld poly", pin->getId(), poly->getLayerNum());
        for (auto it = poly->begin_compact(); it != poly->end_compact(); ++it) {
          d.put(" %lld", *it);
        }
        for (auto h = poly->begin_holes(); h != poly->end_holes(); ++h) {
          d.put(" hole");
          for (auto it = h->begin_compact(); it != h->end_compact(); ++it) {
            d.put(" %lld", *it);
          }
        }
        d.put("\n");
        const auto& rings = pin->getPolygonEdges();
        const auto& cornerRings = pin->getPolygonCorners();
        if (rings.size() != cornerRings.size()) {
          d.put("  RINGS %lld != %lld\n", rings.size(), cornerRings.size());
        }
        for (size_t r = 0; r < rings.size(); r++) {
          const auto& edges = rings[r];
          const size_t n = edges.size();
          d.put("  ring %lld edges", r);
          for (size_t j = 0; j < n; j++) {
            const drt::gcSegment* e = edges[j].get();
            t.edges++;
            t.fixedEdges += e->isFixed();
            d.put(" %lld,%lld>%lld,%lld:%lld", e->low().x(), e->low().y(),
                  e->high().x(), e->high().y(), e->isFixed());
            const bool linked
                = e->getNextEdge() == edges[(j + 1) % n].get()
                  && e->getPrevEdge() == edges[(j + n - 1) % n].get()
                  && e->getLayerNum() == i && e->getPin() == pin.get()
                  && e->getNet() == net;
            if (!linked) {
              d.put(" EDGELINK");
            }
          }
          d.put("\n");
          if (r >= cornerRings.size()) {
            continue;
          }
          const auto& corners = cornerRings[r];
          d.put("  ring %lld corners", r);
          if (corners.size() != n) {
            d.put(" COUNT %lld", corners.size());
          }
          for (size_t j = 0; j < corners.size() && j < n; j++) {
            const drt::gcCorner* c = corners[j].get();
            t.corners++;
            t.fixedCorners += c->isFixed();
            d.put(" %lld,%lld:%lld%lld%lld", c->x(), c->y(),
                  static_cast<int>(c->getType()), static_cast<int>(c->getDir()),
                  c->isFixed());
            const bool linked
                = c->getNextEdge() == edges[j].get()
                  && c->getPrevEdge() == edges[(j + n - 1) % n].get()
                  && edges[j]->getLowCorner() == c
                  && edges[(j + n - 1) % n]->getHighCorner() == c
                  && c->getNextCorner() == corners[(j + 1) % n].get()
                  && c->getPrevCorner() == corners[(j + n - 1) % n].get()
                  && c->getLayerNum() == i && c->getPin() == pin.get();
            if (!linked) {
              d.put(" CORNERLINK");
            }
          }
          d.put("\n");
        }
        d.put("  maxrects");
        for (const auto& mr : pin->getMaxRectangles()) {
          t.maxRects++;
          t.fixedMaxRects += mr->isFixed();
          d.put(" %lld,%lld,%lld,%lld:%lld%lld", gtl::xl(*mr), gtl::yl(*mr),
                gtl::xh(*mr), gtl::yh(*mr), mr->isFixed(), mr->isTapered());
          if (mr->getLayerNum() != i || mr->getPin() != pin.get()
              || mr->getNet() != net) {
            d.put(" RECTLINK");
          }
        }
        d.put("\n");
      }
    }
    for (const auto& sp : net->getSpecialSpcRects()) {
      d.put(" special %lld,%lld,%lld,%lld layer %lld\n", gtl::xl(*sp),
            gtl::yl(*sp), gtl::xh(*sp), gtl::yh(*sp), sp->getLayerNum());
    }
  }
}

void setUpAllocator(char** argv)
{
  const char* t = std::getenv("GLIBC_TUNABLES");
  if (t == nullptr || std::strstr(t, "glibc.malloc.tcache_count") == nullptr) {
    std::string v = t ? std::string(t) + ":" : std::string();
    v += "glibc.malloc.tcache_count=65535";
    setenv("GLIBC_TUNABLES", v.c_str(), 1);
    execv("/proc/self/exe", argv);
    std::perror("gcbench: execv");  // continue with the allocator as it is
  }
  // fill the cache of every size class it holds (up to 1032 bytes)
  std::vector<void*> blocks;
  for (size_t size = 8; size <= 1024; size += 16) {
    const size_t n = size <= 128 ? 32768 : 2048;
    blocks.clear();
    for (size_t i = 0; i < n; i++) {
      blocks.push_back(std::malloc(size));
    }
    for (void* b : blocks) {
      std::free(b);
    }
  }
}

}  // namespace

int main(int argc, char** argv)
{
  if (std::strcmp(argValue(argc, argv, "alloc", "tuned"), "default") != 0) {
    setUpAllocator(argv);
  }
  const std::string scenario = argValue(argc, argv, "scenario", "pgvias");
  const std::string size = argValue(argc, argv, "size", "small");
  const int repeat = std::atoi(argValue(argc, argv, "repeat", "4"));
  const uint64_t seed = std::strtoull(argValue(argc, argv, "seed", "1"), nullptr, 10);
  const char* dumpPath = argValue(argc, argv, "dump", nullptr);
  const bool timing = std::atoi(argValue(argc, argv, "time", "0")) != 0;

  gcbench::Scenario sc;
  if (repeat < 1 || !gcbench::makeScenario(scenario, size, repeat, sc)) {
    std::fprintf(stderr,
                 "usage: gcbench scenario=pgvias|pins|mix|drnets "
                 "size=tiny|small|large [repeat=N>=1] [seed=S] [dump=FILE] "
                 "[time=1] [alloc=default]\n");
    return 2;
  }
  FILE* dump = dumpPath ? std::fopen(dumpPath, "w") : nullptr;
  if (dumpPath && !dump) {
    std::perror(dumpPath);
    return 1;
  }
  Digest digest(dump);
  Totals t;
  double setupSeconds = 0;
  for (size_t w = 0; w < sc.workers.size(); w++) {
    gcbench::Workload wl;
    gcbench::buildWorkload(sc.workers[w], seed, wl);
    drt::frTechObject tech(sc.workers[w].layers);
    drt::FlexGCWorker gc(&tech, &wl.drWorker);
    const auto t0 = std::chrono::steady_clock::now();
    gc.init(&wl.design);
    setupSeconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    digest.put("worker %lld\n", w);
    dumpWorker(digest, gc, t);
  }
  if (dump) {
    std::fclose(dump);
  }
  std::printf("scenario %s size %s repeat %d seed %" PRIu64 ": %zu workers\n",
              scenario.c_str(), size.c_str(), repeat, seed, sc.workers.size());
  std::printf("nets %lld pins %lld edges %lld (fixed %lld) corners %lld (fixed "
              "%lld) maxrects %lld (fixed %lld)\n",
              t.nets, t.pins, t.edges, t.fixedEdges, t.corners, t.fixedCorners,
              t.maxRects, t.fixedMaxRects);
  std::printf("digest %016" PRIx64 "\n", digest.value());
  if (timing) {
    std::fprintf(stderr, "setup %.3f s\n", setupSeconds);
  }
  return 0;
}
