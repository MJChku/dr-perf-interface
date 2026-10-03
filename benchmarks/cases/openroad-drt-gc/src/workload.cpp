// Deterministic DRC-worker inputs for the standalone extraction.
#include "workload.h"

#include <algorithm>

namespace gcbench {

using drt::DesignObj;
using drt::drConnFig;
using drt::drConnFigType;
using drt::drNet;
using drt::Owner;
using odb::Point;
using odb::Rect;

namespace {

constexpr int kM1 = 2, kV1 = 3, kM2 = 4, kV2 = 5, kM3 = 6;

struct Builder
{
  Workload& w;
  int nextOwner = 0;

  const Owner* owner()
  {
    w.owners.push_back(Owner{nextOwner++});
    return &w.owners.back();
  }
  void fixed(int layer, const Rect& r, const Owner* o)
  {
    w.design.objsByLayer[layer].push_back(DesignObj{r, o});
  }
  static drConnFig seg(int layer, const Rect& r)
  {
    drConnFig f;
    f.type = drConnFigType::PathSeg;
    f.layerNum = layer;
    f.box = r;
    return f;
  }
  static drConnFig via(const drt::frViaDef* def, int x, int y)
  {
    drConnFig f;
    f.type = drConnFigType::Via;
    f.viaDef = def;
    f.origin = Point(x, y);
    return f;
  }
};

// Small deterministic generator for the worker's placement in the design.
uint64_t splitmix64(uint64_t& s)
{
  uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

}  // namespace

namespace {

// A random worker for correctness checks: shapes snap to a 70 dbu grid in a
// small area, so they overlap, touch, share corners, repeat and enclose holes.
void buildRandomWorkload(uint64_t seed, int layers, Workload& w, Builder& b)
{
  uint64_t st = seed * 0x2545f4914f6cdd1dULL + 1;
  auto pick = [&](int n) { return static_cast<int>(splitmix64(st) % n); };
  const int topMetal = std::min(layers - 1, 10);  // metal1 (2) .. metal5 (10)
  auto rect = [&]() {
    const int x = 70 * pick(24), y = 70 * pick(24);
    return Rect(x, y, x + 70 * (1 + pick(8)), y + 70 * (1 + pick(3)));
  };
  auto cut = [&]() {
    const int x = 140 * pick(10), y = 140 * pick(10);
    return Rect(x, y, x + 140, y + 140);
  };
  const int nets = pick(14);
  for (int n = 0; n < nets; n++) {
    const Owner* o = b.owner();
    drNet dn;
    dn.frNet = o;
    dn.fixed = pick(8) == 0;
    const bool inDR = pick(3) != 0;
    const int shapes = pick(4) == 0 ? 0 : 1 + pick(20);
    for (int k = 0; k < shapes; k++) {
      const bool fixed = !inDR || pick(2) == 0;
      int metal = 2 * (1 + pick(std::max(1, topMetal / 2)));
      if (metal >= layers) {
        metal = 2;
      }
      switch (pick(6)) {
        case 0: {  // a ring of four rectangles around a hole
          const int x = 70 * pick(16), y = 70 * pick(16);
          const Rect ring[4] = {Rect(x, y, x + 420, y + 70),
                                Rect(x, y + 350, x + 420, y + 420),
                                Rect(x, y, x + 70, y + 420),
                                Rect(x + 350, y, x + 420, y + 420)};
          for (const Rect& r : ring) {
            if (fixed) {
              b.fixed(metal, r, o);
            } else {
              dn.routeFigs.push_back(Builder::seg(metal, r));
            }
          }
          break;
        }
        case 1: {  // a cut rectangle, possibly repeated or sharing corners
          const int cutLayer = metal + 1 < layers ? metal + 1 : 3;
          const Rect c = cut();
          if (fixed) {
            b.fixed(cutLayer, c, o);
          } else if (cutLayer == 3 || cutLayer == 5) {
            const int cx = (c.xMin() + c.xMax()) / 2, cy = (c.yMin() + c.yMax()) / 2;
            dn.routeFigs.push_back(
                Builder::via(cutLayer == 3 ? &w.via1 : &w.via2, cx, cy));
          } else {
            b.fixed(cutLayer, c, o);
          }
          break;
        }
        case 2: {  // a patch wire
          drConnFig f = Builder::seg(metal, rect());
          f.type = drConnFigType::PatchWire;
          if (fixed) {
            b.fixed(metal, f.box, o);
          } else {
            dn.routeFigs.push_back(f);
          }
          break;
        }
        default: {
          const Rect r = rect();
          if (fixed) {
            b.fixed(metal, r, o);
          } else if (pick(2) == 0) {
            dn.extFigs.push_back(Builder::seg(metal, r));
          } else {
            dn.routeFigs.push_back(Builder::seg(metal, r));
          }
        }
      }
    }
    if (inDR) {
      w.drWorker.nets.push_back(std::move(dn));
    }
  }
}

}  // namespace

void buildWorkload(const WorkerSpec& s, uint64_t seed, Workload& w)
{
  w.design.objsByLayer.assign(s.layers, {});
  w.drWorker.nets.clear();
  w.owners.clear();
  // via1 / via2: 140x140 cut, enclosure extended along x below and along y above
  w.via1 = drt::frViaDef{kM1, kV1, kM2, {Rect(-70, -70, 70, 70)},
                         {Rect(-70, -70, 70, 70)}, {Rect(-70, -100, 70, 100)}};
  w.via2 = drt::frViaDef{kM2, kV2, kM3, {Rect(-70, -70, 70, 70)},
                         {Rect(-70, -70, 70, 70)}, {Rect(-70, -100, 70, 100)}};
  Builder b{w};
  if (s.random != 0) {
    buildRandomWorkload(s.random ^ seed, s.layers, w, b);
    return;
  }

  // where this worker's extended box sits in the design (grid of 20 um)
  uint64_t st = seed;
  const int X0 = 40000 * static_cast<int>(splitmix64(st) % 64);
  const int Y0 = 40000 * static_cast<int>(splitmix64(st) % 64);
  int y = Y0;

  // power/ground rails with fixed and routed vias
  for (int n = 0; n < s.pgNets; n++, y += 1400) {
    const Owner* o = b.owner();
    const int slots = s.pgFixedCuts + s.pgRouteCuts;
    const Rect rail(X0 - 200, y - 100, X0 + 400 * slots + 200, y + 100);
    b.fixed(kM1, rail, o);
    b.fixed(kM2, rail, o);
    b.fixed(kM3, rail, o);
    for (int c = 0; c < s.pgFixedCuts; c++) {
      const int x = X0 + 400 * c;
      b.fixed(kV1, Rect(x - 70, y - 70, x + 70, y + 70), o);
      b.fixed(kV2, Rect(x - 70, y - 70, x + 70, y + 70), o);
    }
    drNet dn;
    dn.frNet = o;
    for (int c = s.pgFixedCuts; c < slots; c++) {
      const int x = X0 + 400 * c;
      dn.routeFigs.push_back(Builder::via(&w.via1, x, y));
      dn.routeFigs.push_back(Builder::via(&w.via2, x, y));
    }
    w.drWorker.nets.push_back(std::move(dn));
  }

  // design nets: L-shaped fixed pins on metal1, each reached by a via1 and a
  // metal2 wire; a macro pin of the same net on metal3 that is not routed here
  for (int n = 0; n < s.designNets; n++, y += 2000) {
    const Owner* o = b.owner();
    drNet dn;
    dn.frNet = o;
    for (int p = 0; p < s.pinsPerNet; p++) {
      const int px = X0 + 800 * p;
      b.fixed(kM1, Rect(px, y, px + 140, y + 600), o);
      b.fixed(kM1, Rect(px, y, px + 400, y + 140), o);
      dn.routeFigs.push_back(Builder::via(&w.via1, px + 70, y + 400));
      dn.routeFigs.push_back(
          Builder::seg(kM2, Rect(px, y + 300, px + 140, y + 1400)));
    }
    for (int l = 0; l < s.macroPinLayers; l++) {
      for (int r = 0; r < s.macroPinRects; r++) {
        const int x = X0 + 600 * r;
        b.fixed(kM3 + 2 * l, Rect(x, y + 200, x + 300, y + 500), o);
      }
    }
    w.drWorker.nets.push_back(std::move(dn));
  }

  // instance obstructions on metal1 and metal2
  for (int n = 0; n < s.obsOwners; n++, y += 1000) {
    const Owner* o = b.owner();
    for (int r = 0; r < s.obsRects; r++) {
      const int x = X0 + 600 * r;
      b.fixed(kM1, Rect(x, y, x + 300, y + 300), o);
      b.fixed(kM2, Rect(x, y, x + 300, y + 300), o);
    }
  }

  // routed signal nets: metal2 wire, via2 at its right end, metal3 wire up
  for (int n = 0; n < s.signalNets; n++, y += 1600) {
    const Owner* o = b.owner();
    drNet dn;
    dn.frNet = o;
    for (int k = 0; k < s.wiresPerNet; k++) {
      const int x = X0 + 1200 * k;
      dn.routeFigs.push_back(Builder::seg(kM2, Rect(x, y, x + 1000, y + 140)));
      dn.routeFigs.push_back(Builder::via(&w.via2, x + 930, y + 70));
      if (s.metal3Wires) {
        dn.routeFigs.push_back(
            Builder::seg(kM3, Rect(x + 860, y - 30, x + 1000, y + 800)));
      }
    }
    for (int r = 0; r < s.pinRectsPerNet; r++) {
      const int x = X0 + 600 * r;
      b.fixed(kM1, Rect(x, y + 400, x + 300, y + 540), o);
    }
    w.drWorker.nets.push_back(std::move(dn));
  }

  for (int n = 0; n < s.emptyNets; n++) {
    drNet dn;
    dn.frNet = b.owner();
    w.drWorker.nets.push_back(std::move(dn));
  }
}

namespace {

// The grid in a fixed shuffled order, each worker repeated back to back, so a
// worker's allocations mostly meet the heap an identical worker left behind.
std::vector<WorkerSpec> ordered(std::vector<WorkerSpec> v, int repeat)
{
  uint64_t st = 12345;
  for (size_t i = v.size(); i > 1; i--) {
    std::swap(v[i - 1], v[splitmix64(st) % i]);
  }
  std::vector<WorkerSpec> out;
  for (const WorkerSpec& w : v) {
    out.insert(out.end(), repeat, w);
  }
  return out;
}

}  // namespace

bool makeScenario(const std::string& name,
                  const std::string& size,
                  int repeat,
                  Scenario& out)
{
  const bool large = size == "large";
  const bool tiny = size == "tiny";
  if (!large && !tiny && size != "small") {
    return false;
  }
  out.name = name;
  std::vector<WorkerSpec> grid;
  if (name == "pgvias") {
    // power/ground via rows of growing length
    std::vector<int> ks = tiny    ? std::vector<int>{1, 3}
                          : large ? std::vector<int>{64, 128, 256, 384, 512, 1024}
                                  : std::vector<int>{2, 4, 8, 12, 16, 24};
    std::vector<int> rs = tiny    ? std::vector<int>{0, 2}
                          : large ? std::vector<int>{0, 16, 64}
                                  : std::vector<int>{0, 2, 6};
    for (int k : ks) {
      for (int r : rs) {
        WorkerSpec s;
        s.pgNets = 3;
        s.pgFixedCuts = k;
        s.pgRouteCuts = r;
        grid.push_back(s);
      }
    }
  } else if (name == "pins") {
    // nets with more cell pins and larger macro pins
    std::vector<int> ps = tiny    ? std::vector<int>{1, 2}
                          : large ? std::vector<int>{64, 128, 256}
                                  : std::vector<int>{8, 16, 32};
    std::vector<int> os = tiny    ? std::vector<int>{1, 2}
                          : large ? std::vector<int>{64, 128, 256}
                                  : std::vector<int>{8, 16, 32};
    for (int p : ps) {
      for (int o : os) {
        WorkerSpec s;
        s.designNets = 2;
        s.pinsPerNet = p;
        s.macroPinRects = o;
        grid.push_back(s);
      }
    }
  } else if (name == "mix") {
    // workers with more or fewer nets, some of them with nothing in the box
    const int m = large ? 8 : 1;
    const std::vector<int> es = tiny ? std::vector<int>{1} : std::vector<int>{0, 8, 24};
    const std::vector<int> ss = tiny ? std::vector<int>{1, 2} : std::vector<int>{2, 4, 8};
    const std::vector<int> ws = tiny ? std::vector<int>{1} : std::vector<int>{1, 2};
    const std::vector<int> rs = tiny ? std::vector<int>{1} : std::vector<int>{1, 2};
    for (int e : es) {
      for (int sn : ss) {
        for (int wn : ws) {
          for (int r : rs) {
            WorkerSpec s;
            s.emptyNets = e * m;
            s.signalNets = sn * m;
            s.wiresPerNet = wn;
            s.pinRectsPerNet = r;
            grid.push_back(s);
          }
        }
      }
    }
  } else if (name == "drnets") {
    // detailed-routing workers with more or fewer nets of their own
    const int m = large ? 8 : 1;
    const std::vector<int> es = tiny ? std::vector<int>{1} : std::vector<int>{0, 4, 12};
    const std::vector<int> ss = tiny ? std::vector<int>{1, 2} : std::vector<int>{2, 4, 8};
    const std::vector<int> ds = tiny ? std::vector<int>{1} : std::vector<int>{2, 4, 8};
    const std::vector<int> ws = tiny ? std::vector<int>{1} : std::vector<int>{1, 2};
    for (int e : es) {
      for (int sn : ss) {
        for (int d : ds) {
          for (int wn : ws) {
            WorkerSpec s;
            s.emptyNets = e * m;
            s.signalNets = sn * m;
            s.wiresPerNet = wn;
            s.metal3Wires = false;
            s.designNets = d * m;
            s.pinsPerNet = 2;
            grid.push_back(s);
          }
        }
      }
    }
  } else if (name == "random") {
    // random workers for correctness checks, a few with other layer counts
    const int n = tiny ? 20 : large ? 2000 : 200;
    for (int i = 0; i < n; i++) {
      WorkerSpec s;
      s.random = 1 + i;
      s.layers = (i % 7 == 3) ? 11 : (i % 11 == 5) ? 70 : 21;
      grid.push_back(s);
    }
    out.workers = grid;
    return true;
  } else {
    return false;
  }
  out.workers = ordered(grid, repeat);
  return true;
}

}  // namespace gcbench
