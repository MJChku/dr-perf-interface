// Deterministic DRC-worker inputs for the standalone extraction.
//
// A worker holds, inside its extended box, the fixed objects the design region
// query returns and the nets of the detailed-routing worker. The generator
// builds them from counts, on a Nangate45-like 21-layer stack (metal1 = 2,
// via1 = 3, metal2 = 4, via2 = 5, metal3 = 6), in dbu.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "FlexGC.h"

namespace gcbench {

struct WorkerSpec
{
  int layers = 21;

  // Power/ground nets along a rail: rails on metal1..metal3 from the design,
  // a row of fixed vias (pgFixedCuts cut rectangles on via1 and on via2) and
  // a row of routed vias of the same net (pgRouteCuts per cut layer).
  int pgNets = 0;
  int pgFixedCuts = 0;
  int pgRouteCuts = 0;

  // Nets with fixed L-shaped pins on metal1 (instance pins from the design),
  // each pin reached by a routed via1 and a routed metal2 wire, and a macro
  // pin of macroPinRects fixed rectangles on each of macroPinLayers layers
  // from metal3 up, which this worker does not route.
  int designNets = 0;
  int pinsPerNet = 0;
  int macroPinRects = 0;
  int macroPinLayers = 1;

  // Instances whose obstructions (fixed rectangles on metal1 and metal2) fall
  // in the box; each instance owns a gc net of its own.
  int obsOwners = 0;
  int obsRects = 0;

  // Routed signal nets of the worker: metal2 wires, each with a via2 and a
  // metal3 wire, and pinRectsPerNet fixed pin rectangles on metal1 that the
  // wires in this box do not reach.
  int signalNets = 0;
  int wiresPerNet = 0;
  int pinRectsPerNet = 0;
  bool metal3Wires = true;  // false: the via2 lands on metal3 without a wire

  // Nets of the worker with nothing in the box.
  int emptyNets = 0;

  // When nonzero, the counts above are ignored and the worker is drawn at
  // random from this seed instead: nets with overlapping, touching, ring-shaped
  // and duplicated shapes, fixed and routed, on random layers (for checks).
  uint64_t random = 0;
};

// Everything a worker reads. Owners and via definitions are held by address.
struct Workload
{
  drt::frDesign design;
  drt::FlexDRWorker drWorker;
  std::deque<drt::Owner> owners;
  drt::frViaDef via1, via2;
};

void buildWorkload(const WorkerSpec& spec, uint64_t seed, Workload& out);

// The sequence of workers one run of a scenario processes.
struct Scenario
{
  std::string name;
  std::vector<WorkerSpec> workers;
};

// name: pgvias | pins | mix | drnets | random; size: tiny | small | large.
bool makeScenario(const std::string& name,
                  const std::string& size,
                  int repeat,
                  Scenario& out);

}  // namespace gcbench
