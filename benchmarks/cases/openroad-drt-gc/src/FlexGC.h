// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
// After OpenROAD src/drt/src/gc/FlexGC_impl.h (FlexGCWorker::Impl) at commit
// ec10d069775173bc856608663c11d33c09f3c9b6: only the members that
// FlexGCWorker::Impl::init() uses to build per-net pin structures are kept.
// See ../LICENSE.OpenROAD.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

#include "frBase.h"
#include "gcNet.h"

namespace drt {

// Per-layer set of the fixed polygon vertices of one net (corner pass).
struct PtHash
{
  std::size_t operator()(const std::pair<frCoord, frCoord>& p) const
  {
    return std::hash<uint64_t>()(
        (static_cast<uint64_t>(static_cast<uint32_t>(p.first)) << 32)
        ^ static_cast<uint64_t>(static_cast<uint32_t>(p.second)));
  }
};
using PtSet = std::unordered_set<std::pair<frCoord, frCoord>, PtHash>;
using VertexCache = std::vector<std::unique_ptr<PtSet>>;

// ---------------------------------------------------------------- inputs
// What the worker reads from the design and from the detailed-routing worker.
// Upstream these are frDesign's region query (instance pins, obstructions,
// special nets) and FlexDRWorker's drNets with their connected figures.

// An owner of shapes: a net, an instance (for its obstructions) or a pin.
struct Owner
{
  int id = 0;
};

// One fixed object returned by the design region query on a layer.
struct DesignObj
{
  odb::Rect box;
  const Owner* owner = nullptr;
};

struct frViaDef
{
  frLayerNum layer1Num = 0;
  frLayerNum cutLayerNum = 0;
  frLayerNum layer2Num = 0;
  std::vector<odb::Rect> layer1Figs, cutFigs, layer2Figs;
};

enum class drConnFigType
{
  PathSeg,
  Via,
  PatchWire
};

struct drConnFig
{
  drConnFigType type = drConnFigType::PathSeg;
  odb::Rect box;               // PathSeg, PatchWire
  frLayerNum layerNum = 0;     // PathSeg, PatchWire
  const frViaDef* viaDef = nullptr;  // Via
  odb::Point origin;           // Via: translation of the via definition
};

struct drNet
{
  const Owner* frNet = nullptr;
  bool fixed = false;
  std::vector<drConnFig> extFigs;
  std::vector<drConnFig> routeFigs;
};

struct frDesign
{
  // regionQuery->query(extBox, layer) for every layer, already resolved.
  std::vector<std::vector<DesignObj>> objsByLayer;
  Owner fakeVSS, fakeVDD;
};

struct FlexDRWorker
{
  std::vector<drNet> nets;
};

// ---------------------------------------------------------------- worker

class FlexGCWorker
{
 public:
  FlexGCWorker(const frTechObject* tech, const FlexDRWorker* drWorker)
      : tech_(tech), drWorker_(drWorker)
  {
  }

  // FlexGCWorker::Impl::init without the region-query packing.
  void init(const frDesign* design);

  const frTechObject* getTech() const { return tech_; }
  const FlexDRWorker* getDRWorker() const { return drWorker_; }
  std::vector<std::unique_ptr<gcNet>>& getNets() { return nets_; }
  const std::vector<std::unique_ptr<gcNet>>& getNets() const { return nets_; }

  gcNet* addNet(const Owner* owner = nullptr)
  {
    auto uNet = std::make_unique<gcNet>(getTech()->numLayers());
    auto net = uNet.get();
    net->setOwner(owner);
    net->setId(nets_.size());
    nets_.push_back(std::move(uNet));
    owner2nets_[owner] = net;
    return net;
  }

  void initDesign(const frDesign* design);
  void initDRWorker();
  void initNets();
  void initNet(gcNet* net);

 private:
  const frTechObject* tech_;
  const FlexDRWorker* drWorker_;
  std::map<const Owner*, gcNet*> owner2nets_;
  std::vector<std::unique_ptr<gcNet>> nets_;

  gcNet* getNet(const Owner* owner);
  void initObj(const odb::Rect& box,
               frLayerNum layerNum,
               const Owner* owner,
               bool isFixed);
  gcNet* initDRObj(const drConnFig& fig, const drNet& net);

  void initNet_pins_polygon(gcNet* net);
  void initNet_pins_polygonEdges(gcNet* net);
  void initNet_pins_polygonEdges_getFixedPolygonEdges(
      gcNet* net,
      std::vector<std::set<std::pair<odb::Point, odb::Point>>>&
          fixedPolygonEdges);
  template <typename PointIter>
  void initNet_pins_polygonEdges_helper(
      gcNet* net,
      gcPin* pin,
      PointIter begin,
      PointIter end,
      frLayerNum i,
      const std::vector<std::set<std::pair<odb::Point, odb::Point>>>&
          fixedPolygonEdges);
  void initNet_pins_polygonCorners(gcNet* net);
  void initNet_pins_polygonCorners_helper(gcNet* net,
                                          gcPin* pin,
                                          VertexCache* vcache);
  void initNet_pins_maxRectangles(gcNet* net);
  void initNet_pins_maxRectangles_getFixedMaxRectangles(
      gcNet* net,
      std::vector<std::set<std::pair<odb::Point, odb::Point>>>&
          fixedMaxRectangles);
  void initNet_pins_maxRectangles_helper(
      gcNet* net,
      gcPin* pin,
      const gtl::rectangle_data<frCoord>& rect,
      frLayerNum i,
      const std::vector<std::set<std::pair<odb::Point, odb::Point>>>&
          fixedMaxRectangles);

  // FlexGC_main.cpp
  bool isCornerOverlap(gcCorner* corner,
                       const gtl::rectangle_data<frCoord>& rect);
};

}  // namespace drt
