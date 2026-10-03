// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
//
// Re-implements the parts of FlexGCWorker::Impl::init() that build per-net pin
// structures, after OpenROAD src/drt/src/gc/FlexGC_init.cpp at commit
// ec10d069775173bc856608663c11d33c09f3c9b6 (The OpenROAD Authors, BSD-3-Clause,
// see ../LICENSE.OpenROAD). The containers, the Boost.Polygon calls and the
// order of work follow the upstream functions of the same names; logging,
// region-query packing, pin access, tapering rules and the design database
// are left out.
//
// One deliberate difference from that commit: on routing layers the corner
// pass looks a corner up in a per-layer set of the net's fixed polygon
// vertices, built once per layer, instead of calling isPolygonCorner() (which
// re-extracts the whole fixed polygon set) once per corner. That cache was
// part of the router build the drperf study measured. Define
// GC_UPSTREAM_CORNER_LOOKUP to compile the upstream per-corner lookup instead;
// both give the same fixed flags.

#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "FlexGC.h"

namespace drt {

// ------------------------------------------------------------ gc nets

gcNet* FlexGCWorker::getNet(const Owner* owner)
{
  auto it = owner2nets_.find(owner);
  if (it == owner2nets_.end()) {
    return addNet(owner);
  }
  return it->second;
}

void FlexGCWorker::initObj(const odb::Rect& box,
                           frLayerNum layerNum,
                           const Owner* owner,
                           bool isFixed)
{
  auto currNet = getNet(owner);
  if (getTech()->getLayerType(layerNum) == LayerType::CUT) {
    currNet->addRectangle(box, layerNum, isFixed);
  } else {
    currNet->addPolygon(box, layerNum, isFixed);
  }
}

// Fixed objects of the design inside the extended box, layer by layer.
void FlexGCWorker::initDesign(const frDesign* design)
{
  const int numLayers = getTech()->numLayers();
  for (int i = 0; i < numLayers; i++) {
    for (const DesignObj& obj : design->objsByLayer[i]) {
      initObj(obj.box, i, obj.owner, true);
    }
  }
}

namespace {
odb::Rect translate(const odb::Rect& r, const odb::Point& o)
{
  return odb::Rect(
      r.xMin() + o.x(), r.yMin() + o.y(), r.xMax() + o.x(), r.yMax() + o.y());
}

void addNonTaperedPatches(gcNet* gNet, const std::vector<drConnFig>& figs)
{
  for (const drConnFig& obj : figs) {
    if (obj.type == drConnFigType::PatchWire) {
      const odb::Rect& box = obj.box;
      int z = obj.layerNum / 2 - 1;
      for (auto& nt : gNet->getNonTaperedRects(z)) {
        if (nt.intersects(box)) {
          gNet->addNonTaperedRect(box, z);
          break;
        }
      }
    }
  }
}
}  // namespace

gcNet* FlexGCWorker::initDRObj(const drConnFig& obj, const drNet& dNet)
{
  gcNet* currNet = getNet(dNet.frNet);
  switch (obj.type) {
    case drConnFigType::PathSeg:
    case drConnFigType::PatchWire:
      currNet->addPolygon(obj.box, obj.layerNum, dNet.fixed);
      break;
    case drConnFigType::Via: {
      const frViaDef* viaDef = obj.viaDef;
      frLayerNum layerNum = viaDef->layer1Num;
      for (const odb::Rect& fig : viaDef->layer1Figs) {
        currNet->addPolygon(translate(fig, obj.origin), layerNum, dNet.fixed);
      }
      // push cut layer rect
      layerNum = viaDef->cutLayerNum;
      for (const odb::Rect& fig : viaDef->cutFigs) {
        currNet->addRectangle(translate(fig, obj.origin), layerNum, dNet.fixed);
      }
      // push layer2 rect
      layerNum = viaDef->layer2Num;
      for (const odb::Rect& fig : viaDef->layer2Figs) {
        currNet->addPolygon(translate(fig, obj.origin), layerNum, dNet.fixed);
      }
      break;
    }
  }
  return currNet;
}

// The detailed-routing worker's nets and their connected figures.
void FlexGCWorker::initDRWorker()
{
  if (!getDRWorker()) {
    return;
  }
  for (const drNet& dNet : getDRWorker()->nets) {
    // always first generate gcnet in case owner does not have any object
    auto it = owner2nets_.find(dNet.frNet);
    if (it == owner2nets_.end()) {
      addNet(dNet.frNet);
    }
    gcNet* gNet = nullptr;
    for (const drConnFig& fig : dNet.extFigs) {
      gNet = initDRObj(fig, dNet);
    }
    for (const drConnFig& fig : dNet.routeFigs) {
      gNet = initDRObj(fig, dNet);
    }
    addNonTaperedPatches(gNet, dNet.extFigs);
    addNonTaperedPatches(gNet, dNet.routeFigs);
  }
}

// ------------------------------------------------------------ pins

// Pins are the merged polygons of a routing layer (route and fixed shapes
// together) and the individual rectangles of a cut layer.
void FlexGCWorker::initNet_pins_polygon(gcNet* net)
{
  int numLayers = getTech()->numLayers();
  std::vector<gtl::polygon_90_set_data<frCoord>> layerPolys(numLayers);
  std::vector<gtl::polygon_90_with_holes_data<frCoord>> polys;
  for (int i = 0; i < numLayers; i++) {
    polys.clear();
    using gtl::operators::operator+=;
    layerPolys[i] += net->getPolygons(i, false);
    layerPolys[i] += net->getPolygons(i, true);
    layerPolys[i].get(polys);
    for (auto& poly : polys) {
      net->addPin(poly, i);
    }
  }
  for (int i = 0; i < numLayers; i++) {
    for (auto& rect : net->getRectangles(i, false)) {
      net->addPin(rect, i);
    }
    for (auto& rect : net->getRectangles(i, true)) {
      net->addPin(rect, i);
    }
  }
}

// ------------------------------------------------------------ edges

namespace {
using PointPairSets = std::vector<std::set<std::pair<odb::Point, odb::Point>>>;

template <typename PointIter>
void insertRingEdges(std::set<std::pair<odb::Point, odb::Point>>& edges,
                     PointIter it,
                     PointIter end)
{
  const odb::Point first((*it).x(), (*it).y());
  odb::Point bp = first;
  for (++it; it != end; ++it) {
    const odb::Point ep((*it).x(), (*it).y());
    edges.insert(std::make_pair(bp, ep));
    bp = ep;
  }
  edges.insert(std::make_pair(bp, first));
}
}  // namespace

// Every edge of the net's fixed shapes, per layer, so a pin edge can be
// classified as fixed (it coincides with one) or route.
void FlexGCWorker::initNet_pins_polygonEdges_getFixedPolygonEdges(
    gcNet* net,
    PointPairSets& fixedPolygonEdges)
{
  int numLayers = getTech()->numLayers();
  std::vector<gtl::polygon_90_with_holes_data<frCoord>> polys;
  for (int i = 0; i < numLayers; i++) {
    polys.clear();
    net->getPolygons(i, true).get(polys);
    for (auto& poly : polys) {
      insertRingEdges(fixedPolygonEdges[i], poly.begin(), poly.end());
      for (auto holeIt = poly.begin_holes(); holeIt != poly.end_holes();
           holeIt++) {
        insertRingEdges(fixedPolygonEdges[i], holeIt->begin(), holeIt->end());
      }
    }
  }
  // cut rectangles are not merged: their four edges as they are
  for (int i = 0; i < numLayers; i++) {
    for (auto& rect : net->getRectangles(i, true)) {
      const odb::Point ll(gtl::xl(rect), gtl::yl(rect));
      const odb::Point lr(gtl::xh(rect), gtl::yl(rect));
      const odb::Point ur(gtl::xh(rect), gtl::yh(rect));
      const odb::Point ul(gtl::xl(rect), gtl::yh(rect));
      fixedPolygonEdges[i].insert(std::make_pair(ll, lr));
      fixedPolygonEdges[i].insert(std::make_pair(lr, ur));
      fixedPolygonEdges[i].insert(std::make_pair(ur, ul));
      fixedPolygonEdges[i].insert(std::make_pair(ul, ll));
    }
  }
}

// One closed ring of gcSegments for a polygon outline or hole.
template <typename PointIter>
void FlexGCWorker::initNet_pins_polygonEdges_helper(
    gcNet* net,
    gcPin* pin,
    PointIter it,
    PointIter end,
    frLayerNum i,
    const PointPairSets& fixedPolygonEdges)
{
  std::vector<std::unique_ptr<gcSegment>> tmpEdges;
  const gtl::point_data<frCoord> first = *it;
  gtl::point_data<frCoord> bp = first;
  auto addEdge = [&](const gtl::point_data<frCoord>& ep) {
    auto edge = std::make_unique<gcSegment>();
    edge->setLayerNum(i);
    edge->addToPin(pin);
    edge->addToNet(net);
    edge->setSegment(bp, ep);
    const bool fixed = fixedPolygonEdges[i].find(std::make_pair(
                           odb::Point(bp.x(), bp.y()), odb::Point(ep.x(), ep.y())))
                       != fixedPolygonEdges[i].end();
    edge->setFixed(fixed);
    if (!tmpEdges.empty()) {
      edge->setPrevEdge(tmpEdges.back().get());
      tmpEdges.back()->setNextEdge(edge.get());
    }
    tmpEdges.push_back(std::move(edge));
    bp = ep;
  };
  for (++it; it != end; ++it) {
    addEdge(*it);
  }
  addEdge(first);  // closing edge
  tmpEdges.front()->setPrevEdge(tmpEdges.back().get());
  tmpEdges.back()->setNextEdge(tmpEdges.front().get());
  pin->addPolygonEdges(tmpEdges);
}

void FlexGCWorker::initNet_pins_polygonEdges(gcNet* net)
{
  int numLayers = getTech()->numLayers();
  PointPairSets fixedPolygonEdges(numLayers);
  initNet_pins_polygonEdges_getFixedPolygonEdges(net, fixedPolygonEdges);
  for (int i = 0; i < numLayers; i++) {
    for (auto& pin : net->getPins(i)) {
      auto poly = pin->getPolygon();
      initNet_pins_polygonEdges_helper(
          net, pin.get(), poly->begin(), poly->end(), i, fixedPolygonEdges);
      for (auto holeIt = poly->begin_holes(); holeIt != poly->end_holes();
           holeIt++) {
        initNet_pins_polygonEdges_helper(net,
                                         pin.get(),
                                         holeIt->begin(),
                                         holeIt->end(),
                                         i,
                                         fixedPolygonEdges);
      }
    }
  }
}

// ------------------------------------------------------------ corners

namespace {
void collectPolygonVertices(const gtl::polygon_90_set_data<frCoord>& polySet,
                            PtSet& out)
{
  std::vector<gtl::polygon_90_with_holes_data<frCoord>> polygons;
  polySet.get(polygons);
  for (const auto& polygon : polygons) {
    for (const auto& pt : polygon) {
      out.emplace(pt.x(), pt.y());
    }
    for (auto h = polygon.begin_holes(); h != polygon.end_holes(); ++h) {
      for (const auto& pt : *h) {
        out.emplace(pt.x(), pt.y());
      }
    }
  }
}

#ifdef GC_UPSTREAM_CORNER_LOOKUP
bool isPolygonCorner(frCoord x,
                     frCoord y,
                     const gtl::polygon_90_set_data<frCoord>& polySet)
{
  std::vector<gtl::polygon_90_with_holes_data<frCoord>> polygons;
  polySet.get(polygons);
  for (const auto& polygon : polygons) {
    for (const auto& pt : polygon) {
      if (pt.x() == x && pt.y() == y) {
        return true;
      }
    }
    for (auto h = polygon.begin_holes(); h != polygon.end_holes(); ++h) {
      for (const auto& pt : *h) {
        if (pt.x() == x && pt.y() == y) {
          return true;
        }
      }
    }
  }
  return false;
}
#endif

frCornerDirEnum cornerDir(frDirEnum prev, frDirEnum next)
{
  using D = frDirEnum;
  if ((prev == D::N && next == D::W) || (prev == D::W && next == D::N)) {
    return frCornerDirEnum::NE;
  }
  if ((prev == D::W && next == D::S) || (prev == D::S && next == D::W)) {
    return frCornerDirEnum::NW;
  }
  if ((prev == D::S && next == D::E) || (prev == D::E && next == D::S)) {
    return frCornerDirEnum::SW;
  }
  if ((prev == D::E && next == D::N) || (prev == D::N && next == D::E)) {
    return frCornerDirEnum::SE;
  }
  return frCornerDirEnum::UNKNOWN;
}
}  // namespace

// One gcCorner between every pair of consecutive edges. A corner is fixed when
// it belongs to the net's fixed shapes.
void FlexGCWorker::initNet_pins_polygonCorners_helper(gcNet* net,
                                                      gcPin* pin,
                                                      VertexCache* vcache)
{
  for (auto& edges : pin->getPolygonEdges()) {
    std::vector<std::unique_ptr<gcCorner>> tmpCorners;
    auto prevEdge = edges.back().get();
    auto layerNum = prevEdge->getLayerNum();
    gcCorner* prevCorner = nullptr;
    for (auto& nextEdge : edges) {
      auto uCurrCorner = std::make_unique<gcCorner>();
      auto currCorner = uCurrCorner.get();
      tmpCorners.push_back(std::move(uCurrCorner));
      prevEdge->setHighCorner(currCorner);
      nextEdge->setLowCorner(currCorner);
      currCorner->addToPin(pin);
      currCorner->setPrevEdge(prevEdge);
      currCorner->setNextEdge(nextEdge.get());
      currCorner->setLayerNum(layerNum);
      currCorner->x(prevEdge->high().x());
      currCorner->y(prevEdge->high().y());
      int orient = gtl::orientation(*prevEdge, *nextEdge);
      if (orient == 1) {
        currCorner->setType(frCornerTypeEnum::CONVEX);
      } else if (orient == -1) {
        currCorner->setType(frCornerTypeEnum::CONCAVE);
      } else {
        currCorner->setType(frCornerTypeEnum::UNKNOWN);
      }
      currCorner->setDir(cornerDir(prevEdge->getDir(), nextEdge->getDir()));

      if (getTech()->getLayerType(layerNum) == LayerType::CUT) {
        if (currCorner->getType() == frCornerTypeEnum::CONVEX) {
          currCorner->setFixed(false);
          for (auto& rect : net->getRectangles(true)[layerNum]) {
            if (isCornerOverlap(currCorner, rect)) {
              currCorner->setFixed(true);
              break;
            }
          }
        } else if (currCorner->getType() == frCornerTypeEnum::CONCAVE) {
          currCorner->setFixed(true);
          auto cornerPt = currCorner->getNextEdge()->low();
          for (auto& rect : net->getRectangles(false)[layerNum]) {
            if (gtl::contains(rect, cornerPt, true)
                && !gtl::contains(rect, cornerPt, false)) {
              currCorner->setFixed(false);
              break;
            }
          }
        }
      } else {
#ifdef GC_UPSTREAM_CORNER_LOOKUP
        (void) vcache;
        currCorner->setFixed(isPolygonCorner(currCorner->x(),
                                             currCorner->y(),
                                             net->getPolygons(true)[layerNum]));
#else
        auto& slot = (*vcache)[layerNum];
        if (!slot) {
          slot = std::make_unique<PtSet>();
          collectPolygonVertices(net->getPolygons(true)[layerNum], *slot);
        }
        currCorner->setFixed(
            slot->find({currCorner->x(), currCorner->y()}) != slot->end());
#endif
      }

      if (prevCorner) {
        prevCorner->setNextCorner(currCorner);
        currCorner->setPrevCorner(prevCorner);
      }
      prevCorner = currCorner;
      prevEdge = nextEdge.get();
    }
    auto currCorner = tmpCorners.front().get();
    prevCorner->setNextCorner(currCorner);
    currCorner->setPrevCorner(prevCorner);
    pin->addPolygonCorners(tmpCorners);
  }
}

void FlexGCWorker::initNet_pins_polygonCorners(gcNet* net)
{
  int numLayers = getTech()->numLayers();
  VertexCache vcache(numLayers);
  for (int i = 0; i < numLayers; i++) {
    for (auto& pin : net->getPins(i)) {
      initNet_pins_polygonCorners_helper(net, pin.get(), &vcache);
    }
  }
}

// ------------------------------------------------------------ max rectangles

void FlexGCWorker::initNet_pins_maxRectangles_getFixedMaxRectangles(
    gcNet* net,
    PointPairSets& fixedMaxRectangles)
{
  int numLayers = getTech()->numLayers();
  std::vector<gtl::rectangle_data<frCoord>> rects;
  for (int i = 0; i < numLayers; i++) {
    rects.clear();
    gtl::get_max_rectangles(rects, net->getPolygons(i, true));
    for (auto& rect : rects) {
      fixedMaxRectangles[i].insert(
          std::make_pair(odb::Point(gtl::xl(rect), gtl::yl(rect)),
                         odb::Point(gtl::xh(rect), gtl::yh(rect))));
    }
    for (auto& rect : net->getRectangles(i, true)) {
      fixedMaxRectangles[i].insert(
          std::make_pair(odb::Point(gtl::xl(rect), gtl::yl(rect)),
                         odb::Point(gtl::xh(rect), gtl::yh(rect))));
    }
  }
}

void FlexGCWorker::initNet_pins_maxRectangles_helper(
    gcNet* net,
    gcPin* pin,
    const gtl::rectangle_data<frCoord>& rect,
    frLayerNum i,
    const PointPairSets& fixedMaxRectangles)
{
  auto rectangle = std::make_unique<gcRect>();
  rectangle->setRect(rect);
  rectangle->setLayerNum(i);
  rectangle->addToPin(pin);
  rectangle->addToNet(net);
  if (fixedMaxRectangles[i].find(
          std::make_pair(odb::Point(gtl::xl(rect), gtl::yl(rect)),
                         odb::Point(gtl::xh(rect), gtl::yh(rect))))
      != fixedMaxRectangles[i].end()) {
    rectangle->setFixed(true);
  } else {
    rectangle->setFixed(false);
    int k = i / 2 - 1;
    for (auto& r : net->getTaperedRects(k)) {
      if (rectangle->intersects(r)) {
        rectangle->setTapered(true);
        for (auto& nt : net->getNonTaperedRects(k)) {
          if (rectangle->intersects(nt)) {
            net->addSpecialSpcRect(nt, i, rectangle->getPin(), rectangle->getNet());
          }
        }
        break;
      }
    }
  }
  pin->addMaxRectangle(std::move(rectangle));
}

void FlexGCWorker::initNet_pins_maxRectangles(gcNet* net)
{
  int numLayers = getTech()->numLayers();
  PointPairSets fixedMaxRectangles(numLayers);
  initNet_pins_maxRectangles_getFixedMaxRectangles(net, fixedMaxRectangles);
  std::vector<gtl::rectangle_data<frCoord>> rects;
  for (int i = 0; i < numLayers; i++) {
    for (auto& pin : net->getPins(i)) {
      rects.clear();
      gtl::get_max_rectangles(rects, *(pin->getPolygon()));
      for (auto& rect : rects) {
        initNet_pins_maxRectangles_helper(
            net, pin.get(), rect, i, fixedMaxRectangles);
      }
    }
  }
}

// ------------------------------------------------------------ drivers

void FlexGCWorker::initNet(gcNet* net)
{
  initNet_pins_polygon(net);
  initNet_pins_polygonEdges(net);
  initNet_pins_polygonCorners(net);
  initNet_pins_maxRectangles(net);
}

void FlexGCWorker::initNets()
{
  for (auto& uNet : getNets()) {
    initNet(uNet.get());
  }
}

void FlexGCWorker::init(const frDesign* design)
{
  addNet(&design->fakeVSS);  // [0] floating VSS
  addNet(&design->fakeVDD);  // [1] floating VDD
  initDesign(design);
  initDRWorker();
  initNets();
}

}  // namespace drt
