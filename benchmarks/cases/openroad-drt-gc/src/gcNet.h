// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
// After OpenROAD src/drt/src/db/gcObj/gcNet.h at commit
// ec10d069775173bc856608663c11d33c09f3c9b6. See ../LICENSE.OpenROAD.
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "gcPin.h"
#include "gcShape.h"

namespace drt {

// The shapes of one net inside a DRC worker's extended box, kept per layer:
// polygon sets on routing layers, rectangle lists on cut layers, each split
// into fixed (from the design) and route (from the detailed-routing worker).
class gcNet
{
 public:
  explicit gcNet(const int numLayers)
      : fixedPolygons_(numLayers),
        routePolygons_(numLayers),
        fixedRectangles_(numLayers),
        routeRectangles_(numLayers),
        pins_(numLayers),
        taperedRects_(numLayers),
        nonTaperedRects_(numLayers)
  {
  }

  void addPolygon(const odb::Rect& box, frLayerNum layerNum, bool isFixed = false)
  {
    gtl::rectangle_data<frCoord> rect(
        box.xMin(), box.yMin(), box.xMax(), box.yMax());
    using gtl::operators::operator+=;
    if (isFixed) {
      fixedPolygons_[layerNum] += rect;
    } else {
      routePolygons_[layerNum] += rect;
    }
  }
  void addRectangle(const odb::Rect& box,
                    frLayerNum layerNum,
                    bool isFixed = false)
  {
    gtl::rectangle_data<frCoord> rect(
        box.xMin(), box.yMin(), box.xMax(), box.yMax());
    if (isFixed) {
      fixedRectangles_[layerNum].push_back(rect);
    } else {
      routeRectangles_[layerNum].push_back(rect);
    }
  }
  void addPin(const gtl::polygon_90_with_holes_data<frCoord>& shape,
              frLayerNum layerNum)
  {
    auto pin = std::make_unique<gcPin>(shape, layerNum, this);
    pin->setId(pins_[layerNum].size());
    pins_[layerNum].push_back(std::move(pin));
  }
  void addPin(const gtl::rectangle_data<frCoord>& rect, frLayerNum layerNum)
  {
    gtl::polygon_90_with_holes_data<frCoord> shape;
    std::vector<frCoord> coords
        = {gtl::xl(rect), gtl::yl(rect), gtl::xh(rect), gtl::yh(rect)};
    shape.set_compact(coords.begin(), coords.end());
    auto pin = std::make_unique<gcPin>(shape, layerNum, this);
    pin->setId(pins_[layerNum].size());
    pins_[layerNum].push_back(std::move(pin));
  }
  void addTaperedRect(const odb::Rect& bx, int zIdx)
  {
    taperedRects_[zIdx].push_back(bx);
  }
  void addNonTaperedRect(const odb::Rect& bx, int zIdx)
  {
    nonTaperedRects_[zIdx].push_back(bx);
  }
  void addSpecialSpcRect(const odb::Rect& bx,
                         frLayerNum lNum,
                         gcPin* pin,
                         gcNet* net)
  {
    auto sp = std::make_unique<gcRect>();
    sp->setLayerNum(lNum);
    sp->addToNet(net);
    sp->addToPin(pin);
    sp->setRect(bx);
    specialSpacingRects_.push_back(std::move(sp));
  }
  void setOwner(const void* in) { owner_ = in; }
  void setId(int in) { id_ = in; }

  const std::vector<gtl::polygon_90_set_data<frCoord>>& getPolygons(
      bool isFixed = false) const
  {
    return isFixed ? fixedPolygons_ : routePolygons_;
  }
  const gtl::polygon_90_set_data<frCoord>& getPolygons(frLayerNum layerNum,
                                                       bool isFixed
                                                       = false) const
  {
    return isFixed ? fixedPolygons_[layerNum] : routePolygons_[layerNum];
  }
  const std::vector<std::vector<gtl::rectangle_data<frCoord>>>& getRectangles(
      bool isFixed = false) const
  {
    return isFixed ? fixedRectangles_ : routeRectangles_;
  }
  const std::vector<gtl::rectangle_data<frCoord>>& getRectangles(
      frLayerNum layerNum,
      bool isFixed = false) const
  {
    return isFixed ? fixedRectangles_[layerNum] : routeRectangles_[layerNum];
  }
  const std::vector<std::vector<std::unique_ptr<gcPin>>>& getPins() const
  {
    return pins_;
  }
  const std::vector<std::unique_ptr<gcPin>>& getPins(frLayerNum layerNum) const
  {
    return pins_[layerNum];
  }
  const std::vector<odb::Rect>& getTaperedRects(int z) const
  {
    return taperedRects_[z];
  }
  const std::vector<odb::Rect>& getNonTaperedRects(int z) const
  {
    return nonTaperedRects_[z];
  }
  const std::vector<std::unique_ptr<gcRect>>& getSpecialSpcRects() const
  {
    return specialSpacingRects_;
  }
  const void* getOwner() const { return owner_; }
  int getId() const { return id_; }

 private:
  std::vector<gtl::polygon_90_set_data<frCoord>> fixedPolygons_;  // routing
  std::vector<gtl::polygon_90_set_data<frCoord>> routePolygons_;  // routing
  std::vector<std::vector<gtl::rectangle_data<frCoord>>> fixedRectangles_;  // cut
  std::vector<std::vector<gtl::rectangle_data<frCoord>>> routeRectangles_;  // cut
  std::vector<std::vector<std::unique_ptr<gcPin>>> pins_;
  const void* owner_{nullptr};
  int id_{-1};
  std::vector<std::vector<odb::Rect>> taperedRects_;     // routing, by z
  std::vector<std::vector<odb::Rect>> nonTaperedRects_;  // routing, by z
  std::vector<std::unique_ptr<gcRect>> specialSpacingRects_;
};

}  // namespace drt
