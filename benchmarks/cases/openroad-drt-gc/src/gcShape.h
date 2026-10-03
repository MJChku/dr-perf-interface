// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
// Shapes built by the per-net passes, after OpenROAD
// src/drt/src/db/gcObj/gcShape.h at commit
// ec10d069775173bc856608663c11d33c09f3c9b6 (the virtual gcShape/gcPinFig
// interfaces are dropped; only the data the passes set is kept).
// See ../LICENSE.OpenROAD.
#pragma once

#include "frBase.h"

namespace drt {
class gcSegment;
class gcRect;
class gcPolygon;
class gcPin;
class gcNet;
}  // namespace drt

template <>
struct gtl::geometry_concept<drt::gcSegment>
{
  using type = segment_concept;
};
template <>
struct gtl::geometry_concept<drt::gcRect>
{
  using type = gtl::rectangle_concept;
};
template <>
struct gtl::geometry_concept<drt::gcPolygon>
{
  using type = polygon_90_with_holes_concept;
};

namespace drt {

class gcCorner : public gtl::point_data<frCoord>
{
 public:
  gcCorner* getPrevCorner() const { return prevCorner_; }
  gcCorner* getNextCorner() const { return nextCorner_; }
  gcSegment* getPrevEdge() const { return prevEdge_; }
  gcSegment* getNextEdge() const { return nextEdge_; }
  frCornerTypeEnum getType() const { return type_; }
  frCornerDirEnum getDir() const { return dir_; }
  bool isFixed() const { return fixed_; }
  gcPin* getPin() const { return pin_; }
  frLayerNum getLayerNum() const { return layer_; }

  void setPrevCorner(gcCorner* in) { prevCorner_ = in; }
  void setNextCorner(gcCorner* in) { nextCorner_ = in; }
  void setPrevEdge(gcSegment* in) { prevEdge_ = in; }
  void setNextEdge(gcSegment* in) { nextEdge_ = in; }
  void setType(frCornerTypeEnum in) { type_ = in; }
  void setDir(frCornerDirEnum in) { dir_ = in; }
  void setFixed(bool in) { fixed_ = in; }
  void addToPin(gcPin* in) { pin_ = in; }
  void setLayerNum(frLayerNum in) { layer_ = in; }

 private:
  gcPin* pin_{nullptr};
  gcCorner* prevCorner_{nullptr};
  gcCorner* nextCorner_{nullptr};
  gcSegment* prevEdge_{nullptr};
  gcSegment* nextEdge_{nullptr};
  frCornerTypeEnum type_{frCornerTypeEnum::UNKNOWN};
  frCornerDirEnum dir_{frCornerDirEnum::UNKNOWN};  // away from the polygon
  bool fixed_{false};
  frLayerNum layer_{-1};
};

// A polygon edge, directed from low() to high().
class gcSegment : public gtl::segment_data<frCoord>
{
 public:
  gcSegment* getPrevEdge() const { return prev_; }
  gcSegment* getNextEdge() const { return next_; }
  gcCorner* getLowCorner() const { return lowCorner_; }
  gcCorner* getHighCorner() const { return highCorner_; }
  bool isFixed() const { return fixed_; }
  frLayerNum getLayerNum() const { return layer_; }
  gcPin* getPin() const { return pin_; }
  gcNet* getNet() const { return net_; }

  frDirEnum getDir() const
  {
    if (low().x() == high().x()) {
      return low().y() < high().y() ? frDirEnum::N : frDirEnum::S;
    }
    return low().x() < high().x() ? frDirEnum::E : frDirEnum::W;
  }

  void setSegment(const gtl::point_data<frCoord>& bp,
                  const gtl::point_data<frCoord>& ep)
  {
    low(bp);
    high(ep);
  }
  void setPrevEdge(gcSegment* in) { prev_ = in; }
  void setNextEdge(gcSegment* in) { next_ = in; }
  void setLowCorner(gcCorner* in) { lowCorner_ = in; }
  void setHighCorner(gcCorner* in) { highCorner_ = in; }
  void setFixed(bool in) { fixed_ = in; }
  void setLayerNum(frLayerNum in) { layer_ = in; }
  void addToPin(gcPin* in) { pin_ = in; }
  void addToNet(gcNet* in) { net_ = in; }

 private:
  frLayerNum layer_{-1};
  gcPin* pin_{nullptr};
  gcNet* net_{nullptr};
  gcSegment* prev_{nullptr};
  gcSegment* next_{nullptr};
  gcCorner* lowCorner_{nullptr};
  gcCorner* highCorner_{nullptr};
  bool fixed_{false};
};

class gcRect : public gtl::rectangle_data<frCoord>
{
 public:
  void setRect(const gtl::rectangle_data<frCoord>& in)
  {
    gtl::rectangle_data<frCoord>::operator=(in);
  }
  void setRect(const odb::Rect& in)
  {
    gtl::xl(*this, in.xMin());
    gtl::xh(*this, in.xMax());
    gtl::yl(*this, in.yMin());
    gtl::yh(*this, in.yMax());
  }
  bool isFixed() const { return fixed_; }
  bool isTapered() const { return tapered_; }
  frLayerNum getLayerNum() const { return layer_; }
  gcPin* getPin() const { return pin_; }
  gcNet* getNet() const { return net_; }
  void setFixed(bool in) { fixed_ = in; }
  void setTapered(bool in) { tapered_ = in; }
  void setLayerNum(frLayerNum in) { layer_ = in; }
  void addToPin(gcPin* in) { pin_ = in; }
  void addToNet(gcNet* in) { net_ = in; }
  bool intersects(const odb::Rect& b) const
  {
    return gtl::xl(*this) <= b.xMax() && gtl::xh(*this) >= b.xMin()
           && gtl::yl(*this) <= b.yMax() && gtl::yh(*this) >= b.yMin();
  }

 private:
  frLayerNum layer_{-1};
  gcPin* pin_{nullptr};
  gcNet* net_{nullptr};
  bool fixed_{false};
  bool tapered_{false};
};

class gcPolygon : public gtl::polygon_90_with_holes_data<frCoord>
{
 public:
  gcPolygon(const gtl::polygon_90_with_holes_data<frCoord>& shape,
            frLayerNum layer,
            gcPin* pin,
            gcNet* net)
      : gtl::polygon_90_with_holes_data<frCoord>(shape),
        layer_(layer),
        pin_(pin),
        net_(net)
  {
  }
  frLayerNum getLayerNum() const { return layer_; }

 private:
  frLayerNum layer_{-1};
  gcPin* pin_{nullptr};
  gcNet* net_{nullptr};
};

}  // namespace drt
