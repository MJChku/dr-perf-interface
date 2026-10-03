// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
// Minimal stand-ins for the types these passes use, after OpenROAD
// src/drt/src/frBaseTypes.h and src/odb/include/odb/geom.h at commit
// ec10d069775173bc856608663c11d33c09f3c9b6. See ../LICENSE.OpenROAD.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "boost/polygon/polygon.hpp"

namespace gtl = boost::polygon;

namespace odb {

// odb::Point orders lexicographically (defaulted operator<=> upstream).
class Point
{
 public:
  Point() = default;
  Point(int x, int y) : x_(x), y_(y) {}
  int x() const { return x_; }
  int y() const { return y_; }
  bool operator==(const Point& o) const { return x_ == o.x_ && y_ == o.y_; }
  bool operator<(const Point& o) const
  {
    return x_ < o.x_ || (x_ == o.x_ && y_ < o.y_);
  }

 private:
  int x_ = 0;
  int y_ = 0;
};

class Rect
{
 public:
  Rect() = default;
  Rect(int xl, int yl, int xh, int yh) : xl_(xl), yl_(yl), xh_(xh), yh_(yh) {}
  int xMin() const { return xl_; }
  int yMin() const { return yl_; }
  int xMax() const { return xh_; }
  int yMax() const { return yh_; }
  bool intersects(const Rect& b) const
  {
    return xl_ <= b.xh_ && xh_ >= b.xl_ && yl_ <= b.yh_ && yh_ >= b.yl_;
  }

 private:
  int xl_ = 0;
  int yl_ = 0;
  int xh_ = 0;
  int yh_ = 0;
};

}  // namespace odb

namespace drt {

using frCoord = int;
using frLayerNum = int;

enum class frCornerTypeEnum
{
  UNKNOWN,
  CONCAVE,
  CONVEX
};

enum class frCornerDirEnum
{
  UNKNOWN,
  NE,
  SE,
  SW,
  NW
};

enum class frDirEnum
{
  UNKNOWN = 0,
  D = 1,
  S = 2,
  W = 3,
  E = 4,
  N = 5,
  U = 6
};

enum class LayerType
{
  MASTERSLICE,
  CUT,
  ROUTING
};

// The technology as the passes see it: a layer count and a type per layer.
// Nangate45 as TritonRoute numbers it: 0 masterslice, odd layers cut, even
// layers from 2 up routing (21 layers).
class frTechObject
{
 public:
  explicit frTechObject(int numLayers)
  {
    for (int l = 0; l < numLayers; l++) {
      types_.push_back(l == 0       ? LayerType::MASTERSLICE
                       : (l % 2 == 1) ? LayerType::CUT
                                      : LayerType::ROUTING);
    }
  }
  int numLayers() const { return static_cast<int>(types_.size()); }
  LayerType getLayerType(frLayerNum l) const { return types_[l]; }

 private:
  std::vector<LayerType> types_;
};

}  // namespace drt
