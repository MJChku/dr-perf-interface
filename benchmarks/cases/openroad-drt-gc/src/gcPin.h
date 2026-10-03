// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
// After OpenROAD src/drt/src/db/gcObj/gcPin.h at commit
// ec10d069775173bc856608663c11d33c09f3c9b6. See ../LICENSE.OpenROAD.
#pragma once

#include <memory>
#include <vector>

#include "gcShape.h"

namespace drt {

// One merged polygon of a net on a layer, with the edges, corners and maximal
// rectangles the passes derive from it.
class gcPin
{
 public:
  gcPin(const gtl::polygon_90_with_holes_data<frCoord>& shape,
        frLayerNum layer,
        gcNet* net)
      : polygon_(std::make_unique<gcPolygon>(shape, layer, this, net)), net_(net)
  {
  }
  void setId(int in) { id_ = in; }
  int getId() const { return id_; }
  void addPolygonEdges(std::vector<std::unique_ptr<gcSegment>>& in)
  {
    polygon_edges_.push_back(std::move(in));
  }
  void addPolygonCorners(std::vector<std::unique_ptr<gcCorner>>& in)
  {
    polygon_corners_.push_back(std::move(in));
  }
  void addMaxRectangle(std::unique_ptr<gcRect> in)
  {
    max_rectangles_.push_back(std::move(in));
  }
  gcPolygon* getPolygon() const { return polygon_.get(); }
  const std::vector<std::vector<std::unique_ptr<gcSegment>>>& getPolygonEdges()
      const
  {
    return polygon_edges_;
  }
  const std::vector<std::vector<std::unique_ptr<gcCorner>>>& getPolygonCorners()
      const
  {
    return polygon_corners_;
  }
  const std::vector<std::unique_ptr<gcRect>>& getMaxRectangles() const
  {
    return max_rectangles_;
  }
  gcNet* getNet() const { return net_; }

 private:
  int id_{-1};
  std::unique_ptr<gcPolygon> polygon_;
  gcNet* net_{nullptr};
  std::vector<std::vector<std::unique_ptr<gcSegment>>> polygon_edges_;
  std::vector<std::vector<std::unique_ptr<gcCorner>>> polygon_corners_;
  std::vector<std::unique_ptr<gcRect>> max_rectangles_;
};

}  // namespace drt
