// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors (the code re-implemented here)
// Standalone extraction of OpenROAD's detailed-router DRC worker setup.
// After FlexGCWorker::Impl::isCornerOverlap in OpenROAD
// src/drt/src/gc/FlexGC_main.cpp at commit
// ec10d069775173bc856608663c11d33c09f3c9b6. See ../LICENSE.OpenROAD.

#include "FlexGC.h"

namespace drt {

// True when the corner point coincides with the rectangle's corner on the
// side the corner points to.
bool FlexGCWorker::isCornerOverlap(gcCorner* corner,
                                   const gtl::rectangle_data<frCoord>& rect)
{
  frCoord cornerX = corner->getNextEdge()->low().x();
  frCoord cornerY = corner->getNextEdge()->low().y();
  switch (corner->getDir()) {
    case frCornerDirEnum::NE:
      return cornerX == gtl::xh(rect) && cornerY == gtl::yh(rect);
    case frCornerDirEnum::SE:
      return cornerX == gtl::xh(rect) && cornerY == gtl::yl(rect);
    case frCornerDirEnum::SW:
      return cornerX == gtl::xl(rect) && cornerY == gtl::yl(rect);
    case frCornerDirEnum::NW:
      return cornerX == gtl::xl(rect) && cornerY == gtl::yh(rect);
    default:
      return false;
  }
}

}  // namespace drt
