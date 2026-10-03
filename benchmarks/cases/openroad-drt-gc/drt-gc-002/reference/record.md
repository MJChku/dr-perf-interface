## drt-gc-002: three passes re-extract the same fixed polygons

**Origin.** OpenROAD `ec10d069`, `src/drt/src/gc/FlexGC_init.cpp`,
`FlexGCWorker::Impl::initNet` (lines 943–949) and the passes it calls. Three of
them rebuild a net's fixed polygons on a layer from the Boost.Polygon set:

- `initNet_pins_polygonEdges_getFixedPolygonEdges` (487–548):
  `net->getPolygons(i, true).get(polys)`, to fill a `std::set` of fixed edges;
- the corner pass (`initNet_pins_polygonCorners_helper`, 749–844): the fixed
  vertex lookup. The measured router built a per-layer vertex set from `get()`;
  upstream calls `isPolygonCorner()` (725–747), which re-extracts once per corner;
- `initNet_pins_maxRectangles_getFixedMaxRectangles` (856–878):
  `gtl::get_max_rectangles(rects, net->getPolygons(i, true))`, which copies and
  re-merges the set before covering it.

Each extraction is a clean plus polygon formation, both allocation-heavy. On
layers where the net has only fixed shapes (obstructions, pins nothing is routed
to), all of this only confirms that every edge, corner and maximal rectangle is
fixed.

**Found by drperf** (aes, per-pass per-net regions keyed by pin vertices). The
fix is `GC_FIXEDPOLYS`. Step 1 extracts each layer's fixed polygons once per
`initNet()`. Steps 2–3 add the all-fixed-layer shortcut, a single-source pin
merge and a rule that a rectangle pin is its own maximal rectangle.

    edges on obstruction nets   1,700*verts + 20,984  ->  431*verts + 4,478
    corners                     1,677*verts + 7,219   ->  463*verts + 2,896
    DRC setup on aes            41.3e9 -> 23.6e9 instructions across steps 1-3

### The standalone case

Region `drt-gc-002` is the body of `FlexGCWorker::initNet`, once per net. In
scenario `pins`, each worker has two nets with p L-shaped fixed cell pins on
metal1. Each pin is reached by a routed via1 whose enclosure lies inside the pin,
and by a routed metal2 wire, so metal1 holds fixed and route shapes and metal2
only route shapes. Each net also has a macro pin of o fixed rectangles on metal3
that this worker does not route: an all-fixed layer. Each worker also has the two
floating supply nets. Small: p, o ∈ {8,16,32}. Large: p, o ∈ {64,128,256}. Each
worker spec runs 4 times. The counts are powers of two, where the copying growth
of `std::vector<polygon_90_with_holes_data>` (Boost's type has no move
constructor) is proportional to the count. Between powers of two it adds a
stepwise term.

### Answer key

| PCV | Expression at region entry (computed on copies of the net's sets) |
| --- | --- |
| `vR` | vertices of the pins `initNet` builds on layers where the net has route shapes (merged route + fixed polygons, 4 per cut rectangle) |
| `vF` | vertices of the pins it builds on layers where the net has only fixed shapes |
| `layers` | layers where the net has shapes |
| `nlogn` | Σ over those layers of n·log2 n, n = rectangles added on the layer |

Pins, edges, corners and maximal rectangles are all proportional to vertices
within a layer kind. That is why two vertex counts, split by whether the layer
has route shapes, carry them. `vF` is the term the fix shrinks. `nlogn` stands
for the sorting in `polygon_90_set_data::clean()` and the ordered `std::set`s of
edges and rectangles. Without it the formula still holds (`vR, vF, layers`), but
drperf leaves 4.7% (original) and 4.1% (fixed) unexplained on the small run,
spread over `std::sort` and red-black tree blocks. On this workload `layers` is 0
for the floating nets and 4 for the others. It works as an indicator shared with
the constant, and its sign is not meaningful.

Implementation: `assets/pcvs.patch` (`src/pcv_drt_gc_002.h`). The states are
computed before `perfmark_begin_v` by merging copies of the route and fixed sets.
Reading a set cleans it, so the net's own sets are only inspected through
`value()`, which leaves them unread.

### drperf on the reference annotation

| arm | size | formula | unexplained |
| --- | --- | --- | ---: |
| original | small | `4,759.6*vR + 7,277.2*vF - 984.5*layers + 190.1*nlogn + 40,599.9` | 0.7% |
| original | large | `4,536.9*vR + 7,345.7*vF + 1,575.7*layers + 219.8*nlogn + 179,322` | 0.7% |
| fixed | small | `4,110.3*vR + 2,095.7*vF - 670*layers + 131.8*nlogn + 19,059.6` | 0.3% |
| fixed | large | `3,940.3*vR + 1,972.9*vF - 2,186.9*layers + 159.2*nlogn + 102,210.6` | 0.4% |

- `vF`: 7,277 → 2,096 per vertex on all-fixed layers (−71%). This is the
  shortcut: no fixed-edge set, no vertex cache, no fixed maximal rectangles and
  no lookups there. Rectangle pins also skip `get_max_rectangles`.
- `vR`: 4,760 → 4,110 (−14%). One extraction replaces three, and there are
  per-polygon maximal rectangles.
- The per-net constant halves. The single-source pin merge skips two set copies
  on each of the 21 layers.
- The top functions for `vR` and `vF` are `malloc`, `free` and
  `polygon_90_set_data::clean`, as in drperf's aes report, where the passes spent
  40–63% of their instructions in the allocator.
- **Small to large.** The small formula at the large states is within −0.6%
  (original) and +0.6% (fixed) of the measured total. Region instructions on the
  large run: 1,161,873,694 → 790,056,295 (−32.0%).

### Equivalence

- Step 1. Edges and corners read exactly the list `get()` returns. Merged
  polygons have disjoint interiors and share no edge, so no maximal rectangle
  spans two of them. The union of the per-polygon maximal rectangles is the
  set's, and it only fills a `std::set` used for membership.
- Step 2. With no route shape on a layer, the pins there are the fixed polygons
  or the fixed cut rectangles. Every pin edge, vertex, convex cut corner and
  maximal rectangle is then fixed, and concave cut corners lie in no route
  rectangle. The original lookups would all answer "fixed". Corners the original
  leaves unset stay unset.
- Step 3. Merging a set with an empty set gives that set's polygons.
  `get_max_rectangles` of a rectangle is that rectangle.

`tools/equivalence.sh` finds all outputs identical, gate off vs on vs the neutral
build, on every scenario and size.
