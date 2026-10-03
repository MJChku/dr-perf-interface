## drt-gc-001: cut-layer corners scan every fixed cut rectangle

**Origin.** OpenROAD `ec10d069`, `src/drt/src/gc/FlexGC_init.cpp`,
`FlexGCWorker::Impl::initNet_pins_polygonCorners_helper` (lines 749–844), called
per pin from `initNet_pins_polygonCorners` (846–854). The comparison is
`isCornerOverlap` in `FlexGC_main.cpp` (70–100).

The corner pass gives every pin polygon a ring of `gcCorner`s. On a cut layer, a
convex corner is fixed when some fixed cut rectangle of the net has its corner,
in the corner's direction, at the corner point:

```cpp
currCorner->setFixed(false);
for (auto& rect : net->getRectangles(true)[layerNum]) {
  if (isCornerOverlap(currCorner, rect)) { currCorner->setFixed(true); break; }
}
```

Every cut rectangle is a pin with four convex corners, fixed or routed. A net
with n fixed cut rectangles on a layer therefore does about 4n scans of up to n
rectangles, which is quadratic. The corner of a fixed rectangle stops at its own
entry, so on average it scans half the list. A routed cut scans the whole list.
Power/ground via arrays in a worker's extended box make n large.

**Found by drperf** (aes, per-net sub-region `gc_corner_cut` of the corner pass):

    gc_corner_cut = 376.8*vc + 5.3*q + 1,109      (37% of the cost follows no declared state)
    vc = cut-layer pin vertices,  q = sum over cut layers of pin vertices x fixed cut rectangles

The scan dominated DRC setup on leon3. `GC_CUTCORNERCACHE` puts the corner points
of the fixed rectangles into four hash sets per layer, one per direction, and
looks each corner up. On leon3 (48 threads), the CPU time of DRC worker setup in
routing iteration 0 fell from 4,421 s to 3,273 s (−26%). The routed DEF was
identical (digest `f208c03f4aefb8e3`), and so was the violation trajectory.

### The standalone case

`src/` rebuilds this pass with the passes before it, which create its input. The
region `drt-gc-001` is the body of `FlexGCWorker::initNet_pins_polygonCorners`,
once per net. In scenario `pgvias`, each worker has three power/ground nets:
rails on metal1–3 (fixed), a row of k fixed vias on via1 and via2, and a row of r
routed vias of the same net whose enclosures lie inside the rails. Each worker
also has the two floating supply nets, which have no shapes. Small:
k ∈ {2,4,8,12,16,24}, r ∈ {0,2,6}. Large: k ∈ {64,…,1024}, r ∈ {0,16,64}. Each
worker spec runs 4 times.

### Answer key

| PCV | Expression at region entry (pins and edges exist) |
| --- | --- |
| `vc` | vertices of the net's pins on cut layers (4 per cut rectangle) |
| `vm` | vertices of its pins on the other layers |
| `scan` | over all convex corners of cut-layer pins, the fixed cut rectangles compared up to and including the first match (all of them when none matches) |
| `fcut` | fixed cut rectangles on cut layers where the net has pins |

On this workload `scan = 2·[2k(k+1) + 4rk]`: each fixed cut at list position j
compares j+1 times per corner, each routed cut k times, on two cut layers.
drperf's `q = vc × fixed` is an upper bound that equals `scan` only when no
corner matches. `vm` is 0 for the floating nets and 12 for a power net (three
rectangular rails), so it also carries the per-net cost of the routing-layer
corners. `fcut` has a small coefficient in the original: the exit taken when a
corner matches. It becomes the main term of the fix.

Implementation: `assets/pcvs.patch` (`src/pcv_drt_gc_001.h`), computed before
`perfmark_begin_v` from the pins, without allocating.

### drperf on the reference annotation

`GC_CUTCORNERCACHE=0|1 bin/drperf build/drt-gc-001/{reference,fixed}/gcbench scenario=pgvias size=S`,
instructions per call. The unexplained share comes from `drperf-dev derive` on the same runs.

| arm | size | formula | unexplained |
| --- | --- | --- | ---: |
| original | small | `370*vc + 2,128.8*vm + 5*scan + 7*fcut + 1,062` | 0.0% |
| original | large | `370.2*vc + 2,128.8*vm + 5*scan + 7.1*fcut + 1,043.1` | 0.0% |
| fixed | small | `398.6*vc + 2,328.3*vm + 0.0987*scan + 1,166*fcut + 2,378.7` | 1.1% |
| fixed | large | `401.7*vc + 1,994.8*vm - 0.00247*scan + 1,124.2*fcut + 13,171.5` | 3.8% |

- `scan`: 5 instructions per comparison, all in `initNet_pins_polygonCorners_helper`
  (with `isCornerOverlap` inlined), in both sizes. With the gate on, it drops to
  0.1 (small) and −0.002 (large): the term is gone.
- `fcut`: 7 → 1,166 per fixed cut rectangle. That is four hash-set insertions
  (node allocation, hashing, bucket growth).
- `vc` rises from 370 to 399 per cut vertex, for the lookup that replaces a
  loop setup.
- Fixed-arm unexplained share: `std::unordered_set` rehashing grows its bucket
  arrays in prime-sized steps, not linearly.
- **Small to large.** The small formula evaluated at the large run's states
  gives −0.0% (original) and +1.6% (fixed) of the measured total. In the large
  run the scan is the larger part: region instructions fall from 1,481,080,516
  to 511,470,685 (−65.5%).

The unannotated neutral build shows the region as one observed mean per call:
the task starts from no states.

### Equivalence

`isCornerOverlap(corner, rect)` holds exactly when the corner point
`getNextEdge()->low()` equals `(xh, yh)` of the rectangle for NE, `(xh, yl)` for
SE, `(xl, yl)` for SW and `(xl, yh)` for NW. It never holds for another
direction. Each direction's set holds exactly those points from the same
`getRectangles(true)[layer]`, which the corner pass never modifies. So set
membership is the predicate "some rectangle matches", and the fixed flag is the
same. `tools/equivalence.sh` (gate off vs on vs the neutral build, every
scenario and size, full dumps at the small sizes) finds all outputs identical.
