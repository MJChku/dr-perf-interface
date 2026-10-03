## drt-gc-003: every per-net pass walks all 21 layers

**Origin.** OpenROAD `ec10d069`, `src/drt/src/gc/FlexGC_init.cpp`,
`FlexGCWorker::Impl::initNets` (951–957), which calls `initNet` (943–949) for
every net of the worker. Each pass of `initNet` loops
`for (int i = 0; i < numLayers; i++)`, 21 layers on Nangate45:

- the pin merge (`initNet_pins_polygon`, 460–485) builds two polygon set copies
  and calls `get()` per layer;
- the fixed edges `get()` the fixed set of every layer (495–530), in a vector of
  21 `std::set`s (703);
- the corner pass sizes a 21-slot cache (baseline tree) and walks every layer
  (849);
- the fixed maximal rectangles call `get_max_rectangles` on every layer's fixed
  set (863–877), in another vector of 21 `std::set`s (924).

All of this runs where the net has nothing. On aes a net's shapes sit on 1.92 of
the 21 layers, and 26.5% of the routing worker's nets have no shape in the box.
Measured on the real `FlexGC_init.cpp` (all other gates off), `initNet` of a net
without shapes costs 25,012 instructions; with the fix it costs 1,702.

**Found by drperf** (aes, per-worker region over all nets). The fix is
`GC_EMPTYLAYERS`:

    gc_nets = 352.7*v + 59.7*f + 12.2*scan + 18,281.1*nets    (87% of the cost follows no declared state)
           -> 198.9*v + 15.7*f + 4.9*scan + 1,042.8*nets      (97%)

drperf reported most calls as not modelled. The workers have too many distinct
states for its 128-point limit, so the per-net term was clear but the rest was
not.

### The standalone case

Region `drt-gc-003` is the body of `FlexGCWorker::initNets`, once per worker. In
scenario `mix`, a worker has e nets of the routing worker with nothing in the
box, and s routed nets. Each routed net has w metal2 wires, each with a via2 and
a metal3 wire, and r fixed pin rectangles on metal1 that the wires do not reach.
Each worker also has the two floating supply nets. Small: e ∈ {0,8,24},
s ∈ {2,4,8}, w ∈ {1,2}, r ∈ {1,2}. Large: e and s ×8. Each worker spec runs 4
times. A routed net always uses the same four layers, and w, r ≤ 2 keeps every
per-layer polygon set below `std::sort`'s insertion-sort threshold, so a net's
cost is proportional to its content. Wider ranges cross the threshold, and drperf
then reports up to 8% as unexplained.

### Answer key

| PCV | Expression at region entry |
| --- | --- |
| `nets` | gc nets of the worker, including nets with no shape |
| `layers` | Σ over nets of the layers where the net has shapes |
| `v` | Σ over nets of the vertices of the pins `initNet` builds |
| `f` | Σ over nets of the fixed shapes (fixed polygon rectangles, fixed cut rectangles) |

The per-net term is the one that should not exist: it is proportional to the
technology's layer count, not to anything the net holds. With the generator set
to an 11-layer stack (`WorkerSpec::layers`), the same run gives `10,859*nets`
instead of `20,489*nets`, about 963 instructions per layer per net, with the
other coefficients unchanged. The PCVs keep 21 layers because all workers of a
design share one technology.

Implementation: `assets/pcvs.patch` (`src/pcv_drt_gc_003.h`).

### drperf on the reference annotation

| arm | size | formula | unexplained |
| --- | --- | --- | ---: |
| original | small | `20,489*nets + 918.1*layers + 3,291.1*v + 18,175.3*f + 19,518.4` | 0.0% |
| original | large | `20,489*nets + 947.9*layers + 3,289.5*v + 17,932.7*f + 145,429.8` | 0.2% |
| fixed | small | `2,323*nets + 1,802.3*layers + 3,324.5*v + 18,140.7*f + 8,449.1` | 0.0% |
| fixed | large | `2,323*nets + 1,842.6*layers + 3,322.8*v + 17,898.3*f + 64,207.1` | 0.3% |

- `nets`: 20,489 → 2,323 per net (−89%), identical in both sizes. The original's
  top functions are `initNets`, `boost::polygon::get_polygons` and
  `initNet_pins_maxRectangles`. The fixed arm's are the mask computation (five
  container tests per layer) and `std::vector::operator[]`.
- `v` and `f` do not change. `layers` rises by about 900: the mask walk and
  clearing the reused sets on visited layers.
- **Small to large.** The small formula at the large states is within −1.9%
  (original) and −1.3% (fixed) of the measured total. Region instructions on the
  large run: 979,930,912 → 665,381,167 (−32.1%).

### Equivalence

A polygon set with no vertex data was never added to. It is in its default
state, and `get()`, `get_max_rectangles()` and `+=` of it produce and change
nothing. A layer without polygons, cut rectangles or pins makes every loop body
do nothing. Pins are added only on layers with polygons or cut rectangles, and no
shape list changes during `initNet()`, so the recorded layers stay exactly the
layers with content. They are visited in the same ascending order, so pin ids and
container insertion orders are the same. The reused per-thread sets and caches
are empty when handed out, as fresh ones are. Above 64 layers every layer is
walked. `tools/equivalence.sh` finds all outputs identical on every scenario and
size, including 70-layer workers in `random`.
