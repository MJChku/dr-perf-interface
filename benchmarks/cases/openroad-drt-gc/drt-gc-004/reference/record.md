## drt-gc-004: seven per-layer containers built for every new net

**Origin.** OpenROAD `ec10d069`. The constructor `gcNet::gcNet(numLayers)`
(`src/drt/src/db/gcObj/gcNet.h`, 27–36) runs from
`FlexGCWorker::Impl::addNet` (`FlexGC_impl.h`, 95–104), which `initDesign` and
`initDRWorker` (`FlexGC_init.cpp`, 400–421) call for every owner of shapes:

```cpp
gcNet(const int numLayers)
    : fixedPolygons_(numLayers), routePolygons_(numLayers),
      fixedRectangles_(numLayers), routeRectangles_(numLayers),
      pins_(numLayers), taperedRects_(numLayers), nonTaperedRects_(numLayers) {}
```

That is seven allocations and 147 element constructions per net on 21 layers,
42 of them polygon sets, whether the net holds anything or not, and as many
destructions when the worker ends. On aes a net uses 2.38 of the 7 kinds. The two
floating supply nets never hold anything, and 26.5% of the routing worker's nets
have no shape in the box.

**Found by drperf** (aes, region `gc_drworker` around `initDRWorker`). The fix is
`GC_NETCTOR`: size a kind on its first add, and let reads of an unsized kind see
one shared, never-written vector of default elements.

    gc_drworker = 264.7*fpoly + 936.7*newnets + 871.7   (76% unexplained; gcNet::gcNet 429.9 per new net)
               -> 275.2*fpoly + 523.2*newnets + 1,011.6  (gcNet::gcNet 118.7)

In that report, 165.6 of the `newnets` coefficient (`vector::_M_fill_assign`)
came from per-layer counters of the annotation itself, placed in the constructor.
The standalone case has no such counters.

### The standalone case

Region `drt-gc-004` is the body of `FlexGCWorker::initDRWorker`, once per worker.
Scenario `drnets` has three kinds of routing-worker nets. There are d design nets
the gc worker already holds from `initDesign` (fixed pins on metal1), each with
two routed via1s and metal2 wires. There are s new routed nets with w metal2
wires, each with a via2. And there are e new nets with nothing in the box. Small:
e ∈ {0,4,12}, s ∈ {2,4,8}, d ∈ {2,4,8}, w ∈ {1,2}. Large: e, s, d ×8. Each
worker spec runs 4 times. Every via figure in this scenario comes with one path
segment, so polygon and cut shapes are in a fixed ratio to figures.

### Answer key

| PCV | Expression at region entry |
| --- | --- |
| `drnets` | nets of the routing worker |
| `newnets` | of those, nets whose owner has no gc net yet (`addNet` runs) |
| `kinds` | Σ over routing-worker nets of the shape kinds (route/fixed polygons, route/fixed cut rectangles) the net receives here and did not hold before |
| `figs` | connected figures (path segments, vias, patch wires) |

`newnets` is the construction the fix removes. `kinds` is the first insertion
into an empty per-layer container. In the original that is a vector's first
allocation, and in the fixed arm it also sizes the kind, so the cost moves there.
`figs` covers the per-figure owner lookup and the shape adds.

Implementation: `assets/pcvs.patch` (`src/pcv_drt_gc_004.h`).

### drperf on the reference annotation

| arm | size | formula | unexplained |
| --- | --- | --- | ---: |
| original | small | `94.1*drnets + 1,627.3*newnets + 403.4*kinds + 724.5*figs + 603.4` | 2.8% |
| original | large | `79.7*drnets + 1,586.7*newnets + 419.9*kinds + 724.5*figs + 180.7` | 7.2% |
| fixed | small | `102.1*drnets + 439.2*newnets + 679.9*kinds + 730.5*figs + 622.3` | 2.4% |
| fixed | large | `80.6*drnets + 409.8*newnets + 696.4*kinds + 730.5*figs + 94.9` | 7.7% |

- `newnets`: 1,627 → 439 (−73%). The constructor is inlined into `addNet` here
  (957.6 of the original coefficient; with the gate on it keeps the pointer set-up
  and the map insertion).
- `kinds`: 403 → 680. Sizing moves to the first add.
- `figs` is unchanged.
- **Unexplained share.** Nearly all of it is owner lookup: `owner2nets_` is a
  `std::map`. `initDRObj` looks up the owner once per figure, and `initDRWorker`
  once per net, and each lookup walks O(log nets) levels. On the large grid a
  worker grows from 30 to 226 nets, so those blocks are not proportional to any
  count. `nets_` vector growth, which moves `unique_ptr`s, adds a little. A PCV
  counting map comparisons (tried: an exact replay of the map) does not fit
  within drperf's four states together with the terms above.
- **Small to large.** The small formula underestimates the large total by 6.5%
  (original) and 6.9% (fixed): the same log(nets) lookups. Region instructions on
  the large run: 90,420,044 → 78,891,088 (−12.8%). The saved construction is
  partly offset by sizing the kinds that are used. The larger saving, 7 × 21
  destructions per net when the worker ends, falls outside this region.

### Equivalence

A kind not yet sized has had no add since construction, so in the original every
layer of it is a default-constructed element. The shared vector has the same
length and default-constructed elements, so every getter returns an equal value:
size, iteration, contents, and a polygon set's orientation and dirty/sorted
flags. The shared objects are never written. The only mutators of a gcNet's
per-layer data are the add methods, which write the net's own members, and const
reads of a pristine polygon set do not modify it (`clean()` and `sort()` test
their flags first). No caller keeps a getter reference across an add to the same
net and kind. Sizing on the first add creates what the constructor created.
`tools/equivalence.sh` finds all outputs identical on every scenario and size,
including 11- and 70-layer workers (one shared vector per layer count).
