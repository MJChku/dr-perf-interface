# OpenROAD detailed router: DRC worker setup

Four standalone cases taken from `FlexGCWorker::Impl::init()` in OpenROAD's
detailed router (TritonRoute, `src/drt/src/gc/`), the setup every DRC worker runs
before it checks anything. drperf found all four while profiling the router on
the aes and leon3 designs. Each case here is a small C++17 program that rebuilds
the per-net pin structures the same way upstream does. It has an empty marker,
bounded workloads, correctness tests, a reference annotation with the PCVs and
the formula they give, and the fix behind a run-time gate. The full OpenROAD is
not needed. The program uses Boost.Polygon headers and drperf's `libperfmark`.

| Case | Issue | Marked region (per call) | Scenario | PCVs (answer key) |
| --- | --- | --- | --- | --- |
| [drt-gc-001](drt-gc-001/) | Cut-layer corners scan the net's fixed cut rectangles | `initNet_pins_polygonCorners` (net) | `pgvias` | `vc, vm, scan, fcut` |
| [drt-gc-002](drt-gc-002/) | Three passes re-extract the same fixed polygons | `initNet` (net) | `pins` | `vR, vF, layers, nlogn` |
| [drt-gc-003](drt-gc-003/) | Every pass walks all 21 layers of every net | `initNets` (worker) | `mix` | `nets, layers, v, f` |
| [drt-gc-004](drt-gc-004/) | Seven per-layer vectors built eagerly for every new net | `initDRWorker` (worker) | `drnets` | `drnets, newnets, kinds, figs` |

What drperf reports on the reference annotation (instructions per call, `drperf-dev
derive`; the unexplained share is the cost of blocks that follow no declared state):

| Case | Arm | small | unexpl. | large | unexpl. | small formula at large states |
| --- | --- | --- | ---: | --- | ---: | ---: |
| 001 | original | `370*vc + 2,128.8*vm + 5*scan + 7*fcut + 1,062` | 0.0% | `370.2*vc + 2,128.8*vm + 5*scan + 7.1*fcut + 1,043.1` | 0.0% | -0.0% |
| 001 | fixed | `398.6*vc + 2,328.3*vm + 0.0987*scan + 1,166*fcut + 2,378.7` | 1.1% | `401.7*vc + 1,994.8*vm - 0.00247*scan + 1,124.2*fcut + 13,171.5` | 3.8% | +1.6% |
| 002 | original | `4,759.6*vR + 7,277.2*vF - 984.5*layers + 190.1*nlogn + 40,599.9` | 0.7% | `4,536.9*vR + 7,345.7*vF + 1,575.7*layers + 219.8*nlogn + 179,322` | 0.7% | -0.6% |
| 002 | fixed | `4,110.3*vR + 2,095.7*vF - 670*layers + 131.8*nlogn + 19,059.6` | 0.3% | `3,940.3*vR + 1,972.9*vF - 2,186.9*layers + 159.2*nlogn + 102,210.6` | 0.4% | +0.6% |
| 003 | original | `20,489*nets + 918.1*layers + 3,291.1*v + 18,175.3*f + 19,518.4` | 0.0% | `20,489*nets + 947.9*layers + 3,289.5*v + 17,932.7*f + 145,429.8` | 0.2% | -1.9% |
| 003 | fixed | `2,323*nets + 1,802.3*layers + 3,324.5*v + 18,140.7*f + 8,449.1` | 0.0% | `2,323*nets + 1,842.6*layers + 3,322.8*v + 17,898.3*f + 64,207.1` | 0.3% | -1.3% |
| 004 | original | `94.1*drnets + 1,627.3*newnets + 403.4*kinds + 724.5*figs + 603.4` | 2.8% | `79.7*drnets + 1,586.7*newnets + 419.9*kinds + 724.5*figs + 180.7` | 7.2% | -6.5% |
| 004 | fixed | `102.1*drnets + 439.2*newnets + 679.9*kinds + 730.5*figs + 622.3` | 2.4% | `80.6*drnets + 409.8*newnets + 696.4*kinds + 730.5*figs + 94.9` | 7.7% | -6.9% |

The last column is `drperf-dev derive SMALL --predict LARGE`: the formula fitted
on the small run, evaluated at the large run's states, against the large run's
measured total. Each case's `reference/record.md` explains its numbers. Only
drt-gc-004's large run has a share above 5%. That remainder is the
`std::map` lookup of each figure's owner, which costs O(log nets). The workers
there grow from 30 to 226 nets.

## Layout

```text
README.md, LICENSE.OpenROAD
build.sh                        build a case variant into build/<case>/<variant>/gcbench
src/                            the standalone extraction, no marker (shared by all cases)
tests/run_tests.sh, tests/probe correctness and marker-entry checks of the neutral cases
drt-gc-00N/case.json            identity, origin, region, empty marker, workload, tests
drt-gc-00N/region.patch         the empty marker, against src/
drt-gc-00N/task.md              neutral task
drt-gc-00N/reference/           answer key; keep out of agent exports
  answer.json                   PCVs, factors, formulas
  record.md                     the issue, origin, measurements, the fix, equivalence
  provenance.json               hashes of record.md and assets
  assets/pcvs.patch             reference annotation (applies after region.patch)
  assets/fix.patch              the gated fix (applies after region.patch, with or without pcvs.patch)
  assets/drperf/                what drperf printed for original/fixed, small/large, and the prediction
tools/validate.sh               evaluator: rerun drperf, refresh assets/drperf/ and provenance
tools/equivalence.sh            evaluator: original vs fixed outputs, all cases and scenarios
tools/provenance.py             refresh reference/provenance.json after editing a record
```

**Why this directory and format.** The cases combine two conventions. Like
`benchmarks/regions/`, each has a neutral source with one empty C++ marker, a
`region.patch`, a `task.md` and bounded tests. Like the historical families in
`benchmarks/cases/`, each also has an answer key, a fix and measured formulas
under `reference/`, with `answer.json`, `record.md` and `provenance.json`, and
`case.json` carries the `schema_version/id/title/status/readiness_note` fields.
The regions collection keeps no answer keys or fixes, so the cases live here.
They sit under one family directory because they share one extraction. Each case
directory still has everything `bench.py list` and
`tests/test_benchmark.py::EvidenceTests` read, at the case id
`openroad-drt-gc/drt-gc-00N`.

## Build, test, run

Needs g++ with C++17, Boost headers (built here with Boost 1.83; the drperf study
used 1.89), and a built drperf (`./build.sh` at the repository root).

```sh
cd benchmarks/cases/openroad-drt-gc
./build.sh drt-gc-001                 # neutral: build/drt-gc-001/neutral/gcbench
tests/run_tests.sh                    # all four cases, ~40 s
bin=build/drt-gc-001/neutral/gcbench
../../../bin/drperf $bin scenario=pgvias size=small
```

`gcbench scenario=pgvias|pins|mix|drnets|random size=tiny|small|large [repeat=N]
[seed=S] [dump=FILE] [time=1] [alloc=default]` builds a fixed sequence of workers.
It runs `FlexGCWorker::init()` on each and prints totals and a digest of
everything setup built. The case's scenario and `size=small` are the measurement
envelope. `size=large` is held out for evaluation. `random` exists for the checks.

Reference and fixed arms (evaluator only):

```sh
./build.sh all                                     # neutral, reference, fixed, check
../../../bin/drperf build/drt-gc-001/reference/gcbench scenario=pgvias size=small
GC_CUTCORNERCACHE=1 ../../../bin/drperf build/drt-gc-001/fixed/gcbench scenario=pgvias size=small
tools/validate.sh                                  # all formulas and predictions, ~75 s
tools/equivalence.sh                               # ~60 s
```

The gates are `GC_CUTCORNERCACHE` (001), `GC_FIXEDPOLYS` (002), `GC_EMPTYLAYERS`
(003) and `GC_NETCTOR` (004). They are read once from the environment; unset or `0`
runs the original code. The patches are independent. Each applies to `src/` after
its `region.patch`. The four fixes are not meant to be stacked.

## Equivalence

`tools/equivalence.sh` runs each case's `check` build (src + region + fix) on
every scenario at every size, with the gate off and on, and also runs the neutral
build. The digest covers every net's per-layer shapes, and every pin's polygon,
edge ring and corner ring (type, direction, fixed flag and links). It also covers
every maximal rectangle (fixed and tapered flags) and the special spacing
rectangles. The small sizes also compare the full dumps byte for byte (1.5–4.5 MB
each). The `random` scenario draws overlapping, touching, repeated and
ring-shaped (hole) shapes, fixed and routed, on 11-, 21- and 70-layer stacks. The
last stack exercises the fallback of `GC_EMPTYLAYERS` above 64 layers. Result: all
113 comparisons identical. The digest is sensitive: five hand-made bugs, one or
two per fix (a swapped corner direction, an all-fixed test that ignores route cut
rectangles, a rectangle test that accepts non-rectangles, a layer mask that
ignores cut rectangles, a pin vector left unsized), all changed the digest or
crashed.

## Differences from upstream

- **Corner-vertex cache.** On routing layers, upstream commit `ec10d069` calls
  `isPolygonCorner()` once per corner, and each call re-extracts the whole fixed
  polygon set. The router the drperf study measured had already replaced that
  with a per-layer vertex set (`OPENROAD_DRT_GC_CORNERCACHE`), and the four
  issues were found on that build. `src/` keeps that cache. Compile with
  `-DGC_UPSTREAM_CORNER_LOOKUP` for the upstream lookup; `tools/equivalence.sh`
  checks both give identical results.
- **Left out:** region-query packing, pin access (`initPA0/1`), logging, floating
  VSS/VDD owner resolution, NDR tapering rules (tapered and non-tapered rectangle
  lists stay, always empty, as on aes), the virtual `gcShape` interfaces, and the
  design database. The design's fixed objects and the detailed-routing worker's
  nets come from a deterministic generator (`src/workload.cpp`).
- **Allocator.** `gcbench` re-executes itself with
  `GLIBC_TUNABLES=glibc.malloc.tcache_count=65535`. Before the first worker it
  fills glibc's per-thread cache, and it repeats each worker spec back to back.
  Otherwise a call's malloc/free cost depends on what earlier workers left in the
  free lists. With `alloc=default`, drperf reports 31.8%, 20.2%, 9.7% and 25.0%
  of the small runs of 001–004 as unexplained, instead of 0.0–2.8%. The code of
  the regions is unchanged.
- **Workload sizes** are chosen so that the per-layer polygon sets stay on one
  side of `std::sort`'s insertion-sort threshold, and so that the vectors of
  polygons grow in powers of two (Boost's polygon type has no move constructor,
  so a reallocation copies). Other sizes add per-block nonlinearities that drperf
  reports as unexplained. The records name the ones that remain.
- Flags follow the OpenROAD release build: `-O3 -DNDEBUG
  -Wp,-D_GLIBCXX_ASSERTIONS -flto` (C++17 here instead of C++20).

## Catalog entry (not added)

`benchmarks/catalog.json` is not edited. To include the cases, append:

```json
"openroad-drt-gc/drt-gc-001",
"openroad-drt-gc/drt-gc-002",
"openroad-drt-gc/drt-gc-003",
"openroad-drt-gc/drt-gc-004"
```

`benchmarks/tests/test_benchmark.py` asserts the catalog holds exactly 29 cases,
so that number would become 33. The provenance check it runs (`record.md` and
`assets/*` hashes) passes on these directories. `bench.py prepare` has no adapter
for C++ drivers, so the cases stay `reference-only` there.

## License

`src/frBase.h`, `gcShape.h`, `gcPin.h`, `gcNet.h`, `FlexGC.h`, `FlexGC_init.cpp`
and `FlexGC_main.cpp` re-implement code from OpenROAD (BSD-3-Clause, copyright The
OpenROAD Authors). The re-implementation follows the upstream functions closely:
the same containers, Boost.Polygon calls and branch structure. The files keep the
SPDX identifier and copyright line, and the license text is in
`LICENSE.OpenROAD`. The fix patches carry over code and equivalence arguments
from the patch scripts used in the drperf study. `workload.*`, `main.cpp`,
`tests/` and `tools/` are new.
