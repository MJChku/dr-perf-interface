# Per-net pin structures in DRC worker setup

Discover the performance-critical variables (PCVs) of this workload. Locate the
regions whose cost they explain and express the PCVs as computations on state
available at region entry. The cost interface is affine in those expressions;
you do not need to supply its coefficients.

Use code inspection and the measurements allowed by your assigned condition.
Choose experiments that distinguish competing explanations, including changes
to independent inputs. Record your current hypotheses before each measurement,
and submit your revised findings after it. Explain remaining uncertainty.

The task is discovery. Preserve program behavior. Changes for observation and
annotations are allowed; performance fixes are outside this task. Do not replace
a PCV with a computation that re-executes the region or measures its cost.

## Target

Region `drt-gc-002` marks the body of `FlexGCWorker::initNet` in
`src/FlexGC_init.cpp`, the four passes that turn a net's shapes into pins, edges,
corners and maximal rectangles. This is a standalone extraction of the DRC worker
setup of OpenROAD's detailed router. The region runs once per net of each worker.

```sh
./build.sh drt-gc-002            # src/ + region.patch -> build/drt-gc-002/neutral/gcbench
build/drt-gc-002/neutral/gcbench scenario=pins size=small
```

Bounded trigger: nets with cell pins that the router reaches through vias, and a
macro pin that this worker does not route. Measurements use
`scenario=pins size=small`; you may change `repeat=` and `seed=`. Edit the
generator only for observation. `tests/run_tests.sh drt-gc-002` checks the
setup's results and the marker entry. Keep it passing.
