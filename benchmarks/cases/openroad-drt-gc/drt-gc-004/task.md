# Routing-worker nets in DRC worker setup

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

Region `drt-gc-004` marks the body of `FlexGCWorker::initDRWorker` in
`src/FlexGC_init.cpp`. It gives each net of the detailed-routing worker a gc net
and adds that net's routed shapes. This is a standalone extraction of the DRC
worker setup of OpenROAD's detailed router. The region runs once per worker.

```sh
./build.sh drt-gc-004            # src/ + region.patch -> build/drt-gc-004/neutral/gcbench
build/drt-gc-004/neutral/gcbench scenario=drnets size=small
```

Bounded trigger: routing workers whose nets are partly known from the design and
partly new, some with nothing in the box. Measurements use
`scenario=drnets size=small`; you may change `repeat=` and `seed=`. Edit the
generator only for observation. `tests/run_tests.sh drt-gc-004` checks the
setup's results and the marker entry. Keep it passing.
