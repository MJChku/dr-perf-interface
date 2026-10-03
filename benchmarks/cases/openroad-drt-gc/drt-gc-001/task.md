# Pin corner classification in DRC worker setup

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

Region `drt-gc-001` marks the body of `FlexGCWorker::initNet_pins_polygonCorners`
in `src/FlexGC_init.cpp`. This is a standalone extraction of the DRC worker setup
of OpenROAD's detailed router. The region runs once per net of each worker.

```sh
./build.sh drt-gc-001            # src/ + region.patch -> build/drt-gc-001/neutral/gcbench
build/drt-gc-001/neutral/gcbench scenario=pgvias size=small
```

Bounded trigger: workers holding power/ground rails with rows of fixed vias and
routed vias of the same net. Measurements use `scenario=pgvias size=small`; you
may change `repeat=` and `seed=`. Edit the generator only for observation.
`tests/run_tests.sh drt-gc-001` checks the setup's results and the marker entry.
Keep it passing.
