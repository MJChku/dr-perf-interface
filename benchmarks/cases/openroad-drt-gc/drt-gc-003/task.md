# Net initialization of a DRC worker

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

Region `drt-gc-003` marks the body of `FlexGCWorker::initNets` in
`src/FlexGC_init.cpp`, which builds the pin structures of every net of a worker.
This is a standalone extraction of the DRC worker setup of OpenROAD's detailed
router. The region runs once per worker.

```sh
./build.sh drt-gc-003            # src/ + region.patch -> build/drt-gc-003/neutral/gcbench
build/drt-gc-003/neutral/gcbench scenario=mix size=small
```

Bounded trigger: workers with more or fewer routed nets, some of them nets of the
worker with nothing in its box. Measurements use `scenario=mix size=small`; you
may change `repeat=` and `seed=`. Edit the generator only for observation.
`tests/run_tests.sh drt-gc-003` checks the setup's results and the marker entry.
Keep it passing.
