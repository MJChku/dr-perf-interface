# GPU resource interfaces during GX application execution

Run the actual application's CUDA launches through GX. Keep its `perfmark`
regions and PCVs, or use the `drperf::gpu::Region` class. Do not replace launches
with `drperf_gpu_launch` or load the standalone `libdrperf_gpu.so`.

## Build and use

Build GX's `gx_cuda.so` and Lean translator from the updated GX checkout using
its normal build environment. In DrPerf:

```sh
sh gpu/gx/build.sh
```

The build also provides `build/libperfmark.so` for existing CPU annotations,
without building DynamoRIO. GPU-only users can link
`gpu/build/libdrperf_gpu_gx.so` and use `drperf_gpu.h`'s `Region` class.

When bundled as GX's `tools/drperf` submodule, the normal command is:

```sh
gx explain gpu --gpu H100 --out gpu-report -- python3 app.py
```

GX builds these small libraries on first use, configures the marker module,
runs the application through its existing CUDA path and fits the resulting
observations. Reports are `gpu-report/report/report.txt` and
`gpu-report/report/profile.gpu.drperf.json`; `gx explain report gpu-report`
reads the result. No separate kernel harness is needed.

For development outside GX's command line, this remains available:

```sh
bin/drperf --gpu --gx /path/to/GX/NEX --output gpu-report -- python3 app.py
```

`--gx-library /path/to/gx_cuda.so` selects a separate runtime build. This direct
command places reports immediately inside `gpu-report/`. Both entry points
collect every marked invocation and fit means at distinct PCV states, one PTX
basic block at a time. They do not need DynamoRIO, GPU timing, or real GPUs.
GX functional execution can be expensive; start with small application inputs.

The default enables GX kernel-check execution and real-backed device buffers.
Keep application inputs small enough for host memory. Existing GX settings are
preserved; selecting compute-skipping execution produces explicitly incomplete
resource reports. Chrome event tracing defaults off. Counts are aggregated in
memory per launch/region, not logged per instruction.

## Attribution and checking

- Entry PCVs are copied on the application thread. Kernel submission retains
  the innermost region, including when a stream executes after the region ends.
- A region is emitted only after its host scope closes and every submitted
  kernel completes. Child costs remain separate and use existing composition.
- Stream workers carry the submitted context. Multiple streams and submitting
  threads do not share the host marker stack.
- CUDA graph capture does not count execution. Each replay binds its kernels
  to the region active at replay submission, with fresh PCVs.
- Counter IDs include the translated artifact identity, function and basic
  block. Kernels with the same symbol in different modules cannot alias blocks.
- Prediction-only or uninstrumented kernels, omitted lanes and fake payload
  execution remain visible as unmodeled events. Open regions, cancelled or
  unfinished launches reject the capture.

Memory metrics are **requested PTX bytes**, not measured HBM traffic. Host CUDA
copies are not kernel load/store instructions. The collector fits operations
and bytes, not latency. Identical-state invocation variation remains reported.

## Current coverage

Scalar/vector memory, scalar arithmetic, static-shape tensor arithmetic and
basic-block accounting run on GX's existing translated execution path. This
integration does **not yet connect** all specialized C++ helper counters from
the standalone collector: TMA/bulk-copy descriptors, tcgen05, and WGMMA's shared
operand reads mark their invocations incomplete. They must not appear as fully
explained zero cost. Device-side region markers, unique-address footprints and
lane-span estimates are also not collected by this host-region observer.

Thus this supports end-to-end application launches and repeated observations;
it does not promise complete resource accounting for every serving model yet.
Unsupported PTX still follows GX's normal failure behavior.

## Integration with `gx explain gpu`

The optional C ABI in `gpu/include/gx_resource_observer.h` has three callbacks:
submit on the host thread, count executed resources, finish the launch. GX's
`common/resource_capture.cc` loads the collector only when
`GX_EXPLAIN_GPU_LIBRARY` is set. Its generated counter calls compile out when
capture is off; instrumentation mode is part of the JIT cache key.

The GX command calls `lib/gpu_backend.py`'s `gx_environment()` to compose
its environment, set `DRPERF_GPU_CAPTURE` to an output prefix, run the application,
and call `report_capture()` on the resulting `prefix.<pid>.jsonl` files. Or use
its existing GX launcher and set the two collector variables directly. The
fitter/report format is shared with standalone captures; no new fitting engine
or application harness is necessary.

## Tests

```sh
python3 -m unittest gpu.tests.test_gx -v
GX_HOME=/path/to/GX/NEX GX_TEST_LIBRARY=/path/to/gx_cuda.so \
  python3 -m unittest gpu.tests.test_gx.GXApplication -v
```

The real-GX test is an ordinary CUDA driver application. It checks numerical
outputs, four PCV states with repeated invocations, nested regions, barriers,
two streams, concurrent submitting threads, graph replay, and skipped kernels.
Setting `GX_TEST_PERFMARK=/path/to/libperfmark.so` repeats it through the existing
CPU marker ABI. Observer tests also exercise delayed completion, unequal costs
at identical states, cancelled/pending launches, and unmarked kernels.
