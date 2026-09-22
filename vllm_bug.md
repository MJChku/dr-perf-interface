# DynamoRIO `-native_exec_list gx_cuda.so` crashes under the GX GPU emulator (blocks fast drperf runs of vLLM + the ditto tier)

Status 2026-09-22: the whole-module native option still fails. The follow-up
below implements and tests a narrower native-work boundary. Earlier runs
reported 19x startup and 41x serving overhead under DynamoRIO and an eight-rank
Kimi-K3 collective timeout. The original attribution of all that slowdown to
emulator threads was incomplete: debugging also found a pathological
DynamoRIO 11.3 `close_range` scan, already fixed in GX's DynamoRIO fork.

## Why we want the option

drperf counts instructions per `perfmark` region. The GX emulator library
`gx_cuda.so` (LD_PRELOADed CUDA driver replacement; its threads interpret the
PTX of real kernels and the NCCL collectives) is already excluded from the
counts (`-exclude_cuda_module gx_cuda.so`) and holds no region, so translating
it through the code cache buys nothing. DynamoRIO's `-native_exec_list` runs
listed modules natively and retakes control when they call back or return;
drperf exposes it as `DRPERF_NATIVE_EXEC_MODULES` (lib/runner.py,
bin/drperf-dev, README "measurement scope"). With GX it crashes, in both attach
modes.

## Where to run it

- Host: `ssh ubuntu@icdslab2.epfl.ch` (128 cores, ~45 GB free RAM; other people's
  containers `nex-mpi-*`, `avadrt`, `orfs_drt` must not be touched; never
  `docker save` there, it filled the disk once).
- Image: `gx-kimi-k3:latest` (id `a5acee962ec2`, built 2026-08-06). Python is
  `/home/jma/bridge-venv2/bin/python` (3.12.3), glibc 2.39, GX at
  `/home/jma/GX` (`tools/gx_run.sh <cmd>` sets the GX environment and
  `LD_PRELOAD=/home/jma/GX/src/sims/gpu/gx_cuda.so`, then execs `gx <cmd>`).
- GX library: the image's own `gx_cuda.so` (Aug 1 build) segfaults with CuPy,
  so a newer build is bind-mounted over it:
  `/home/ubuntu/ditto_kv_gx/gxvm_runtime/gx_cuda_mixedfake.so` (12,164,104
  bytes, md5 `f3ad0d042733...`, GX commit 80e7c88 (Sep 18) plus the
  mixed-payload NCCL patch in `~/GX/NEX/src/sims/gpu/emu/nccl/preprocessing.cc`
  on the controller host). Same crash is expected with any GX build; not yet
  checked with the image's own library.
- drperf: staged at `/home/ubuntu/ditto_kv_gx/drperf` on the host, mounted
  read-only at `/home/ubuntu/drperf` inside the container (the prebuilt
  `build/libdrperf_attach.so` has RUNPATH `/home/ubuntu/drperf/third_party/dynamorio/lib64/release`,
  so the path must be the same). DynamoRIO 11.3.0 build 1 (`third_party/dynamorio`
  -> `DynamoRIO-Linux-11.3.0-1`, only bin64/lib64/ext copied; the "incomplete
  installation" warnings about lib32 are harmless). Client `build/libdrperf.so`,
  markers `build/libperfmark.so`, Python extension `build/_perfmark.so`.

Container used for every run (the ditto tier tree is at `/w`):

```
docker run -d --name drperf-tier --cpus 16 --memory 40g --memory-swap 40g \
  --ipc host --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --cap-add=SYS_ADMIN \
  -v /home/ubuntu/ditto_kv_gx:/w \
  -v /home/ubuntu/ditto_kv_gx/drperf:/home/ubuntu/drperf:ro \
  -v /home/ubuntu/ditto_kv_gx/gxvm_runtime/gx_cuda_mixedfake.so:/home/jma/GX/src/sims/gpu/gx_cuda.so:ro \
  --entrypoint /bin/bash gx-kimi-k3:latest -lc "<command>"
```

For a throwaway shell use `docker run --rm ... -lc "<command>"` with the same
mounts. `--ipc host` is what the GX launcher expects (SysV arena); do not run
two GX jobs at once on the host.

## Minimal reproducer (no vLLM)

Inside the container (`--rm` shell as above):

```
cd /home/jma/GX
export GX_HOME=/home/jma/GX GX_NUM_LOCAL_GPUS=1 GX_DEVICE_MODEL=A100 DRP=/home/ubuntu/drperf
export GX_PTX_CACHE_ROOT=/w/cache/gxptx
export PYTHONPATH=$DRP/perfmark/python:/w/pydeps PERFMARK_LIB=$DRP/build/libperfmark.so PERFMARK_CALIBRATE=1 DRPERF=1
export LD_LIBRARY_PATH=/home/jma/bridge-venv2/lib/python3.12/site-packages/nvidia/cu13/lib
cat > /tmp/mini.py <<'PY'
import perfmark, torch
x = torch.zeros(1024, device="cuda")
with perfmark.region("mini", n=1024):
    y = (x + 1).sum().item()
print("mini ok", y)
PY

# A: instrumented emulator (works; prints "mini ok 0.0", writes /tmp/miniA.<pid>.json)
bash tools/gx_run.sh $DRP/third_party/dynamorio/bin64/drrun \
  -c $DRP/build/libdrperf.so -o /tmp/miniA.%p.json -exclude_cuda_module gx_cuda.so -no_follow_threads \
  -- /home/jma/bridge-venv2/bin/python /tmp/mini.py

# B: emulator native (crashes; rc=139)
bash tools/gx_run.sh $DRP/third_party/dynamorio/bin64/drrun -native_exec_list gx_cuda.so \
  -c $DRP/build/libdrperf.so -o /tmp/miniB.%p.json -exclude_cuda_module gx_cuda.so -no_follow_threads \
  -- /home/jma/bridge-venv2/bin/python /tmp/mini.py
```

B's only output (GX's own crash handler, `~/GX/NEX/src/sims/gpu/interpose.c`
line 57 is the `LOGE` in `crash_handler`, it does not name the faulting code):

```
Caught signal 11 (Segmentation fault) at address 0xc33008	in /interpose.c:57
```

The handler writes `crash_trace.log` (backtrace) into the process CWD
(`/home/jma/GX`); in the vLLM run that file held only the signal line, no
frames. `interpose.c` also interposes `dlopen`/`dlclose`/`sigaction`
(`dlopen_orig`, `dlclose_orig`, `sigaction_orig`), which is a plausible place
for a native/instrumented transition to go wrong.

## The vLLM reproduction (what we actually need)

Harness (in the ditto tier repo, `/home/ubuntu/compression/ditto_kv` on the
controller host, synced to `/home/ubuntu/ditto_kv_gx` = `/w` on icdslab2):

- `exp/gx_kimi/drperf_offline.sh`: in-container driver. `PRESET=qwen`
  (Qwen2.5-0.5B, one emulated A100, one process: vLLM `LLMEngine` with
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`) or `PRESET=kimi` (Kimi-K3, 8 emulated
  H200s, 8 spawned TP workers). `DRPERF_MODE=late|early|native`.
  `DRPERF_DR_OPTS` are extra DynamoRIO core options.
- `exp/gx_kimi/drperf_serve.py`: builds the DynamoRIO command/environment
  (late: `LD_PRELOAD=libdrperf_attach.so ...` + `DYNAMORIO_OPTIONS`; early:
  `drrun ... -- python drperf_offline.py ...`).
- `exp/gx_kimi/drperf_offline.py`: the workload (sessions x turns of prefix
  reuse through `engine.add_request`/`engine.step`).

Failed run 1, late attach + native exec
(`/home/ubuntu/ditto_kv_gx/runs/drperf-offline-qwen-late6-nativeexec-20260922-205456/{driver,serve,server}.log`):

```
RUN=/w/runs/drperf-offline-qwen-late6-nativeexec-20260922-205456 PRESET=qwen DRPERF_MODE=late \
DRPERF_DR_OPTS="-no_follow_children -takeover_attempts 50 -takeover_timeout_ms 60000 -unsafe_ignore_takeover_timeout -native_exec_list gx_cuda.so" \
bash /w/exp/gx_kimi/drperf_offline.sh
```
which ran the engine with
```
LD_PRELOAD=/home/ubuntu/drperf/build/libdrperf_attach.so /home/jma/GX/src/sims/gpu/gx_cuda.so
DRPERF_LATE=1
DYNAMORIO_OPTIONS=-code_api -no_follow_children -takeover_attempts 50 -takeover_timeout_ms 60000 -unsafe_ignore_takeover_timeout -native_exec_list gx_cuda.so -client_lib '/home/ubuntu/drperf/build/libdrperf.so;0;-o /w/runs/.../raw/run.%p.json -blocks -max_slots 2097152 -exclude_cuda_module gx_cuda.so -no_follow_threads'
```
The engine booted natively (26 s), then at the first region (DynamoRIO attach
via `dr_app_setup_and_start` from `libdrperf_attach.so`):
```
<Application /usr/bin/python3.12 (18).  drperf internal crash at PC 0x0000000071076199.  Please report this at https://github.com/DynamoRIO/dynamorio/issues.  Program aborted.
Received SIGSEGV at pc 0x0000000071076199 in thread 60
Base: 0x0000000071000000
Registers:eax=0x00000000ffffffff ebx=0x00007ff6c904fac0 ecx=0x0000000000000000 edx=0x0000000000000000
	esi=0x00007ff6c904fa00 edi=0x0000000000000004 esp=0x00007ff6c904f080 ebp=0x000000007120ac70
	r8 =0x00007ff6c9036480 r9 =0x0000000000000008 r10=0x0000000000000008 r11=0x0000000000000246
	r12=0x00007ff6c904f100 r13=0x00007ff6c904fab8 r14=0x00007ff6c904fbe8 r15=0x00007ff6c88e0100
	eflags=0x0000000000010246
version 11.3.0, build 1
```
(PC is inside `libdynamorio.so`, base 0x71000000; thread 60 is not the main
thread, i.e. one of the emulator's or torch's threads being taken over.)

Failed run 2, injection at start + native exec
(`/home/ubuntu/ditto_kv_gx/runs/drperf-offline-qwen-early-nativeexec-20260922-205850/`):
```
RUN=... PRESET=qwen DRPERF_MODE=early DRPERF_DR_OPTS="-no_follow_children -native_exec_list gx_cuda.so" bash /w/exp/gx_kimi/drperf_offline.sh
```
i.e. `setarch -R drrun -no_follow_children -native_exec_list gx_cuda.so -c libdrperf.so -o run.%p.json -blocks -max_slots 2097152 -exclude_cuda_module gx_cuda.so -no_follow_threads -- python drperf_offline.py ...`.
Segfault within a minute, before any region, same signature as the minimal
reproducer (`Caught signal 11 ... in /interpose.c:57`, rc = -11).

## What works (for comparison)

- Late attach with the takeover options but without native exec: Qwen boots in
  25 s natively, the 48-request workload takes 153 s under DynamoRIO (3.7 s
  native), counts are written at exit (`run.18.json`), five clean runs today.
- Injection at start without native exec: Qwen boot 484 s under DynamoRIO,
  workload 22 s (this run had `-no_follow_children`; the emulator ran inside
  the code cache the whole time).
- Late attach on Kimi-K3 (8 workers) hangs (workers stall at the first request
  with vLLM's shm broadcast timing out), so Kimi uses injection at start:
  boot 29 min with 48 CPUs; then the collective watchdog fired
  (`WorkNCCL(SeqNum=590, OpType=ALLREDUCE, NumelIn=3584) ran for 600072 ms`).

## What to look at

- Whether DynamoRIO's native-module gateways interact badly with GX's
  interposition of `dlopen`/`dlclose`/`sigaction` or with the emulator's own
  signal handler (`crash_handler` installs itself via `sigaction`; GX likely
  wraps `sigaction` for other reasons too).
- Get a real backtrace: `crash_trace.log` in `/home/jma/GX` inside the container
  (the container filesystem survives until `docker rm`; `docker cp
  drperf-tier:/home/jma/GX/crash_trace.log .`), or run drrun with
  `-msgbox_mask 15`/a debug build with `-loglevel 3` on the minimal reproducer.
- Variants worth a 2-minute test on the minimal reproducer: `-native_exec_list`
  with `-no_native_exec_opt`, `-no_native_exec_dircalls`,
  `-no_native_exec_callcall`, `-native_exec_syscalls`; listing only the
  emulator's internal worker-thread entry module if it is separate; DynamoRIO
  11.x latest release instead of 11.3.0.
- If native exec cannot be made to work, the measurement has to be sized for
  the slowdown: `--distributed-timeout-seconds 7200` and 2,048-token steps for
  Kimi (prepared in `drperf_offline.sh`, PRESET=kimi), roughly 2 to 3 hours per
  run.

## Debugging follow-up: explicit native-work boundary (2026-09-22)

The launch chain above is **plain GX functional emulation, not GXVM**:
`gx_run.sh -> gx -> drrun -> Python`. The DynamoRIO instance belongs to drperf.
There is no timing simulator to disable in this experiment.

Reproduced in `drperf-gx-debug` on icdslab2 using the same image and mounted GX
library as above. The ordinary minimal run completes in 39.130 s and records
23 marker triggers, including `mini`. Whole-module native execution crashes
before `import torch` finishes. Disabling native syscall interception, direct
call gateways, or call-call gateways does not fix it. A debug build also fails.

GDB catches the actual failing instruction in the **application libc allocator**,
not the GX crash handler: libc offset `0xaca29`, `mov %rax,0x8(%r9)`, with
`r9=0xc33000`; the fault is at `0xc33008`. The process maps show a hole from
`0xc33000` to `0xc3d000`. This and the following experiment implicate the
interaction between native execution and DynamoRIO's early-injection `brk`
emulation; they do not constitute a complete upstream root-cause fix.

Adding `-no_emulate_brk` avoids that crash and prints `mini ok` in 15.243 s,
but the report has **`marker_seen=false`, zero trace records, and no regions**.
DynamoRIO lost application coverage. That apparent speedup is invalid. The
combination `-native_exec_opt -native_exec_retakeover` also crashes. Do not use
successful process exit as the acceptance test for native-module execution.

The narrower implementation in `client/drperf.c` adds `-native_gx` (runner:
`DRPERF_NATIVE_GX=1`, requiring `DRPERF_EXCLUDE_CUDA_MODULE=gx_cuda.so`). It uses
`drwrap_replace_native` only at GX's exported `gxvm_gpu_native_run` entry.
The replacement switches to application state, executes the original entry,
restores DynamoRIO state, and uses `drwrap_replace_native_fini` to resume
instrumentation. It marks blocking native work safe to suspend. The loader,
CUDA dispatch, Python, and the Rust ditto FTL remain instrumented. This borrows
GX's existing boundary ABI; it does not load GXVM or enable timing simulation.

Validation so far:

- A two-thread synthetic emulator exercises the real callback, heap allocations,
  argument forwarding, and marked host work after every native call. Early and
  late attachment both pass; host instruction counts match with native work
  enabled and disabled. Missing configuration is rejected.
- The real PyTorch mini run completes in 39.717 s, retaining all 23 triggers.
  The boundary is installed but `native_gx_calls=0`: these compute kernels do
  not exercise the native NCCL boundary, so this is a coverage check, not a
  speedup result.
- `exp/gx_kimi/drperf_serve.py` in the ditto repository accepts
  `DRPERF_NATIVE_GX` and an explicit `DRPERF_CLIENT` override for testing a client
  without replacing a shared installation. It also rejects an otherwise
  successful run with no application regions or invalid counter metadata.

The integration results and runtime diagnosis are recorded below. Whole-module native execution remains unsupported for this GX
configuration.

### A separate startup bottleneck: DynamoRIO's descriptor scan

The early-injection Qwen run on stock 11.3 spent minutes in subprocess creation.
GDB attached to a child found:

```
handle_close_generic_pre(fd=240987821, set_return_val=false)
handle_close_range_pre(fd=240987821)
pre_system_call
handle_system_call
```

The parent was waiting in `kernel_clone`. These are possible descriptor numbers,
not hundreds of millions of open files. The old runtime scans the entire numeric
range when Python closes descriptors before exec. Stopping this exploratory run
required terminating the test container; no usable measurement was retained.

GX's DynamoRIO fork already contains `90b6203` (sparse `close_range` handling)
and `697b30c` (TLS-safe asynchronous takeover). A separately staged GX DynamoRIO
11.91 runtime was used with a rebuilt **drperf** client; no GXVM client runs.
The client now supports both old core event registration and newer drmgr-owned
exit/syscall-filter registration. All 18 drperf tests pass against both runtimes.
`DRPERF_OUTPUT_DIRECTORY` in CMake allows the alternate client to be built without
replacing the existing client. Runner/harness overrides are `DRPERF_DRRUN`,
`DRPERF_CLIENT`, and `DRPERF_ATTACH`; the attach library must link to the selected
runtime as well.

### Real translated NCCL with host coverage preserved

`tests/gx_nccl_probe.c` uses two simulated A100s in one process, initializes
rank-dependent data on the host, and checks all-reduce results against host
expectations for 4, 16, 64, 256, and 1,024 floats. It then performs marked CPU
loops after each collective. Both configurations use the same GX library and
patched DynamoRIO runtime, with no GPU and no timing simulation.

| configuration | observed process time | native boundary calls | host counts after NCCL |
|---|---:|---:|---|
| fully instrumented emulator | 2.179 s | 0 | 35, 107, 395, 1,547, 6,155 |
| `-native_gx` | 1.720 s | 10 | 35, 107, 395, 1,547, 6,155 |

Every payload value matched in both runs. Both reports retain all ten region
triggers, with no slot overflow, denied counters, unmatched markers, or dropped
trace records. These are single small-probe timings, not an eight-rank Kimi
performance result. The key validation is exact host-count agreement after
native work and correct nonzero NCCL payloads.

### vLLM + ditto-ftl: verified early-attach coverage

The full Qwen2.5-0.5B workload completed under GX mode 0 using dummy weights
and the existing `DITTO_FRAMEWORK_TEST=1` ratio-controlled codec output. This
checks execution and CPU coverage, not model or compression numerics.

- 48 requests completed, with zero failed requests; their summed latency under
  instrumentation was 22.3 s. The engine's initialization timer reported 89.1 s
  (excluding Python imports before that timer). The old-runtime exploratory run
  was stopped in the descriptor scan, so there is no matched end-to-end speedup
  estimate from that run.
- 103 application regions and 14,792 application-region calls were captured,
  excluding calibration and `perf.*` annotation helpers. Including those helpers,
  the report contains 107 region names and 20,269 triggers.
- `_ditto_ftl_core.abi3.so` contributes 29,035,972 instructions in application
  regions (33,221,580 when annotation helpers are included).
  `ftl.bind_store_batch` ran 23 times, `ftl.plan_load` 35 times,
  `gc.prepare_compaction` 14 times, and `ftl.adopt` 14 times.
- The report has zero slot overflows, denied counters, unmatched ends, state
  overflows, depth overflows, and dropped trace records. No `gx_cuda.so`
  instructions are attributed to the regions.
- Measurement coverage does not mean complete cost explanation. In particular,
  59 `rt.query` calls are recorded but omitted from its fitted state table because
  load-path annotations omit the `released` PCV present on its store path. This
  is an annotation-schema mismatch, not the state limit. The selected FTL report
  also preserves unexplained costs and insufficient-point diagnostics.
- The native boundary is installed but has zero calls for this single-rank
  workload. Its correctness and execution benefit are established separately
  by the two-rank NCCL probe, not by this Qwen run.
- Strict late attachment with the same runtime still fails to take over all
  threads (exit 40, no reports). The harness no longer automatically adds
  `-unsafe_ignore_takeover_timeout`; early attachment is the validated route.
  Eight-rank Kimi has not been rerun with this combination.

The harness also had a non-daemon 180-second shutdown timer that kept a completed
run alive. It is now a daemon, retaining the fallback exit without creating the
wait itself. The early run above loaded the old timer before this change; its
launch-to-exit wall time includes that shutdown delay and report writing, and
must not be treated as inference latency. The shell driver now propagates the
measurement process's failure code.

Artifacts on `ubuntu@icdslab2.epfl.ch`:

```
/home/ubuntu/ditto_kv_gx/runs/drperf-gx-boundary-qwen/
  summary.json              # counts, validity metadata, binary hashes
  report.txt                # selected FTL/compaction interfaces and unexplained costs
  raw/run.151.json{,.blocks,.slots,.trace}
  client.json               # all 48 requests
  native-debug.tar.gz       # minimal/GDB/NCCL probes and their raw reports
  late/                    # strict takeover failure, no usable measurement
```

Reproduce inside the container with the mounts documented above:

```bash
RUN=/w/runs/drperf-gx-boundary-qwen-repeat \
PRESET=qwen DRPERF_MODE=early DRPERF_NATIVE_GX=1 \
DRPERF_CLIENT=/w/gxvm_runtime/libdrperf_gxdr.so \
DRPERF_DRRUN=/w/gxvm_runtime/drperf-dynamorio/bin64/drrun \
DRPERF_ATTACH=/w/gxvm_runtime/libdrperf_attach_gxdr.so \
bash /w/exp/gx_kimi/drperf_offline.sh
```

The directory name `gxvm_runtime` is historical staging only. The process loads
GX, the DynamoRIO core/extensions, and the drperf client; it does not load or
run a GXVM timing client. The exact staged runtime SHA-256 is
`3f3d4a3f84e4dfbe8c4905dec081a3a9a36f620dc429e4236cffbe85440a684e`.

## Strict late attachment fixed (2026-09-23)

The late-attach blocker was a missing runtime option in the drperf harness,
not an inherent inability to attach to GX/vLLM. GXVM already supplies
`-attach_unmask_suspend_signal` in `NEX/tools/gxvm/gxvm` and its timeline
launchers. Reusing the newer DR runtime without this option was insufficient.
Immediately before attachment, the diagnostic found 189 existing threads:
`jemalloc_bg_thd` alone blocked SIGILL (`SigBlk=fffffffe7ffbfeff`). SIGILL is
DR's takeover signal. Ptrace-assisted unmasking allowed strict attachment in
0.309166 seconds; no unsafe-ignore option was used.

The Qwen harness now defaults to late attachment, enables
`DRPERF_ATTACH_UNMASK_SIGNAL=1`, and prefers the staged matching newer
DR/client/attach bundle when present. Explicit runtime overrides win. Kimi TP8
continues to default to early attachment until separately validated.
The general drperf runner accepts the same opt-in environment variable.
Stock DR 11.3 does not implement this runtime option; ptrace must be permitted.
No GXVM timing client runs in these experiments, and no GX runtime source
change was necessary for this fix.

Three full Qwen runs completed 48 requests each with zero failures:

| Run | Engine init | Summed requests | Whole GX launcher wall time |
|---|---:|---:|---:|
| late-unmask | 9.7 s | 24.6 s | 77.699 s |
| late-confirm (diagnostic attach wrapper) | 9.9 s | 24.4 s | 77.553 s |
| late-default (updated harness defaults) | 9.8 s | 24.4 s | 77.656 s |

Earlier early-attach engine initialization took 89.1 s. That earlier run's
whole-process time also included a subsequently fixed watchdog delay, so it
must not be used as a controlled end-to-end speedup comparison. Request times
are not improved by late attachment; startup is the saving.

The `late-unmask` report has all 103 application region names and exactly the
same per-region call counts as early attachment (14,792 application calls;
20,269 total triggers including helpers). It counts 29,045,070 Rust FTL
instructions in application regions. There are no validity errors, overflows,
unmatched ends, or dropped triggers. It uses 191,651 basic-block slots versus
1,509,956 for the earlier early-attach run. These are workload coverage checks,
not a claim that every cost has an accepted affine formula.

Artifacts on icdslab2:
`/home/ubuntu/ditto_kv_gx/runs/drperf-gx-boundary-qwen/{late-unmask,late-confirm}/`.
`late-unmask/summary.json` records region counts, module totals, and validity.
`late-confirm/server.log` records the pre-attach blocked mask and attach duration.
`late-default/` validates the updated harness without explicit mode/runtime
option overrides. The temporary diagnostic wrapper remains inside the debug
container only; production uses the unmodified attach library.

Regression: `tests/late_blocked.c` starts two workers before the first marker,
one blocking all maskable signals. Without unmasking, strict late attach
fails. With unmasking, both workers' four measured states have exactly the same
instruction counts as early attachment. `tests/test_late_blocked.py` passes
against the newer runtime and explicitly skips the coverage test on stock
11.3, which lacks the option.

## Bundled runtime and automatic GX handling (2026-09-23)

`./build.sh` now builds the GX DynamoRIO fork at
`7c0717f412f8aae9199b49aff95557ada2cf3413` in `third_party/`, replacing the
11.3 release as the default. It fetches the pinned source and submodules from
GitHub, not from a local GX worktree. The attach library uses a relative runtime
search path so this bundle can be staged together. Client build failures now
propagate rather than being hidden by a shell output filter.

Signal unmasking is enabled by default. Loading `gx_cuda.so` automatically
installs CUDA counting exclusion and the `gxvm_gpu_native_run` native boundary,
including when a launcher loads GX after drperf starts. These defaults do not
run GXVM timing. Loader/dispatch stubs still run through DynamoRIO; this is not
whole-module native execution. `DRPERF_NATIVE_GX=0` disables automatic handling.

Validation of the newly built bundle:

- All 21 drperf unit/integration tests passed, including blocked-thread takeover,
  automatic GX detection in early and late modes, native worker execution,
  identical host counts after native return, PCV expansion, and state budgets.
- The fork's `gxvm_close_range.py` regression passed with 0 and 128 existing
  threads (ten subprocess launches each, 0.18 and 0.19 seconds).
- Full Qwen/ditto under functional GX: 48 requests, no failures, engine init
  9.7 s, summed requests 24.3 s. The report has all 103 application regions and
  14,792 calls, 29,040,265 Rust FTL instructions, and no validity errors.
  `native_gx=true` and one hook are detected automatically; TP1 makes zero native
  boundary calls, so the NCCL test below separately validates native execution.
- Real GX/NCCL integration, two simulated ranks: all-reduce values correct at
  4/16/64/256/1024 floats, ten native boundary calls. Host counts after NCCL are
  exactly 35/107/395/1547/6155 in both instrumented and native modes. Single-run
  wall times were 2.219 s and 1.679 s, respectively, not a controlled benchmark.

The full application evidence is on icdslab2 under
`/home/ubuntu/ditto_kv_gx/runs/drperf-gx-boundary-qwen/bundled-auto/`.
The NCCL probe is checked in as `tests/gx_nccl_probe.c`; its raw validation files
are in the debug container's `/tmp/drperf-debug/nccl-bundled-{auto,control}.*`.
