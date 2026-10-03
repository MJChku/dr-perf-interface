# Causal video inference: GX screening and drperf

This study runs pretrained causal-video models through Inferix, FastVideo,
LightX2V, FlashDreams, Causal Forcing++, Matrix-Game 3, Scope, StreamDiffusionV2, TeleFuser, SkyReels-V2, DiffSynth-Studio, VideoX-Fun, WanVideoWrapper, Wan2GP, and vLLM-Omni, profiles their kernels on an A100,
then screens framework overhead with GX
`partial_sync` and drperf in separate runs. It uses the existing Torch 2.11 / CUDA
12.8 environment. Source and model revisions are pinned in `pins.json`.
The CPU search also covers Qwen3-TTS, VoxCPM2, Chatterbox Turbo, Kokoro, F5-TTS, Soprano, ZipVoice, MeloTTS, Marvis, Dia and Outlines.
After kernel calibration, discovery and candidate evaluation use GX; real GPUs
are used again only to validate selected candidates.
CosyVoice2 adds a modest, validated autocast-cache result with an explicit GPU
memory tradeoff; its CPU dispatch and GPU conversion work both change.

## Recorded CPU optimization results

GX predictions and physical A100 validation are recorded separately below.
These rows use different frameworks and offload policies; compare before/after
within a row, not performance across frameworks. Each linked case records its
source/runtime identities, all native samples, numerical checks, memory use,
compact traces, and prediction limitations.

| CPU change | GX mode 2, before -> after | GX mode 3, before -> after | A100 median, before -> after |
| --- | ---: | ---: | ---: |
| [VideoX-Fun: upload scheduling and allocator reuse, 8 steps](#videox-fun-cpu-scheduling-during-sequential-offload) | 13.789 -> 6.866 s | 14.491 -> 7.563 s | 12.578 -> 7.590 s |
| VideoX-Fun: same change and database, 50 steps | 70.000 -> 31.347 s | 74.009 -> 35.461 s | 59.188 -> 35.051 s |
| [WanVideoWrapper: cache CPU weight casts](#wanvideowrapper-repeated-cpu-checkpoint-conversions) | 6.445 -> 5.791 s | 7.010 -> 6.349 s | 10.030 -> 9.648 s |
| [DiffSynth: asynchronous uploads, over the normalization fix](evidence/diffsynth-cpu.json) | 8.057 -> 7.266 s | 8.660 -> 7.796 s | 9.758 -> 9.232 s |
| [Wan2GP: skip redundant empty-model unload](#wan2gp-redundant-garbage-collection-mixed-native-timing-result) | 5.454 -> 5.277 s | 6.640 -> 6.483 s | 5.733 -> 5.543 s |
| [vLLM-Omni: batch component weight uploads](#vllm-omni-cpu-waits-between-component-weight-uploads) | 3.709 -> 3.422 s (worker) | 4.182 -> 3.875 s (request) | 3.900 -> 3.793 s |
| [Qwen3-TTS: avoid redundant single-position checks](#qwen3-tts-cpu-control-checks-during-autoregressive-decoding) | 13.393 -> 12.625 s | 17.425 -> 16.124 s | 8.304 -> 7.557 s |
| [Chatterbox Turbo: fused activation dispatch and CPU inference mode](#chatterbox-turbo-cpu-dispatch-and-watermark-autograd) | 1.141 -> 1.015 s | 1.392 -> 1.207 s | 0.649 -> 0.560 s |
| [Kokoro: prepare inference weights and CPU length metadata](#kokoro-inference-preparation-and-cpu-length-metadata) | 85.918 -> 82.764 ms | 115.175 -> 110.320 ms | 36.290 -> 32.902 ms |
| [F5-TTS: reuse rotary calculations and specialize solver checks](#f5-tts-rotary-reuse-and-fixed-grid-solver-checks) | 0.994 -> 0.974 s | 1.308 -> 1.265 s | 0.602 -> 0.582 s |
| [Soprano: scope generation with inference mode](#soprano-cpu-bookkeeping-during-autoregressive-generation) | 0.957 -> 0.904 s | 1.196 -> 1.148 s | 0.534 -> 0.477 s |
| [MeloTTS: avoid per-sample Python waveform objects](#melotts-cpu-waveform-concatenation) | 69.984 -> 52.352 ms | 90.045 -> 72.370 ms | 40.485 -> 34.450 ms |
| [Marvis: CPU cache-position bookkeeping](#marvis-cpu-cache-position-bookkeeping) | 4.867 -> 4.685 s | 8.952 -> 8.206 s | 2.878 -> 2.718 s |
| [Outlines: opt-in CPU inference bookkeeping](#outlines-cpu-bookkeeping-in-structured-generation) | 1.493 -> 1.387 s | 2.690 -> 2.208 s | 0.972 -> 0.871 s |
| [CosyVoice2: autocast cache, CPU dispatch and GPU casts](#cosyvoice2-autocast-caching-a-modest-native-improvement) | 7.347 -> 6.977 s | 10.426 -> 9.528 s | 4.614 -> 4.526 s |

**Wan2GP has a mixed native result:** its mean worsened from 5.717 to 5.983 s,
including a retained 9.987 s candidate sample; the median alone does not
establish a mean-latency improvement. DiffSynth's row uses the fixed repeat
comparison; its earlier mixed result is also retained. All paired measured native
outputs in these cases match. Qwen3-TTS had one differing cold baseline warmup
hash, retained separately from its 20 matching measured outputs. GX generally overestimates the selected gains;
predicted reductions are not substitutes for the native measurements.

The [native Inferix comparison](NATIVE_RESULTS.md) measures 41.29 → 12.65 seconds
with byte-identical saved output after selecting GPU KV residency and removing
a discarded decode. A valid baseline drperf instruction summary is recorded
in [the native results](NATIVE_RESULTS.md). With [GX copy timing](INFERIX_COPY_TIMING.md),
the host-KV baseline models 38.418 seconds, versus 41.291 seconds on A100. The
earlier 12.268-second estimate omitted copy service. A [bounded KV-transfer
candidate](INFERIX_KV_TRANSFER.md) retains offload and reduces host/device traffic
by 63.54%, giving **22.307 seconds in GX (41.94% lower)**. Its 144 paired CPU
cache-equivalence checks pass. Native A100 medians are **41.279 -> 25.292 seconds
(38.73% lower)** with exactly matching saved video tensors and unchanged offload.
The baseline/candidate retain 168/42 cuDNN prediction misses, respectively.
A native result or kernel database alone is not evidence of a successful GX
timing simulation.

The [matched FastVideo native comparison](FASTVIDEO_NATIVE_RESULTS.md) found
no convincing gain from resident DiT weights: median latency changed from
15.424 to 15.360 seconds, with 2.697 GB more peak allocated GPU memory.
Its [valid drperf collection](FASTVIDEO_DRPERF.md) measures 8.63 billion marked
host instructions; existing RoPE table caching already handles this workload.
The [final FastVideo partial run](FASTVIDEO_PARTIAL_RESULTS.md) models 97.39%
GPU busy time, but 84 cuDNN launch mismatches prevent a matched replay claim.
The [native GPU databases and reports](evidence/kernel-databases/README.md)
are included as small compressed artifacts.

## LightX2V: GX discovery, drperf interface, A100 validation

The [harness](lightx2v_runner.py) runs the public Self-Forcing pipeline with
21 pixel frames (two causal chunks), block weight offload, KV offload, FA2,
and asynchronous VAE decoding. Offloading is an explicit workload choice;
the upstream Self-Forcing configuration defaults to resident weights/cache.
GX replays calibrated scheduler decisions and text lengths while retaining
the operations that produce them; it does not validate GPU arithmetic.

GX's copy inventory identified full-capacity KV uploads, including unused
slots. [Loading only valid entries](cpu_patches/lightx2v_valid_kv.patch) removes
6.04 GB H2D (35% of KV uploads), retaining the offload buffers and CUDA events.
A separate drperf run then identified 162 million CPU instructions spent
clearing the cache at request startup. Full-model GX/drperf requests with
21/45/69 frames accept this interface for the fixed 30-layer BF16 configuration:

```text
reset_instructions = 0.09375 * cache_bytes + 335369
```

The [reset patch](cpu_patches/lightx2v_reset.patch) retains synchronization and
metadata reset while removing payload clearing. Reset work changes from
162.076/323.817/485.557 million instructions to 307782/307067/307266 at the three
capacities. The residual fitted slope is negligible (-2.78e-8 instructions per
byte); all payload-fill instructions disappear. Both interfaces have zero
unexplained share within the per-block tolerance at the observed states.

| Candidate, 21-frame request | GX virtual seconds | A100 median seconds |
|---|---:|---:|
| Original | 4.178 | 5.691 |
| Valid-entry loading | 4.129 | 5.237 |
| Valid-entry loading + metadata-only reset | 4.042 | 5.193 |

The selected combination lowers native latency by **8.75%** against the original
reference. Reset alone adds **0.83%** over valid-entry loading. Each native
measurement uses three warmed generations. Final output hashes match exactly,
and peak allocated memory remains 7,404,322,816 bytes. The [full reports](evidence/kernel-databases/lightx2v-validation.json.gz)
and [GX/drperf evidence](evidence/lightx2v-screen.json) retain individual timings,
source identities, launch audits and trace paths. Baseline/final GX have six
cuDNN prediction misses each; model timing is not a matched native replay.
Independent copy-engine contention is not modeled. The timing discrepancy is
retained rather than treated as validated prediction accuracy.

A [deferred-writeback experiment](cpu_patches/lightx2v_kv_commit.patch) removed
another 6.90 GB D2H (80% of KV writeback), but its native median was 5.282 s,
versus 5.237 s for valid-entry loading. It is **excluded from the selected
candidate**: less traffic did not establish a latency improvement.

The CPU oracle passes 116,640 attention-tensor comparisons, including ring
wraps, fixed sinks, repeated writes, request resets, step isolation, and
head-shard shapes. It executes the source cache methods with CPU backing:

```bash
python3 examples/causal_video/validate_lightx2v_kv.py out/causal-video/LightX2V
```

Distributed execution is not validated. The CPU oracle does not check CUDA
scheduling; native generation provides the separate numerical check. The GX
cuDNN issue records the stream-query repair needed by asynchronous VAE decoding
and remaining launch mismatches: `GX/NEX/issues/cudnn-emulation-video-inference/`.
For 81 frames (seven chunks), reusing the same kernel database gives **24.61 →
17.19 virtual seconds (30.16% lower)** and **96.61 GB fewer H2D bytes**. GX wall
time for the measured generation is 26.60/24.30 seconds, excluding load/warmup.
Copy-only modeled time falls from 16.07 to 9.05 seconds. This larger pair has
27/88 prediction misses, including larger-shape kernels and differing cuDNN
plans. A subsequent A100 check measures **31.051 → 23.666 seconds (23.78%
lower latency)** across three warmed generations per version. The final video
hashes match exactly, and peak allocated GPU memory stays at 7,841,622,016
bytes. GX predicts 7.422 seconds saved; native CUDA measures 7.385 seconds
saved. Absolute GX latency remains optimistic. No new kernel profile was
collected for this longer request.

For this fixed model/shape, let C be the number of causal chunks and B =
862,617,600 bytes be one chunk of K/V across all layers. The original uploads
`5*B*C*C` bytes; valid-entry loading uploads `B*(5*C*C + 3*C)/2`. These formulas
match both GX inventories. They describe transfer bytes derived from the cache
schedule, separately from the drperf instruction interface above.

## FlashDreams: CPU initialization, confirmed on A100

The default Self-Forcing 1.3B pipeline completes nine chunks (105 frames) under
GX on the existing Torch stack, with compilation and CUDA graphs enabled.
drperf found redundant CPU weight conversions when the text encoder reloads
at request initialization: the loader defaults to FP32, then FlashDreams converts
the model to BF16. Loading directly in the requested dtype removes **69.1% of
marked instructions**, from 29.81 billion to 9.22 billion, while preserving the
encoder release/reload policy.

Across 242 tensors and five observed sizes, the original weight-materialization
interface is approximately `1.5625 * elements + 86082` instructions, with 3.1%
maximum unexplained cost. Its total drops from 8.90 billion to 19.5 million
instructions with the [opt-in patch](cpu_patches/flashdreams_encoder_dtype.patch).
The remaining fitted size coefficient is small and negative, not literally zero.
Other regions are not yet fully explained. The
[CPU parity check](validate_flashdreams_dtype.py) passes for FP32/BF16 checkpoint
and requested-dtype combinations.

Separate A100 GPU1 confirmation measures **28.668 -> 18.000 seconds median
(37.2% lower latency)** over three warmed requests per version, following two
warmups. The final video hashes match exactly. Peak allocated GPU memory remains
26,697,003,008 bytes. The only configuration change is opting into direct-dtype
loading: encoder release/reload, model settings, compilation, CUDA graphs,
prompt, seed, and output transfer are preserved. Generation includes request
initialization and CPU output, and excludes file encoding. This is one prompt,
seed, and model/shape; it is not a production workload distribution.

A second [opt-in patch](cpu_patches/flashdreams_tokenizer_retention.patch), applied
on top of direct-dtype loading, retains one **CPU tokenizer per pipeline** while
still releasing encoder weights. It removes repeated loading/construction of a
256,300-entry tokenizer and its associated cleanup work. Marked instructions fall
from 9.22 billion to 4.22 billion (54.2%); the tokenizer-loading regions no longer
run during the warmed request. This is a fixed-vocabulary stage measurement,
not a discovered growth formula over vocabulary size.

A separate matched A100 pair measures **17.979 -> 16.787 seconds (6.6% lower
latency)**. Final hashes and GPU peak memory again match exactly. The
[lifecycle check](validate_flashdreams_tokenizer.py) verifies exact embeddings,
weight release, pipeline isolation, model-path invalidation, and cleanup when
retention is disabled or the pipeline is deleted. Retention assumes immutable
tokenizer assets at a given path and intentionally keeps CPU memory alive. Enable
it with `--retain-cpu-tokenizer`; direct-dtype loading uses
`--load-in-requested-dtype`. Both options default off.

For tokenizer retention, GX reuses the corrected kernel database and aggregate
CPU calibration: **8.572 -> 7.390 virtual seconds**, predicting **1.182 seconds
saved**, versus **1.192 seconds** measured on A100. The GX before/after launch
name/shape/count inventories match. Compute prediction coverage remains 52.4%,
so agreement in this initialization-cost saving does not validate absolute GX
latency. No further GPU kernel profiling was performed for retention.

Across the two separately paired experiments, the final median is **16.787
seconds versus the original 28.668 seconds (41.4% lower)**. These are warmed
requests: initial tokenizer construction still happens once per pipeline.

GX supplied the execution environment for discovery and the instruction checks.
Its first partial timeline exposed initialization headroom, but the numerical
115.005 -> 7.462 second result is **rejected**: kernel-space instruction samples
distorted per-module CPU calibration, and compute prediction coverage was only
46.7%. The follow-up gives **11.383 -> 8.646 virtual seconds** using aggregate CPU
calibration and a corrected kernel database, at roughly 52.4% prediction coverage.
This is a sensitivity check, not a repaired detailed timing model; cuDNN launch
choices also differ between the two GX runs. The native gain above does not establish
GX timing accuracy for this case.

The investigation also exposed missing legacy CUDA exports in cuDNN initialization
and missing compute recording during CUDA-graph capture. Isolated repairs pass
raw-kernel and BF16 cuBLAS replay checks. Corrected kernel calibration contains
307 signatures, 825 samples, and 40,943 observed launches; graphs were disabled
only for kernel sampling. Generated cuDNN names and different Triton autotune
choices still limit prediction matching. See the
[evidence](evidence/flashdreams-screen.json), [runner](flashdreams_runner.py), and
[calibration/native reports](evidence/kernel-databases/flashdreams-profile.json.gz).

## VideoX-Fun: CPU scheduling during sequential offload

GX exposes CPU overhead in VideoX-Fun's real pretrained Wan 1.3B pipeline.
For 17 frames at 256x448 with eight denoising steps, sequential CPU offload
repeatedly waits for individual weight uploads and clears the CUDA allocator
cache. The [opt-in Accelerate patch](cpu_patches/videoxfun_async_offload.patch)
queues weight/input transfers without a host wait and reuses cached allocations.
The [runner](videoxfun_runner.py) still offloads each layer after use, retains
blocking output copies, and releases unused cache at request boundaries. It
also releases the large text-embedding allocation immediately after use to
avoid fragmentation under the same **2.25 GiB allocator budget**. A simpler
version without that release failed at this budget and is excluded.

| Complete warmed request | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 1, GPU-only timing | 6.8356 s | 6.8393 s | No improvement |
| GX mode 2, mean of two requests | 13.7892 s | 6.8656 s | 50.21% |
| GX mode 3, mean of two requests | 14.4915 s | 7.5632 s | 47.81% |
| A100, median of six requests per variant, ABBA order | 12.5777 s | 7.5905 s | 39.65% |

Both versions transfer exactly the same bytes: 68.953 GB H2D and 0.0117 GB
D2H per request, plus 60.644 GB of device-to-device work. Mode-2 launch
inventories also match across the pair. Native peak allocated memory stays
at 2,120,920,064 bytes; peak reserved memory is 2,157,969,408 bytes before and
2,126,512,128 bytes after. All twelve native output hashes match. Validation
uses four fresh processes in before/after/after/before order, each with one
warmup and three measured requests. This is a CPU scheduling/cache-management
change with unchanged offload policy and traffic, not a reduction in model work.
The allocator cap is not a partition of total physical GPU memory.

GX diagnostic runs separate the mechanisms. Asynchronous transfers alone give
9.5369 seconds while modeled CPU work remains 8.2688 seconds; allocator-cache
reuse alone gives 8.6539 seconds and lowers modeled CPU work to 2.9798 seconds.
The combination gives 6.8656 seconds, with 2.9784 seconds of modeled CPU work
and only about 0.0300 seconds outside modeled GPU spans. All directional copy
counts and bytes match. The diagnostic runs select different cuDNN plans and
have no separate native validation, so these are modeled mechanism checks,
not measured attribution of the native speedup.

Open these compact Chrome/Perfetto traces to see the difference:

- Whole-request mode-2 activity: [before](evidence/videoxfun-mode2-before.chrome.json.gz),
  [after](evidence/videoxfun-mode2-after.chrome.json.gz). These contain both
  measured requests, with activity averaged into 10 ms bins. CPU and GPU
  categories overlap; their percentages must not be added together.
- First 250 ms of the first request, with individual CPU/GPU spans:
  [before](evidence/videoxfun-mode2-before-250ms.chrome.json.gz),
  [after](evidence/videoxfun-mode2-after-250ms.chrome.json.gz).
  Event-record markers and dependency arrows are omitted from these views.
  CPU spans show GX instruction-based modeled time, not measured native CPU utilization.

The [evidence](evidence/videoxfun-cpu.json) retains full-trace paths, checksums,
source identities, per-run measurements, launch audits and native hashes.
The [reproduction bundle](evidence/videoxfun-reproduce.tar.gz) contains the
exact deployment configurations, controllers and native commands; adjust host
and asset paths for another machine. Apply the patch to Accelerate 1.14.0,
put that package and the pinned VideoX-Fun source on `PYTHONPATH`, and run
`videoxfun_runner.py` with `--allocator-gib 2.25`. Add `--async-offload` for the
candidate. The patch assumes immutable CPU weights and stream-ordered GPU use;
it is not validated for arbitrary training or concurrent weight mutation.

A longer 50-step run reuses the eight-step kernel database. Mode 2 predicts
**70.0002 -> 31.3474 seconds (55.22% lower)**, with identical before/after
copy summaries and compute inventories. Mode 3 predicts 74.0092 -> 35.4611 seconds (52.09% lower). A100 ABBA validation measures 59.1884 -> 35.0510 seconds median (40.78% lower), with all twelve long-request output hashes identical. No additional kernel calibration was collected.
The eight-step mode-2 pair has 40 unpredicted cuDNN launches out of 96,000;
mode 1 and mode 3 select different cuDNN plans across their pairs. The 50-step
pair has 20 misses out of 273,120 launches per variant. These limitations and
kernel-signature aliases preclude a claim of exact native replay or a universal
prediction-accuracy guarantee. No GPU arithmetic is validated by GX itself.

Paper-ready description of this CPU case:

> GX also identified CPU overhead in VideoX-Fun's sequential CPU-offload path:
> repeated host waits for weight uploads and allocator-cache clearing delayed
> GPU submission. We queued uploads asynchronously and reused allocator memory,
> retaining layer offload and releasing large unused allocations when needed
> under the same 2.25-GiB allocator budget. For pretrained Wan 1.3B generating
> 17 frames at 256x448 with eight denoising steps, GX mode 2 predicted request
> latency decreasing from 13.79 to 6.87 seconds (50.2%); mode 3 predicted 14.49
> to 7.56 seconds (47.8%). A100 validation measured 12.58 to 7.59 seconds median
> (39.7%), with identical output hashes, unchanged transfer traffic, and unchanged
> peak allocated GPU memory. Reusing the same kernel database for 50 denoising
> steps, GX modes 2 and 3 predicted reductions of 55.2% and 52.1%, respectively;
> A100 validation measured 59.19 to 35.05 seconds (40.8%). Each native comparison
> used six measured requests per variant in ABBA process order. GX identified
> a useful CPU optimization, although it overestimated the latency reduction
> in these experiments.

## WanVideoWrapper: repeated CPU checkpoint conversions

GX exposed repeated FP32-to-BF16 checkpoint conversions before model-weight
uploads in ComfyUI-WanVideoWrapper. The [opt-in patch](cpu_patches/wanwrapper_cpu_cast_cache.patch)
caches these CPU conversions, invalidating entries when the source identity,
dtype or tensor version changes. It retains CPU offload and GPU execution in
ComfyUI's `inference_mode`. The [runner](wanwrapper_runner.py) executes the public
loading, text-encoding, sampling and VAE-decoding nodes headlessly; it excludes
the UI, graph queue, previews and file encoding. This is pretrained Wan 1.3B,
17 frames at 256x448, eight Euler steps, CFG 5 and seed 42.

| Warmed complete request | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, mean of two requests | 6.4450 s | 5.7909 s | 10.15% |
| GX mode 3, mean of two requests | 7.0103 s | 6.3487 s | 9.44% |
| A100, median of ten requests per variant | 10.0304 s | 9.6484 s | 3.81% |
| A100, mean of the same ten requests per variant | 10.0042 s | 9.7339 s | 2.70% |

Native validation uses ABBA process order, two warmups and five measured requests
per process. All twenty output hashes match. Whole-pipeline peak allocated GPU
memory remains 11,636,556,288 bytes, and all directional GX transfer counts and
bytes match. The cache retains **2,836,681,856 bytes of extra host tensors**;
native peak process RSS rises from 35.96–36.03 GB to 38.30–38.35 GB. This is a
host-memory-for-CPU-time tradeoff. GPU residency is unchanged.

GX overestimates the measured improvement. Mode-2 predicted kernel inventories
match, but ten unpredicted cuDNN transforms have differing malformed launch
dimensions. Mode 3 also selects different cuDNN plans, with ten versus forty
prediction misses. CPU calibration has 21.74% unresolved samples and uses a
different CPU model from the A100 host. The result establishes a modest native
gain for this workload, not an accurate prediction of its magnitude.

The [evidence](evidence/wanwrapper-cpu.json) records every native sample, source
identities, memory measurements and GX audits. Compact mode-2 traces show
[before](evidence/wanwrapper-mode2-before.chrome.json.gz) and
[after](evidence/wanwrapper-mode2-after.chrome.json.gz) activity in 10 ms bins;
these are modeled intervals, not native CPU utilization. The
[reproduction bundle](evidence/wanwrapper-reproduce.tar.gz) and archived kernel
and CPU databases preserve the tested configuration. Enable the candidate with
`WANWRAPPER_CPU_CAST_CACHE=1`; the default is the original path. Validation is
limited to the unquantized checkpoint and this node sequence, with no concurrent
weight mutation or training.

Paper-ready description:

> In ComfyUI-WanVideoWrapper, GX identified repeated CPU dtype conversions of
> checkpoint weights before GPU upload. Caching these conversions while retaining
> CPU offload reduced predicted request latency from 6.445 to 5.791 seconds in
> mode 2 (10.1%) and from 7.010 to 6.349 seconds in mode 3 (9.4%). For pretrained
> Wan 1.3B generating 17 frames at 256x448 with eight denoising steps, A100
> validation measured 10.030 to 9.648 seconds median (3.8%; 2.7% by mean), using
> ten requests per variant in ABBA process order. All twenty output hashes
> matched, with unchanged transfer traffic and peak allocated GPU memory.
> The optimization retained 2.84 GB of additional host tensors. This case
> demonstrates a useful CPU optimization and its memory tradeoff, while also
> showing that GX can overestimate the size of the native improvement.

## CPU screens with limited modeled headroom

The 17-frame, 256x448, eight-step pretrained Wan workload also runs through
xDiT, SGLang Diffusion, and FastGen. Their solvers and residency policies differ, so these are
independent screens rather than a performance ranking between frameworks.

| Framework/configuration | GX mode-2 request | Time outside modeled GPU spans |
| --- | ---: | ---: |
| xDiT, all model components resident | 1.4891 s | 0.0354 s (2.38%) |
| SGLang Diffusion, resident DiT and offloaded T5/VAE | 5.1176 s | 0.6428 s (12.56%) |
| FastGen, resident parameters, UniPC | 1.3778 s | 0.0206 s (1.50%) |

These screens deprioritized CPU optimization in those configurations. No
optimized variant or native speedup is claimed. The percentages are a
prioritization heuristic, not measured device utilization or a general upper
bound on CPU optimization. All retain cuDNN prediction misses; SGLang's
steady CPU calibration has 900 unresolved instruction samples out of 5,436.
[Evidence](evidence/cpu-screen-negative.json) includes exact launch audits,
source/model pins, and the SGLang FA2 compatibility patch. Their kernel and
CPU calibration databases are in the existing kernel-databases bundle.

FastGen's native kernel calibration checks finite output and all scoped
GPU-derived control values; it is not a native latency validation. Its GX
trace additionally omits 18 solver launches per request (110.6 microseconds
summed isolated native kernel medians) and three initial-index launches
(17.4 microseconds). Neither sum is an end-to-end correction. A CPU-cached
clamp candidate passes 1,444 source-equivalence checks, but was not timed
because the baseline showed little modeled CPU headroom. There is no mode-3
or optimized A100 comparison for this screen. The
[compact mode-2 trace](evidence/fastgen-mode2.chrome.json.gz) and
[reproduction bundle](evidence/fastgen-reproduce.tar.gz) retain this result.

VoxCPM2 provides another low-gain screen, using its original compiled speech
pipeline and a 3.84-second generated waveform. GX mode 2 predicts 0.7945 seconds,
with 13.0 ms (1.64%) outside modeled GPU spans. Reusing Euler-sampler scratch
buffers reduces modeled CPU time from 235.9 to 220.8 ms (6.4%), but request time
only from 0.7945 to 0.7884 seconds (0.77%). We stopped after this screen; there
is no mode-3 or optimized A100 result. Initial native kernel calibration and
64 CPU source-equivalence cases are recorded separately.

This screen required a private GX repair: predicted kernels and cuBLAS calls
were not retained during CUDA graph capture. The original 0.1563-second estimate
is rejected. The repaired trace still has 5,962 prediction misses across two
requests, differing Triton launch configurations and omitted FP32 GEMV work,
so the small gain is not a hardware speedup claim. The
[before](evidence/voxcpm-mode2-before.chrome.json.gz) and
[after](evidence/voxcpm-mode2-after.chrome.json.gz) traces,
[reproduction bundle](evidence/voxcpm-reproduce.tar.gz), and
[evidence](evidence/cpu-screen-negative.json) retain both the negative result
and these limitations.

## DiffSynth-Studio: CPU module copying and checkpoint loading

GX exposes two CPU costs in DiffSynth-Studio's real pretrained Wan 1.3B pipeline.
For a 17-frame, 256x448 request with eight denoising steps and CPU offload,
every offloaded normalization call recursively copies a Python module before
converting its parameters. The [normalization patch](cpu_patches/diffsynth_norm_copy.patch)
copies the known stateless leaf module and its independent parameter storage
without recursively traversing Python state. Hooks, buffers, custom conversions,
and mutable state retain the original path. CPU offload remains enabled.

| Normalization copying, warmed request | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, mean of two iterations | 8.4412 s | 8.0498 s | 4.64% |
| GX mode 3, mean of two iterations | 9.0513 s | 8.6556 s | 4.37% |
| A100, median of six iterations per variant, ABBA order | 10.0762 s | 9.8088 s | 2.65% |

The controlled comparison sets `vram_limit=0` on both sides, retaining full CPU
offload independently of transient allocator occupancy. Mode-2 before/after compute
launch inventories match, and every directional copy-byte total matches in both modes.
Mode 3 selected different cuDNN plans across the pair, which limits attribution of
its exact gain; the mode-2 comparison has no such inventory difference. The combined two-iteration GX
totals are 137.796 GB H2D and 0.024 GB D2H on each side. Native peak
allocated memory remains 2,662,645,760 bytes; all twelve video hashes are identical. Native validation uses four fresh
processes in baseline/candidate/candidate/baseline order, each with one warmup
and three measured requests.
This is a modest gain on one fixed prompt, seed, and shape. An earlier adaptive
residency experiment changed copy traffic and is excluded from this CPU claim.

A separate [checkpoint-loading patch](cpu_patches/diffsynth_checkpoint_loading.patch)
removes full payload materialization when discovering checkpoint keys/shapes,
uses mapped binary storage when supported, and avoids a redundant CPU copy after
dtype conversion. GX mode-0 host timing drops from **30.151 to 13.517 seconds
median (55.17%)**, over three fresh processes per variant using already cached
checkpoint files. Timing covers `from_pretrained`, excluding imports and inference.
All **1,261 loaded tensors (14,453,599,334 bytes)** have the same aggregate hash.
This result concerns model startup, not warmed request latency or virtual-time
prediction. Mapped checkpoint files must remain immutable while in use; legacy
archives fall back to the ordinary loader.

The [evidence](evidence/diffsynth-cpu.json) records all repetitions, source hashes,
managed run identities, exact-copy checks, and calibration provenance. GPU
calibration was collected once; both GX timing modes reuse it. Launch coverage
is 99.9478% for both mode-2 variants: cuDNN plan misses remain unresolved and
receive no predicted service. The mode-3 baseline instead has ten malformed
transform launches across two requests, while its candidate has forty other
cuDNN misses. These plan differences also occur across runs with the same offload
setting and are recorded as a GX limitation, not credited to the CPU change.
The initial GPU calibration used a 2-GiB adaptive residency budget, which is a
limitation when interpreting absolute full-offload timings. CPU calibration also
has unresolved instruction samples. The real A100 result independently checks
both output equivalence and the direction of the small generation improvement.

Reproduce with the [pipeline runner](diffsynth_runner.py), the pinned upstream
revision in the evidence, and the existing Wan checkpoint/tokenizer directory.
The same runner supports GX discovery, late attachment, kernel calibration,
and native validation. The [normalization checks](validate_diffsynth_norm.py)
verify values, storage isolation, hooks, fallback conversions, and gradients;
the [loader checks](validate_diffsynth_loader.py) compare zip/legacy checkpoints,
dtypes, tensor layouts, and repeated-read mutation isolation.

An incremental experiment keeps the hardened normalization copy and makes
inference weight uploads asynchronous using the
[upload patch](cpu_patches/diffsynth_async_upload.patch). CPU weights remain
immutable and pipeline-owned, and GPU consumption stays ordered on the same
stream. Full CPU offload remains enabled; all directional copy counts and bytes
are unchanged. Mode 2 predicts **8.0570 -> 7.2663 seconds (9.81%)** and mode 3
predicts **8.6597 -> 7.7958 seconds (9.98%)**. Unlike the normalization change,
this primarily reduces host waiting: modeled instruction CPU work increases
slightly, while time outside modeled GPU spans falls from 1.3715 to 0.5844 seconds
per request. Both pairs select different cuDNN plans, limiting exact attribution.

The initial A100 ABBA batch has identical output hashes in all twelve requests
and unchanged peak allocated memory. Its median falls from 10.0445 to 9.2886
seconds (7.53%), but a 15.3931-second candidate request makes the **mean 1.77%
worse**. That sample is retained. A separate fixed-length ABBA repeat with two
warmups and five measured requests per process gives **9.7585 -> 9.2319 seconds
median (5.40%)**, and **9.7330 -> 9.2117 seconds mean (5.36%)**. All twenty
repeat output hashes match the initial batch, with the same peak allocated
memory. The first batch is not replaced. The `async_uploads` entry in the existing
[evidence](evidence/diffsynth-cpu.json) records both aggregation methods and all
samples. Compact mode-2 activity traces are available
[before](evidence/diffsynth-async-mode2-before.chrome.json.gz) and
[after](evidence/diffsynth-async-mode2-after.chrome.json.gz), with exact full-trace
paths in the evidence. The [reproduction bundle](evidence/diffsynth-async-reproduce.tar.gz)
preserves the configurations, commands, and historical runners. This comparison
starts from the hardened normalization implementation; its percentages must
not be added to the earlier independently measured normalization result.

## StreamDiffusionV2: repeated whole-video conversion

The documented staged API passes the original CPU video into `encode_chunk`
for every chunk. Each call converts and uploads that entire video again, even
though adaptive noise reads only the current four frames and their preceding
frame. The [patch](cpu_patches/streamdiffusionv2_input_window.patch) normalizes
that five-frame window. It preserves the denoiser, VAE, adaptive-noise rule,
cache policy, and first-chunk behavior; file-path inputs retain the original path.

The [harness](streamdiffusionv2_runner.py) uses the published pretrained checkpoint,
two denoising steps, stream batching, FA2, Wan VAE, and 480×832 CPU FP32 inputs.
With four-frame chunks, let `F = 1 mod 4` be input length and `H,W` resolution.
The source implies these total converted-element counts:

```text
before = 3*H*W * F*(F+3)/4
 after = 3*H*W * (9*F-5)/4
```

Thus the redundant conversion grows quadratically before the patch and linearly
after it. drperf on 17/33/65 frames accepts approximately
`2.0624*elements + 48693` instructions per original conversion, with a maximum
0.0109% unexplained share. Window views select a different CPU conversion path;
refining the PCVs to `contiguous_elements` and `strided_elements` accepts
`2.0625*contiguous_elements + 2.1192*strided_elements + 77568`, with maximum
0.0276% unexplained share. Only one strided window size was observed: its
coefficient is not a demonstrated scaling law for arbitrary layouts. Other
marked regions remain partly unexplained or underidentified.

Across all three requests, conversion instructions fall from **3.676 to
0.642 billion (82.5%)**, and total marked instructions from **8.647 to 5.628
billion (34.9%)**. GX excludes emulator instructions from those counts.
At 65 frames, modeled H2D traffic falls from **2.648 to 0.347 GB**.
For a 257-frame input, the same source-derived formula predicts **40.028 GB →
1.383 GB** of input uploads without running that larger case. This is a byte-count
prediction for the staged API, not an extrapolated latency or a drperf guarantee
outside the measured states.

| Input frames | GX before → after, seconds | A100 before → after, seconds | A100 reduction |
| --- | --- | --- | --- |
| 17 | 2.328 → 2.291 | 1.973 → 1.903 | 3.5% |
| 33 | 4.578 → 4.405 | 4.207 → 3.833 | 8.9% |
| 65 | 9.373 → 8.629 | 9.175 → 7.696 | 16.1% |

The A100 values are three-run medians after warmup, with **all nine paired output
hashes identical**. This final native comparison restores upstream GPU metadata;
its only source difference is the input-window patch. Peak allocated GPU memory
falls by 144 MB, while allocator reservation rises by 42 MB. Loading, input
file decoding/resizing, and output video encoding are outside the timing.
The public stream-batching warmup yields 13/29/61 output frames from these inputs.
The [CPU oracle](validate_streamdiffusionv2_window.py) passes 540 comparisons
across dtypes, 4D/5D layouts, contiguous/strided inputs and chunk sizes.

GX is a qualified screen here. Its original page calibration assigns a conversion
page 11.36 ns/instruction, with 26.06% unresolved instruction samples; the resulting
39.79-second baseline estimate at 65 frames is rejected. The table uses the same
aggregate-only calibration, 0.239946 ns/instruction, for both variants. This is a
sensitivity model, not a calibration repair. Both GX inventories match each
other, but 112 cuDNN launches differ from native and receive no predicted service;
FA2 signatures also combine different input-dependent durations. Accordingly,
these GX latencies are neither a matched native replay nor an error bound.
GX retains the CPU metadata compatibility patch and replays native scalar
controls for exactly the calibrated inputs and request order.

The updated GX runtime now passes full-sync at a 10-us epoch and matched
partial-sync controls. The initial failures mixed old runtime artifacts with
updated GX and then omitted the existing private cuDNN/cuBLAS compatibility
layer during rebuild; these were experiment integration errors. Both modes
below use the same rebuilt runtime, source variants, kernel database, native
control tape and aggregate CPU calibration. No GPU was used for this rerun.

| Input frames | Updated partial-sync before → after | Full-sync before → after | Earlier native A100 medians before → after |
| --- | --- | --- | --- |
| 17 | 1.775 → 1.739 s | 1.932 → 1.897 s | 1.973 → 1.903 s |
| 33 | 3.594 → 3.421 s | 3.897 → 3.723 s | 4.207 → 3.833 s |
| 65 | 7.526 → 6.781 s | 8.111 → 7.366 s | 9.175 → 7.696 s |

At 65 frames, partial-sync predicts 9.9% improvement, full-sync 9.2%, and native
A100 measured 16.1%. Full-sync gives closer absolute times for these inputs, but
both GX modes predict about 0.745 s saved versus native 1.479 s. Synchronization
alone does not resolve the discrepancy. Each new run predicts 142,985 of 143,013
compute launches (99.98%); the 28 remaining cuDNN transform launches receive no
predicted service and have different launch grids between runs and from native.
Predicted kernel inventories match across all four runs. CPU calibration and
FA2 duration aliasing remain limitations, and there is one GX capture per
variant/mode versus three native measurements. The earlier partial table above
uses the previous runtime and is retained as historical evidence.

The evidence's `full_sync` record contains all four audits, configurations,
runtime hashes, host wall times and compressed trace paths. Full-sync observed
CPU/GPU spans include blocking and worker overhead; they are not hardware
utilization measurements.

The [evidence](evidence/streamdiffusionv2-screen.json) records full instruction
summaries, timings, source hashes, limitations and local compressed trace paths.
The [kernel database and replay bundle](evidence/kernel-databases/README.md)
include the native control tape and both CPU calibrations, so replay does not
require another GPU run for these inputs. The supplied input is
`examples/original.mp4` at the pinned upstream revision, with prompt
`A dog walks on the grass, realistic`.

```bash
# In a pinned StreamDiffusionV2 checkout: apply the CPU metadata patch for GX.
# Apply the input-window patch only for the candidate.
python3 examples/causal_video/validate_streamdiffusionv2_window.py \
  --tree out/causal-video/StreamDiffusionV2
```

## TeleFuser / ABot-World: CPU image layout, modest latency gain

The public `ABotWorldInteractivePipeline` runs the real 5B ABot checkpoint with
resident weights, eager Torch SDPA, and twelve three-latent-frame control blocks
at 480x832. Each block returns twelve PIL images. The test includes cache growth
and eviction in the default 18-frame window; HTTP/WebRTC transport is excluded.
GPU1 was used for one kernel calibration and selected final confirmation. The
kernel database is reused unchanged throughout the GX comparison.

drperf first pointed at output conversion: NumPy's strided copies dominated
CPU work. `tensor2video` keeps planar RGB storage after its dtype conversion,
then `Image.fromarray` packs each strided image into interleaved RGB bytes.
The [patch](cpu_patches/telefuser_planar_output.patch) lets PIL merge the three
contiguous channel planes directly. GPU operations, D2H bytes, model arithmetic,
attention, and cache policy are unchanged. Small images below 4,096 pixels,
non-planar storage and other channel counts retain the original path.

| Measurement | Before | After |
|---|---:|---:|
| Full-model marked CPU instructions | 11.226 billion | 6.876 billion |
| GX virtual time, 12 generation blocks | 8.043 s | 7.672 s |
| A100 median, 12 generation blocks | 9.111 s | 8.919 s |
| A100 median, session creation plus 12 blocks | 9.240 s | 9.048 s |

This is **38.8% fewer marked CPU instructions but 2.1% lower native generation
latency**. All 36 paired native output-chunk hashes match; final native runs
restore the original framework controls and change only `base_pipeline.py`.
The baseline GX screen has roughly 10% modeled device idle during generation,
including a roughly 43 ms output-CPU segment per block. Most other CPU work
overlaps GPU execution, so this case is deprioritized for further CPU-only work.

The small-fixture interface explains why the change helps video-sized images
and why it needs a small-image fallback. The actual post-NumPy PIL conversion
tail is measured with `pixels`, `rows`, and `frames` as PCVs. Omitting `rows`
initially left 9.3% unexplained cost. Adding it reduced that to below 0.4%.
On the final eight fast-path states, each with five repetitions:

```text
before: 79.1*pixels + 90.6*rows +  64,200*frames + 10,300 + unexplained
 after: 12.1*pixels + 43.2*rows + 182,000*frames + 15,000 + unexplained
```

Both final fits leave at most about 0.2% unexplained cost. The lower pixel
coefficient comes with higher per-image setup cost; tiny images can lose,
which motivated the fallback. These are local CPU fixture observations, not
instruction formulas for the entire model or guarantees at unseen sizes. The
fixtures use at most 96x160 images, 26 times fewer pixels per frame than the
full-model request; they exclude GPU math, D2H, NumPy casting,
and image destruction. Several full-model annotations remain underidentified
because the standard session uses one block size.

The [oracle](validate_telefuser_planar_output.py) passes 264 exact comparisons
across RGB/RGBA, dtypes, layouts, resizing, empty sequences and output ownership;
48 unsupported FP16 CPU resize inputs retain their original errors. It applies
the saved patch to the pinned source before testing. `--measure-pil` exposes the
actual source-derived PIL tail to drperf for the small-fixture experiment.

```bash
# Requires Torch, NumPy, Pillow and einops (the isolated local dependency below).
PYTHONPATH=out/causal-video/telefuser/deps \
  python3 examples/causal_video/validate_telefuser_planar_output.py \
  --tree out/causal-video/TeleFuser
```

The [evidence](evidence/telefuser-screen.json) contains the exact launch configs,
source hashes, instruction summaries, PCV fits, native reports and trace paths.
The [runner](telefuser_runner.py) supplies GX control metadata while retaining
original GPU reads; the native calibration checks cache and RoPE controls.
A later audit caught the prompt-padding scalar index: using the CPU tokenizer's
length, 13, restores the native fill range. Both final GX runs have **100%
prediction coverage and exactly the same 121,321-launch inventory as native**.
Initial runs with the incorrect padding boundary are excluded from the final pair.

The [kernel bundle](evidence/kernel-databases/README.md) contains 324 signatures,
855 native samples and the CPU calibration. Matching inventories still do not
establish exact timing: the same PyTorch FlashAttention signature covers growing
self-attention key lengths and fixed cross-attention lengths, and CPU calibration
uses the GX host rather than the A100 host. The GX latency reduction is a screen;
the separate A100 pair is the confirmation.

## Causal Forcing++: small modeled CPU headroom

The public framewise one-step pipeline runs under GX with the actual 1.3B EMA
checkpoint: 81 frames at 480x832, four denoising steps on the first latent frame,
one on each later frame, and a cache-refresh pass on every frame. A100 GPU1 was
used for kernel calibration and to check control metadata; discovery and the
candidate comparison used GX.

The baseline `partial_sync` request has **7.253 virtual seconds**, including
**0.183 seconds of modeled device idle time (2.5%)**. A small
[candidate patch](cpu_patches/causal_forcing_metadata_cpu.patch) keeps two cache
position tensors on CPU. It removes 2,700 eight-byte D2H reads, 60 initial H2D
copies, and 2,700 GPU fill kernels. KV payloads remain on GPU. The candidate gives
**7.128 virtual seconds**; drperf's marked exclusive instructions fall only
**0.49%, from 5.360 billion to 5.334 billion**.

This case is **deprioritized for further CPU-only work**. Small modeled idle is a
screening signal, not a proven upper bound on native improvement. The 1.7% virtual
change is not a clean speedup estimate: cuDNN launch variants differ, and a single
FlashAttention launch signature covers key lengths from 512 to 32,760. Prediction
coverage exceeds 99.9%, but that does not resolve this input aliasing. No candidate
native latency or numerical-equivalence claim is made.

GX skips scalar GPU arithmetic, so the [runner](causal_forcing_runner.py) supplies
CPU-known cache positions and tokenizer lengths while still executing the
original GPU reads, writes, and reductions. The native calibration checked those
values. Both GX variants use the A100's text-offload policy and the same
memory-efficient BF16 loading setup, outside measured inference. At fixed query
geometry, drperf finds useful conditional relations across 21 context lengths;
the full declared PCV models remain rank-deficient.

The [evidence](evidence/causal-forcing-screen.json) records both trace paths,
instruction summaries, validity checks, source hashes, and limitations. The
[kernel database and profiling report](evidence/kernel-databases/README.md)
contain 332 signatures, 866 samples, and 118,865 observed launches. GPU arithmetic
and video quality are not validated by GX.

## Matrix-Game 3: output copies dominate CPU work

The [harness](matrix_game_runner.py) runs the real pretrained interactive model
at 704x1280, with three denoising steps, INT8 projections, FA2, and the compiled
MG-LightVAE decoder. The 97-frame GX `partial_sync` request models **23.383 seconds**:
21.264 seconds GPU busy and **2.119 seconds idle (9.1%)**, clipped to the measured
request. Most modeled idle is near output handling. The compressed Chrome trace
is 6.3 MiB; its path is recorded in the [evidence](evidence/matrix-game-screen.json).

A separate 177-frame drperf run records **20.36 billion** marked CPU instructions.
The outer generation region accounts for 14.13 billion exclusive instructions;
two NumPy copy functions alone account for 11.48 billion. Camera-history selection
accounts for only 105 million (0.52%). Across 200, 400, and 600 candidate/reference
pairs, its observed interface is:

```text
memory_selection_instructions = 80058.61 * candidate_reference_pairs + 3057821
```

The maximum unexplained share is 0.12%. Other regions lack enough independent
states for an identified interface. This is **deprioritized for further CPU-only
optimization** under the small-headroom screening rule; no candidate speedup is
claimed. Prediction coverage is 98.88%, but FA2 aliases self- and cross-attention
inputs to one signature with up to 21.7x sample-duration variation. Native kernel
calibration is not an unprofiled latency baseline, and modeled idle is not a bound
on native improvement.

This run also exposed a drperf late-attachment failure when a worker blocks the
suspension signal. [prepare_drperf_gx.py](prepare_drperf_gx.py) builds an isolated
bundle against GX's patched DynamoRIO and enables its signal-unmasking attachment
options. It leaves the normal drperf runtime unchanged. A blocked-signal worker
probe times out with the stock runtime; the new bundle captures exactly
`4*n + 14` instructions at n=10, 100, and 1000. The full Matrix-Game collection
then passes with zero validity errors and zero counted GX-module instructions.

```bash
python3 examples/causal_video/prepare_drperf_gx.py out/drperf-gx \
  --dynamorio /home/ubuntu/GX/NEX/build/gxvm-observed-install/dynamorio
```

## Scope / LongLive: host dispatch mostly overlaps GPU work

The [public streaming pipeline](scope_runner.py) runs the actual LongLive 1.3B
checkpoint and built-in rank-256 performance LoRA. Six chunks produce 69 frames
at 320x576, switching prompts after the third chunk. Both steady streaming and
prompt-triggered recaching execute. The workload uses FA2, the Wan VAE, and the
public constructor's BF16 text-encoder option, with resident weights.

GX `partial_sync` gives **4.849 virtual seconds**, with **0.273 seconds modeled
GPU idle (5.6%)**. Summed CPU work is 2.309 seconds, but most overlaps device work;
0.195 seconds of the idle interval is near final output handling. This case is
**deprioritized for further CPU-only optimization**. No candidate was selected.

drperf records **5.294 billion** marked exclusive instructions. LoRA linear calls
account for 1.619 billion across 7,620 calls. Their observed interface is almost
constant, about 213,000 instructions per call, with less than 0.1% unexplained
cost across seven input states. Self-attention and transformer-block regions
also fit within 0.3% unexplained cost. Cache setup and decoding remain poorly
explained by their initial PCVs; this is not a complete performance interface.

The [evidence](evidence/scope-screen.json) includes trace paths, all region
diagnostics, and limitations. The kernel database contains 485 signatures,
1,250 samples, and 103,099 observed launches. GX has 17 prediction misses, but
FA2 input aliasing produces up to 9.9x sample-duration differences within one
signature. CPU calibration has 6.7% unresolved instruction samples. These limits
prevent treating the modeled idle fraction as a native speedup bound.

GX control adapters retain original GPU operations while supplying known cache
positions, token lengths, and calibrated single-prompt blending branches. A
separate native control check verifies those branches and reproduces the kernel
calibration's final video hash exactly. The initial GX runs that took incorrect
blending branches are excluded from the timing baseline. Native GPU use was
limited to calibration and this control check; no optimization speedup is claimed.

## SkyReels-V2: simulator overhead exposed, little modeled CPU headroom

The [public diffusion-forcing runner](skyreels_runner.py) uses the actual
DF-1.3B-540P checkpoint, 544x960, CFG 6, 30 steps, default bidirectional FA2,
and resident weights. CPU calibration, drperf, and native kernel calibration
cover 33/65/97 frames. The bounded GX timeline covers 33 frames only.

The first timeline was misleading: **282.938 virtual seconds**, including
278.835 summed CPU seconds and 231.335 seconds GPU idle. The database contained
43,200 identical occupancy-query rows. GX repeatedly queried those records;
274.153 modeled CPU seconds occurred before 7,200 CUB scan-kernel submissions.
This was simulator work, not a supported SkyReels optimization opportunity.

A diagnostic database copy removes only identical occupancy-query rows. Every
other table, kernel sample and distinct query result is preserved. With the
same source, harness, launch inventory and copy inventory, GX now gives
**51.882 virtual seconds**, **5.137 summed CPU seconds**, and **0.279 seconds
modeled GPU idle (0.54%)**. Measured capture wall time falls from 618.50 to
23.10 seconds; warmup falls from 158.78 to 6.42 seconds. This is a GX diagnostic,
not an application speedup. The original database and rejected trace remain
available. The issue is recorded in GX's `issues/flashdreams-cpu-calibration.md`
under the separate occupancy-replay section.

This screen **deprioritizes further CPU-only optimization**. drperf records
31.244 billion marked exclusive instructions across the three sizes. Transformer
calls account for 17.155 billion, attention for 9.289 billion. With 30 steps held
fixed, generation overhead fits approximately `7.11 million * frames + 132 million`
with at most 0.3% unexplained cost, while scheduler setup fits approximately
`1.28 million * frames + 3.43 million` within 0.4%. Several other declared models
remain underidentified; this is not a complete performance interface.

The [evidence](evidence/skyreels-screen.json) includes both timelines, diagnostic
equivalence checks, region diagnostics and launch configurations. The compressed
Chrome traces are approximately 16 MB each. The bundled database has 430 kernel
signatures, 1,230 samples, and 415,485 native launches with zero skips. GX has
98.55% prediction coverage and 1,800 misses for one fused Triton signature.
FA2 samples sharing one signature differ by up to 78.2x. The three-size native
inventory is not directly comparable to the single-size GX inventory. These
limits prevent treating the idle fraction as a native speedup bound, despite
51.882 virtual seconds being close to the 53.345-second native calibration run.
That native run includes profiling and is not an unprofiled candidate comparison.

GX retains original scheduler GPU operations and scalar reads, substituting the
known initial index and successful order-two solver status. Native calibration
asserts those controls. Its output-finiteness flag inspects uint8 output, not
prequantization floats or visual quality. No application patch is included.

## Workloads and boundaries

- **Inferix:** its public `run_streaming_generation` API, TRUE_STREAMING, the
  pretrained 1.3B Self-Forcing DMD checkpoint, UMT5-XXL, and Wan VAE. The main
  request uses 21 latent frames at 480 × 832 with four denoising steps per block.
  The pinned streaming implementation returns seven chunks of nine pixel frames
  (63 total); this study preserves and reports that behavior.
- **FastVideo:** its public `VideoGenerator.generate_video` API and pretrained
  SFWan causal DMD pipeline, 81 pixel frames at 480 × 832, four denoising steps
  per block. Its prompt differs from Inferix's. The frameworks are independent
  optimization studies, not a controlled quality or throughput comparison.

Load and warmup precede measurement. Generation includes text encoding,
causal denoising, VAE decode, and output transfer; encoding a video file is
excluded. Native tensors are checked for finiteness. GX output values are never
used for numerical validation because GPU arithmetic is skipped.

`native_run.py` uses the existing managed A100 container helper. `config.py`
generates GX managed-launcher configurations; run them using
`GX/NEX/container_cmds/run.py` from that repository. Model assets are staged in
`/workspace/causal`, with the shared Wan checkpoint in the existing workspace.
`asset_identity.py` hashes the actual Inferix checkpoint/tokenizer files on each
host before comparing their identities. FastVideo's downloader records both
source and stored hashes; its fp32 text encoder weights are stored as bf16 to
fit disk, and both native and GX use those same files.
The [runbook](RUNBOOK.md) gives commands for repeating the separate measurements
in these staged workspaces.

## Separate measurements

1. **Native:** warmed, unprofiled A100 generation and saved output.
2. **GPU profile:** a separate real A100 run, profiler enabled only around
   generation. Use zero per-signature profiler warmup after application warmup
   so a signature occurring once still gets a sample. Profiler wall time is not
   the native latency reference.
3. **CPU profile:** GX emulation with native CPU IPC sampling, then `gxvm collect`.
4. **Partial timing:** GX `partial_sync` with the GPU and CPU databases; obtain
   virtual iteration durations from `gxvm report`, not application clocks.
   FastVideo's timeline markers are in its GPU worker. The public parent call
   also builds frame grids after the worker responds, so its worker virtual
   interval is not the full parent-call latency.
5. **drperf:** GX emulation without GX timing instrumentation. The drperf CUDA
   boundary excludes emulated CUDA calls in `gx_cuda.so`; application/library
   host dispatch remains measured. Markers cover transformer, attention, RoPE,
   VAE, and KV cache regions. Check drperf validity before interpreting counts.
6. **Validation:** compare candidate and baseline native outputs and repeat
   warmed unprofiled timings. `validate_inferix.py` additionally captures
   callbacks, latent tensors, and final VAE cache state; these extra copies make
   its host timings unsuitable as performance results.

## Adaptations must remain visible

`prepare_inferix.py` makes unused UI imports lazy. `compat.py` handles the
removed torchvision video writer and the Diffusers single-GPU configuration
keyword collision. Memory-mapped checkpoint reads reduce setup RAM without
changing the state dictionary or measured generation.

`inferix_metadata.py` moves token lengths and Python KV-cache position counters
to host storage. GX needs actual control metadata even though it skips GPU
arithmetic. This change also removes real GPU synchronization costs, so native
and GX comparisons must use the same adapted source. Retain original native
results separately and validate the adaptation numerically.

`inferix_decode.py` is a candidate patch: TRUE_STREAMING already decodes every
block, but the pinned code also decodes the whole sequence and discards that
second result. The patch skips only that redundant decode. Source inspection
is a lead; native semantic validation and measured results are required before
claiming an optimization.

## Interpreting partial_sync

The timeline retains GPU stream/event dependencies and CPU work, but does not
reconstruct all CPU synchronization. Current runs configure HBM and host-link
rates for modeled copies; missing copy metadata, callbacks, and missing kernel
predictions can still contribute zero time. It is an
optimistic approximation, **not a guaranteed hardware lower bound**.

`summarize_partial.py` reports marked intervals, prediction coverage, CPU
sampling coverage, device busy/idle spans, and large CPU segments before GPU
submissions. CPU/GPU work sums can overlap: neither their sums nor an idle span
alone proves a CPU bottleneck. Validate suspicious regions with drperf and
real A100 measurements. Launch-inventory comparison is an additional gate;
matching shapes and counts cannot prove numerical equivalence.

For each partial run, require successful managed cleanup, a completed model
report with the intended measured request (21 latent frames for Inferix; 81
pixel frames for FastVideo), a nonempty marked virtual
interval from `gxvm report`, no CPU IPC coverage/trace errors, and kernel
prediction hits for every measured launch. Select trace launches by host
`create_ts_us` inside the worker's measured host interval: the task `ts`
switches to a logical clock after adoption. `audit_kernel_trace.py` checks
native-profile versus GX source/settings, launch counts, unambiguous
signatures, metadata, usable samples and hit status. A failed inventory gate
means the virtual duration cannot be presented as a matched replay. The
currently configured Inferix GPU profile has 106,617 observed API rows,
including 8,838 cuBLAS wrapper rows. Its 97,779 non-cuBLAS kernel launches
match the prior 21-frame GX CPU-profile count, but only 97,611 match at the
exact signature; 168 use different cuDNN bf16/tf32 plan variants. Those
variants still fail the [Inferix partial audit](PARTIAL_RESULTS.md). The cuBLAS
wrapper audit found one GX surrogate task per intercepted call, without nested
emulated `cuLaunch*` work; matching that wrapper count does not resolve the
cuDNN signature differences.

CPU calibration currently runs on the GX host (AMD EPYC 7702P), whereas the
A100 host has AMD EPYC 9554 CPUs. Partial timings are therefore screening
estimates for this calibration, not calibrated predictions of the A100 host.
CPU instruction counts remain a separate measurement from CPU elapsed time.

The current Inferix GX-host IPC shard recorded 14,390 resolved instruction
samples and 6,684 unresolved samples: **31.7% of all instruction samples**
were unresolved (`6,684 / (14,390 + 6,684)`). The unresolved count is 46.4%
*relative to resolved samples*, a different denominator. The collector maps
sampled instruction pointers to executable file mappings when it writes the
profile; anonymous/JIT code and mappings gone by then can remain unresolved.
Its fallback cost is therefore material. `native_cpu_inferix_entry.py` prepares
an opt-in measurement of real GPU generation on the A100 host after warmup,
but the managed native container currently has `perf_event_paranoid=4`, no
effective capabilities, and no staged native IPC profiler library. Native CPU
profiling is unavailable under that configuration.

A native-host profile would measure 9554 CPU costs, but would not by itself
prove replay fidelity on the 7702P GX host. GXVM indexes CPU cost by hash of
the absolute mapped module path and a 4 KiB page offset, then falls back to
module or global means. The database does not verify binary identity or ISA
dispatch. Native GPU and GX emulation can execute different host paths; Torch
or libc may also select different code on Zen 4 and Zen 2. Compare module
paths/builds, measured launch inventory, replay page/module/global coverage,
and real A100 times before treating a partial estimate as calibrated.

## Wan2GP: redundant garbage collection, mixed native timing result

GX identified a full CPU garbage collection in MMGP's model-unload path even
when no model was active. The [opt-in guard](cpu_patches/wan2gp_empty_unload.patch)
skips that empty unload while retaining cleanup when active models must be
removed, ordinary GC, and generation-boundary cleanup. The
[runner](wan2gp_runner.py) uses pretrained Wan 1.3B, 17 frames at 256x448,
eight Euler steps, CFG 5, seed 42, and MMGP profile 2 with CPU offload and no
quantization. Measurements cover warmed repeated prompts through VAE decoding
and CPU uint8 output, excluding the UI and file encoding.

| Warmed complete request | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, mean of two requests | 5.4537 s | 5.2772 s | 3.24% |
| GX mode 3, mean of two requests | 6.6395 s | 6.4830 s | 2.36% |
| A100, median of ten requests per variant | 5.7331 s | 5.5433 s | 3.31% |
| A100, mean of the same ten requests per variant | 5.7173 s | 5.9828 s | **-4.64%** |

A100 validation uses ABBA process order, two warmups and five measured requests
per process. All twenty output hashes match. One optimized request took
9.9865 seconds; it is retained in every aggregate. Thus the median improves,
but the sample mean worsens. This is **not a demonstrated mean-latency gain**.
The output check compares quantized uint8 videos, not intermediate floating
point values. Peak allocated GPU memory remains 3,141,723,136 bytes, and peak
reserved GPU memory remains 3,202,351,104 bytes. Native peak host RSS remains
about 21.56 GB. Directional GX copy counts and bytes match in both modes.

Mode 2 reduces modeled CPU work from 3.2600 to 3.0799 seconds per request and
time outside GPU activity from 0.9796 to 0.8006 seconds. Its cuDNN plans differ
(40 versus 10 unpredicted launches); mode 3 has matching full compute launch
inventories, with 40 unpredicted launches in each run. CPU calibration has
2.04% unresolved samples and uses a different CPU model from the A100 host.
The empty unload also clears allocator caches and updates a memory-check
clock; skipping it is opt-in, with other cleanups retained. Unchanged memory
peaks here do not establish safety under unrelated allocation pressure.

The [evidence](evidence/wan2gp-cpu.json) retains all samples, source identities,
coverage, memory, and a narrow GC diagnostic. Compact traces are available for
mode 2 [before](evidence/wan2gp-mode2-before.chrome.json.gz) and
[after](evidence/wan2gp-mode2-after.chrome.json.gz), and mode 3
[before](evidence/wan2gp-mode3-before.chrome.json.gz) and
[after](evidence/wan2gp-mode3-after.chrome.json.gz). They show modeled activity
in 10 ms bins; they are not native CPU-utilization traces. The
[reproduction bundle](evidence/wan2gp-reproduce.tar.gz) and archived calibration
databases preserve the configuration. Enable the candidate with
`MMGP_SKIP_EMPTY_UNLOAD=1`.

Paper-ready description, including the negative result:

> In Wan2GP, GX exposed redundant CPU garbage collection when the MMGP offload
> manager attempted to unload an empty active-model set. Skipping this cleanup
> reduced predicted request latency from 5.454 to 5.277 seconds in mode 2
> (3.2%) and from 6.640 to 6.483 seconds in mode 3 (2.4%), with unchanged
> transfer traffic and GPU-memory peaks. For pretrained Wan 1.3B generating
> 17 frames with eight denoising steps, A100 validation gave a 3.3% lower
> median across ten requests per variant, but a 4.6% higher mean because one
> optimized request took 9.987 seconds. All twenty output hashes matched.
> We retain this mixed result rather than treating the predicted reduction
> as evidence of a native mean-latency improvement.

A separate GX screen, starting from this candidate, cached tiny read-only
FlashAttention sequence-length tensors within a generation. It removed 3,836
small H2D copies across two requests, but predicted latency changed only from
5.2795 to 5.2692 seconds (0.20%). We deprioritized this variant without native
validation; its patch, configuration, and measurements are retained in the
same evidence and reproduction bundle.

## vLLM-Omni: CPU waits between component weight uploads

The public multiprocess `DiffusionEngine.step` in vLLM-Omni v0.22.0 runs the
pretrained Wan 1.3B model with component CPU offload, eager FA2 attention,
17 frames at 256x448, eight Euler steps, CFG 5 and seed 42. It includes the
original scheduler, worker IPC and CPU video postprocessing; UI and file
encoding are excluded. The [opt-in patch](cpu_patches/omni_async_upload.patch)
queues CUDA weight uploads without per-tensor host waits, retaining the final
synchronization before forward execution. CPU offload, transfer bytes, and
peak GPU memory remain unchanged. HSDP and non-CUDA paths retain blocking
uploads. The measured improvement comes from submission scheduling and overlap;
mode-2 CPU instruction cost is essentially unchanged at 1.262 s per request.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, worker interval | 3.7086 s | 3.4217 s | 7.74% |
| GX mode 3, same worker interval | 4.1447 s | 3.8383 s | 7.39% |
| GX mode 3, whole client request | 4.1820 s | 3.8755 s | 7.33% |
| A100, whole-request median | 3.8999 s | 3.7929 s | 2.74% |
| A100, whole-request mean | 3.8998 s | 3.7959 s | 2.66% |

GX mode 2 reduces time outside worker GPU spans from 0.3306 to 0.0399 s.
Its client and worker clocks are independent, so the client mode-2 markers
are **not** an end-to-end latency prediction. Mode 3 synchronizes the processes
and provides the whole-request comparison. Both GX modes use two measured
requests per variant and reuse one native kernel database. Their whole
application wall times, including imports, model loading and warmup, are
82.4/81.0 s in mode 2 and 135.9/129.8 s in mode 3 (before/after).

Native validation uses ABBA process order, two warmups and five measured
requests per process: ten samples per variant, with all twenty finite float32
output hashes identical. Peak allocated GPU memory is 11,863,954,944 bytes and
peak reserved memory is 11,932,794,880 bytes on both sides. Worker peak host RSS
is 22.386–22.390 GB before and 22.402–22.419 GB after. The GPU container was
stopped after validation. All samples and source/runtime identities are in
[the evidence](evidence/omni-cpu.json); the [runner](omni_runner.py) and
[reproduction bundle](evidence/omni-reproduce.tar.gz) preserve the setup.

Compact Chrome/Perfetto activity traces, each 5–11 KB compressed:

- Mode 2: [before](evidence/omni-mode2-before.chrome.json.gz),
  [after](evidence/omni-mode2-after.chrome.json.gz).
- Mode 3: [before](evidence/omni-mode3-before.chrome.json.gz),
  [after](evidence/omni-mode3-after.chrome.json.gz).

These views average activity into 10 ms bins and omit event markers/arrows.
CPU, copy and compute counters overlap and must not be summed as utilization.
Mode-3 CPU spans include elapsed blocking time. Full trace paths and checksums
are retained in the evidence.

GX overestimates the native gain. The mode-2 pair has 40/10 cuDNN prediction
misses and mode 3 has 10/40; cuDNN plans differ, so neither pair is exact native
replay. CPU calibration has 17.69% unresolved samples, and GX/native host CPU
models differ. GX also substitutes the known first Euler index for a
GPU-data-dependent lookup, omitting three CUB launches totaling 17.407 us per
native calibrated request plus unquantified scalar-read/CPU bookkeeping.
It retains the comparison and preceding-stream wait; native validation retains
and checks the original lookup. These limitations remain part of the result.

An earlier direct-pinned-D2H candidate improved GX worker time by only 0.028%
and was not sent for A100 validation. A GX metadata probe showed that this
Torch stack already returns pinned memory for nonblocking D2H, so the following
`pin_memory()` reuses storage. The negative screen is retained in the evidence.

Paper-ready description:

> In vLLM-Omni, GX identified CPU waits between component weight uploads.
> Queuing these uploads asynchronously, while retaining the synchronization
> before model execution, reduced mode-2 worker time from 3.709 to 3.422 seconds
> (7.7%). Mode 3 predicted whole-request latency decreasing from 4.182 to 3.875
> seconds (7.3%). Single-A100 ABBA validation measured 3.900 to 3.793 seconds
> median (2.7%), using ten requests per variant. All twenty returned float32
> videos were finite and byte-identical, with unchanged transfer traffic and
> peak GPU memory. This improvement came from CPU submission scheduling and
> overlap. GX overestimated its magnitude; differing cuDNN plans and a limited
> scheduler-control adapter prevent an exact-replay interpretation.


## Qwen3-TTS: CPU control checks during autoregressive decoding

This extends the CPU search to speech inference. The workload uses the original
`Qwen3TTSModel.generate_custom_voice` API with the pretrained 0.6B CustomVoice
checkpoint, resident BF16 parameters, FlashAttention2, seed 42, and one English
sentence with speaker Ryan. It returns 165,120 float32 samples at 24 kHz (6.88 s
of speech). Timings include tokenization, both autoregressive decoders, speech
waveform decoding and return to CPU; they exclude model loading and file encoding.

The GX timeline exposed repeated device-to-host scalar reads in Transformers'
packed-sequence detection. A batch containing a single integer position cannot
contain multiple packed sequences, so a shape/type check can answer this on the
CPU. Keeping Qwen's binary-mask position indices as integers extends that fast
path to its talker as well as its code predictor; rotary embedding still performs
its original FP32 conversion. The two patches remove 8,428 scalar checks per
request. They retain model weights, sampling, attention computation and audio
output behavior in the tested requests.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, request | 13.393359 s | 12.625325 s | 5.73% |
| GX mode 3, request, 10 us epochs | 17.424990 s | 16.123590 s | 7.47% |
| A100, public API without control hooks, median | 8.303864 s | 7.557471 s | 8.99% |
| A100, public API without control hooks, mean | 8.307572 s | 7.553126 s | 9.08% |
| A100, with matching control-validation hooks, median | 8.618006 s | 7.825970 s | 9.19% |

Each native comparison uses ABBA process order, two warmups and five measurements
per process: ten measured samples per variant, with no samples removed. All 20
uninstrumented measured waveforms have the same SHA256; seven of eight warmups
also match. The first baseline warmup has a different hash, with the same length,
finite status and extrema; its numerical error was not retained. This is not a
claim of cold-start determinism. All 28 outputs in the separate control-checked
comparison match. Peak allocated GPU memory remains 2,378,813,952 bytes, with
2,457,862,144 bytes reserved, in the recorded comparisons.

GX uses the scalar-control tape captured during initial kernel calibration for
this fixed request, since it skips GPU arithmetic. It preserves original scalar
reads and waits except those removed by the patch. Native validation retains
original tensor operations. Known-unique token suppression and known-empty
padding need scoped GX adapters; omitted kernel work is quantified in the
[evidence](evidence/qwen-tts-cpu.json). The candidate also introduces 117 uncovered
integer-operation launches per request, and cuDNN plans/grids can differ. CPU
calibration has 4.42% unresolved instruction samples. Mode 3 uses 10 us epochs;
the many tiny operations, different CPU hosts and control hooks limit absolute
latency comparisons. Both GX modes overestimate native latency here. These are
recorded predictions with coverage limitations, not exact hardware replays.

The integer-only guard alone predicts 13.393359 -> 12.831406 s in mode 2 (4.20%);
it was not separately validated on A100. Source-level CPU oracles cover 33 packed
position cases and 54 binary-mask position-index cases. Native correctness is
limited to the stated prompt, seed and request shape.

Artifacts:

- [Results, all native samples, provenance and limitations](evidence/qwen-tts-cpu.json).
- Mode-2 CPU/GPU activity: [before](evidence/qwen-tts-mode2-before.chrome.json.gz), [after](evidence/qwen-tts-mode2-after.chrome.json.gz).
- Mode-3 CPU/GPU activity: [before](evidence/qwen-tts-mode3-before.chrome.json.gz), [after](evidence/qwen-tts-mode3-after.chrome.json.gz).
- Patches: [single-position check](cpu_patches/qwen_tts_single_position.patch), [integer position bookkeeping](cpu_patches/qwen_tts_integer_positions.patch).
- [Reproduction bundle](evidence/qwen-tts-reproduce.tar.gz), [GX harness](qwen_tts_runner.py), and [kernel/CPU calibration databases](evidence/kernel-databases/README.md).

The compact traces use 10 ms activity bins; each category is a union of modeled
intervals, and overlapping categories must not be added as hardware utilization.
Full detailed traces are linked by path in the evidence. Completed raw captures
are losslessly compressed and SHA256-verified before their raw duplicates are
removed.

Paper text to append:

> We also used GX to identify CPU control overhead in Qwen3-TTS. During autoregressive decoding, its attention interface repeatedly read device-side position metadata to decide whether a sequence was packed, even when a single integer position made the answer known from its shape. A single-position fast path, combined with integer position bookkeeping, removed 8,428 host-visible checks per request. For a fixed request producing 6.88 seconds of speech, GX mode 2 predicted latency decreasing from 13.393 to 12.625 seconds (5.7%), and mode 3 predicted 17.425 to 16.124 seconds (7.5%). Subsequent uninstrumented A100 validation measured a median decrease from 8.304 to 7.557 seconds (9.0%) across ten samples per variant, with identical waveforms in all 20 measured requests and unchanged peak GPU allocation. This optimization avoids redundant control checks and their supporting GPU operations without changing model computation or bulk data transfers. The recorded-request simulation has incomplete kernel coverage, and one cold baseline warmup had a different waveform hash; the numerical comparison applies to the measured steady-state requests.

## Chatterbox Turbo: CPU dispatch and watermark autograd

The original public Chatterbox Turbo API generates 3.4 seconds of audio
(81,600 FP32 samples at 24 kHz) for a fixed English request, built-in voice,
and seed 42. We retain sampling, the pretrained resident model, the audio
decoder and the CPU Perth watermark. Input reference encoding and file encoding
are outside this workload. Source, model and dependency revisions are pinned in
[the evidence](evidence/chatterbox-cpu.json).

GX mode 2 attributes 1.137 seconds of modeled CPU work to a 1.141-second request;
0.595 seconds lie outside modeled GPU activity. The original GPT2 activation
uses eight separate tensor operations. Selecting PyTorch's fused tanh-GELU
removes 13,944 kernel launches per request and their host dispatch work.
Adding `torch.inference_mode()` to the public generation method also avoids
unused autograd recording in the CPU watermark. A CPU oracle on the recorded
native audio checks identical watermark output and observes 27 saved tensors
in the original call, totaling 21.24 MB of tensor references. That count includes
parameters and is not an additional-allocation measurement.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2 request, mean of two | 1.141208 s | 1.015024 s | 11.06% |
| GX mode 2 modeled CPU work | 1.136740 s | 1.009619 s | 11.18% |
| GX mode 3 request, mean of two | 1.391560 s | 1.206600 s | 13.29% |
| A100 request, median of ten per variant | 0.648909 s | 0.559553 s | 13.77% |

A100 mean latency improves by 13.96%. Native validation uses ABBA process order,
two warmups and five measurements per process, with original tensor and FFT
methods and no control-tape hooks. All 20 measured outputs and eight warmups
have the same waveform hash. A separate guarded pair also checks the native
control values, CPU waveform inputs and FFT shapes; its timing is excluded.
Host/device transfer counts and bytes are unchanged in both GX modes. Native
peak allocated GPU memory is 2,969,894,912 versus 2,970,853,376 bytes; peak reserved
memory is 3,024,093,184 versus 3,017,801,728 bytes. All samples are retained.

The initial A100 database includes the original model's kernels and two
additional FP32 fused-GELU input shapes, calibrated before candidate timing.
All optimization timing before final A100 validation uses GX. The fused
activation preserves the mathematical tanh approximation but changes rounding:
primitive calibration observed a maximum absolute difference of 2.38e-7.
Matching audio hashes on this request do not establish bitwise equivalence for
all prompts or seeds.

There are material simulation limits. GX fails in the vocoder's cuFFT calls,
so fixed STFT/ISTFT shape adapters omit their compute and dispatch work. Native
validation uses the original FFTs. Native-only FFT, FP32 GEMM and differing
cuDNN launches sum to 15.70 ms of isolated kernel medians per request; this is
not an end-to-end correction. GX retains 24/22 prediction misses across the
two before/after requests in each mode. CPU calibration has 7.44% unresolved
instruction samples, and the GX and A100 CPU hosts differ. Mode 3 uses 10-us
epochs. Absolute GX times overestimate native latency; the validated result is
the improvement, not exact replay accuracy.

The compact Chrome traces show CPU, GPU and copy activity in 1-ms bins:
[mode 2 before](evidence/chatterbox-mode2-before.chrome.json.gz),
[mode 2 after](evidence/chatterbox-mode2-after.chrome.json.gz),
[mode 3 before](evidence/chatterbox-mode3-before.chrome.json.gz), and
[mode 3 after](evidence/chatterbox-mode3-after.chrome.json.gz).
See the [patch](cpu_patches/chatterbox_fused_inference.patch),
[runner](chatterbox_runner.py), [reproduction bundle](evidence/chatterbox-reproduce.tar.gz),
and [calibration manifest](evidence/kernel-databases/manifest.json).

Paper-ready paragraph:

> In Chatterbox Turbo, GX identified CPU dispatch overhead in the autoregressive
> speech-generation loop: mode 2 placed 1.137 seconds of CPU work on a
> 1.141-second request. Replacing an eight-operation GELU expression with a fused
> implementation of the same tanh approximation and disabling unused autograd
> recording in the CPU watermark stage removed 13,944 kernel launches per
> request without changing transfer traffic. GX predicted latency reductions
> from 1.141 to 1.015 seconds in mode 2 (11.1%) and from 1.392 to 1.207 seconds
> in mode 3 (13.3%). Subsequent A100 validation measured 0.649 to 0.560 seconds
> median (13.8%; 14.0% by mean), with identical waveform hashes across all twenty
> measured requests. This example exposes CPU overhead beyond cache transfers.
> GX required FFT shape adapters and retained uncovered operations; absolute
> simulated times overestimated native latency, so the measurements validate
> the optimization rather than exact timing accuracy.


## Kokoro: inference preparation and CPU length metadata

GX mode 2 shows about 46 ms outside modeled GPU activity in an 86 ms request
for the original Kokoro pipeline. Keeping shape-derived lengths on the CPU and
skipping a one-element packed-RNN sort alone saves only 0.665% in GX. We retain
that low-gain screen. Adding an opt-in `prepare_for_inference()` call after
checkpoint loading, final device/dtype placement and `eval()` materializes the
weights of 89 weight-normalized layers once. Together these changes remove
114 GPU launches and nine metadata-related GPU reads per request.

| Measurement | Original | Prepared inference | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, request mean | 85.918 ms | 82.764 ms | 3.67% |
| GX mode 2, modeled CPU time | 83.537 ms | 79.192 ms | 5.20% |
| GX mode 3, request mean | 115.175 ms | 110.320 ms | 4.22% |
| A100, longer-warmup median, 40 samples/variant | 36.290 ms | 32.902 ms | 9.33% |
| A100, longer-warmup mean | 36.354 ms | 32.996 ms | 9.24% |

The workload uses the original public `KPipeline`, resident FP32 Kokoro-82M,
`af_heart`, seed 42 and “Hello. This is a test of speech generation.” It includes
English text processing, duration prediction, packed RNNs, waveform generation,
CPU output transfers and word timestamps. Each request produces 73,800 samples
at 24 kHz. The initial two-warmup/five-measured-per-process ABBA comparison gave
38.386 -> 36.147 ms median (5.83%), with startup variation. A fixed repeat uses
ten warmups and twenty measured requests per process in ABBA order. All 80
measured and 40 warmup outputs in that repeat match exactly; the initial 20
measured and eight warmup outputs also match. Both comparisons and all samples
are retained. Peak allocated GPU memory changes from 412.53 to 413.35 MB.

This is an explicit inference preparation API: its state dictionary contains
plain materialized weights. Reload the original model to train or load the
original parametrized checkpoint. Preparation cost is outside warmed request
latency. The change removes both CPU dispatch and weight-normalization kernels;
it does not preserve an identical GPU workload.

GX cannot execute the original cuFFT path in this runtime. Its fixed 20-point,
hop-5 FFT shape adapters and recorded duration/output controls are disclosed in
the evidence. `repeat_interleave` receives the native output size of 123 because
GX does not compute speech durations. Native timing uses original FFT operations
and no tensor hooks; separate guarded native checks verify the recorded shapes
and controls. Baseline native-only kernel signatures sum to 0.472 ms/request
of isolated medians, including adapters, FP32 GEMV omissions and cuDNN plan
differences; this is not an end-to-end correction. Six before/four after
uncovered launches remain per two-request capture. The CPU profile has 1.60%
unresolved instruction samples. GX and A100 use different CPU hosts, and mode 3
uses 10-us epochs. Absolute GX latency overpredicts this short native workload.

Evidence and reproduction:

- [Separate GX and A100 results, all samples and limitations](evidence/kokoro-cpu.json)
- [Mode 2 before](evidence/kokoro-mode2-before.chrome.json.gz) and [after](evidence/kokoro-mode2-after.chrome.json.gz)
- [Mode 3 before](evidence/kokoro-mode3-before.chrome.json.gz) and [after](evidence/kokoro-mode3-after.chrome.json.gz)
- [Source patch](cpu_patches/kokoro_prepared_inference.patch), [runner](kokoro_runner.py) and [reproduction bundle](evidence/kokoro-reproduce.tar.gz)

The compact traces include CPU/GPU interval counters at 0.1-ms resolution.
They show the reduced dispatch work; counters are not additive utilization.
Full traces and launch audits remain referenced by the evidence.

Paper-ready addition:

> In Kokoro, GX identified CPU dispatch overhead from repeatedly normalizing
> fixed inference weights. Materializing these weights once, together with
> keeping known sequence lengths on the CPU and avoiding single-sequence
> sorting, reduced simulated request latency from 85.92 to 82.76 ms in mode 2
> (3.7%) and from 115.18 to 110.32 ms in mode 3 (4.2%). Subsequent A100 validation
> measured 36.29 to 32.90 ms median latency (9.3%), with identical outputs across
> all 80 measured requests. The optimization removes repeated host dispatch and
> normalization kernels; GX used recorded output shapes and FFT adapters, so
> the result validates the optimization rather than absolute simulation timing.


## F5-TTS: rotary reuse and fixed-grid solver checks

The original public `F5TTS.infer` path uses pretrained F5TTS_v1_Base (FP16)
and Vocos (FP32), the supplied English reference and transcript, one short
English generation request, 32 Euler steps, CFG 2, and seed 42. Reference
preprocessing, text conversion, the original worker thread, decoding and CPU
waveform return remain in the measured path; ASR and file encoding are excluded.
The output is 81,920 samples at 24 kHz.

GX exposed repeated rotary trigonometric calculations and GPU scalar checks in
the fixed-grid ODE solver. The candidate caches rotary frequencies, sine and
cosine within the existing thread-local inference cache and clears them at the
request boundary. When requested output times are exactly the integration grid,
the solver directly stores each linear-interpolation endpoint. Other grids and
interpolation modes retain their original path. The patches pass 108 solver
trajectory/gradient/callback cases and 72 rotary helper cases, plus a gradient
fallback check. The original source is retained.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, one worker request | 0.994051 s | 0.973704 s | 2.05% |
| GX mode 3, mean caller request | 1.308250 s | 1.265465 s | 3.27% |
| A100, median request | 0.602235 s | 0.582314 s | 3.31% |
| A100, mean request | 0.602400 s | 0.582698 s | 3.27% |

Native validation uses ABBA order, three warmups and seven measured requests per
process (14 measured requests per variant). All 28 measured and 12 warmup audio
and mel hashes match exactly; four additional guarded outputs check GPU controls
and actual FFT/shape behavior. All samples are retained, including variation
between the two candidate processes. This is a modest gain. The solver-only GX
ablation improves mode 2 by 0.30%, with no independent native validation.
The combined change removes 3,158 GPU launches per request, so its benefit includes
both reduced CPU dispatch and reduced GPU work.

Open these compact Chrome traces to compare CPU and GPU lanes:

- Mode 2: [before](evidence/f5-tts-mode2-before.chrome.json.gz),
  [after](evidence/f5-tts-mode2-after.chrome.json.gz).
- Mode 3: [before](evidence/f5-tts-mode3-before.chrome.json.gz),
  [after](evidence/f5-tts-mode3-after.chrome.json.gz).

Mode 2 uses one measured request per capture because this API creates a fresh
worker each request; independent partial clocks made the initial multi-request
worker durations unsuitable. Mode 3 reports caller intervals, with 10 us epochs.
The trace counters use 1 ms bins; full compressed traces and the initial clock
diagnostic are linked from the evidence. CPU spans include framework dispatch and
synchronization, and do not represent pure Python work.

GX uses recorded GPU controls and CPU return values after retaining their original
transfers, plus explicit FFT and GPU-dependent shape adapters. Native timing uses
the original numerical operations with all hooks removed. The baseline omits 21
native launches whose isolated medians sum to 0.176 ms per request; this is not an
end-to-end correction. There are no selected kernel prediction misses. CPU
calibration has 3.10% unresolved instruction samples; CPU hosts differ and GX
absolute times exceed native times. Results apply to this fixed request.

[Full evidence and all native samples](evidence/f5-tts-cpu.json),
[rotary patch](cpu_patches/f5_tts_rotary_cache.patch),
[solver patch](cpu_patches/torchdiffeq_fixed_grid.patch),
[runner](f5_tts_runner.py), and
[reproduction bundle](evidence/f5-tts-reproduce.tar.gz) retain source and runtime
identities, control tapes, oracle checks and launch omissions. Initial GPU and CPU
calibrations are in [kernel-databases](evidence/kernel-databases/README.md).

Paper-ready text to append:

> In F5-TTS, GX identified repeated rotary-position calculations and redundant
> GPU scalar checks in the fixed-grid ODE solver. Reusing rotary trigonometric
> values within each inference request and directly storing solver endpoints
> removed 3,158 GPU launches per request. On a fixed 32-step speech-generation
> workload, GX mode 2 predicted request latency decreasing from 0.994 to 0.974
> seconds (2.0%), and mode 3 from 1.308 to 1.265 seconds (3.3%). Subsequent A100
> validation measured median latency decreasing from 0.602 to 0.582 seconds
> (3.3%), with identical audio and mel outputs across all 28 measured requests.
> This modest gain includes reduced CPU dispatch and GPU work; absolute GX times
> differ from native timing, and the simulation uses recorded control values and
> explicit FFT/shape adapters.


## Soprano: CPU bookkeeping during autoregressive generation

This case uses Soprano's officially supported Transformers backend (4.57.3,
SDPA), with its pretrained 79.7M-parameter BF16 Qwen3 model and 30.4M-parameter
FP32 decoder. The default automatic backend prefers LMDeploy; these measurements
apply to the Transformers path. The original `SopranoTTS.infer` request includes
text normalization, generation, decoding and CPU waveform return. Its built-in
constructor warmup remains outside measured requests. The fixed English request
produces 94,208 samples at 32 kHz, with seed 42 and default generation settings.

GX's baseline trace showed substantial time outside GPU spans. The selected
one-line change replaces `torch.no_grad()` with `torch.inference_mode()` around
the existing `model.generate` call. It suppresses unnecessary inference tensor
bookkeeping. Post-generation stacking remains outside that scope, so the returned
hidden-state tensors retain their ordinary tensor type. Original GPU arithmetic,
control checks, transfers and output handling remain the same.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 2, mean request | 0.957353 s | 0.903567 s | 5.62% |
| GX mode 3, mean request | 1.196030 s | 1.147665 s | 4.04% |
| A100, median request | 0.533662 s | 0.476706 s | 10.67% |
| A100, mean request | 0.534156 s | 0.477347 s | 10.64% |

The selected GX pairs each execute the same 99,846 GPU launches across two
requests, with identical launch inventories and transfer bytes and no prediction
misses. Mode-2 GPU spans remain 0.461936 seconds per request, while modeled CPU
time falls from 0.956610 to 0.901921 seconds. Native peak GPU allocation remains
403,574,272 bytes and peak reserved memory remains 499,122,176 bytes.

A100 validation uses ABBA order, three warmups and seven measured requests per
process, retaining all 14 samples per variant. All 28 measured and 12 warmup output
hashes match exactly, as do four additional guarded outputs. Guarded runs verify
the 197 recorded GPU controls per request and original native FFT shapes; their
timing is excluded. The two baseline process medians are 0.535360/0.532600 seconds,
and the candidate medians are 0.474616/0.479510 seconds.

A separate candidate batches 48 post-generation token-validity checks into one
host transfer. It passes 108 CPU selection/finish-reason cases, but improves GX
mode 2 by only 0.08%. It was not selected for native validation or included in the
reported inference-mode change.

Compact Chrome traces show the CPU and GPU lanes:

- Mode 2: [before](evidence/soprano-mode2-before.chrome.json.gz),
  [after](evidence/soprano-mode2-after.chrome.json.gz).
- Mode 3: [before](evidence/soprano-mode3-before.chrome.json.gz),
  [after](evidence/soprano-mode3-after.chrome.json.gz).

These compact traces use 1 ms occupancy bins. The evidence links full compressed
traces and all per-request intervals. Mode 3 uses 10 us epochs; synchronization
and granularity can change GPU trace spans despite identical launch inventories.
GPU scalar reads and waveform transfers remain in GX, with recorded values
replayed after each transfer. An explicit ISTFT shape adapter omits FFT-related
work. The baseline also omits FP32 GEMV launches: all 64 missing launches per
request have summed isolated medians of 0.571 ms, which is not an end-to-end
correction. CPU calibration has 4.14% unresolved instruction samples. Different
CPU hosts and these simulation limitations prevent an absolute-time accuracy
claim; the native comparison independently confirms the gain for this request.

[Full evidence and all samples](evidence/soprano-cpu.json),
[selected patch](cpu_patches/soprano_inference_mode.patch),
[runner](soprano_runner.py), and
[reproduction bundle](evidence/soprano-reproduce.tar.gz) record pinned source,
model and runtime identities, original control tapes, the unselected ablation,
and validation checks. Initial calibration files are in
[kernel-databases](evidence/kernel-databases/README.md).

Paper-ready text to append:

> In Soprano's Transformers inference backend, GX exposed CPU bookkeeping
> overhead during autoregressive generation. Replacing the existing no-gradient
> context with inference mode around generation preserved the GPU launch
> inventory and transfer volume. GX mode 2 predicted request latency decreasing
> from 0.957 to 0.904 seconds (5.6%), and mode 3 from 1.196 to 1.148 seconds (4.0%).
> Subsequent A100 validation measured median latency decreasing from 0.534 to
> 0.477 seconds (10.7%), with identical outputs across all 28 measured requests
> and unchanged peak GPU memory. This CPU optimization applies to the tested
> Transformers backend; Soprano's preferred LMDeploy backend was not evaluated.
> The simulation uses recorded GPU controls and an explicit FFT shape adapter,
> and its absolute times differ from native measurements.

## ZipVoice: inconclusive CPU screen with missing SGEMM timing

The original public `generate_sentence` request uses FP32 ZipVoice and Vocos,
16 steps, CFG1, seed42, a supplied reference recording/transcript, and the
supported PyTorch Swoosh fallback. The request includes CPU text processing,
feature extraction, postprocessing and WAV encoding. Initial A100 calibration
recorded 370 kernel signatures and 33,576 launches; all calibration outputs
matched. This is calibration, not an optimization validation.

GX mode2 reported 0.797164 seconds/request. Moving GPU lengths to the CPU once
before the existing Python indexing loop changed this to 0.792371 seconds
(0.60%). Guarding sampled attention diagnostics when DEBUG logging is disabled
changed it to 0.797068 seconds (0.012%). The metadata helper passed 360 CPU
oracle cases; the diagnostic guard preserved enabled DEBUG messages and RNG
state in its oracle. Neither candidate received full-model A100 validation.

**The timing screen is inconclusive:** GX omitted 2,369 native launches/request,
including 2,352 FP32 SGEMMs, despite reporting zero prediction misses. The
omitted kernels' isolated medians sum to 146.668 ms/request, which is not an
end-to-end correction. GX's apparent 26.45% outside-GPU fraction therefore
cannot establish actual CPU headroom. The FEMU SGEMM entry points return success
without submitting predicted compute; this is reported in
`GX/NEX/issues/sgemm-timing-zipvoice.md`. No mode3 or optimized native run was
performed for these sub-1% candidates.

[Full evidence and limitations](evidence/cpu-screen-negative.json),
[reproduction bundle](evidence/zipvoice-reproduce.tar.gz),
[baseline trace](evidence/zipvoice-mode2-before.chrome.json.gz),
[CPU-length candidate trace](evidence/zipvoice-mode2-metadata-after.chrome.json.gz),
and [debug-guard trace](evidence/zipvoice-mode2-debug-after.chrome.json.gz).
The compact traces use 1 ms bins; detailed captures and exact configurations
are retained privately, with compressed calibration databases in
[evidence/kernel-databases](evidence/kernel-databases/README.md).


## MeloTTS: CPU waveform concatenation

MeloTTS's public `TTS.tts_to_file` converted every generated float32 audio sample
into a Python float, appended silence as Python integers, constructed a float64
NumPy array, then converted that array back to float32. Keeping samples in array
storage and concatenating the chunks and silence directly removes this CPU
allocation and conversion work. The patch retains the original fallback for
other input dtypes and does not change model execution, cache policy or transfers.

This case uses English-v3 with the pretrained FP32 synthesizer and English BERT,
seed42, public synthesis defaults, and the text “Hello. This is a test of speech
generation.” The request includes text/phoneme processing, BERT feature
extraction, synthesis, the original `empty_cache`, and audio concatenation. It
returns 113,309 samples at 44.1 kHz; file encoding and model loading are excluded.
Torch2.11/CUDA12.8 and Transformers4.57.3 are retained.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode2 request mean | 69.984 ms | 52.352 ms | 25.19% |
| GX mode2 modeled CPU execution | 68.206 ms | 50.584 ms | 25.84% |
| GX mode3 request mean, 10 us epochs | 90.045 ms | 72.370 ms | 19.63% |
| A100 unprofiled request median | 40.485 ms | 34.450 ms | 14.91% |
| A100 unprofiled request mean | 40.521 ms | 34.485 ms | 14.90% |

The selected GX pairs have identical GPU launch inventories and transfer bytes:
4,806 predicted launches across two requests, with zero prediction misses.
Mode2 GPU spans remain 37.727 ms/request; the improvement is CPU work after
waveform transfer. Native ABBA validation used five warmups and twenty measured
requests per process, forty measurements per variant. All eighty measured
outputs, twenty warmups and four separately guarded outputs match the initial
calibration hash `c07e45f51c5390cf6b5203027719dc0fd37826e3fa9639cdfd4007269f6ccabd`.
Peak allocated/reserved GPU memory remains 773.390/843.055 MB. The actual helper
also passed 185 CPU oracle cases, covering multiple chunks, dtypes, noncontiguous
arrays, empty input and special values.

GX's numerical-free execution required replaying native Boolean-index counts
inside the stochastic duration predictor. That adapter omits GPU nonzero/count
work. Together with omitted FP32 SGEMMs, the baseline has 167 native-only
launches/request (74 SGEMMs), whose isolated native medians sum to 6.348 ms. This
is not an end-to-end correction. CPU calibration has 5.72% unresolved instruction
samples, and CPU calibration and A100 validation use different hosts. Both GX
modes overpredict absolute latency and the size of the gain; the independent
A100 comparison confirms the CPU optimization's benefit. Native timing uses the
original indexing and removes all control-replay hooks.

[Recorded GX and A100 evidence](evidence/melo-cpu.json),
[patch](cpu_patches/melo_audio_concat.patch), [runner](melo_runner.py), and
[reproduction bundle](evidence/melo-reproduce.tar.gz). Compact Chrome traces are
2–3 KB each:

- Mode2: [before](evidence/melo-mode2-before.chrome.json.gz),
  [after](evidence/melo-mode2-after.chrome.json.gz).
- Mode3: [before](evidence/melo-mode3-before.chrome.json.gz),
  [after](evidence/melo-mode3-after.chrome.json.gz).

The traces use 1 ms activity bins and show a shorter CPU tail after the waveform
copy. Individual kernel names and dependency arrows are omitted from these
viewing copies; the evidence JSON points to retained detailed traces.

Paper-ready text:

> In MeloTTS, GX identified CPU postprocessing overhead after speech synthesis:
> the framework expanded every output sample into a Python float list before
> rebuilding the waveform as a NumPy array. Concatenating the existing float32
> arrays directly reduced GX mode-2 request latency from 69.98 to 52.35 ms
> (25.2%) and mode-3 latency from 90.05 to 72.37 ms (19.6%), with identical GPU
> launch inventories and transfer bytes. Subsequent A100 validation measured
> a reduction from 40.49 to 34.45 ms (14.9% median over forty measurements per
> variant), with identical output hashes for all eighty measured requests.
> This is a CPU optimization rather than a reduction in GPU computation or
> host-device traffic. GX overestimated the gain; recorded indexing adaptations,
> missing kernel timing and different CPU hosts limit absolute timing accuracy.


## Marvis: CPU cache-position bookkeeping

Marvis's native TorchTune KVCache read its current GPU position for every
bounds check and reset. In eager inference the caller already knows the
position from the sequence lengths it inserts. An opt-in CPU counter replaces
these scalar reads while preserving GPU cache contents, position updates,
reset behavior and bounds enforcement. It requires exclusive cache ownership,
with every mutation going through update/reset; direct cache-position writes
and torch.compile are unsupported while enabled. The default path is unchanged.

The workload uses the native 568.9M-parameter model in BF16 and its original
79.3M-parameter FP32 Mimi codec, with `Hello.`, speaker0, temperature0.9,
topk50 and seed42. It generates the full 32-frame/2.56-second cap, so this is
a bounded request, not a full-utterance quality study. Text processing, the
original semantic-codebook diagnostic print, sampling, decoding and output
transfer are included; loading and file encoding are excluded. BF16 is the
supported `create_model` default; the repository CLI explicitly uses FP32.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode2 request mean | 4.867310 s | 4.685244 s | 3.74% |
| GX mode2 modeled CPU execution | 4.862768 s | 4.685093 s | 3.65% |
| GX mode3 request mean, 10 us epochs | 8.952495 s | 8.205815 s | 8.34% |
| A100 unprofiled request median | 2.877589 s | 2.718174 s | 5.54% |

The change eliminates 4,298 device scalar reads and 8,320 small GPU metadata
kernels per request. It retains the model's arithmetic and cache data updates.
Native ABBA validation uses three warmups and seven measured requests per
process, fourteen measurements per variant. All 28 measured waveforms,
twelve warmups and four separately guarded outputs match the initial hash
`0ac4f85ba03a30969ebfbf25d1e5250519411e01db8528fbb1e05a0ab84b01d8`.
The actual cache helper also passes 54 CPU cases and 528 updates, including
reset cycles, capacity errors, different shapes/dtypes and activation at a
known nonzero position.

Each GX mean contains two requests. Mode3's individual before samples are
8.56445/9.34054 seconds and after samples are 7.86253/8.54910 seconds;
retain that variation when interpreting its 8.34% prediction. Both modes
overpredict absolute A100 latency, and they differ on the size of the gain.
The CPU calibration has 4.66% unresolved instruction samples and uses a
different CPU host from A100 validation. The baseline also omits 64 native
SGEMM/split-K launches per request, with an isolated median sum of 1.114 ms;
that sum is not an end-to-end correction. Selected GX pairs have zero
prediction misses. GPU scalar/list values and output arrays are replayed
under GX after the original transfers/waits; native latency has those hooks
removed. Torch2.11/CUDA12.8 is retained despite Moshi's declared Torch<2.10
constraint, and this unquantized workload is validated independently.

A separate deferred-codebook-concatenation candidate produced no useful GX
gain (4.872912 seconds, 64 prediction misses). It is retained as an unselected
screen, with its own 72-case CPU oracle and no optimized A100 claim.

[Recorded GX and A100 evidence](evidence/marvis-cpu.json),
[patch](cpu_patches/marvis_host_positions.patch), [runner](marvis_runner.py),
and [reproduction bundle](evidence/marvis-reproduce.tar.gz).
Compact traces use 1 ms bins and are 83–342 KB:

- Mode2: [before](evidence/marvis-mode2-before.chrome.json.gz),
  [after](evidence/marvis-mode2-after.chrome.json.gz).
- Mode3: [before](evidence/marvis-mode3-before.chrome.json.gz),
  [after](evidence/marvis-mode3-after.chrome.json.gz).

Paper-ready text:

> In Marvis, GX exposed repeated GPU-to-CPU reads of cache-position metadata
> during autoregressive generation. Keeping an explicit position counter on
> the CPU eliminated 4,298 scalar device reads per bounded request while
> preserving all cache data updates. GX mode2 predicted a reduction from
> 4.867 to 4.685 seconds (3.7%); mode3 predicted 8.952 to 8.206 seconds (8.3%).
> Subsequent A100 validation measured 2.878 to 2.718 seconds (5.5% median over
> fourteen requests per variant), with identical hashes for all 28 measured
> outputs. The optimization is restricted to exclusively owned eager caches.
> GX correctly identified a useful bookkeeping change, although both modes
> overestimated absolute latency and differed in the predicted gain.


## Dia: a low-gain CPU contraction screen

The original native Dia 1.6B pipeline was evaluated on a bounded 128-token,
greedy request with CFG3 and its original DAC codec. Its FP16 configuration
retains FP32 normalization parameters; the compact checkpoint preserves those
actual state dtypes. The workload produces 57,344 samples at 44.1 kHz and
includes text processing, model decoding, codec and CPU output. The first cold
native calibration waveform differs from the second warmup and measured
calibration output; all hashes are retained.

Replacing trailing-axis `DenseGeneral` contractions with equivalent reshapes
and matrix multiplies lowers GX mode2 from 3.033322 to 3.007375 seconds (0.86%).
The GPU launch inventories and transfer bytes are identical, and modeled GPU
spans stay at 1.809861 seconds. The actual helper passes 99 CPU output cases
and two gradient checks, retaining the general path for scalar outputs after
a preliminary shortcut exposed a rounding difference. The gain is small;
this candidate has no mode3 or optimized A100 validation and no full-model
candidate correctness claim. Other Dia CPU opportunities remain unexplored.

The baseline has zero prediction misses but omits 5,974 native launches per
request, including 4,572 FP32 GEMVs and Boolean-index cardinality work, and
adds 24 index kernels through shape replay. Omitted isolated kernel medians
sum to 47.471 ms; this is not an end-to-end correction. CPU calibration has
3.55% unresolved instruction samples. The native kernel calibration is
recorded separately from optimization validation.

[Full screen record](evidence/cpu-screen-negative.json),
[reproduction bundle with candidate patch](evidence/dia-reproduce.tar.gz),
and compact mode2 traces: [before](evidence/dia-mode2-before.chrome.json.gz),
[after](evidence/dia-mode2-dense-after.chrome.json.gz).


## Outlines: CPU bookkeeping in structured generation

The public Outlines generator runs SmolLM2-135M in BF16 with SDPA on four
identical prompts, `Return a 64 digit integer:`, constrained by `[0-9]{64}`.
Greedy generation takes 65 decode steps including EOS. Timing includes prompt
tokenization, grammar masking, generation and output decoding; model loading,
grammar compilation and file I/O are excluded.

An opt-in `inference_mode` flag wraps Torch model generation to reduce CPU
bookkeeping. Its default is false; other backends retain their original path.
The flag is appropriate only when model and logits callbacks do not need
Autograd or produce tensors intended for later Autograd use. Both runners have
identical timed request bodies; the candidate adds only a load-time keyword.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode2 request mean | 1.492917 s | 1.387279 s | 7.08% |
| GX mode2 modeled CPU execution | 1.492446 s | 1.386774 s | 7.08% |
| GX mode3 request mean, 10 us epochs | 2.690205 s | 2.207910 s | 17.93% |
| A100 unprofiled request median | 0.971696 s | 0.871300 s | 10.33% |

Both GX comparisons retain identical GPU launch inventories and transfer bytes;
mode2 GPU spans remain 0.808975 seconds. Native ABBA validation uses three
warmups and seven measured batches per process, fourteen per variant. All 28
measured batches, twelve warmups and four separately guarded batches match the
initial output hash. Native mean latency improves by 10.26%. Nine CPU wrapper
checks cover context restoration, exceptions and backend guards.

The two GX modes overestimate absolute latency and differ substantially on the
gain; mode3 is not treated as hardware ground truth. CPU calibration has 5.00%
unresolved instruction samples and uses a different CPU host. Native compiled
masking uses a 192-block grid while GX uses 96 blocks, producing 520 prediction
misses across two requests. The baseline omits 325 native launches/request
(including 260 native-grid mask launches and 65 FP32 BLAS launches) and adds
260 GX-grid mask launches. Isolated omitted kernel medians total 2.397 ms per
request; this is not an end-to-end correction. No new GPU profiling or synthetic
latency alias was added during optimization.

A separate guard check initially reused a warm compiler cache and skipped a
compilation-only scalar read from the cold calibration tape. Repeating only
the guarded checks with fresh compiler caches resolved that mismatch. All
unhooked timing samples were retained unchanged. An independent mask-retention
candidate removes 260 small D2H calls/request but gives only 0.43% in mode2;
it is recorded separately and has no optimized A100 validation.

[Recorded GX and A100 results](evidence/outlines-cpu.json),
[patch](cpu_patches/outlines_inference_mode.patch), [runner](outlines_runner.py),
and [reproduction bundle](evidence/outlines-reproduce.tar.gz).
Compact CPU/GPU traces:

- Mode2: [before](evidence/outlines-mode2-before.chrome.json.gz),
  [after](evidence/outlines-mode2-after.chrome.json.gz).
- Mode3: [before](evidence/outlines-mode3-before.chrome.json.gz),
  [after](evidence/outlines-mode3-after.chrome.json.gz).

Paper-ready text:

> We also used GX to identify CPU bookkeeping overhead in Outlines structured
> generation. Opting into inference mode around model generation preserved
> all GPU launches and transfer bytes while reducing CPU work. For a batch of
> four regex-constrained 64-digit outputs, GX mode2 predicted 1.493 to 1.387
> seconds (7.1%), and mode3 predicted 2.690 to 2.208 seconds (17.9%). Subsequent
> A100 validation measured 0.972 to 0.871 seconds (10.3% median over fourteen
> batches per variant), with identical outputs in all 28 measured batches.
> The opt-in applies to callbacks that do not require Autograd. This case
> illustrates useful optimization guidance despite discrepancies in absolute
> latency and predicted gain; recorded kernel-coverage gaps limit precision.


## CosyVoice2: autocast caching, a modest native improvement

GX mode-2 traces of the original CosyVoice2 text-streaming API showed repeated
on-device FP32-to-FP16 weight-conversion kernels during autoregressive decoding.
On the pinned Torch 2.11 stack, the `inference_mode` decorator disables autocast's weight
cache. Replacing that one decorator with `no_grad` permits reuse within the
existing outer autocast context. Model parameters and arithmetic precision
remain unchanged; both CPU dispatch and GPU conversion work change.

This removes 74,115 conversion launches per request. All other kernel-signature
counts and host/device memcpy bytes are unchanged in the mode-2 comparison.
It retains half-precision weights:
native peak allocated GPU memory increases from 2.954 to 3.679 GB
(+725.1 MB). The change is specific to this Torch version and execution
path; callers already in inference mode or using frozen parameters may not
benefit. The evaluated model is CosyVoice2, not the inheriting CosyVoice3 path.

| Measurement | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| GX mode 1, GPU-only | 3.666409 s | 2.642673 s | 27.92% |
| GX mode 2, caller-marked request | 7.346833 s | 6.977407 s | 5.03% |
| GX mode 3, synchronized request | 10.425775 s | 9.528135 s | 8.61% |
| A100, unhooked median | 4.613696 s | 4.526336 s | 1.89% |
| A100, unhooked mean | 4.623781 s | 4.524514 s | 2.15% |

The native ABBA comparison retains all ten samples per variant. All 20 measured
outputs, 12 warmups and four separate guarded requests produce exactly the
same 285,120-sample waveform (11.88 seconds at 24 kHz). The workload uses the
official four text chunks, a cached reference speaker, the original RAS
sampler, seed 42, and batched audio output. Model loading and cached-speaker
preparation are excluded. No kernel database was updated during optimization.

This is a small improvement with a memory tradeoff. GX overpredicts absolute
latency and the selected gain. It also omits cuFFT/associated work through a
recorded-shape adapter after an actual cuFFT emulation failure; the native
validation retains these operations. Baseline native-only launches have a
1.883 ms sum of isolated medians per request, which is not an end-to-end timing
correction. Mode 2 retains four prediction misses per variant across two requests.
Mode 3 additionally has a cuDNN algorithm/coverage difference: its baseline
records two malformed transform launches (grid.x 697,008,384, block.x 544),
while its candidate has the four unmatched transform/convolution launches
seen in mode 2. This is reported in the existing GX cuDNN issue. The mode-3
pair is coverage-limited; no guessed geometry or duration correction is used.
The mode-1 candidate has the same symptom with a different malformed grid,
so its device-only comparison is also coverage-limited. These observations
refer to the archived runtime, not an untested newer GX revision.
CPU calibration has 4.16% unresolved instruction samples. Mode 2 has separate
caller/worker clocks: raw worker spans are not request latencies. The reported
mode-2 interval is on the caller; mode 3 synchronizes both threads.
The GX CPU model includes Python control-replay bookkeeping; native latency
uses unhooked requests. This is another source of timing distortion, not a
measured application bottleneck or an additive correction to these results.

[GX/A100 results, samples, hashes and limitations](evidence/cosyvoice-cpu.json),
[one-decorator patch](cpu_patches/cosyvoice_autocast_cache.patch),
[runner](cosyvoice_runner.py),
[reproduction bundle](evidence/cosyvoice-reproduce.tar.gz).
The compact Chrome activity views show CPU execution, GPU work and copies:
[mode-2 before](evidence/cosyvoice-mode2-before.chrome.json.gz) /
[after](evidence/cosyvoice-mode2-after.chrome.json.gz), and
[mode-3 before](evidence/cosyvoice-mode3-before.chrome.json.gz) /
[after](evidence/cosyvoice-mode3-after.chrome.json.gz).
They aggregate interval coverage into 5 ms bins; original captures remain
archived and are referenced by hash. These are simulated timelines.

Two other independent screens were not selected: replacing GPU scalar-list
construction with stacking reduced mode-2 latency by 1.61%, with new prediction
misses; constructing only the used causal-mask row changed latency by 0.0025%.
Their CPU equivalence checks and complete GX audits are retained. Neither has
optimized native validation. Omitting the known-unpadded streaming mask reduced
mode-2 latency by 0.54%, so this third screen was also stopped before
native validation. It preserves the nonstreaming path after an oracle found a
cached-mask semantic difference there. The earlier
[mask source/oracle bundle](evidence/cosyvoice-candidate.tar.gz) remains available.

The separate vLLM path copies `token_ids` into a new Python list to read its
last element at each decode step. This remains an unmeasured source lead;
the vLLM backend was not used in the reported workload.

Paper text for this modest result:

> In CosyVoice2, GX exposed repeated weight conversions caused by an interaction
> between inference mode and autocast caching on Torch 2.11. Allowing the existing
> cache to reuse these conversions removed 74,115 GPU launches per request.
> GX mode 2 predicted 7.347 to 6.977 seconds (5.0%);
> mode 3 predicted 10.426 to 9.528 seconds (8.6%), with a
> recorded cuDNN coverage difference between the two runs.
> Independent A100 validation measured a smaller median reduction from
> 4.614 to 4.526 seconds
> (1.9%), with identical outputs in all 20 measured
> requests. The change increased peak GPU memory by 725 MB. This case
> records the distinction between a GX optimization prediction and its measured
> benefit, including the memory tradeoff and simulation coverage limitations.

Follow-on FunASR source inspection is retained in the
[source-screen record](evidence/cpu-screen-negative.json). A redundant reset
of the published SenseVoice configuration costs about 53 microseconds in a
CPU-only scale check, so this lead was deprioritized before GPU calibration.
The other decoding leads remain unmeasured; this is not a pretrained FunASR
timing result.
