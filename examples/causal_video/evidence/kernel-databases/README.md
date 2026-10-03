# Native A100 kernel databases

These are the actual databases used for the final baseline GX partial runs,
compressed without changing their contents. `manifest.json` records compressed
and uncompressed SHA-256 hashes. Decompress into an ignored output directory
and use the resulting SQLite file as `config.py --gpu-db`:

```bash
mkdir -p out/causal-video/shipped-kernels
gzip -dc examples/causal_video/evidence/kernel-databases/fastvideo-offload.sqlite.gz \
  > out/causal-video/shipped-kernels/fastvideo-offload.sqlite
```

Each adjacent `.report.json.gz` contains its native profiling report: model
settings, source hashes, runtime configuration, and output summary. The
database records measured kernel samples and observed launch inventories;
it contains no model weights or output videos.

- **Inferix:** host KV offload, original streaming decode, metadata adaptation,
  21 latent frames at 480×832, four denoising steps per block.
- **FastVideo:** layerwise DiT offload, metadata and worker-output adaptations,
  81 pixel frames at 480×832, worker intra/inter-op threads both one.

Both profiles used the real pretrained models on A100 GPU1. Native sampling
coverage does not guarantee emulated launch parity: the audits document
cuDNN variants missing from these databases. GPU-resident/optimized variants
need their own matched profiles. See the study's native and partial results
for scope and limitations.


LightX2V's `lightx2v-offload.sqlite.gz` adds the Self-Forcing 1.3B profile with
block weight/KV offload, FA2, asynchronous Wan VAE, and 21 pixel frames. It
contains 294 signatures, 785 usable samples, and 22,104 observed launches.
`lightx2v-validation.json.gz` bundles its `kernel_profile` report and separate
native before/after reports at 21 and 81 frames. The longer GX case reuses the
21-frame database; no 81-frame GPU profiling was performed. See the
[LightX2V study](../../README.md#lightx2v-gx-discovery-drperf-interface-a100-validation)
for prediction misses, configuration, and numerical/latency validation.

FlashDreams' `flashdreams.sqlite.gz` contains the corrected Self-Forcing 1.3B
calibration: 307 signatures, 825 usable samples, and 40,943 observed launches,
nine chunks (105 frames), default cuDNN SDPA and compilation. CUDA graphs were
disabled only to expose individual kernels to the profiler; the GX target uses
graphs. `flashdreams-profile.json.gz` bundles the incomplete initial calibration,
corrected calibration, and separate native before/after reports for direct-dtype
loading and CPU-tokenizer retention. The retention experiment reuses the same
kernel database. The initial
229-signature capture bypassed extended-driver resolver paths and lacked launch
counts. Corrected interception fixes that gap; generated cuDNN names and different
Triton autotune choices still prevent full GX matching. Hashes and limitations
are in `../flashdreams-screen.json`.

Causal Forcing++'s `causal-forcing.sqlite.gz` contains the framewise one-step
1.3B calibration: 332 signatures, 866 samples, and 118,865 observed launches for
81 frames at 480x832. `causal-forcing-profile.json.gz` bundles the native kernel
profiling report and the verified EMA checkpoint conversion. It contains no
native candidate comparison. Attention context lengths alias to one launch
signature, and GX cuDNN variants still differ. This was a
[CPU-headroom screen](../../README.md#causal-forcing-small-modeled-cpu-headroom),
not a validated GX speedup estimate.

Matrix-Game 3's `matrix-game.sqlite.gz` contains the pretrained distilled model
calibration: 810 signatures, 1,909 samples, and 50,264 observed launches for
97 frames at 704x1280, INT8 projections, FA2, and compiled MG-LightVAE.
`matrix-game-profile.json.gz` is the native kernel calibration report, including
control-adapter assertions and output hashes. It is not a candidate comparison.
FA2 input aliasing and 561 GX prediction misses limit timing interpretation.
See the [screen](../../README.md#matrix-game-3-output-copies-dominate-cpu-work).

Scope's `scope.sqlite.gz` contains LongLive 1.3B with the built-in performance
LoRA: 485 signatures, 1,250 samples, and 103,099 observed launches for 69 frames
at 320x576 with a midstream prompt switch. `scope-profile.json.gz` bundles kernel
calibration, an unprofiled control-adapter validation with the same output hash,
and verified checkpoint conversions. No candidate comparison is included.
See the [screen](../../README.md#scope--longlive-host-dispatch-mostly-overlaps-gpu-work).


StreamDiffusionV2 adds `streamdiffusionv2.sqlite.gz` (410 signatures, 1,230 native
samples, 143,013 launches, zero skips) and `streamdiffusionv2-profile.json.gz`
(kernel calibration, final native before/after reports, and source audit).
`streamdiffusionv2-replay.json.gz` contains the exact calibrated control `phases`,
plus `cpu_profile` and `cpu_profile_global` strings. The StreamDiffusionV2 runner
accepts this compressed bundle directly with `--tape`; extract the selected CPU
profile string to a file for GXVM replay. The aggregate profile is a sensitivity
model because the original page calibration is biased. These controls cover
only the pinned input and 17/33/65-frame request order. See the case README for
cuDNN/FA2 prediction limitations and the distinction between GX metadata
adaptation and final upstream-native validation.

TeleFuser's `telefuser.sqlite.gz` contains ABot-World 5B with eager Torch SDPA,
480x832, twelve three-latent-frame control blocks, default 18-frame KV window,
and resident weights: 324 signatures, 855 samples, 121,321 observed launches,
zero skips. `telefuser-profile.json.gz` bundles that calibration, final native
before/after reports with original framework controls, and model provenance.
`telefuser-cpu.db.gz` is the CPU calibration used unchanged in the paired GX runs.
Both final GX inventories match native exactly and have full prediction coverage;
attention input lengths still alias under a single signature. The output-layout
patch adds no GPU kernels and needs no new calibration. See the
[case](../../README.md#telefuser--abot-world-cpu-image-layout-modest-latency-gain).

SkyReels-V2's `skyreels.sqlite.gz` contains DF-1.3B-540P at 544x960, 30 steps,
33/65/97 frames: 430 signatures, 1,230 samples, 415,485 launches, zero skips.
`skyreels-replay.sqlite.gz` removes identical occupancy-query records only;
all kernel samples and distinct query results are unchanged. Use that copy for
the diagnostic replay to avoid repeated scans of 43,200 duplicate records.
`skyreels-profile.json.gz` includes native calibration and checkpoint conversion
provenance; `skyreels-cpu.db.gz` holds the unchanged CPU calibration. The bounded
GX screen uses only 33 frames, has 1,800 misses, and retains FA2 signature aliasing.
No application optimization or candidate comparison is included. See the
[screen](../../README.md#skyreels-v2-simulator-overhead-exposed-little-modeled-cpu-headroom).

DiffSynth-Studio uses `diffsynth-wan.sqlite.gz` and the associated native report
for pretrained Wan 1.3B, 17 frames at 256x448, and eight steps. Its
`diffsynth-wan.cpu.db.gz` is the separate GX-host CPU instruction calibration,
not a GPU database. The kernel profile used a 2-GiB adaptive residency budget;
the full-offload CPU comparison reuses it and records unmatched cuDNN plans.
See `../diffsynth-cpu.json` for repetitions, source identities, and limitations.

VideoX-Fun uses `videoxfun-wan.sqlite.gz`, its native calibration report
`videoxfun-wan.report.json.gz`, and instruction calibration
`videoxfun-wan.cpu.db.gz`. The actual pretrained Wan 1.3B profile has 17 frames,
256x448 resolution, eight steps, and sequential CPU offload. Both GX candidates
reuse it, including the 50-step experiment. cuDNN plan misses remain; matching
launch names/dimensions do not imply unique input shapes or exact hardware timing.

The `xdit-wan-*` and `sglang-wan-*` artifacts retain kernel/CPU calibration and
native calibration reports for the negative CPU screens. SGLang uses the
stable-name kernel database and the corrected continuous steady CPU profile;
these are not databases from a validated optimization comparison.

WanVideoWrapper's `wanwrapper-wan.sqlite.gz` is the initial A100 kernel
calibration for pretrained Wan 1.3B, 17 frames at 256x448 and eight Euler steps.
`wanwrapper-wan.cpu.db.gz` is the separate GX-host CPU calibration under
ComfyUI inference mode. The native kernel report used `no_grad`; its recorded
GPU peak covers only the post-reset VAE phase. Correct whole-pipeline peaks and
all separate A100 validation samples are in `../wanwrapper-cpu.json`. Neither
the calibration report's instrumented latency nor its partial peak is used as
the native before/after result.

Wan2GP's `wan2gp-wan.*` calibration files cover unquantized pretrained Wan 1.3B,
17 frames at 256x448, eight Euler steps, and MMGP profile 2. The initial kernel
calibration and later timing runner have different metadata-control adapters;
the original runner is retained in the reproduction bundle. A separate native
control run checked the corrected tokenizer lengths and Euler indices and
reproduced the calibration output hash. The calibrated GPU operations are
retained by these adapters. Performance claims use the matched later runners,
not the instrumented calibration request's wall time.

`omni-wan.*` records vLLM-Omni v0.22.0 Wan 1.3B, 17 frames at 256x448,
8 Euler steps, component CPU offload and FA2. The native database has 300
signatures and 25,771 observed launches, with no skipped launches. Calibration
is not an unprofiled latency baseline. GX's known-index adapter omits three
CUB launches and scalar-read work; native validation retains the original
lookup. See `../omni-cpu.json` for mode-2/mode-3 misses, all native samples,
CPU calibration limitations, and source/runtime identities.

`fastgen-wan.*` records FastGen's resident pretrained Wan 1.3B path with
17 frames at 256x448 and eight UniPC steps: 320 signatures, 36,575 native
launches, no skipped calibration launches. It is a negative CPU screen,
not an optimized native comparison. The scoped solver/status and timestep
adapters passed native checks. GX omits 18 solver launches per request in
addition to the three declared initial-index launches, and has five malformed
cuDNN-grid misses. See the FastGen entry in `../cpu-screen-negative.json`.

Qwen3-TTS CustomVoice0.6B: `qwen-tts.sqlite.gz`, `qwen-tts.report.json.gz`, and
`qwen-tts.cpu.db.gz` contain the initial A100 kernel calibration, its report,
and the GX CPU instruction calibration. The fixed-request control tape and
scoped GX adapters are documented in `../qwen-tts-cpu.json` and included in
`../qwen-tts-reproduce.tar.gz`. Kernel misses and omitted work remain explicit;
these databases do not establish numerical output correctness.

Chatterbox Turbo retains separate initial model and fused-GELU primitive shards,
their merged predictor, the native calibration report and CPU instruction
calibration. See the `chatterbox` entry in `manifest.json`; final A100 latency
validation is recorded separately in `../chatterbox-cpu.json`.


Kokoro: `kokoro.sqlite.gz`, `kokoro.report.json.gz` and `kokoro.cpu.db.gz`
contain the original 240-signature / 3,085-launch A100 calibration and the
continuous GX CPU instruction calibration. Checksums are in `manifest.json`.
See `../kokoro-cpu.json` for selected before/after mode-2/mode-3 audits, the
metadata-only screen, and both complete native ABBA comparisons. FFT and
shape-discovery adapters are explicit; kernel calibration is separate from
unprofiled native latency validation.

F5-TTS: `f5-tts.sqlite.gz`, `f5-tts.report.json.gz`, and `f5-tts.cpu.db.gz`
contain the initial A100 calibration (140 signatures, 42,004 launches, no skips)
and the CPU instruction calibration. Native control calibration subsequently
recorded concrete GPU-dependent shapes without changing the kernel database.
See `../f5-tts-cpu.json` for mode-2/mode-3 results, FFT/shape omissions, all
A100 validation samples, and `../f5-tts-reproduce.tar.gz` for both calibration
runner versions, final runner and exact source/dependency identities.

Soprano: `soprano.sqlite.gz`, `soprano.report.json.gz`, and `soprano.cpu.db.gz`
contain the initial A100 calibration (159 signatures, 49,987 launches, no skips)
and CPU instruction calibration for the official Transformers backend. The same
runner and original control tape serve the selected inference-mode comparison.
See `../soprano-cpu.json` for omitted native work, mode-2/mode-3 results and all
A100 samples, and `../soprano-reproduce.tar.gz` for exact source/runtime identities.

- **MeloTTS:** English-v3 public request with English BERT, 234 native signatures, 2,570 launches, no profiler skips. GX Boolean-index shape replay and omitted FP32 SGEMMs leave 167 native-only launches/request; use the accompanying audit rather than assuming zero prediction misses implies full coverage.
- **ZipVoice:** original 16-step FP32 public request, 370 signatures, 33,576 native launches. Material SGEMM omission makes this an inconclusive CPU screen; no optimized A100 result is claimed.

- **Marvis:** native BF16 TorchTune model with FP32 Mimi, 225 signatures, 279,580 calibrated launches and no skips. The selected eager CPU position counter removes 8,320 small metadata kernels/request. The baseline omits 64 SGEMM/split-K launches/request; mode2/mode3 and A100 results, 4.66% unresolved CPU samples, exact source/dependency identities and all latency samples are in `../marvis-cpu.json`.

- **Dia:** original native1.6B FP16 model with FP32 norms and DAC,191signatures208806native launches,no skips. First cold warmup waveform differs; secondwarmup/measured calibration agree. Low-gain0.86% GX contraction screen, no optimizednative/mode3 validation. Baseline omitted4572FP32GEMVs plusBoolean-indexwork; see `../cpu-screen-negative.json`.

- **Outlines:** SmolLM2-135M BF16 constrained generation, 142 signatures and 81,455 native launches, no profiler skips. Identical GPU inventory/transfer bytes for the selected inference-mode pair. Compiled-mask grid mismatch causes 520 GX prediction misses/two requests; 325 native-only launches/request. Separate mode2/mode3 predictions, all 28 A100 timings, guard-cache recovery and exact output checks are in `../outlines-cpu.json`.

- **CosyVoice2:** pretrained 0.5B LLM plus flow and HiFT, 1,521 signatures and 462,701 native launches, no profiler skips. The original database is reused for every candidate. Mode 2 preserves all non-cast launch counts for the autocast-cache pair; modes 1 and 3 have recorded cuDNN geometry/algorithm differences. GX substitutes recorded FFT shapes and omits associated work. Separate predictions, all 20 unhooked A100 timings, exact outputs, memory costs and control-replay limitations are in `../cosyvoice-cpu.json`; `cosyvoice.sqlite.gz`, `cosyvoice.report.json.gz`, and `cosyvoice.cpu.db.gz` hold the calibrations.
