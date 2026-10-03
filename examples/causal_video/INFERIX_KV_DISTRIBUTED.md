# Distributed Inferix KV transfers

The follow-up to the [CPU-offload transfer case](INFERIX_KV_TRANSFER.md)
removes the single-GPU restriction. It uses Inferix's existing distributed
attention and collectives, with the same full-capacity working buffers and CPU
KV backing. Production changes remain in the same two Self-Forcing files.

## Cache coordinates

Let `U` be Ulysses degree and `R` ring degree. Input activations are sharded
across `U*R` ranks. After Ulysses all-to-all, the cache stores `H/U` heads and
`C/R` token slots per rank. Consequently:

```text
cache start = global current_start / R
new cache tokens = local input tokens * U
sink slots = sink frames * global tokens per frame / R
```

These quantities are used consistently for fetching history, detecting
whether history must be rolled, updating cache metadata, and dirty writeback.
`CoreAttention` already writes K/V after redistribution; distributed
Self-Forcing must not first write the incompatible pre-redistribution tensors
into the same cache. The single-GPU path continues writing directly.

The earlier upstream path divided start by the total world size and wrote
full-head local tensors into head-sharded storage. Removing the transfer guard
alone would therefore be incorrect for Ulysses.

## Numerical regression

Run in the fork checkout:

```bash
python3 tests/unit/test_self_forcing_kv_transfer.py
python3 tests/unit/test_self_forcing_kv_distributed.py
```

The distributed test passes with U2/R1, U1/R2 and U2/R2: 24 scenario combinations
and 480 rank-local forwards, covering offload on/off, sink 0/1, one-/three-frame
blocks, repeats, fill, eviction and reset. It executes the actual block,
self-attention, cache wrapper and `CoreAttention.forward` source methods. Gloo
all-to-all and all-gather transport actual CPU tensor values. Dense attention
and identity projections/RoPE isolate cache and communication semantics.
An independent cache model checks exact K/V contents and transfer ranges;
outputs agree within 1e-6. Unread scratch is poisoned with NaNs.
The earlier original-versus-candidate CPU regression also passes all 144 paired
steps for the single-GPU path.

The multi-frame Ulysses test preserves the existing rank-major stored-prefix
sink behavior. It does not redefine that prefix as a chronologically complete
first-frame sink. Distributed tests use batch 1, as required by CoreAttention.

## Validation scope

GX completion checks allocation, CUDA/NCCL execution and transfer bounds; GX
skips GPU arithmetic and cannot prove generated-video equality. The earlier
native 41.2787 -> 25.2925 second measurement belongs to the single-GPU revision
`def4cb8`, before this distributed extension. No multi-GPU native speedup or
full-model numerical equality is claimed here.

## Full-model GX results

Both managed emulation-only runs pass with empty cleanup-error lists. Each runs
the real pretrained Self-Forcing 1.3B pipeline at 480x832 for six latent frames,
with original decoding, four denoising forwards plus clean-context forward per
block, and CPU KV offload enabled. Two three-frame blocks exercise history.

| Layout | Per-rank capacity | History reads, first / second block | Write size per forward | Per-rank collectives |
| --- | ---: | ---: | ---: | --- |
| U1/R2 | 16,380 tokens | 0 / 2,340 | 2,340 | 300 batched point-to-point calls; 10 all-gathers |
| U2/R1 | 32,760 tokens | 0 / 4,680 | 4,680 | 1,200 all-to-alls; 10 all-gathers |

Every rank records 300 reads and 300 writes, exactly matching the expected
ranges. Both patched source file hashes match PR commit `50ee5ab`. These are
actual CUDA/NCCL paths under GX, not mocked communications. No GPU predictor,
CPU replay or timing simulation is enabled.

The test-only compatibility loader restores Inferix's distributed constructor
when newer Diffusers consumes its parallel_config keyword. It also initializes
the text encoder on meta and strictly assigns the actual mmap checkpoint,
checking that no meta tensors remain; this avoids simultaneous temporary FP32
encoder allocations and stays within the GX host's unchanged memory guard.
The original pipeline converts that encoder to BF16 before inference. These
loader adaptations are outside the production PR and are archived separately.

Nonzero ranks skip streaming callbacks and therefore run the existing full-video
decode fallback: rank0 returns 18 chunk-decoded frames, rank1 21 full-decoded
frames. This is the source pipeline's behavior, not a GX numerical assertion.

## Four-rank GX integration

U2/R2 also passes using actual constructed Inferix attention blocks, cache
managers, FlashAttention and NCCL. All four ranks complete 32 forwards covering
offload on/off, sink0/1, repeat/fill/evict/reset. Each executes 128 all-to-alls
and 32 batched point-to-point calls. This smaller integration test does not load
the full pretrained model on four ranks. Source hashes match commit `50ee5ab`.

The [draft PR](https://github.com/MJChku/Inferix/pull/1) includes the production
change and CPU regressions. GX scripts, configurations and per-rank reports are
archived in `/home/ubuntu/GX/NEX/Example/infrix_kvcache/distributed/`.

## Eight-rank partial-sync comparison

A matched U2/R4 pair completes the full pretrained 1.3B model with 21 latent
frames and CPU KV offload retained. Both variants include the distributed
layout correction. The controlled baseline restores full-capacity transfers;
the candidate fetches needed history and writes changed ranges only.

The initial exact-match predictor gives 18.894638 -> 16.955283 virtual seconds,
10.264% lower. Aggregate H2D+D2H payload falls from 336.637308 to 125.295996 GB:
the 211.341312 GB saving exactly matches the cache-transfer calculation.
Each GPU saves 26.417664 GB, about 2.03 seconds of host-link service at the
configured 13 GB/s. Transfers on eight GPUs run concurrently, explaining why
the gain is smaller than the single-GPU case's roughly 16-second saving.

Both managed runs and all-rank dependency reconstruction pass. Each has 38
matched collective groups and 117,600 matched point-to-point transfers. Timing
uses the maximum marked iteration over all eight ranks. This is GX virtual
timing, not native hardware timing. Exact kernel launch coverage is only
37.933% before and 37.454% after; unmatched kernels contribute zero time in
these initial reports. A completed follow-up with explicit launch dimensions
also gives about 10% lower virtual time across its scaling sensitivities.

`GX_APP_REAL_MAX_BYTES=8388608` keeps bulk K/V payloads fake-backed while
NCCL control allocations remain real. Higher application thresholds caused
mixed-backing launches or excessive host memory usage. No NCCL check was
disabled. The decoder follows the existing pipeline: rank 0 emits 63 streaming
frames; nonzero ranks perform a full-decode fallback producing 81 frames.
Both variants execute the same decoding behavior.

The complete trace pair, configurations, per-rank reports and compressed raw
captures are in
`/home/ubuntu/GX/NEX/Example/infrix_kvcache/eight_gpu_partial_sync/`.
They are separate from the historical single-GPU trace pair.

## Weightless 14B on eight emulated GPUs

The completed matched-runtime follow-up gives **35.5258 -> 26.4905 seconds
(25.43%) in mode 2** and **39.7815 -> 30.5189 seconds (23.28%) in mode 3**.
Both use the same 45 archived kernel-duration extrapolations during execution;
mode 3 uses a 10-microsecond epoch. Aggregate host/device traffic remains
1,485.765 -> 546.470 GB. There is **no native eight-GPU validation**. The
`matched_current_runtime` entry in the [evidence](evidence/inferix-kv-eight-timing.json)
records timing, measured/estimated/missing launch counts, and checksummed paths
to the complete GX results and trace archive. The 32.9922 -> 23.8884 result below
is retained as historical evidence from the older runtime, not a matched sample
for the new mode-3 result.

The same optimization was evaluated using Inferix's actual Wan T2V-14B
architecture (40 layers, 40 heads, width 5120), with parameters allocated
without loading checkpoints. Both runs use U2/R4, 21 latent frames at 480x832,
and CPU KV offload. A harness override makes the cache loops use all 40 layers.

| GX result | Full-cache transfers | Bounded transfers |
| --- | ---: | ---: |
| Maximum marked virtual interval | 32.9922 s | 23.8884 s |
| H2D + D2H traffic, all ranks | 1,485.765 GB | 546.470 GB |
| Peak host container memory | 38.068 GiB | 19.205 GiB |
| Peak virtual GPU allocation, rank 0 | 45.004 GiB | 45.004 GiB |

Virtual latency falls **27.6%**. The **939.295 GB** traffic saving exactly
matches the cache-range calculation. Both traces retain 38 collective groups
and 156,800 point-to-point transfers. All eight ranks pass source, harness,
configuration, shape and memory-capacity checks. This is a weightless GX
estimate, not a native eight-GPU measurement or numerical validation.

The model uses explicit 80 GiB device capacity with the existing A100 profile;
the 45 GiB allocation peak would not fit the earlier 40 GiB configuration.
Changed kernel launch configurations use same-name grid scaling; cuBLAS
uses an explicitly estimated matrix-work model. Unresolved predictions remain
visible in the result. The full application took 15m49s before and 14m19s
after on the GX host, including warmup. A short pre-attachment CPU sample
points to NCCL emulation, Boost fibers and metadata lookup as substantial
execution overhead; warmup is not instrumented by DynamoRIO.

The existing GX case directory above contains `14b-before.chrome.json.gz`,
`14b-after.chrome.json.gz`, `14b-results.json`, two verified capture archives
and one reproduction archive. Intermediate command logs and regenerable
trace/report files are removed after preserving the final evidence.

The same pair also passes in GX `gpu_only_partial_sync`, with zero recorded
CPU work and no DynamoRIO. Maximum marked virtual time is **30.7016 -> 21.6616 s**
(29.44% lower); measured generation wall time is **408.612 -> 376.826 s**,
or **13.31x -> 17.40x** wall/virtual slowdown. Including setup and warmup,
the applications take 13m46s and 12m02s. The optimized generation uses 24.08%
less wall time than the earlier CPU-enabled run. The runtime was also rebuilt
for the new mode, so this does not isolate instrumentation overhead alone.
GPU prediction assumptions are unchanged, including explicit kernel estimates
and unresolved predictions. The GPU-only artifacts were removed during cleanup;
the numerical summary remains in [the timing evidence](evidence/inferix-kv-eight-timing.json).
