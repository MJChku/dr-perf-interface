# Inferix KV transfer optimization with CPU offload retained

The [distributed follow-up](INFERIX_KV_DISTRIBUTED.md) extends the PR to ring and
Ulysses layouts. The measurements and archived patch below describe the initial
single-GPU revision; distributed validation is recorded separately.

GX predicts **38.4180 s → 22.3073 s**, a **41.94% reduction** in virtual time
(1.72× speedup), for the same full Self-Forcing request with KV offload enabled.
Host/device traffic falls **63.54%**, without selecting GPU-resident KV backing.
A subsequent native A100 PCIe 40GB comparison measures **41.2787 -> 25.2925 s**
median across three generations after full-size warmup: **38.73% lower latency
(1.63x)**. Saved video tensors match exactly at zero tolerance. Peak allocated
GPU memory is essentially unchanged (24.8587 -> 24.8580 GB); allocator-reserved
peak increases from 26.8918 to 27.4685 GB. Offload remains enabled in both.
Native validation covers one prompt/seed/resolution and 21 latent frames, with
shared lazy-import and host-metadata compatibility adaptations. The two patched
source files are the only differences between the measured variants.

| Measured quantity in GX | Baseline | Bounded transfers |
| --- | ---: | ---: |
| Marked virtual duration | 38.418029 s | 22.307320 s |
| H2D bytes | 211,435,745,184 | 90,669,281,184 |
| D2H bytes | 121,162,752,000 | 30,587,904,000 |
| H2D + D2H service, summed | 25.584500 s | 9.327476 s |
| D2D bytes | 512,578,119,680 | 512,578,119,680 |
| CPU work, summed across threads in marked interval | 4.707262 s | 4.825387 s |
| Prediction misses | 168 | 42 |

The application command, model, prompt, seed, 21 latent frames, resolution,
denoising schedule, streaming/decode behavior, CPU/kernel databases, GX runtime,
and 13 GB/s host-link calibration are unchanged. No new GPU profiling was used.
The two application source edits are staged as an immutable managed asset.

## What changes

The original layer fetches its entire allocated self-attention KV cache before
each forward and writes the entire active prefix back afterwards. In the normal
path, attention overwrites only the current block's KV. The candidate:

1. Fetches only history preceding the current block, including when a denoising
   step rewrites the same block. Current-block entries are overwritten before
   attention, and unused cache capacity is not read.
2. Writes back only the modified range. When a sliding window evicts tokens,
   it also writes the shifted history while preserving the sink prefix.
3. Retains the original full-transfer path for GPU-resident and distributed
   configurations. Bounded transfers apply to CPU-offloaded, single-GPU inference.

`enable_kv_offload=True` and the CPU backing allocations are retained. GPU working
buffers keep their original capacity; Inferix already retains these in
`kv_cache_meta` between calls. This candidate preserves that behavior and adds
no persistent GPU KV allocation. It does not implement a dynamic memory-pressure
policy. Native peak allocated and allocator-reserved memory are reported above.

H2D copy count increases from 5,333 to 6,083 because K and V histories are copied
separately; their much smaller payload reduces modeled service. D2H count stays
2,167. Every recorded copy has metadata and calibrated service. Fixed copy
latency and contention between independent copy engines are not modeled by GX.

## Symbolic transfer model

For uniform blocks without eviction, let `L` be layers, `R` forwards per block,
`B` generated blocks, `T` tokens per block, `C` cache capacity in tokens, and `Q`
bytes per token for both K and V. Self-attention cache traffic is:

| Direction | Original bytes | Candidate bytes |
| --- | --- | --- |
| H2D | `L * R * B * C * Q` | `L * R * T * B*(B-1)/2 * Q` |
| D2H | `L * R * T * B*(B+1)/2 * Q` | `L * R * B * T * Q` |

Here `L=30`, `R=5` (four denoising forwards plus one clean-context forward),
`B=7`, `T=4680`, `C=32760`, and `Q=6144`. Self-cache traffic changes from
211.341 GB H2D / 120.766 GB D2H to 90.575 GB / 30.192 GB. The measured reduction
matches these formulas exactly: **211,341,312,000 bytes saved**. Other transfers,
including output and cross-attention traffic, remain. Writeback growth becomes
linear in block count in this no-eviction regime.

## Validation and timing coverage

[The CPU regression](validate_inferix_kv.py) executes the original and candidate
source methods for the whole attention block, self-attention, and cache wrapper.
It substitutes dense CPU attention and simple projections/RoPE to isolate cache
semantics. All **144 paired steps across 12 scenarios** pass exact output and
active-cache comparisons: two requests, repeated denoising and clean rewrites,
variable block sizes, resets, full-cache attention, sliding-window eviction with
and without a sink, and offload-disabled fallback.

Unread candidate scratch is filled with NaNs; attention and outputs must stay
finite. Per-call read/write ranges are checked. Both `parallel_config=None` and
`world_size=1` are exercised. This validates cache semantics on CPU, not native
GPU transfers or pretrained-model numerical outputs. `world_size>1` is not tested.

Both GX captures contain 106,617 prediction-bearing launches. The candidate has
106,575 hits, 42 misses, and zero prediction errors. Its remaining missing
signature is a cuDNN transform with malformed launch dimensions, part of the
existing cuDNN emulation issue. The baseline had seven missing signature variants
covering 168 launches; cuDNN plan selection therefore differs between captures.
Neither run passes the full native launch-inventory audit. Both also lack the
trace exporter's expected `gxClock` footer. The measured traffic reduction is
exact for these executions; the timing comparison retains these modeling limits.

The candidate's controller ran out of disk space while collecting artifacts,
after the application completed with return code zero. The managed launcher
stopped the container. Generated build sandboxes were removed, then the completed
remote artifacts were recovered and independently verified by SHA-256 and size.
The empty launcher `result.json` was not replaced with a fabricated PASS. A
separate recovery record documents application success, artifact verification,
and the stopped container. Complete markers and dependency reconstruction pass.

## Reproduction and artifacts

Copy the metadata-adapted baseline into a fresh isolated tree, then apply:

```bash
python3 examples/causal_video/inferix_kv_transfer.py \
  --root out/causal-video/Inferix-kvtransfer --apply \
  --patch out/causal-video/kv-transfer.patch
python3 examples/causal_video/validate_inferix_kv.py
```

The applicator checks both source hashes before writing. The reviewable
[patch](cpu_patches/inferix_kv_transfer.patch) changes only the Self-Forcing
attention/cache integration and its cache wrapper. Baseline and benchmark source
files, along with the native validation harness, remain unchanged.

- [Compact evidence](evidence/inferix-kv-transfer.json), including source hashes,
  CPU validation, modeled costs, and recovery provenance.
- Candidate run: `/home/ubuntu/GX/NEX/build/experiments/inferix-kv-transfer-1789644983222209540/`.
- Config, reports, launch audit, validation and recovery records:
  `/home/ubuntu/drperf/out/causal-video/inferix-kv-transfer/`.
- Baseline: [Inferix with GX copy timing](INFERIX_COPY_TIMING.md).
- [Managed runbook](RUNBOOK.md) for launch and report commands. For this recovered
  capture, report directly from `host-0/d0/timeline`; consult `recovery.json`
  rather than the incomplete controller result.

[Draft PR on the fork](https://github.com/MJChku/Inferix/pull/1). The complete case package, including
before/after GX Chrome traces and native reports, is at
`/home/ubuntu/GX/NEX/Example/infrix_kvcache/`.
