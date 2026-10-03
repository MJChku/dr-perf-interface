# Inferix with GX copy timing

**Follow-up:** [Bounded KV transfers with offload retained](INFERIX_KV_TRANSFER.md)
reduce virtual time from **38.4180 s to 22.3073 s** and host/device traffic by
**63.54%** for the same request.

The September 17 rerun increases the same host-KV Self-Forcing request from
**12.2677 s to 38.4180 s of partial-sync virtual time**. The existing native A100
median is **41.2907 s**. GX now estimates 6.96% below that reference, compared
with 70.29% below before copy timing.

| Measurement | Seconds |
| --- | ---: |
| Previous GX, copies had zero service | 12.267723 |
| Updated GX, copies modeled | 38.418029 |
| Native A100, median of three warm requests | 41.290692 |
| Increase in virtual duration | 26.150306 |

This is the same pretrained Self-Forcing 1.3B workload: 21 latent frames,
480×832, four denoising steps per block, one segment, TRUE_STREAMING, host KV,
and the original decode behavior. The application command, image, model assets,
CPU calibration database, and native kernel database were retained. No new GPU
run was needed. Reprocessing the old capture with the updated reporter still
produces 12.267723 s.

## Copy accounting

The new capture exactly matches the native Nsight trace's directional copy
counts and byte totals:

| Direction | Copies | Bytes | Updated GX summed service | Native summed service |
| --- | ---: | ---: | ---: | ---: |
| H2D | 5,333 | 211,435,745,184 | 16.2643 s | 16.1573 s |
| D2H | 2,167 | 121,162,752,000 | 9.3202 s | 9.2259 s |
| D2D | 9,605 | 512,578,119,680 | 0.6593 s | 0.7511 s |

All 17,105 copies have metadata and modeled service; no host-link calibration
is missing. The old capture contained only 4,342 copy nodes and no copy service.
The updated memory path also records transfers involving fake GPU buffers.

The GX issue's reproduction configuration supplies 13 GB/s in each host-link
direction and 1.555 TB/s HBM bandwidth. Native aggregate effective rates were
13.09 GB/s H2D and 13.13 GB/s D2H. This is therefore a comparison using a
native-informed bandwidth calibration, not an independent blind prediction.
D2D service charges both the read and write against HBM. Summed service is not
an additive end-to-end penalty when operations overlap.

## Screening result and coverage

Within the marked request, modeled GPU work occupies **38.1594 s**, leaving
**0.2587 s** of modeled device idle time: **99.33% busy** when both kernels and
copies count as work. Summed CPU execution is **4.7073 s**, overlapping device
work. **This is a positive screen for optimizing framework-generated copy
traffic.** Small device idle time limits gains from reducing CPU instruction
overhead alone; it is not a reason to stop investigating Inferix. The busy
figure includes copies and is not SM utilization.

The host-link transfers account for 25.5845 s of summed modeled service. Source
inspection identifies two candidates while preserving host KV residency:

- Write back only modified KV ranges. The ordinary attention path modifies
  `local_start_index:local_end_index`, but the caller writes the whole active
  prefix starting at zero. Unchanged historical KV is repeatedly copied back.
  Sliding-window eviction also modifies history and needs separate handling.
- Fetch only the historical KV needed by attention, instead of transferring
  the full allocated cache, including unused capacity and entries that will be
  overwritten by the current forward.

These are source-supported candidates, not measured optimized results. The
next comparison should keep offload enabled and check transfer bytes, virtual
time, and correctness. The 25.5845 s is not an estimate of removable cost:
some transfers remain necessary under that memory policy.

Kernel prediction coverage remains **106,449 hits, 168 misses, zero errors**.
The native and GX launch totals both remain 106,617. The observable kernel audit
still fails on the previously observed seven cuDNN signature variants; matching
copy totals do not resolve that separate issue. It also flags the updated trace
exporter's missing `gxClock` footer metadata. The selected ordered GPU launch
sequence hash exactly matches the previous GX capture, and application source
hashes are unchanged. Partial synchronization retains
its intended approximate dependency model, and this run does not establish the
cause of the remaining 2.873 s gap to native time.

## Runtime and artifacts

The freshly built shared GX library initially failed before inference because
its cuDNN stubs reported runtime version `(0, 0, 0)` to Torch. The successful run
uses an isolated copy of the updated GX source with the existing real-cuDNN
dispatch patch reapplied. The patch's obsolete trace-footer hunk was omitted
because GX now uses a different trace exporter. Existing logger and kernel-
attribute fixes were already present. Copy timing code was preserved. Neither
the shared GX source nor application code was changed for this experiment.

- Successful managed run: `/home/ubuntu/GX/NEX/build/experiments/inferix-copy-cudnn-1789639536204396324/`.
- Result: PASS; cleanup errors empty; container retained stopped.
- GX library SHA-256: `8aaefd86c8ba11d64b3265e8d85c82e017e286d3acf7cfde7584a80fb0d08bcd`.
- CUDA CPU-exclusion counts match the prior run: 1,551 exports, 2,498,473 calls,
  2,167,059 outer scopes.
- Config, input hashes, compatibility patch, build log, report, and launch audit:
  `/home/ubuntu/drperf/out/causal-video/inferix-copy-timing/`.
- [Compact evidence](evidence/inferix-copy-timing.json).

Reproduce with the retained `config-cudnn.json` through `container_cmds/run.py`,
then generate the report using `tools/gxvm/gxvm report RUN --output NEW_DIRECTORY`.
See [the runbook](RUNBOOK.md) for the managed launch workflow and
[the native results](NATIVE_RESULTS.md) for the unchanged hardware reference.
