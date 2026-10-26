# glm53-decode-redundant-requests

**Contribution: NEGATIVE.** Kernel: GLM decode glue: decode_gather_v32, attention_iw_prenorm, dc_selected_address, moe_front_prenorm.

Custom decode kernels around attention and the MoE front.

## Declared PCVs

rows w 1..4, history T 32..512, selected K 16..64

## DrPerf finding

No custom decode kernel scales with history T. Requested-byte waste: gather 2368 B per selected id vs 576 B needed (every lane recomputes the address), iw_prenorm re-reads its weight row per input row (4.7 MB at w4), selected-address re-reads view metadata (90 B/id), MoE front re-reads input rows per expert CTA.

## Fix

Fixed the first three (bitwise under GXPTX): 2368 -> 616 B/id, 4.7 -> 2.4 MB, 90 -> 40 B/id.

Fix status: **Kept as bitwise cleanups; no time effect.**

## Measured timing (not from DrPerf)

B300 harness w4 at chunk 4096: kernel fixes -0.00..-0.08 ms (noise). The repeats were cache hits.

## Credit

DrPerf found the redundancy; footprint metrics (added later) show requested >> unique, i.e. cache-resident.

## Reproduce

- /home/ubuntu/gen-glm53-decode-step/build/drperf-audit2

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b98
- /home/ubuntu/gen-glm53-decode-step/build/drperf-audit/NOTES.md

## Copied sources

- `src/driver3.cc` from /home/ubuntu/gen-glm53-decode-step/build/drperf-audit2/driver3.cc
- `src/kernels.cu` from /home/ubuntu/gen-glm53-decode-step/build/drperf-audit2/kernels.cu

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
