# glm53-flashmla-combine-latency

**Contribution: FOUND.** Kernel: FlashMLA split-KV combine (smxx::decode::flash_fwd_mla_combine_kernel, pinned 6bc49418).

Merges 11 FP32 split partials per request into the BF16 decode attention output; 78 calls per decode step.

## Declared PCVs

w (requests) 1..4, splits per request (2,3,5,8,11 and the real FlashMLA schedules), wsplits = total splits

## DrPerf finding

Learned interface (original): global_read_bytes = 16384*w + 131328*wsplits, blocks = 8*w, sum_launch_max_block_lane_sequence_span = 29*splits + 253. The combine is launched for 64 padded heads although only 32 are consumed, sizes its LSE loop from 48 launched parts (MAX_SPLITS 64) instead of the 11 actual splits, and one warp per head walks the splits with a prefetch-depth-1 loop. Latency model (v1): modeled critical path 5749 with 12 exposed global loads (11 split-loop waits).

## Fix

Wide combine (gen-glm53-decode-step branch flashmla-combine-trim b1833846, combine_glm53.cuh): 32 heads, MAX_SPLITS from a host replica of the scheduler, 4 warps per head with all split loads issued up front. DrPerf: bitwise-equal output and LSE on 1920 (request, head) rows incl. the real schedules; unique reads/writes halved; span 7*splits+238; latency model path 2033 with 2 exposed loads.

Fix status: **Merge-ready (token-exact on GPU); not yet in a serving build.**

## Measured timing (not from DrPerf)

B300 decode harness w4: uniform 15.74 -> 15.44 ms, hot 14.22 -> 13.93 ms (-0.29..-0.31 ms/step); combine 9.0 -> 5.6 us/call (CUPTI). Token-exact vs control in every matched run.

## Credit

Padding and split sizing: DrPerf interface. Serial load chain: first read from source, now flagged by the latency model. Ranking: CUPTI timeline.

## Reproduce

- see gpu/examples/flashmla_decode (combine_cmp.ptx/so, results/combine_cmp2)
- gpu/build/ext-work3 (latency-model acceptance run)

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b102
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b103
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101

## Copied sources

- `src/combine_cmp.cu` from /home/ubuntu/drperf/gpu/examples/flashmla_decode/drivers/combine_cmp.cu
- `src/combine_cmp_driver.cc` from /home/ubuntu/drperf/gpu/examples/flashmla_decode/drivers/combine_cmp_driver.cc
- `src/combine_glm53.cuh` from /home/ubuntu/gen-glm53-decode-step/third_party/flashmla_glm53/combine_glm53.cuh

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
