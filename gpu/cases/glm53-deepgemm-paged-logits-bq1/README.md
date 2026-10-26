# glm53-deepgemm-paged-logits-bq1

**Contribution: FOUND/POTENTIAL.** Kernel: DeepGEMM sm100_paged_mqa_logits (indexer scoring, pinned 8b1392b9).

FP8 index-key scoring over ~100K history per request on 21 full-indexer layers; 12.8 us/call.

## Declared PCVs

padded tokens L, splits S, pages, heads 1..32, dim 8..128, CTAs, in-bounds query rows

## DrPerf finding

Exact: 132 B K+scale reads per padded token, 4 B logits writes; tensor_float_flops = 32768*L, i.e. 4x the useful work, because BLOCK_Q = 128/heads = 4 at H32 while decode has one query per request.

## Fix

BLOCK_Q=1 instantiation of the existing core (Codex dg_bq1.cu): 8192 tensor FLOPs per token (exactly 4x less), identical KV/logit bytes, value oracles pass on 48 cases.

Fix status: **Prototype passes on CPU; not integrated or timed.**

## Measured timing (not from DrPerf)

Logits kernel 12.8 us/call x21 = 0.27 ms/step (b101).

## Credit

DrPerf.

## Reproduce

- /home/ubuntu/gen-glm53-codex-audit/run2.py dgbq1build / capture

## Receipts

- /home/ubuntu/gen-glm53-codex-audit/NOTES2.md
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101

## Copied sources

- `src/dg_bq1.cu` from /home/ubuntu/gen-glm53-codex-audit/drivers2/dg_bq1.cu
- `src/dg_bq1_driver.cc` from /home/ubuntu/gen-glm53-codex-audit/drivers2/dg_bq1_driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
