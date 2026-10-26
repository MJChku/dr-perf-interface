# glm53-cooperative-topk

**Contribution: QUANTIFIED.** Kernel: vLLM cooperative_topk_cs16<2048> (indexer top-k).

Top-2048 over indexer scores per request; 14.1 us/call x21.

## Declared PCVs

T (scaled RF and large/TMA branches), w, K, tie populations

## DrPerf finding

One global score pass at 4 B/token (not 8), shared rescan, bounded tie refinement; fallback per-row kernel has a bounded quadratic insertion sort.

## Fix

None needed.

Fix status: **Closed.**

## Measured timing (not from DrPerf)

14.1 us/call (b101).

## Credit

DrPerf.

## Reproduce

- /home/ubuntu/gen-glm53-codex-audit/run2.py topkbuild

## Receipts

- /home/ubuntu/gen-glm53-codex-audit/NOTES.md
- /home/ubuntu/gen-glm53-codex-audit/NOTES2.md

## Copied sources

- `src/topk2048.cu` from /home/ubuntu/gen-glm53-codex-audit/drivers2/topk2048.cu
- `src/topk2048_driver.cc` from /home/ubuntu/gen-glm53-codex-audit/drivers2/topk2048_driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
