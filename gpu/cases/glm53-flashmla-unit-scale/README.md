# glm53-flashmla-unit-scale

**Contribution: POTENTIAL.** Kernel: FlashMLA V32 decode core dequant producer.

The GLM gather writes literal 1.0f scales; the core still loads 16 B of scales and does 512 BF16 multiplies per token.

## Declared PCVs

as glm53-flashmla-core-interface

## DrPerf finding

Unit-scale variant (Codex workspace, diagnostic lowering at the time): global_read_bytes 660N -> 644N, BF16 scaling FLOPs -> 0, tensor work unchanged; value/LSE oracles pass on 33 states.

## Fix

Prototype only.

Fix status: **Untimed on GPU; small expected gain (dequant producer).**

## Measured timing (not from DrPerf)

None.

## Credit

DrPerf counts plus source reading.

## Reproduce

- /home/ubuntu/gen-glm53-codex-audit/drivers2

## Receipts

- /home/ubuntu/gen-glm53-codex-audit/NOTES2.md

## Copied sources

- `src/flash_unit_scale_kernel.cuh` from /home/ubuntu/gen-glm53-codex-audit/drivers2/flash_unit_scale_kernel.cuh
- `src/make_flash_unit_scale.py` from /home/ubuntu/gen-glm53-codex-audit/drivers2/make_flash_unit_scale.py

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
