# glm53-history-scaling-audit

**Contribution: QUANTIFIED.** Kernel: All custom GLM decode kernels at integration-18 (12 kernels + paged cache/metadata helpers).

Check for hidden O(history) work, the class found by the glm53_scaling example.

## Declared PCVs

T 32..512 (small states only), w, K; extrapolated analytically to T=100K

## DrPerf finding

Every interface is O(w), O(K) or O(pages) as designed; no history term. Paged FP8 append requests 77 B per appended byte (per-scalar metadata), ids_to_slots 4wK+28V, decode tables O(T/128) once per step.

## Fix

None needed.

Fix status: **Closed.**

## Measured timing (not from DrPerf)

None.

## Credit

DrPerf.

## Reproduce

- /home/ubuntu/gen-glm53-codex-audit/run.py

## Receipts

- /home/ubuntu/gen-glm53-decode-step/build/drperf-audit/NOTES.md
- /home/ubuntu/gen-glm53-codex-audit/NOTES.md

## Copied sources

- `src/metadata_driver.cc` from /home/ubuntu/gen-glm53-codex-audit/metadata_driver.cc
- `src/driver.cc` from /home/ubuntu/gen-glm53-codex-audit/driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
