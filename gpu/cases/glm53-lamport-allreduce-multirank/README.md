# glm53-lamport-allreduce-multirank

**Contribution: QUANTIFIED.** Kernel: Lamport one-shot all-reduce (+residual, +RMSNorm): GLM ar_residual_oneshot, FlashInfer allreduce_fusion_kernel_oneshot_lamport.

Decode tensor-parallel exchange, 150 calls per step.

## Declared PCVs

hidden 32..512, w 1..4, ranks 2/4, previous-call size C

## DrPerf finding

Executed as concurrent ranks on host threads with peer-mapped buffers (gpu/examples/multirank_allreduce). Compulsory payload 12N+2C (all-reduce) and 18N+2C (fused residual/norm) where C = 2*hidden*previous_w: the clear size follows the previous call. Polling retries reported separately (timing-dependent).

## Fix

None needed.

Fix status: **Closed.**

## Measured timing (not from DrPerf)

7.8 us/call exclusive in b101 (vLLM 6.4).

## Credit

DrPerf (also motivated ld.acquire/st.release host fences).

## Reproduce

- see gpu/examples/multirank_allreduce

## Receipts

- /home/ubuntu/drperf/gpu/examples/multirank_allreduce/README.md

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
