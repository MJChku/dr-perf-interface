# glm53-decode-glue-latency-chains

**Contribution: FOUND.** Kernel: GLM decode glue after the byte fixes: dc_selected_address, indexer_decode_tables, decode_gather_v32, attention_iw_prenorm.

Small per-layer kernels between the indexer, FlashMLA and the MoE front (4.6, 2.9, 5.8 and 11.3 us/call in b101).

## Declared PCVs

rows w 1..4, history T 64..512, selected K 16..64 (driver3, small states only)

## DrPerf finding

Latency model, PTX-order schedule (uncalibrated units, 400 per exposed global load): dc_selected_address 4975 / 11 exposed loads per launch, indexer_decode_tables 4718 / 11, decode_gather_v32 3585 / 8, attention_iw_prenorm 58231 / 132. Site attribution (cumulative path and exposed count per statement) shows where each wait lands: the address/gather/tables chains are setp comparisons each consuming a freshly loaded PagedFp8LayerView field (|| validation, then page_table -> page_epochs lookups per slot). Under the new basic-block schedule (ptxas-like) those chains survive (control dependence), while attention_iw_prenorm drops to 8037 / 23: its chain was a PTX-ordering artifact that ptxas already removes. This also explains why the earlier 2-4x requested-byte reductions in the same kernels changed nothing (glm53-decode-redundant-requests).

## Fix

GLM53_ENABLE_DECODE_EAGER_CHECKS (gen-glm53-decode-step branch glue-eager-checks ce81e1e6): bitwise view validation in checked_address; dc_selected_address issues view fields, the four page-table entries and the four epochs as three rounds before any use. DrPerf: identical outputs and error codes on 80 A/B cases covering every failure path (null/wrong views, reserved limit, bad table entry, epoch mismatch, table/lease/page-token faults, bad row view); block schedule address 4906 -> 2470 (11 -> 5 exposed), gather 3507 -> 1504 (8 -> 4).

Fix status: **Implemented; GPU microbenchmark requested (decode_step).**

## Measured timing (not from DrPerf)

CUPTI b101: address 0.36, gather 0.35 (exclusive), tables ~0.06, iw 0.20 ms/step; candidate not yet timed.

## Credit

DrPerf latency model with site attribution found the chains; the basic-block schedule (added for this case) separates real dependence chains from PTX-order artifacts.

## Reproduce

- see gpu/examples/glm53_decode_glue; DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu ...; python3 gpu/tools/latency_report.py <report>
- A/B: build/ab_driver build/kernels.so build/kernels_eager.so (kernels_eager = -DGLM53_ENABLE_DECODE_EAGER_CHECKS)

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b100

## Copied sources

- `src/kernels.cu` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/kernels.cu
- `src/driver3.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/driver3.cc
- `src/ab_driver.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/ab_driver.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
