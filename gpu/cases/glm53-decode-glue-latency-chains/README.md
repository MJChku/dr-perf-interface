# glm53-decode-glue-latency-chains

**Contribution: FOUND.** Kernel: GLM decode glue after the byte fixes: dc_selected_address, indexer_decode_tables, decode_gather_v32, attention_iw_prenorm.

Small per-layer kernels between the indexer, FlashMLA and the MoE front (4.6, 2.9, 5.8 and 11.3 us/call in b101).

## Declared PCVs

rows w 1..4, history T 64..512, selected K 16..64 (driver3, small states only)

## DrPerf finding

Latency model, PTX-order schedule (uncalibrated units, 400 per exposed global load): dc_selected_address 4975 / 11 exposed loads per launch, indexer_decode_tables 4718 / 11, decode_gather_v32 3585 / 8, attention_iw_prenorm 58231 / 132. Site attribution (cumulative path and exposed count per statement) shows where each wait lands: the address/gather/tables chains are setp comparisons each consuming a freshly loaded PagedFp8LayerView field (|| validation, then page_table -> page_epochs lookups per slot). Under the new basic-block schedule (ptxas-like) those chains survive (control dependence), while attention_iw_prenorm drops to 8037 / 23: its chain was a PTX-ordering artifact that ptxas already removes. This also explains why the earlier 2-4x requested-byte reductions in the same kernels changed nothing (glm53-decode-redundant-requests).

## Fix

GLM53_ENABLE_DECODE_EAGER_CHECKS (gen-glm53-decode-step branch glue-eager-checks ce81e1e6): bitwise view validation in checked_address; dc_selected_address issues view fields, the four page-table entries and the four epochs as three rounds before any use. DrPerf: identical outputs and error codes on 80 A/B cases covering every failure path (null/wrong views, reserved limit, bad table entry, epoch mismatch, table/lease/page-token faults, bad row view); block schedule address 4906 -> 2470 (11 -> 5 exposed), gather 3507 -> 1504 (8 -> 4).

Fix status: **Implemented; address/gather token-exact; kv-post extension pending a hot-routing determinism check.**

## Measured timing (not from DrPerf)

B300 b105r CUPTI: dc_selected_address 4.70 -> 4.25 us/call, decode_gather_v32 5.82 -> 5.31 us/call (-0.08 ms/step together); harness step w4 -0.02..+0.02 ms (noise), under tracer -0.06..-0.10 ms; hot-routing tokens exact. Calibration: removing 6+4 modeled exposed loads saved ~0.45-0.5 us per kernel, i.e. ~75-110 ns per load (L1/L2-resident view metadata), against 400 model units; the remaining ~4 us per call is launch/dependency overhead the model does not represent. Extension to the kv-post body inside attention_qrank_kv_ik{,_pk} (validation + page lookup once, eager; drperf 48 A/B cases identical incl. 44 faulted error cells; model 21 -> 11.5 exposed loads per kv-post CTA): b106 CUPTI qrank_kv_ik 9.95 -> 9.15 us, pk 9.86 -> 9.26 us, step -0.05..-0.09 ms; hot sorted tokens differ at step 28 of 32 (1 row), under investigation. Also tried and reverted: batched activation loads in prenorm_dot (moe_front_prenorm/attention_iw_prenorm) - bitwise equal but the model got worse (t < m ? load : 0 lowers to predicated loads/branches; no gain predicted).

## Credit

DrPerf latency model with site attribution found the chains; the basic-block schedule (added for this case) separates real dependence chains from PTX-order artifacts.

## Reproduce

- see gpu/examples/glm53_decode_glue; DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu ...; python3 gpu/tools/latency_report.py <report>
- A/B: build/ab_driver build/kernels.so build/kernels_eager.so (kernels_eager = -DGLM53_ENABLE_DECODE_EAGER_CHECKS)

## Receipts

- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b101
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b100
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b105r
- /home/ubuntu/gen-glm53/eval/profiles/gpu/glm53-gap-b106

## Copied sources

- `src/kernels.cu` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/kernels.cu
- `src/driver3.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/driver3.cc
- `src/ab_driver.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/ab_driver.cc
- `src/ab_kvpost.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/ab_kvpost.cc
- `src/front_driver.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/front_driver.cc
- `src/ab_prenorm.cc` from /home/ubuntu/drperf/gpu/examples/glm53_decode_glue/ab_prenorm.cc

Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.
