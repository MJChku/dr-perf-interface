#!/usr/bin/env python3
"""Generate the GLM-5.3 DrPerf GPU case records under gpu/cases/glm53-*.

Each case gets README.md, glm53_case.json and copies of its small driver/kernel
sources (when they live outside this tree). Metadata is named glm53_case.json,
not case.json, so the DeepSeek-V4.1 run_all.py (which runs every */case.json
fixture with its pinned toolchain) is unaffected. Labels follow the ds41
convention: FOUND (DrPerf first identified it), EXPLAINED (DrPerf explained a
timing symptom structurally), QUANTIFIED (found elsewhere, DrPerf supplied
counts), POTENTIAL (no measured fix yet), BLOCKED (tool coverage limit), and
NEGATIVE (fixed as DrPerf suggested, no measured time change).
"""
import json, shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLM = '/home/ubuntu/gen-glm53'
PROF = GLM + '/eval/profiles/gpu'
CODEX = '/home/ubuntu/gen-glm53-codex-audit'
DSTEP = '/home/ubuntu/gen-glm53-decode-step'
EX = '/home/ubuntu/drperf/gpu/examples'

CASES = [
 dict(case='glm53-flashmla-combine-latency', label='FOUND',
  kernel='FlashMLA split-KV combine (smxx::decode::flash_fwd_mla_combine_kernel, pinned 6bc49418)',
  role='Merges 11 FP32 split partials per request into the BF16 decode attention output; 78 calls per decode step.',
  pcvs='w (requests) 1..4, splits per request (2,3,5,8,11 and the real FlashMLA schedules), wsplits = total splits',
  finding=('Learned interface (original): global_read_bytes = 16384*w + 131328*wsplits, blocks = 8*w, '
           'sum_launch_max_block_lane_sequence_span = 29*splits + 253. The combine is launched for 64 padded heads although '
           'only 32 are consumed, sizes its LSE loop from 48 launched parts (MAX_SPLITS 64) instead of the 11 actual splits, '
           'and one warp per head walks the splits with a prefetch-depth-1 loop. Latency model (v1): modeled critical path '
           '5749 with 12 exposed global loads (11 split-loop waits).'),
  fix=('Wide combine (gen-glm53-decode-step branch flashmla-combine-trim b1833846, combine_glm53.cuh): 32 heads, '
       'MAX_SPLITS from a host replica of the scheduler, 4 warps per head with all split loads issued up front. DrPerf: '
       'bitwise-equal output and LSE on 1920 (request, head) rows incl. the real schedules; unique reads/writes halved; '
       'span 7*splits+238; latency model path 2033 with 2 exposed loads.'),
  fix_status='Merge-ready (token-exact on GPU); not yet in a serving build.',
  timing=('B300 decode harness w4: uniform 15.74 -> 15.44 ms, hot 14.22 -> 13.93 ms (-0.29..-0.31 ms/step); combine '
          '9.0 -> 5.6 us/call (CUPTI). Token-exact vs control in every matched run.'),
  receipts=[PROF+'/glm53-gap-b102', PROF+'/glm53-gap-b103', PROF+'/glm53-gap-b101'],
  files=[EX+'/flashmla_decode/drivers/combine_cmp.cu', EX+'/flashmla_decode/drivers/combine_cmp_driver.cc',
         DSTEP+'/third_party/flashmla_glm53/combine_glm53.cuh'],
  reproduce=['see gpu/examples/flashmla_decode (combine_cmp.ptx/so, results/combine_cmp2)',
             'gpu/build/ext-work3 (latency-model acceptance run)'],
  credit='Padding and split sizing: DrPerf interface. Serial load chain: first read from source, now flagged by the latency model. Ranking: CUPTI timeline.'),

 dict(case='glm53-flashmla-core-interface', label='QUANTIFIED',
  kernel='FlashMLA SM100 sparse decode core (sm100::decode::head64::flash_fwd_splitkv_mla_fp8_sparse_kernel<V32>)',
  role='Decode attention over 2048 selected tokens per request, split into ~11 partials; 14.4 us/call, 78 calls/step.',
  pcvs='w 1..4, K 64/128/256, parts 1..8, tasks, split_tasks, blocks64 = w*K/64',
  finding=('First complete official capture of the unmodified kernel after DrPerf fixes (ld.global.nc with eviction + '
           'prefetch-size hints, scalar mul.bf16, mov.b128, div.approx.f32, opcode[operand] lexing, b128 store bytes, '
           'weight-stationary MMA with all-zero column mask). Exact fits: tensor_float_flops = 8912896*blocks64 '
           '(64 heads for 32 useful = 2x), global_read_bytes = 42240*blocks64 + 12288*parts + 75264*tasks '
           '(660 B per selected token; query requested once per task but unique once per request), '
           'global_write_bytes = 65536*split_tasks + 65792*tasks (FP32 partial per split). At w4/K2048: '
           '11.6 MB partial round trip vs 5.4 MB KV; 44 of 148 SMs carry work.'),
  fix='None yet (H32 specialization or 32-head epilogue proposed).',
  fix_status='Open.', timing='Core 14.4 us/call in the b101 CUPTI trace (unchanged by the combine fix).',
  receipts=[PROF+'/glm53-gap-b101', CODEX+'/NOTES2.md'],
  files=[EX+'/flashmla_decode/drivers/flash_core.cu', EX+'/flashmla_decode/drivers/flash_core_bounded_driver.cc',
         EX+'/flashmla_decode/drivers/flash_params.h'],
  reproduce=['see gpu/examples/flashmla_decode (results/grid_aligned)'],
  credit='Interface and padding counts: DrPerf. Tool gaps fixed in gpu/vendor (Statement.lean, prelude.h, tcgen05.cc).'),

 dict(case='glm53-flashmla-unit-scale', label='POTENTIAL',
  kernel='FlashMLA V32 decode core dequant producer',
  role='The GLM gather writes literal 1.0f scales; the core still loads 16 B of scales and does 512 BF16 multiplies per token.',
  pcvs='as glm53-flashmla-core-interface',
  finding=('Unit-scale variant (Codex workspace, diagnostic lowering at the time): global_read_bytes 660N -> 644N, BF16 '
           'scaling FLOPs -> 0, tensor work unchanged; value/LSE oracles pass on 33 states.'),
  fix='Prototype only.', fix_status='Untimed on GPU; small expected gain (dequant producer).', timing='None.',
  receipts=[CODEX+'/NOTES2.md'], files=[CODEX+'/drivers2/flash_unit_scale_kernel.cuh', CODEX+'/drivers2/make_flash_unit_scale.py'],
  reproduce=[CODEX+'/drivers2'], credit='DrPerf counts plus source reading.'),

 dict(case='glm53-deepgemm-paged-logits-bq1', label='FOUND/NEGATIVE',
  kernel='DeepGEMM sm100_paged_mqa_logits (indexer scoring, pinned 8b1392b9)',
  role='FP8 index-key scoring over ~100K history per request on 21 full-indexer layers; 12.8 us/call in graph (b101).',
  pcvs='padded tokens L, splits S, pages, heads 1..32, dim 8..128, CTAs, in-bounds query rows',
  finding=('Exact: 132 B K+scale reads per padded token, 4 B logits writes; tensor_float_flops = 32768*L, i.e. 4x the '
           'useful work, because BLOCK_Q = 128/heads = 4 at H32 while decode has one query per request. Critical-lane '
           'chain (latency model with tensor pipe, source lines via -lineinfo): the page loop is an MMA <-> epilogue '
           'ping-pong - the epilogue waits at sm100_mqa_logits.cuh:423 on the MMA issued at :334 [tensor pipe], the '
           'MMA warp waits for the epilogue to free the accumulator; with BLOCK_Q=1 the MMA wait vanishes (path -14% '
           'at 2 pages/request, per-page cost 375 -> 219 units). Every warp role also starts with three dependent '
           'scheduler loads (sm100_paged_mqa_logits.cuh:46 ceil_div on context_lens).'),
  fix=('BLOCK_Q=1 instantiation of the existing core (dg_bq1.cu): 8192 tensor FLOPs per token (exactly 4x less), '
       'identical KV/logit bytes; 40/40 drperf reference states pass.'),
  fix_status='Measured in graph: -2%, not kept (HBM-bound in the step).',
  timing=('B300 standalone b112 (bitwise, 400222 logits): batch 4 hot 12.48 -> 10.78 us (1.16x), cold HBM 17.06 -> '
          '16.86 us (1.01x); batch 1 hot 10.14 -> 8.58 us. In graph (b113 harness, GLM53_INDEXER_PAGED_BQ1): '
          '12.83 -> 12.54 us/call (-2%), step unchanged, hot tokens exact - the step runs it in the HBM-bound regime, '
          'so the 4x tensor-work cut and the modeled MMA/epilogue chain are hidden behind KV traffic.'),
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b112', PROF+'/glm53-gap-b113'],
  files=[EX+'/deepgemm_paged_logits/drivers/dg_bq1.cu', EX+'/deepgemm_paged_logits/drivers/dg_scoring.cu',
         EX+'/deepgemm_paged_logits/drivers/dg_bq4_driver.cc', EX+'/deepgemm_paged_logits/drivers/dg_bq1_driver.cc'],
  reproduce=['DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu -- build/dg_bq{4,1}_driver build/dg_bq{4,1}.so',
             'python3 gpu/tools/latency_report.py results/lat_bq4_li --region "deepgemm*" --stalls 6'],
  credit='DrPerf found the 4x tensor work, the ping-pong chain and checked the prototype; GPU confirmed the L2-resident gain.'),
 dict(case='glm53-decode-redundant-requests', label='NEGATIVE',
  kernel='GLM decode glue: decode_gather_v32, attention_iw_prenorm, dc_selected_address, moe_front_prenorm',
  role='Custom decode kernels around attention and the MoE front.',
  pcvs='rows w 1..4, history T 32..512, selected K 16..64',
  finding=('No custom decode kernel scales with history T. Requested-byte waste: gather 2368 B per selected id vs 576 B '
           'needed (every lane recomputes the address), iw_prenorm re-reads its weight row per input row (4.7 MB at w4), '
           'selected-address re-reads view metadata (90 B/id), MoE front re-reads input rows per expert CTA.'),
  fix='Fixed the first three (bitwise under GXPTX): 2368 -> 616 B/id, 4.7 -> 2.4 MB, 90 -> 40 B/id.',
  fix_status='Kept as bitwise cleanups; no time effect.',
  timing='B300 harness w4 at chunk 4096: kernel fixes -0.00..-0.08 ms (noise). The repeats were cache hits.',
  receipts=[PROF+'/glm53-gap-b98', DSTEP+'/build/drperf-audit/NOTES.md'],
  files=[DSTEP+'/build/drperf-audit2/driver3.cc', DSTEP+'/build/drperf-audit2/kernels.cu'],
  reproduce=[DSTEP+'/build/drperf-audit2'],
  credit='DrPerf found the redundancy; footprint metrics (added later) show requested >> unique, i.e. cache-resident.'),

 dict(case='glm53-history-scaling-audit', label='QUANTIFIED',
  kernel='All custom GLM decode kernels at integration-18 (12 kernels + paged cache/metadata helpers)',
  role='Check for hidden O(history) work, the class found by the glm53_scaling example.',
  pcvs='T 32..512 (small states only), w, K; extrapolated analytically to T=100K',
  finding=('Every interface is O(w), O(K) or O(pages) as designed; no history term. Paged FP8 append requests 77 B per '
           'appended byte (per-scalar metadata), ids_to_slots 4wK+28V, decode tables O(T/128) once per step.'),
  fix='None needed.', fix_status='Closed.', timing='None.',
  receipts=[DSTEP+'/build/drperf-audit/NOTES.md', CODEX+'/NOTES.md'], files=[CODEX+'/metadata_driver.cc', CODEX+'/driver.cc'],
  reproduce=[CODEX+'/run.py'], credit='DrPerf.'),

 dict(case='glm53-cooperative-topk', label='QUANTIFIED',
  kernel='vLLM cooperative_topk_cs16<2048> (indexer top-k)',
  role='Top-2048 over indexer scores per request; 14.1 us/call x21.',
  pcvs='T (scaled RF and large/TMA branches), w, K, tie populations',
  finding='One global score pass at 4 B/token (not 8), shared rescan, bounded tie refinement; fallback per-row kernel has a bounded quadratic insertion sort.',
  fix='None needed.', fix_status='Closed.', timing='14.1 us/call (b101).',
  receipts=[CODEX+'/NOTES.md', CODEX+'/NOTES2.md'], files=[CODEX+'/drivers2/topk2048.cu', CODEX+'/drivers2/topk2048_driver.cc'],
  reproduce=[CODEX+'/run2.py topkbuild'], credit='DrPerf.'),

 dict(case='glm53-lamport-allreduce-multirank', label='QUANTIFIED',
  kernel='Lamport one-shot all-reduce (+residual, +RMSNorm): GLM ar_residual_oneshot, FlashInfer allreduce_fusion_kernel_oneshot_lamport',
  role='Decode tensor-parallel exchange, 150 calls per step.',
  pcvs='hidden 32..512, w 1..4, ranks 2/4, previous-call size C',
  finding=('Executed as concurrent ranks on host threads with peer-mapped buffers (gpu/examples/multirank_allreduce). '
           'Compulsory payload 12N+2C (all-reduce) and 18N+2C (fused residual/norm) where C = 2*hidden*previous_w: the '
           'clear size follows the previous call. Polling retries reported separately (timing-dependent).'),
  fix='None needed.', fix_status='Closed.', timing='7.8 us/call exclusive in b101 (vLLM 6.4).',
  receipts=[EX+'/multirank_allreduce/README.md'], files=[], reproduce=['see gpu/examples/multirank_allreduce'],
  credit='DrPerf (also motivated ld.acquire/st.release host fences).'),

 dict(case='glm53-fused-a-gemm-sm-coverage', label='NEGATIVE/FOUND',
  kernel='TRT-LLM dsv3FusedAGemm (vLLM csrc copy, src/lib_fused_a_gemm.cu)',
  role='Small-M BF16 GEMM for the attention prefix (6144->2624/2752, 78 calls/step) and q_b (2048->8192/12288, 78 calls/step).',
  pcvs='M (hd_out) 64..256, K (hd_in) 512..2048, tile_m 16/32, tokens 1..4 (small instantiations; production extrapolated)',
  finding=('Exact: global_unique_read_bytes = 2*M*K + 2*tokens*K, tensor_float_flops = 16*M*K for any tokens, blocks = '
           'M/tile_m. Hypothesis from CTA counts: the prefix shapes launch 82/86 CTAs on 148 SMs (one CTA/SM), so they '
           'should be 1.72x slower than balanced. A standalone B300 benchmark (HBM-resident weights) refuted it: tile_m 16 '
           '(164/172 CTAs) is 0.68-0.97x of the current speed; current 7.3-7.9 us = 4.1-4.4 TB/s. The same sweep found '
           'that the hd_in 2048 shapes run faster with 4 pipeline stages instead of 8 (98 KB -> 49 KB shared, 2 -> 4 '
           'CTAs/SM, 512/768 CTAs from ~1.7/2.6 waves to one): 2048->8192 9.00 -> 7.65 us, 2048->12288 12.5 -> 11.4 us, '
           'bitwise. The new latency_report residency columns (CTAs/SM from shared memory and threads, waves) expose '
           'that wave structure directly.'),
  fix='GLM53_FUSED_A_K2048_STAGE4 (gen-glm53-decode-step glue-eager-checks e74147cd).',
  fix_status='Implemented, token-exact, merge-ready.',
  timing=('Standalone B300 (b108): see finding. In-graph (b109 CUPTI): 2048->8192 13.38 -> 12.1 us, 2048->12288 17.4 -> 16.4 us '
          '(-0.095 ms/step), harness step -0.05..-0.13 ms uniform, hot tokens exact.'),
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b108', PROF+'/glm53-gap-b109'],
  files=[EX+'/fused_a_gemm/kernels.cu', EX+'/fused_a_gemm/driver.cc'],
  reproduce=['see gpu/examples/fused_a_gemm (results/small2); python3 gpu/tools/latency_report.py gpu/examples/fused_a_gemm/results/small2'],
  credit=('DrPerf supplied the exact work interface and the (wrong) coverage hypothesis; the GPU sweep found the stage '
          'effect; the residency view in latency_report was added afterwards so DrPerf flags wave structure.')),
 dict(case='glm53-moe-routing-block-latency', label='EXPLAINED (corrected)',
  kernel='FlashInfer 0.6.18 routingCustom::routingIndicesBlockKernel<float, bf16, 256, 8, SigmoidBias + ScaledSumNormalize>',
  role='DeepSeekV3-style MoE routing for <=4 tokens (top-8 of 256 experts, permutation/histogram outputs); 75 calls per decode step; identical kernel and launch (1x256) in vLLM.',
  pcvs='tokens 1..4, local experts 128/256, tile tokens 8/16, TK = tokens*8',
  finding=('One CTA of 256 lanes. Exact: global_read_bytes = 260*TK, unique = 128*TK + 1024, fp32_flops = 649*TK, '
           'block barriers = 9 per launch (2304 lane arrivals), collective arrivals 84*TK + 608. The first latency model '
           'reported a fixed chain of 17 dependent global loads (10344..10596 units); with the critical-lane chain and '
           '-lineinfo source lines this turned out to be a model artifact: the loads are guarded by short forward branches '
           '(RoutingKernel.cuh:127 score loads behind idx<numExperts, RoutingCustomPolicy.cuh:234 bias loads in a dtype '
           'switch) that ptxas predicates. With if-conversion modeled the path is 2763 units (one exposed load plus '
           'barriers): the kernel is short on its own, consistent with vLLM\'s 4.5 us, so our 22 us/call in graph '
           '(1.57 ms/step) is queuing - the shared-expert GEMMs are enqueued first on the side stream and the 1-CTA '
           'routing kernel waits for an SM.'),
  fix=('Scheduling (gen-glm53-decode-step): GLM53_DECODE_ROUTED_FIRST enqueues the routed path before the shared '
       'expert; GLM53_DECODE_ROUTED_PRIORITY gives the main decode streams the greatest priority, the shared-expert '
       'stream the least, and instantiates graphs with cudaGraphInstantiateFlagUseNodePriority (1654 high / 300 low '
       'priority kernel nodes captured).'),
  fix_status='Priority variant merge-ready (token-exact).',
  timing=('b114 B300 harness: routing 23.0 -> 17.0 us/call (routed first) / 16.6 us (priority); step -0.22..-0.24 ms '
          '(routed first) and -0.26..-0.32 ms (priority) uniform, -0.24..-0.32 ms hot; the deprioritized shared down GEMM '
          'slows 5.3 -> 24.9 us off the critical path. CUPTI interval analysis: routing launches ~5 us before its '
          'producer moe_front_prenorm ends (PDL); measured from the producer end it takes 14.4 us (control) -> 8.8 us '
          '(either fix), still ~4.3 us over vLLM while sharing SMs with already-running GEMM CTAs. Forking the shared '
          'expert only after routing (b115) trimmed routing to 7.7 us but cost +0.36..+0.50 ms/step (the shared expert '
          'then contends with the routed GEMMs): dropped.'),
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b54', PROF+'/glm53-gap-b114'],
  files=[EX+'/moe_routing/kernel.cu', EX+'/moe_routing/driver.cu', EX+'/moe_routing/routing_block_kernel.cuh'],
  reproduce=['see gpu/examples/moe_routing (results/run1); latency: python3 gpu/tools/latency_report.py results/run1'],
  credit=('DrPerf explained why the kernel has a fixed ~latency floor; the contention with the side stream comes from the '
          'CUPTI timeline. Tool gaps found: drperf_gpu.h dim3 clashed with CUDA headers in nvcc-compiled drivers (fixed); '
          'latency metrics are per-launch diagnostics without per-site attribution of exposed loads.')),
 dict(case='glm53-decode-glue-latency-chains', label='FOUND',
  kernel='GLM decode glue after the byte fixes: dc_selected_address, indexer_decode_tables, decode_gather_v32, attention_iw_prenorm',
  role='Small per-layer kernels between the indexer, FlashMLA and the MoE front (4.6, 2.9, 5.8 and 11.3 us/call in b101).',
  pcvs='rows w 1..4, history T 64..512, selected K 16..64 (driver3, small states only)',
  finding=('Latency model, PTX-order schedule (uncalibrated units, 400 per exposed global load): dc_selected_address '
           '4975 / 11 exposed loads per launch, indexer_decode_tables 4718 / 11, decode_gather_v32 3585 / 8, '
           'attention_iw_prenorm 58231 / 132. Site attribution (cumulative path and exposed count per statement) shows '
           'where each wait lands: the address/gather/tables chains are setp comparisons each consuming a freshly loaded '
           'PagedFp8LayerView field (|| validation, then page_table -> page_epochs lookups per slot). Under the new '
           'basic-block schedule (ptxas-like) those chains survive (control dependence), while attention_iw_prenorm drops '
           'to 8037 / 23: its chain was a PTX-ordering artifact that ptxas already removes. This also explains why the '
           'earlier 2-4x requested-byte reductions in the same kernels changed nothing (glm53-decode-redundant-requests).'),
  fix=('GLM53_ENABLE_DECODE_EAGER_CHECKS (gen-glm53-decode-step branch glue-eager-checks ce81e1e6): bitwise view '
       'validation in checked_address; dc_selected_address issues view fields, the four page-table entries and the four '
       'epochs as three rounds before any use. DrPerf: identical outputs and error codes on 80 A/B cases covering every '
       'failure path (null/wrong views, reserved limit, bad table entry, epoch mismatch, table/lease/page-token faults, '
       'bad row view); block schedule address 4906 -> 2470 (11 -> 5 exposed), gather 3507 -> 1504 (8 -> 4).'),
  fix_status='Implemented and token-exact (address, gather, kv-post); decode_tables extension A/B-verified, untimed.',
  timing=('B300 b105r CUPTI: dc_selected_address 4.70 -> 4.25 us/call, decode_gather_v32 5.82 -> 5.31 us/call '
          '(-0.08 ms/step together); harness step w4 -0.02..+0.02 ms (noise), under tracer -0.06..-0.10 ms; hot-routing '
          'tokens exact. Calibration: removing 6+4 modeled exposed loads saved ~0.45-0.5 us per kernel, i.e. ~75-110 ns '
          'per load (L1/L2-resident view metadata), against 400 model units; the remaining ~4 us per call is launch/'
          'dependency overhead the model does not represent. Extension to the kv-post body inside attention_qrank_kv_ik{,_pk} '
          '(validation + page lookup once, eager; drperf 48 A/B cases identical incl. 44 faulted error cells; model 21 -> 11.5 '
          'exposed loads per kv-post CTA): b106 CUPTI qrank_kv_ik 9.95 -> 9.15 us, pk 9.86 -> 9.26 us, step -0.05..-0.09 ms; '
          'b106 hot tokens differed at step 28 once; b107 (e1s/e2s twice each, 32 steps) all equal, so that was run-to-run '
          'nondeterminism and the change is token-exact. decode_tables_kernel: loop-invariant view fields read once (36 A/B '
          'cases incl. 30 faulted error cells identical). Also tried and reverted: batched '
          'activation loads in prenorm_dot (moe_front_prenorm/attention_iw_prenorm) - bitwise equal but the model got worse '
          '(t < m ? load : 0 lowers to predicated loads/branches; no gain predicted).'),
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b100', PROF+'/glm53-gap-b105r', PROF+'/glm53-gap-b106', PROF+'/glm53-gap-b107'],
  files=[EX+'/glm53_decode_glue/kernels.cu', EX+'/glm53_decode_glue/driver3.cc', EX+'/glm53_decode_glue/ab_driver.cc', EX+'/glm53_decode_glue/ab_kvpost.cc', EX+'/glm53_decode_glue/front_driver.cc', EX+'/glm53_decode_glue/ab_prenorm.cc', EX+'/glm53_decode_glue/ab_tables.cc'],
  reproduce=['see gpu/examples/glm53_decode_glue; DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu ...; python3 gpu/tools/latency_report.py <report>',
             'A/B: build/ab_driver build/kernels.so build/kernels_eager.so (kernels_eager = -DGLM53_ENABLE_DECODE_EAGER_CHECKS)'],
  credit=('DrPerf latency model with site attribution found the chains; the basic-block schedule (added for this case) '
          'separates real dependence chains from PTX-order artifacts.')),
 dict(case='glm53-prenorm-dot-row-branches', label='FOUND',
  kernel='GLM prenorm_dot (moe_front_prenorm router CTAs, attention_iw_prenorm)',
  role='BF16 activation x FP32 weight-row dots: 256 router CTAs per MoE layer (75/step, ~10.8 us/call) and 32 indexer-weight CTAs (21/step).',
  pcvs='rows 1..4 at production width 6144 (front_driver; one launch per width)',
  finding=('Latency model (basic-block schedule) with site attribution: the router CTA critical path at w4 is 11945 units with '
           '32 exposed global loads, one per (activation step, row): the activation loop loads one row, branches on '
           '`if (t < m)`, consumes it, then loads the next row, so the four rows\' loads never overlap. A first attempt '
           '(branch-free `t < m ? load : 0` selects) lowered to predicated loads and did not help in the model; a '
           'compile-time row count (switch on m once, template body) removes the per-row branches.'),
  fix=('GLM53_ENABLE_DECODE_PRENORM_STATIC_ROWS (gen-glm53-decode-step glue-eager-checks d55406d8): bitwise outputs '
       '(drperf A/B: packed NVFP4, scales, router logits and indexer weights identical for rows 1..4); model w4 6625 / 6 '
       'exposed, constant across widths.'),
  fix_status='Implemented, token-exact; small measured gain.',
  timing=('b110 CUPTI: moe_front_prenorm 10.4 -> 10.0 us/call (-0.03 ms/step), attention_iw_prenorm 11.1 -> 10.7-11.0 us; '
          'harness step -0.03..-0.18 ms (noisy); hot tokens exact. The model over-predicts: removing 26 exposed waits '
          'saved ~0.4 us, because the activations are L2-resident (re-read by 256 CTAs) and ~8 warps per SM hide much of '
          'the per-lane chain - the per-lane model has no warp-level latency hiding.'),
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b110'],
  files=[EX+'/glm53_decode_glue/front_driver.cc', EX+'/glm53_decode_glue/ab_prenorm.cc'],
  reproduce=['DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu -- build/front_driver build/kernels{,_sr}.so (kernels_sr = -DGLM53_ENABLE_DECODE_PRENORM_STATIC_ROWS)'],
  credit='DrPerf latency model + site attribution found it and checked the fix before GPU time.'),
 dict(case='glm53-flashmla-core-critical-chain', label='EXPLAINED/NEGATIVE',
  kernel='FlashMLA SM100 head64 V32 sparse decode core (flash_fwd_splitkv_mla_fp8_sparse_kernel)',
  role='Attention core, 78 calls/step, ~14.4 us/call (b101); at w4 x top-k 2048 a CTA handles only 1-2 64-token blocks.',
  pcvs='w 1..4, K 64..256 selected tokens, parts 1..8 (33 states; examples/flashmla_decode grid)',
  finding=('Four DrPerf gaps blocked this kernel and were fixed first: the translated kernel overflowed 128 KiB lane '
           'stacks (SIGSEGV); tcgen05.mma completion was free (commit arrived at issue); failed mbarrier try_wait probes '
           'counted as work (18% of instructions, path inflated 29%); and the per-site path maximum could not say what '
           'a lane waited on. With the critical-lane chain the core reads: consumer (thread 0) waits on the QK MMA '
           '[tensor pipe, thread 128]; the MMA issuer waits on the dequant warpgroup (thread 256), the KV gather TMA '
           '(thread 192) and the softmax arrive (thread 31); the index/scale warp (thread 224) feeds the gathers. '
           'Per task the path is exactly 5326 + 1787 per 64-token block: the steady-state recurrence is QK MMA (~576) '
           '-> softmax (~630) -> PV MMA (~512) on the attention loop, not the producers. The index/scale warp loads a '
           'block\'s indices, waits, then issues the dependent scale loads and waits again; its 4-buffer lookahead hides '
           'that in steady state, but a one-block CTA (the production case) pays it at start-up.'),
  fix=('GLM53_FLASHMLA_INDEX_PREFETCH prototype (gen-glm53-decode-step third_party/flashmla_glm53/'
       'decode_head64_kernel_glm53.cuh): indices two blocks ahead, scales one block ahead, same arithmetic. DrPerf: '
       '33/33 reference states pass; one-block path 7113 -> 6756 (-5%), slope unchanged.'),
  fix_status='Measured: no gain (not kept).',
  timing=('B300 harness b112: core 14.46 -> 14.46/14.75 us/call (rank0/rank1), step unchanged, hot tokens exact. The '
          'modeled start-up chain of a one-block CTA is not what bounds the real core: calibration note - the model '
          'charges every global load 400 units on an otherwise idle SM, while the real core overlaps CTAs and is '
          'bandwidth/occupancy bound.'),
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b112'],
  files=[EX+'/flashmla_decode/drivers/flash_core.cu', EX+'/flashmla_decode/drivers/flash_core_prefetch.cu',
         EX+'/flashmla_decode/drivers/decode_head64_kernel_glm53.cuh'],
  reproduce=['DRPERF_GPU_LATENCY_SCHEDULE=block bin/drperf --gpu -- build/core_driver build/flash_core{,_pf}.so build/flash_core{,_pf}.ptx build/flash_aux.so build/flash_aux.ptx grid',
             'python3 gpu/tools/latency_report.py results/grid_lat --region flashmla.core.small --stalls 8'],
  credit='DrPerf (after four tool fixes) located the chain and checked the prototype before GPU time.'),

 dict(case='glm53v2-own-decode-moe-fp4-dequant', label='QUANTIFIED/NEGATIVE',
  study='GLM-5.3 GX study v2 (/home/ubuntu/gen-glm53-v2, gx-only agent), 2026-10-28',
  kernel='Own decode NVFP4 MoE (glm_moe_dec_gemm1_k: routed gemm1 + SwiGLU + NVFP4 requant; glm_moe_dec_gemm2_k: gemm2 + weighted finalize)',
  role='Replacement attempt for TRT-LLM-gen fused NVFP4 block-scale MoE in decode (T=4 rows, 8 of 256 experts per row); 78 calls per step.',
  pcvs='None declared: one fixture state (T=4, C4 routing). The report flags SINGLE STATE, so no scaling was learned.',
  finding=('gx explain gpu (drperf-gpu) on the gx check fixture: gemm1 21.1M PTX lane instructions for 3.17 MB of requested '
           'global reads (6.7 lane instructions per weight byte): 5.05M conversion ops + 6.31M integer ops (E2M1 nibble '
           'extraction) + 4.28M fp16 flops (block-scale multiplies) vs 58.7M tensor flops on mma.sync. The FP4->f16 dequant '
           'path is issue-bound; block scales cannot be factored out of mma.sync k-tiles (4 lanes hold different blocks).'),
  fix=('None: matching TRT-LLM-gen needs tcgen05 block-scaled FP4 MMA (kind::mxf4nvf4). Kernel kept behind GLM_MOE_DEC=1, '
       'default off (commit f9a0743 in gen-glm53-v2).'),
  fix_status='Shelved (negative result).',
  timing='Native cold A/B on B300 (agent batch 18, C4 routing, T=4): own 57.2 us vs TRT-LLM-gen 29.9 us per call.',
  receipts=['receipts-v2/glm53v2-own-decode-moe/drperf-gpu-report.txt', 'receipts-v2/glm53v2-own-decode-moe/result.json',
            '/home/ubuntu/gen-glm53-v2/eval/WORKLOG.md (2026-10-28 17:05)', '/home/ubuntu/gen-glm53-v2/eval/GX-FEEDBACK.md'],
  files=['/home/ubuntu/gen-glm53-v2/src/kernels/moe_decode.cu', '/home/ubuntu/gen-glm53-v2/src/tests/t_moedec.cu'],
  reproduce=['cd /home/ubuntu/gen-glm53-v2 && gx explain gpu --gpu B300 -q --out RUN -- build/t_moedec'],
  credit='Instruction mix and the issue-bound diagnosis: DrPerf GPU resource interface (found by the study agent). Timing: native B300 A/B.'),
]

def write_case(c):
    d = HERE / c['case']
    d.mkdir(exist_ok=True)
    copied = []
    for f in c['files']:
        p = Path(f)
        if p.exists():
            (d / 'src').mkdir(exist_ok=True)
            shutil.copy2(p, d / 'src' / p.name)
            copied.append({'source': f, 'copy': 'src/' + p.name})
    meta = dict(c, copied=copied, study=c.get('study', 'GLM-5.3 serving (/home/ubuntu/gen-glm53), integration-18 era, 2026-10-25/26'))
    (d / 'glm53_case.json').write_text(json.dumps(meta, indent=2) + '\n')
    lines = [f"# {c['case']}", '', f"**Contribution: {c['label']}.** Kernel: {c['kernel']}.", '', c['role'], '',
             '## Declared PCVs', '', c['pcvs'], '', '## DrPerf finding', '', c['finding'], '',
             '## Fix', '', c['fix'], '', f"Fix status: **{c['fix_status']}**", '',
             '## Measured timing (not from DrPerf)', '', c['timing'], '', '## Credit', '', c['credit'], '',
             '## Reproduce', ''] + [f'- {r}' for r in c['reproduce']] + ['', '## Receipts', ''] + \
            [f'- {r}' for r in c['receipts']] + ([''] + ['## Copied sources', ''] + [f"- `{x['copy']}` from {x['source']}" for x in copied] if copied else []) + \
            ['', 'Requested bytes are not HBM traffic; the latency model is an uncalibrated dependency model, not measured time.', '']
    (d / 'README.md').write_text('\n'.join(lines))
    return d

if __name__ == '__main__':
    rows = []
    for c in CASES:
        write_case(c)
        rows.append(f"| [{c['case']}]({c['case']}/README.md) | {c['kernel'].split(' (')[0]} | {c['label']} | {c['fix_status']} | {c['timing'][:90]} |")
    (HERE / 'GLM53.md').write_text('\n'.join([
        '# GLM-5.3 DrPerf GPU cases', '',
        'Cases from the GLM-5.3 decode study (`/home/ubuntu/gen-glm53`, 2xB300, C4 = 4 concurrent 100K/1K requests). '
        'Generated by `glm53_cases.py`; metadata in each `glm53_case.json`. Labels as in README.md, plus NEGATIVE '
        '(fixed as DrPerf suggested, no measured time change).', '',
        'Roll-up (B300 decode harness, batch glm53-gap-b111): all merge-ready items from these cases plus compact step '
        'metadata together, against the integration-18 baseline: step -0.79..-0.97 ms (uniform routing), -0.76..-0.90 ms '
        '(hot); w4 15.97 -> 15.12 ms and 14.43 -> 13.63 ms; hot sorted-ID tokens exact over 40 steps.', '',
        '| Case | Kernel | Label | Fix status | Timing (not from DrPerf) |', '|---|---|---|---|---|'] + rows + ['']))
    print('\n'.join(rows))
