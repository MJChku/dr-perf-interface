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

 dict(case='glm53-deepgemm-paged-logits-bq1', label='FOUND/POTENTIAL',
  kernel='DeepGEMM sm100_paged_mqa_logits (indexer scoring, pinned 8b1392b9)',
  role='FP8 index-key scoring over ~100K history per request on 21 full-indexer layers; 12.8 us/call.',
  pcvs='padded tokens L, splits S, pages, heads 1..32, dim 8..128, CTAs, in-bounds query rows',
  finding=('Exact: 132 B K+scale reads per padded token, 4 B logits writes; tensor_float_flops = 32768*L, i.e. 4x the '
           'useful work, because BLOCK_Q = 128/heads = 4 at H32 while decode has one query per request.'),
  fix=('BLOCK_Q=1 instantiation of the existing core (Codex dg_bq1.cu): 8192 tensor FLOPs per token (exactly 4x less), '
       'identical KV/logit bytes, value oracles pass on 48 cases.'),
  fix_status='Prototype passes on CPU; not integrated or timed.', timing='Logits kernel 12.8 us/call x21 = 0.27 ms/step (b101).',
  receipts=[CODEX+'/NOTES2.md', PROF+'/glm53-gap-b101'], files=[CODEX+'/drivers2/dg_bq1.cu', CODEX+'/drivers2/dg_bq1_driver.cc'],
  reproduce=[CODEX+'/run2.py dgbq1build / capture'], credit='DrPerf.'),

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
  fix_status='Implemented; harness benchmark requested.',
  timing='Standalone B300 (b108): see finding. In-graph b101 CUPTI (inclusive, PDL): 2624 7.8, 2752 11.8, 8192 14.2, 12288 17.9 us.',
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b108'],
  files=[EX+'/fused_a_gemm/kernels.cu', EX+'/fused_a_gemm/driver.cc'],
  reproduce=['see gpu/examples/fused_a_gemm (results/small2); python3 gpu/tools/latency_report.py gpu/examples/fused_a_gemm/results/small2'],
  credit=('DrPerf supplied the exact work interface and the (wrong) coverage hypothesis; the GPU sweep found the stage '
          'effect; the residency view in latency_report was added afterwards so DrPerf flags wave structure.')),
 dict(case='glm53-moe-routing-block-latency', label='EXPLAINED',
  kernel='FlashInfer 0.6.18 routingCustom::routingIndicesBlockKernel<float, bf16, 256, 8, SigmoidBias + ScaledSumNormalize>',
  role='DeepSeekV3-style MoE routing for <=4 tokens (top-8 of 256 experts, permutation/histogram outputs); 75 calls per decode step; identical kernel and launch (1x256) in vLLM.',
  pcvs='tokens 1..4, local experts 128/256, tile tokens 8/16, TK = tokens*8',
  finding=('One CTA of 256 lanes. Exact: global_read_bytes = 260*TK, unique = 128*TK + 1024, fp32_flops = 649*TK, '
           'block barriers = 9 per launch (2304 lane arrivals), collective arrivals 84*TK + 608. Latency model: critical '
           'path 10344..10596 units with 17 exposed global loads per launch, essentially constant in tokens: a fixed serial '
           'chain of global round trips and block barriers on a single CTA. It is latency-bound by construction, which '
           'explains the ~4.5 us floor seen in vLLM for 4 tokens.'),
  fix=('No kernel change from DrPerf. The CUPTI timeline shows our instance at 8..15 us when it overlaps the shared-expert '
       'GEMMs on the side stream (a single-CTA latency chain sharing SMs with GEMM CTAs); candidate fixes: higher priority '
       'for the main stream, or folding routing into the preceding MoE-front kernel.'),
  fix_status='Open (scheduling, not kernel work).',
  timing='vLLM 4.5 us/call; ours 13.8 us exclusive / 18.4 us inclusive median, 8 us when overlap is small (b101).',
  receipts=[PROF+'/glm53-gap-b101', PROF+'/glm53-gap-b54'],
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
    meta = dict(c, copied=copied, study='GLM-5.3 serving (/home/ubuntu/gen-glm53), integration-18 era, 2026-10-25/26')
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
        '| Case | Kernel | Label | Fix status | Timing (not from DrPerf) |', '|---|---|---|---|---|'] + rows + ['']))
    print('\n'.join(rows))
