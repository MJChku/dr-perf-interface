# Measured optimisation cases, rerunnable

Nine optimisations from the drperf case study (`~/drperf-cases/CASES.md`) that
were measured but are not yet in the paper's table of measured cases.  Each
directory holds the driver, the marker script, the patch and the equivalence
check of one case, and a `run.sh` that reproduces its before/after formulas:

    ./prepare.sh                 # build work/<case>-base and work/<case>-fix from the study's trees
    ./vllm_admission/run.sh      # one case: two instrumented runs, derive, formulas side by side, equivalence
    QUICK=1 ./wan_host/run.sh    # Wan cases: a smaller state grid
    ./run_all.sh                 # everything

Each `run.sh` runs the driver under `tools/drperf-dev run --blocks` against
the base tree and the fix tree, derives every region, prints the cost formulas
of the regions the case is about, and then runs the case's equivalence check
natively on both trees.  Outputs go to `out/` (`<case>-base/`, `<case>-fix/`,
`derive.txt` inside, `.json`/`.equiv` dumps beside).

| case | system | what the formula showed | fix | expected (from the study) |
|---|---|---|---|---|
| `vllm_admission` | vLLM 0.28 CPU | `process_inputs = 315.6*num_tokens + 372,903`: the slope is `max`/`min` over the ids, the constant a `deepcopy` of SamplingParams; `engine_add_request` primes the detokenizer with the whole prompt | one C pass over the ids; prime with the last 7 tokens | 315.6 -> 109.2 per token; 544.8 -> 227.5 per token; 32 outputs identical |
| `vllm_prepare_inputs` | vLLM | `prepare_inputs = 758*reqs + 107*tokens + 464,928`: a fixed tax of ~35 tensor ops per step; `cached_request_data = 4,907*running + 13,108`: a per-step rebuild of unchanged state | skip `copy_to_gpu` when the buffers alias; hoist and skip the rebuild | constant 465k -> 205k; 4,916 -> 1,671 per running request; outputs identical |
| `vllm_input_batch` | vLLM | `add_request = 275*prompt_tokens + 46k`: the prompt is converted element by element; `condense` costs per row moved | cached `array("i")` memoryview rows; memoised metadata; batched condense | 275 -> 203 per token; metadata rebuild -46%; condense -28% per moved row |
| `vllm_output_path` | vLLM | the surprise was the markers (5 nested regions per request per step); the real cost is 8,248 per request-step in FINAL_ONLY | skip the early-return call instead of entering it | 8,247.5 -> 6,989.4 per request-step (-15%); 752 records identical |
| `vllm_logprobs` | vLLM | `gather_logprobs = 1,009,495*num_reqs + 115,043`: one request's `logprobs=5` runs the vocabulary top-k over every row | top-k on the asking rows only | other-thread part -29.1%; outputs and top-5 dicts identical; wall unmoved at 8 threads |
| `wan_host` | Wan 2.1 (diffusers), CPU | `rope = 29.9*frames + 312,303` rebuilt every forward; `cond_embed = 3,855*batch + 617,350` re-projected every step | cache the rotary table on shape; memoise the prompt projection; list-then-cat in the VAE | rope 482,987 -> 58,612 per call (-88%); cond_embed -28%; latents and video byte-identical |
| `wan_more` | Wan 2.1 | `dgd_init = 14.2*numel + 114,569`: `exp` over the latent for values never read; `callback_step = 14,020*ninputs + 10,873`: `locals()` per name | lazy posterior statistics; capture locals once; fewer rotary temporaries | dgd_init 40x on the slope; callback 33x; attn_rope -1.5% instructions, -25% wall |
| `wan_xattn` | Wan 2.1 | a per-block constant no state explained, attributed partly to an sgemm: cross-attention re-projects the fixed prompt in every block of every forward | memoise cross-attention K/V on tensor identity + version | `ax_qkv` 378,712 -> 137,932 (-64%); 14.2 TFLOP per video removed; latents identical |
| `wan_t5` | Wan 2.1 text encoder | `t5_enc = 24,980*ptok + 46,060*sq + 27,715*lsq + ...`: the real prompt's coefficient is -1.2, everything is paid per padded token and per padded pair | encode at the real length (rounded to 8), re-pad after | 384.8M -> 8.07M per call at 8 real tokens (47.7x); embeddings bit-identical |

## Prerequisites

* `third_party/vllm-cpu/.venv` (run `third_party/vllm-cpu/setup.sh`; ~15 min)
  and `facebook/opt-125m` under `third_party/hf` (`HF_HOME=third_party/hf
  python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m')"`).
* The study's trees and the diffusers venv under `$CASES_ROOT`
  (default `/home/ubuntu/drperf-cases`): `vllm-cpu-{req,rb,batch,out,spec}`,
  `videogen/{wan-src,wan-more,wan-tax,wan-t5,.venv}`.  `prepare.sh` copies
  them (vLLM trees as hardlink farms, only patched files unshared) and never
  writes into them.
* ~1 GB under `out/` per full run of all cases.

## How a tree pair is built

The study's trees carry their markers and, in most cases, the fix.  `prepare.sh`
derives the pair per case: vLLM patches apply on the marked tree (`-p1`), so
base = fix with the patch reversed.  The Wan patches were made against
unmarked files, so base = unmark, reverse, re-mark with the same marker script;
`wan_more`'s base is `wan-src` plus the second-round markers; `wan_t5`'s base
and fix are the two templates of `mark_t5.py`.  Every `run.sh` prints the two
trees' formulas from the same driver, same workload, same marker set.

## Rerun results (2026-10-02, this machine, 115-core CPU box, no GPU)

Every case rebuilt from `prepare.sh` and run with `run.sh` (Wan cases with
`QUICK=1`).  Numbers are drperf instruction counts per call, own cost unless
stated; the study's numbers are in each case's README.

| case | region, quantity | base | fix | change | equivalence |
|---|---|---|---|---|---|
| `vllm_admission` | `process_inputs`, per prompt token | 318 | 115 | -64% | 32 outputs identical |
| `vllm_prepare_inputs` | `prepare_inputs`, fixed part per step | 486,995 | 223,838 | -54% | 10 + 12 requests identical |
| | `cached_request_data`, per running request | 4,930 | 1,660 | -66% | |
| `vllm_input_batch` | `add_request`, per prompt token | 266 | 188 | -29% | 24 requests identical; differential test passes |
| `vllm_output_path` | `process_outputs`, per request-step (FINAL_ONLY) | 8,037 | 6,758 | -16% | 753 records identical |
| `vllm_logprobs` | `gather_logprobs`, other-thread part per step (8 requests, 4 asking) | 14.62M | 11.53M | -21% | outputs and top-5 dicts identical |
| `wan_host` | `rope`, per call | 363,852..459,921 | 57,323..67,299 | -84..-86% | latents and video identical |
| | `cond_embed`, per call | 778,804 | 579,915 | -26% | |
| `wan_more` | `callback_step`, per callback input | 15,369 | 434 | -97% | all digests identical |
| | `dgd_init`, per latent element | 14 | 0 | -100% | |
| | `attn_rope`, per sequence element | 5,761 | 5,672 | -1.5% | |
| `wan_xattn` | `attn`, constant per attention call | 457,574 | 351,848 | -23% | latents and video identical |
| `wan_t5` | `t5_enc`, per encode call at max_len 512 (quick grid) | 61.0M | 10.35M | -83% | 26 arrays bitwise identical |

Two semantics to keep in mind when comparing with `CASES.md`: the current
`derive` reports a region's *own* cost (nested marked regions excluded), where
the study's parent-region numbers (`engine_add_request`, `wan_call`, `block`)
were inclusive; and the Wan quick grids are narrow, so their plane
coefficients differ from the study's while the per-call means agree.  Run the
Wan cases without `QUICK` for paper numbers.  The study's `ax_qkv`/`tf_cond`
probes (Wan C) anchor on the unfixed code and are not reproduced; the
cross-attention memo shows at `attn`.

## Reading the output

`derive` lines are instructions per call on all threads, nested marked regions
excluded, OpenMP runtime waiting excluded; the bracket gives the block counts
and the irregular share.  Instruction counts are exact and reproducible to the
instruction on a deterministic single-threaded path; where the study reports a
wall-clock number (the FramePack, Kimi and spec2lean rows of the paper, or
`attn_rope`'s -25% here) it is a separate native measurement, and these scripts
do not repeat it.  The Wan drivers build tiny random-init models, so their
kernels land in the irregular term and the affine coefficients are the host
orchestration; read those.
