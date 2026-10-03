# drperf causal-video summary

Valid collection; total marked exclusive instructions: 18,581,951,706.

| Region | Calls | Exclusive instructions | Mean/call | PCVs | States | Fit | Max unexplained |
|---|---:|---:|---:|---|---:|---|---:|
| wan_flash_attention | 6,000 | 3,993,314,583 | 665,552 | q_tokens, k_tokens | 2 | insufficient | — |
| wan_block | 3,000 | 2,846,827,912 | 948,943 | elements | 1 | insufficient | — |
| wan_model_to | 50 | 2,352,966,688 | 47,059,334 | modules | 1 | insufficient | — |
| wan_rope_apply | 6,000 | 2,252,293,350 | 375,382 | elements | 1 | insufficient | — |
| wan_self_attention | 3,000 | 2,170,064,197 | 723,355 | elements | 1 | insufficient | — |
| wan_cross_attention | 3,000 | 1,798,751,485 | 599,584 | elements | 1 | insufficient | — |
| wan_rms_norm | 12,000 | 1,660,919,124 | 138,410 | elements | 2 | insufficient | — |
| wan_layer_norm | 9,100 | 448,391,936 | 49,274 | elements | 1 | insufficient | — |
| wan_transformer | 100 | 375,822,157 | 3,758,222 | tokens | 1 | insufficient | — |
| wan_vae_residual | 294 | 200,210,715 | 680,989 | elements | 7 | identified | 41.3% |
| wan_vae_conv | 692 | 101,369,585 | 146,488 | elements | 9 | identified | 26.7% |
| wan_scheduler | 50 | 94,580,738 | 1,891,615 | step | 50 | identified | 87.0% |
| wan_t5_encoder | 2 | 47,220,869 | 23,610,434 | elements | 1 | insufficient | — |
| wan_generate | 1 | 46,452,669 | 46,452,669 | frames, steps | 1 | insufficient | — |
| wan_vae_resample | 63 | 34,832,771 | 552,901 | elements | 4 | identified | 64.5% |
| wan_vae_decoder | 21 | 33,194,161 | 1,580,674 | elements | 1 | insufficient | — |
| wan_t5_attention | 48 | 30,606,437 | 637,634 | elements | 1 | insufficient | — |
| wan_head | 100 | 27,115,706 | 271,157 | elements | 1 | insufficient | — |
| wan_t5_ffn | 48 | 18,862,842 | 392,976 | elements | 1 | insufficient | — |
| wan_time_embedding | 100 | 18,394,173 | 183,942 | batch | 1 | insufficient | — |
| wan_vae_attention | 21 | 11,703,928 | 557,330 | elements | 1 | insufficient | — |
| wan_tokenize | 2 | 9,896,155 | 4,948,078 | chars | 2 | insufficient | — |
| wan_vae_clear_cache | 2 | 4,189,001 | 2,094,500 | modules | 1 | insufficient | — |
| wan_vae_decode | 1 | 3,390,034 | 3,390,034 | elements | 1 | insufficient | — |
| wan_encode_text | 2 | 580,490 | 290,245 | chars | 2 | insufficient | — |

## Relations and diagnostics

- **wan_flash_attention:** too few distinct states for an affine relation; rank-deficient PCVs: q_tokens
- **wan_block:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_model_to:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: modules
- **wan_rope_apply:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_self_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_cross_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_rms_norm:** too few distinct states for an affine relation
- **wan_layer_norm:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_transformer:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens
- **wan_vae_residual:** -1.98e-06 × elements + 3.94e+05 instructions/call; max unexplained 41.3%.
- **wan_vae_conv:** 5.16e-07 × elements + 9.43e+04 instructions/call; max unexplained 26.7%.
- **wan_scheduler:** 0 × step + 2.56e+05 instructions/call; max unexplained 87.0%.
- **wan_t5_encoder:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_generate:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, steps
- **wan_vae_resample:** -6.4e-05 × elements + 1.95e+05 instructions/call; max unexplained 64.5%.
- **wan_vae_decoder:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_t5_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_head:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_t5_ffn:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_time_embedding:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: batch
- **wan_vae_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_tokenize:** too few distinct states for an affine relation
- **wan_vae_clear_cache:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: modules
- **wan_vae_decode:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: elements
- **wan_encode_text:** too few distinct states for an affine relation

## Largest exclusive functions

- `python3.12:_PyEval_EvalFrameDefault`: 2,279,742,008
- `python3.12:PyDict_Contains+?`: 489,987,386
- `ld-linux-x86-64.so.2:__tls_get_addr`: 429,522,648
- `python3.12:_PyObject_GenericGetAttrWithDict`: 386,507,733
- `libc.so.6:__default_morecore+?`: 374,079,048
- `libc.so.6:pthread_mutex_lock`: 302,177,608
- `python3.12:PyObject_Free`: 284,888,994
- `python3.12:_PyTuple_Resize+?`: 258,502,092
- `python3.12:_PyType_Lookup`: 251,268,744
- `libc.so.6:malloc`: 232,137,053
- `python3.12:PyEval_EvalCode+?`: 223,030,147
- `libc.so.6:__pthread_mutex_unlock`: 222,595,879

The relation describes observed PCV states only. It does not measure GPU execution time or identify a critical-path bottleneck.
