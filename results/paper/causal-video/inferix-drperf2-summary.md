# drperf causal-video summary

Valid collection; total marked exclusive instructions: 5,583,431,458.

| Region | Calls | Exclusive instructions | Mean/call | PCVs | States | Fit | Max unexplained |
|---|---:|---:|---:|---|---:|---|---:|
| inferix_self_attention | 1,050 | 1,547,674,440 | 1,473,976 | tokens, context_tokens | 7 | insufficient | — |
| inferix_block | 1,050 | 1,207,777,307 | 1,150,264 | tokens | 1 | insufficient | — |
| inferix_cross_attention | 1,050 | 911,990,548 | 868,562 | tokens, context_tokens | 1 | insufficient | — |
| inferix_causal_rope | 2,100 | 749,846,300 | 357,070 | tokens, heads | 1 | insufficient | — |
| inferix_vae_decoder | 42 | 643,134,590 | 15,312,728 | frames | 1 | insufficient | — |
| inferix_transformer | 35 | 203,512,389 | 5,814,640 | frames, batch, layers | 1 | insufficient | — |
| inferix_kv_set | 1,050 | 137,220,281 | 130,686 | cache_tokens | 7 | identified | 4.5% |
| inferix_text_encode | 1 | 50,083,253 | 50,083,253 | prompts | 1 | insufficient | — |
| inferix_vae_clear_cache | 17 | 35,419,187 | 2,083,482 | — | 1 | insufficient | — |
| inferix_kv_get | 1,050 | 30,343,601 | 28,899 | — | 1 | insufficient | — |
| inferix_inference | 1 | 23,881,055 | 23,881,055 | frames, initial_frames, batch | 1 | insufficient | — |
| inferix_generator | 35 | 16,850,913 | 481,455 | frames, batch | 1 | insufficient | — |
| inferix_vae_cached_decode | 32 | 10,118,686 | 316,209 | frames | 2 | insufficient | — |
| inferix_vae_decode_to_pixel | 8 | 6,504,860 | 813,108 | frames, batch, cached | 2 | insufficient | — |
| inferix_kv_allocate_slots | 60 | 4,743,300 | 79,055 | tokens, layers | 2 | insufficient | — |
| inferix_cross_kv_set | 30 | 3,686,028 | 122,868 | cache_tokens | 1 | insufficient | — |
| inferix_cross_kv_get | 30 | 639,941 | 21,331 | — | 1 | insufficient | — |
| inferix_kv_free | 1 | 4,779 | 4,779 | — | 1 | insufficient | — |

## Relations and diagnostics

- **inferix_self_attention:** conditional on tokens=4680, -0.00996 × context_tokens + 1.46e+06 instructions/call; max unexplained 1.2%. Full declared-PCV model remains unidentifiable.
- **inferix_block:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens
- **inferix_cross_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, context_tokens
- **inferix_causal_rope:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, heads
- **inferix_vae_decoder:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames
- **inferix_transformer:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch, layers
- **inferix_kv_set:** -0.00499 × cache_tokens + 1.25e+05 instructions/call; max unexplained 4.5%.
- **inferix_text_encode:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: prompts
- **inferix_vae_clear_cache:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **inferix_kv_get:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **inferix_inference:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, initial_frames, batch
- **inferix_generator:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch
- **inferix_vae_cached_decode:** too few distinct states for an affine relation
- **inferix_vae_decode_to_pixel:** too few distinct states for an affine relation; rank-deficient PCVs: batch, cached
- **inferix_kv_allocate_slots:** too few distinct states for an affine relation; rank-deficient PCVs: layers
- **inferix_cross_kv_set:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: cache_tokens
- **inferix_cross_kv_get:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **inferix_kv_free:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation

## Largest exclusive functions

- `python3.12:_PyEval_EvalFrameDefault`: 495,688,420
- `libc.so.6:__default_morecore+?`: 141,644,994
- `ld-linux-x86-64.so.2:__tls_get_addr`: 130,828,416
- `python3.12:PyDict_Contains+?`: 117,328,332
- `python3.12:_PyObject_GenericGetAttrWithDict`: 107,346,237
- `libc.so.6:pthread_mutex_lock`: 106,806,492
- `libc.so.6:malloc`: 88,425,879
- `python3.12:PyObject_Free`: 78,944,455
- `libc.so.6:__pthread_mutex_unlock`: 78,416,610
- `python3.12:_PyArg_ParseStack+?`: 72,711,492
- `python3.12:_PyType_Lookup`: 69,794,172
- `python3.12:_PyTuple_Resize+?`: 67,439,141

The relation describes observed PCV states only. It does not measure GPU execution time or identify a critical-path bottleneck.
