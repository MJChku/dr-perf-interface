# drperf causal-video summary

Valid collection; total marked exclusive instructions: 5,227,639,115.

| Region | Calls | Exclusive instructions | Mean/call | PCVs | States | Fit | Max unexplained |
|---|---:|---:|---:|---|---:|---|---:|
| inferix_self_attention | 1,050 | 1,547,620,803 | 1,473,925 | tokens, context_tokens | 7 | insufficient | — |
| inferix_block | 1,050 | 1,203,924,216 | 1,146,594 | tokens | 1 | insufficient | — |
| inferix_cross_attention | 1,050 | 911,861,228 | 868,439 | tokens, context_tokens | 1 | insufficient | — |
| inferix_causal_rope | 2,100 | 752,732,316 | 358,444 | tokens, heads | 1 | insufficient | — |
| inferix_vae_decoder | 21 | 311,728,733 | 14,844,225 | frames | 1 | insufficient | — |
| inferix_transformer | 35 | 203,821,523 | 5,823,472 | frames, batch, layers | 1 | insufficient | — |
| inferix_kv_set | 1,050 | 136,549,472 | 130,047 | cache_tokens | 7 | identified | 4.3% |
| inferix_text_encode | 1 | 50,186,626 | 50,186,626 | prompts | 1 | insufficient | — |
| inferix_vae_clear_cache | 15 | 31,149,907 | 2,076,660 | — | 1 | insufficient | — |
| inferix_inference | 1 | 23,802,729 | 23,802,729 | frames, initial_frames, batch | 1 | insufficient | — |
| inferix_kv_get | 1,050 | 18,045,912 | 17,187 | — | 1 | insufficient | — |
| inferix_generator | 35 | 16,940,120 | 484,003 | frames, batch | 1 | insufficient | — |
| inferix_vae_cached_decode | 21 | 5,555,638 | 264,554 | frames | 1 | insufficient | — |
| inferix_kv_allocate_slots | 60 | 4,988,920 | 83,149 | tokens, layers | 2 | insufficient | — |
| inferix_vae_decode_to_pixel | 7 | 4,786,019 | 683,717 | frames, batch, cached | 1 | insufficient | — |
| inferix_cross_kv_set | 30 | 3,660,443 | 122,015 | cache_tokens | 1 | insufficient | — |
| inferix_cross_kv_get | 30 | 279,707 | 9,324 | — | 1 | insufficient | — |
| inferix_kv_free | 1 | 4,803 | 4,803 | — | 1 | insufficient | — |

## Relations and diagnostics

- **inferix_self_attention:** conditional on tokens=4680, -0.00705 × context_tokens + 1.46e+06 instructions/call; max unexplained 0.7%. Full declared-PCV model remains unidentifiable.
- **inferix_block:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens
- **inferix_cross_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, context_tokens
- **inferix_causal_rope:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, heads
- **inferix_vae_decoder:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames
- **inferix_transformer:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch, layers
- **inferix_kv_set:** -0.00499 × cache_tokens + 1.25e+05 instructions/call; max unexplained 4.3%.
- **inferix_text_encode:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: prompts
- **inferix_vae_clear_cache:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **inferix_inference:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, initial_frames, batch
- **inferix_kv_get:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **inferix_generator:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch
- **inferix_vae_cached_decode:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames
- **inferix_kv_allocate_slots:** too few distinct states for an affine relation; rank-deficient PCVs: layers
- **inferix_vae_decode_to_pixel:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch, cached
- **inferix_cross_kv_set:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: cache_tokens
- **inferix_cross_kv_get:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **inferix_kv_free:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation

## Largest exclusive functions

- `python3.12:_PyEval_EvalFrameDefault`: 472,895,460
- `libc.so.6:__default_morecore+?`: 133,879,799
- `ld-linux-x86-64.so.2:__tls_get_addr`: 122,221,548
- `python3.12:PyDict_Contains+?`: 112,172,254
- `libc.so.6:pthread_mutex_lock`: 100,827,728
- `python3.12:_PyObject_GenericGetAttrWithDict`: 99,653,806
- `libc.so.6:malloc`: 82,854,925
- `python3.12:PyObject_Free`: 74,753,251
- `libc.so.6:__pthread_mutex_unlock`: 73,994,044
- `python3.12:_PyArg_ParseStack+?`: 68,912,620
- `python3.12:_PyType_Lookup`: 65,791,898
- `python3.12:_PyTuple_Resize+?`: 65,050,095

The relation describes observed PCV states only. It does not measure GPU execution time or identify a critical-path bottleneck.
