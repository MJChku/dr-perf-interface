# drperf causal-video summary

Valid collection; total marked exclusive instructions: 4,955,856,063.

| Region | Calls | Exclusive instructions | Mean/call | PCVs | States | Fit | Max unexplained |
|---|---:|---:|---:|---|---:|---|---:|
| inferix_self_attention | 1,050 | 1,542,337,048 | 1,468,892 | tokens, context_tokens | 7 | insufficient | — |
| inferix_block | 1,050 | 1,197,518,614 | 1,140,494 | tokens | 1 | insufficient | — |
| inferix_cross_attention | 1,050 | 908,298,888 | 865,047 | tokens, context_tokens | 1 | insufficient | — |
| inferix_causal_rope | 2,100 | 498,905,046 | 237,574 | tokens, heads | 1 | insufficient | — |
| inferix_vae_decoder | 21 | 310,711,725 | 14,795,796 | frames | 1 | insufficient | — |
| inferix_transformer | 35 | 203,344,694 | 5,809,848 | frames, batch, layers | 1 | insufficient | — |
| inferix_kv_set | 1,050 | 135,614,805 | 129,157 | cache_tokens | 7 | identified | 5.4% |
| inferix_text_encode | 1 | 50,015,579 | 50,015,579 | prompts | 1 | insufficient | — |
| inferix_vae_clear_cache | 15 | 31,133,758 | 2,075,584 | — | 1 | insufficient | — |
| inferix_inference | 1 | 23,745,529 | 23,745,529 | frames, initial_frames, batch | 1 | insufficient | — |
| inferix_kv_get | 1,050 | 18,160,822 | 17,296 | — | 1 | insufficient | — |
| inferix_generator | 35 | 16,896,516 | 482,758 | frames, batch | 1 | insufficient | — |
| inferix_vae_cached_decode | 21 | 5,513,393 | 262,543 | frames | 1 | insufficient | — |
| inferix_kv_allocate_slots | 60 | 4,967,922 | 82,799 | tokens, layers | 2 | insufficient | — |
| inferix_vae_decode_to_pixel | 7 | 4,761,831 | 680,262 | frames, batch, cached | 1 | insufficient | — |
| inferix_cross_kv_set | 30 | 3,645,715 | 121,524 | cache_tokens | 1 | insufficient | — |
| inferix_cross_kv_get | 30 | 279,384 | 9,313 | — | 1 | insufficient | — |
| inferix_kv_free | 1 | 4,794 | 4,794 | — | 1 | insufficient | — |

## Relations and diagnostics

- **inferix_self_attention:** conditional on tokens=4680, -0.00862 × context_tokens + 1.45e+06 instructions/call; max unexplained 1.1%. Full declared-PCV model remains unidentifiable.
- **inferix_block:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens
- **inferix_cross_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, context_tokens
- **inferix_causal_rope:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, heads
- **inferix_vae_decoder:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames
- **inferix_transformer:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch, layers
- **inferix_kv_set:** -0.00499 × cache_tokens + 1.22e+05 instructions/call; max unexplained 5.4%.
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

- `python3.12:_PyEval_EvalFrameDefault`: 481,438,634
- `libc.so.6:__default_morecore+?`: 119,836,770
- `ld-linux-x86-64.so.2:__tls_get_addr`: 113,415,828
- `python3.12:PyDict_Contains+?`: 111,963,809
- `python3.12:_PyObject_GenericGetAttrWithDict`: 99,905,806
- `libc.so.6:pthread_mutex_lock`: 90,738,464
- `libc.so.6:malloc`: 75,537,796
- `python3.12:PyObject_Free`: 73,044,489
- `libc.so.6:__pthread_mutex_unlock`: 66,685,420
- `python3.12:_PyTuple_Resize+?`: 63,182,843
- `python3.12:_PyType_Lookup`: 62,944,833
- `python3.12:_PyArg_ParseStack+?`: 60,886,420

The relation describes observed PCV states only. It does not measure GPU execution time or identify a critical-path bottleneck.
