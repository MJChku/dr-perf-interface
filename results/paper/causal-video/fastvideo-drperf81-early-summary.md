# drperf causal-video summary

Valid collection; total marked exclusive instructions: 8,632,102,827.

| Region | Calls | Exclusive instructions | Mean/call | PCVs | States | Fit | Max unexplained |
|---|---:|---:|---:|---|---:|---|---:|
| fastvideo_transformer | 35 | 4,995,926,811 | 142,740,766 | frames, batch, layers | 1 | insufficient | — |
| fastvideo_causal_denoise | 1 | 1,066,177,352 | 1,066,177,352 | latent_frames, blocks, steps | 1 | insufficient | — |
| fastvideo_cross_attention | 1,050 | 694,798,419 | 661,713 | tokens, context_tokens | 1 | insufficient | — |
| fastvideo_self_attention | 1,050 | 569,299,108 | 542,190 | tokens, context_tokens | 7 | insufficient | — |
| fastvideo_rope | 2,100 | 485,999,779 | 231,428 | tokens, heads | 1 | insufficient | — |
| fastvideo_vae_decoder | 21 | 360,225,832 | 17,153,611 | latent_frames | 1 | insufficient | — |
| fastvideo_pipeline | 1 | 343,708,288 | 343,708,288 | frames, batch | 1 | insufficient | — |
| fastvideo_text_encode | 1 | 98,389,714 | 98,389,714 | prompts | 1 | insufficient | — |
| fastvideo_decode_stage | 1 | 6,773,289 | 6,773,289 | latent_frames | 1 | insufficient | — |
| fastvideo_kv_init | 1 | 3,431,926 | 3,431,926 | batch, layers, capacity_tokens | 1 | insufficient | — |
| fastvideo_vae_decode | 1 | 2,967,790 | 2,967,790 | latent_frames | 1 | insufficient | — |
| fastvideo_vae_clear_cache | 2 | 2,257,744 | 1,128,872 | — | 1 | insufficient | — |
| fastvideo_cross_kv_init | 1 | 1,578,006 | 1,578,006 | batch, layers, text_tokens | 1 | insufficient | — |
| fastvideo_rope_tables | 35 | 568,769 | 16,251 | frames, height, width, head_dim, start_frame | 7 | insufficient | — |

## Relations and diagnostics

- **fastvideo_transformer:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch, layers
- **fastvideo_causal_denoise:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: latent_frames, blocks, steps
- **fastvideo_cross_attention:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, context_tokens
- **fastvideo_self_attention:** conditional on tokens=4680, -0.00763 × context_tokens + 5.41e+05 instructions/call; max unexplained 0.3%. Full declared-PCV model remains unidentifiable.
- **fastvideo_rope:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: tokens, heads
- **fastvideo_vae_decoder:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: latent_frames
- **fastvideo_pipeline:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: frames, batch
- **fastvideo_text_encode:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: prompts
- **fastvideo_decode_stage:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: latent_frames
- **fastvideo_kv_init:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: batch, layers, capacity_tokens
- **fastvideo_vae_decode:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: latent_frames
- **fastvideo_vae_clear_cache:** no numeric declared PCV; no cost relation can be identified; one observed state; a constant cost is not a discovered relation
- **fastvideo_cross_kv_init:** one observed state; a constant cost is not a discovered relation; rank-deficient PCVs: batch, layers, text_tokens
- **fastvideo_rope_tables:** conditional on frames=3, height=30, width=52, head_dim=128, 0 × start_frame + 1.63e+04 instructions/call; max unexplained 0.0%. Full declared-PCV model remains unidentifiable.

## Largest exclusive functions

- `python3.12:_PyEval_EvalFrameDefault`: 938,514,317
- `libtorch_cpu.so:at::native::templates::cpu::(anonymous namespace)::normal_fill_16<>`: 625,322,880
- `libm.so.6:f64xsubf128+?`: 328,032,562
- `libtorch_cpu.so:at::CPUGeneratorImpl::random`: 238,828,800
- `python3.12:PyDict_Contains+?`: 187,152,741
- `libc.so.6:__default_morecore+?`: 165,198,256
- `libtorch_cpu.so:at::native::templates::cpu::(anonymous namespace)::normal_kernel<>`: 163,016,290
- `ld-linux-x86-64.so.2:__tls_get_addr`: 157,642,608
- `python3.12:_PyObject_GenericGetAttrWithDict`: 152,141,188
- `python3.12:PyObject_Free`: 142,707,881
- `libc.so.6:pthread_mutex_lock`: 118,549,460
- `python3.12:PyObject_Malloc`: 101,170,932

The relation describes observed PCV states only. It does not measure GPU execution time or identify a critical-path bottleneck.
