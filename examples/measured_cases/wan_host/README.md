# Wan 2.1 in diffusers, the host side

A tiny random-init `WanPipeline` (2-layer transformer of 64 hidden, VAE base
dim 8, precomputed prompt embeddings) so a full call runs on CPU in 0.3-5 s;
twenty regions from `mark_wan.py`; driver `driver.py` over the grid
frames {5, 9, 17} x steps {4, 6, 8} x resolution {128, 192, 256}.  The
matmul and attention kernels are counted too and land in the irregular term;
the affine coefficients and constants are the host orchestration.

Study ("Video generation: Wan 2.1 in diffusers"):

    rope         29.9*frames + 312,303.4  (35.2%)   ->   0.5*frames + 44,711.8   (22.9%)   mean per call 482,987 -> 58,612
    cond_embed   3,855.4*batch - 37*text_len + 617,350.4   ->   3,943*batch + 513,647.5     mean per call 780,728 -> 564,337
    wan_call     4,537,754*num_frames + 11,431,003*steps + 27,397,966   ->   4,501,651*num_frames + 10,750,513*steps + 27,537,440

* `WanRotaryPosEmbed.forward` depends only on the latent shape and ran on
  every forward: `wan_rope.patch` caches the (cos, sin) tables on
  (frames, height, width, device, dtype);
* `text_embedder(encoder_hidden_states)` ran on every forward and CFG branch
  on an input that does not change within a call: `wan_text_embed.patch`
  memoises it on tensor identity and version, bypassed under autograd;
* the VAE decode did `torch.cat` per latent frame: `wan_vae_cat.patch`
  collects and concatenates once (927.5 -> 92.3 ms at production width; at
  this tiny VAE it is ~1%, and the study records that the rising per-frame
  cost it was first blamed for was the first chunk's different shape);
* `wan_misc.patch`: the full-size `torch.ones` mask built per call for a path
  Wan 2.1 does not take.

The study's `wan-src` carries the four fixes under its markers; `prepare.sh`
unmarks, reverses them, and re-marks for the base.  Equivalence: `equiv.py`
builds the same pipeline from each tree and prints sha256 of the latents and
the decoded video at several grid points (and encode/decode separately).

## Rerun (2026-10-02, `run.sh`, `QUICK=1`)

```
rope        base  7,156*frames + 886*height + 298,091    18% irregular      per call 363,852 .. 459,921
            fix   1,582*frames + 360*height + 45,496      5%               per call  57,323 ..  67,299
cond_embed  base  5,212*batch + 36*text_len + 612,039    42%               per call 778,804
            fix   6,015*batch + 35*text_len + 511,397    26%               per call 579,915
```
Per call: rope -84..-86% (study -87.9% on the full grid), cond_embed -25.5%
(study -27.7%).  The quick grid's shape range is narrow, so the rope plane's
coefficients are not the study's; the per-call means are the comparable
numbers.  Latents and video identical between the trees, and the latents
digest at (5 frames, 4 steps, 128) is the study's `c64589693a19156c`.
