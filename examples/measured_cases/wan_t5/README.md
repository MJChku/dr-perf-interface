# Wan 2.1 text encoder: the cost follows the padding, quadratically

`WanPipeline._get_t5_prompt_embeds` tokenises with `padding="max_length"` at
`max_sequence_length` (512), runs the UMT5 encoder on the padded batch, then
slices each row to its real length and zero-pads again.  Tree `wan-t5`,
markers `mark_t5.py <tree>/diffusers sq2` (states `tag, ptok, sq, lsq`, with
`sq = batch*max_len^2/64`, `lsq = max_len^2/64`); driver `driver.py mode=6`
(encode-prompt only) over real tokens {8, 32, 128, 480} x max length
{64, 128, 256, 512} x batch {1, 2}.

Study ("Wan case D"):

    t5_embeds   25,688.3*ptok + 46,070.5*sq + 27,692*lsq + 6,565,689   ->   693.5*ptok + 34.8*sq - 27.7*lsq + 6,694,193
    per call, batch 1, max_len 512:  8 real tokens 384.8M -> 8.07M (47.7x); 32 -> 8.46M; 128 -> 17.3M; 480 -> 124.5M (3.1x)

The real prompt's coefficient was -1.2: it is free.  Everything was paid per
padded token (projections, FFN) and per padded pair (attention), plus an
unbatched L^2 term because UMT5 materialises its relative position bias per
layer.  The fix encodes at the longest real length rounded up to a multiple
of 8 (floor 16) and zero-pads afterwards; T5's position bias is relative and
masked scores are shifted by `finfo.min`, so real-token outputs do not depend
on the padding -- bit-identical in fp32 at ten encode points and three full
calls.  The other half (shortening `kv_len` into the transformer) is not
behaviour-preserving and is not done: the model attends to its padding.

`mark_t5.py` rewrites the method from a template; without `--fix` it is the
original padding (base), with `--fix` the short-pad encode (fix).
Equivalence: `equiv.py <tree> OUT.npz` records embeddings, latents and video
over a spread of points; `cmp_t5.py` compares two recordings.

## Rerun (2026-10-02, `run.sh`, `QUICK=1`)

```
t5_enc (own thread, nested included)   ptok=512   base  61,002,035   fix  10,353,970   (5.9x)
                                       ptok=64    base  12,241,860   fix   8,810,152
```
The quick grid has one point per padded length (rtok 8 and 128 share a state
under the `sq2` set), so the formula is not identified; the full grid gives
the study's `ptok`/`sq`/`lsq` plane.  Embeddings, latents and video over 26
arrays bitwise identical.
