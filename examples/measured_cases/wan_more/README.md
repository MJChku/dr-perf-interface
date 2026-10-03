# Wan 2.1, second round: the paths the first pass did not mark

24 more regions (`mark_more.py`) on the attention processor (qkv / rope-apply
/ sdpa / out), the per-step host work, the VAE encode and
`DiagonalGaussianDistribution`, tiling, and post-processing; driver
`driver.py` with `mode=0 cb=1` (full call with the step callback on) and
`mode=1` (VAE encode).

Study ("Wan case B"):

    dgd_init        14.2*numel + 114,569.1       ->   0.349*numel + 70,030      (40x on the slope)
    callback_step   14,019.5*ninputs + 10,872.6  ->   425*ninputs + 26,940      (33x per name)
    attn_rope       5,764.3*seq_len + 320,506    ->   5,675.3*seq_len + 323,964 (-1.5% instructions; -25% wall at production)

* `DiagonalGaussianDistribution.__init__` computed `std = exp(0.5*logvar)` and
  `var = exp(logvar)` eagerly; Wan reads `mode()`, which touches neither
  (`wan_more_dgd_lazy.patch`: compute on first read);
* the step callback evaluated `locals()` once per requested name
  (`wan_more_cb_locals.patch`);
* the rotary application allocated temporaries per call
  (`wan_more_rope_apply.patch`): the fix removes memory traffic, not
  instructions, so the instruction formula reports -1.5% where the clock
  reports -25% -- the study's own caveat that instructions are not time.

Base tree = `wan-src` (first-round fixes, 20 regions) plus the 24 second-round
regions; fix tree = the study's `wan-more`.  `prepare.sh` warns if the two
region sets differ.  Equivalence: `equiv.py` drives the rotary application at
a spread of shapes, `vae.encode` statistics, tiled decode/encode and
post-processing, and prints digests.

## Rerun (2026-10-02, `run.sh`, `QUICK=1`)

```
callback_step  base  15,369*ninputs + 2*frames + 11,504    0% irregular
               fix      434*ninputs - 2*frames + 27,441    0%
dgd_init       base  14*numel + 126,000                    2%
               fix    0*numel +  71,874                    0%
attn_rope      base  5,761*seq_len + 334,379               0%
               fix   5,672*seq_len + 337,132               0%
```
Per callback name 15,369 -> 434 (study 14,019.5 -> 425); the posterior slope
14 -> 0 per element (study 14.2 -> 0.349); the rotary application -1.5% per
element, the fix that removes memory traffic rather than instructions.  The
`cb` state (callback tensor inputs 1..3) is what gives `ninputs` its range;
with one input the fix costs slightly more than the original and the
formula says so (crossover at 1.2 names).  All digests identical.
