# Wan 2.1, the per-step host constant decomposed

Fifty-five temporary probes split `denoise_step`'s 12M-instruction constant
into sub-steps; the tree `wan-tax` keeps the probe regions (`ax_qkv`,
`tf_cond`, `bl_norm2`, `at_out`, `bl_mod`, `tf_shift`, `at_rope`, ...) and
`wan_tax.patch` the fixes.  Driver `driver.py`, same grid as `wan_host`.

Study ("Wan case C"):

    ax_qkv (cross-attn K/V)   378,712 -> 137,932   (-63.6%)
    tf_cond                   632,627 -> 416,146   (-34.2%)
    block                12,424.5*seq_len + 2,177,148.7   ->   12,424*seq_len + 1,967,858.2
    transformer_forward  54,619.5*frames + 27,605*seq_len + 5,678,082   ->   54,911*frames + 27,604.2*seq_len + 5,005,886.4

Cross-attention re-projected the fixed prompt in every block of every forward
(3,000x per video instead of 60x); the timestep embedding, modulation, fp32
upcast and `Dropout(p=0)` were redone per CFG branch.  All memoised on tensor
identity plus `_version`, bypassed under autograd, or provably identity.  At
production: 653M fewer host instructions per video (-9.8% of the constant
tax) and 2,940 of 3,000 cross-attention projections gone, 14.2 TFLOP per video,
for 189 MB of bf16 residency.  A per-block constant that no state explained,
attributed partly to an sgemm kernel, was the tell.

`wan_tax.patch` applies on the marked tree (`-p1`); base = fix reversed.
Equivalence: `equiv.py` prints sha256 of latents and decoded video from each
tree.

## Rerun (2026-10-02, `run.sh`, `QUICK=1`)

```
attn                 base  547*seq_len + 758*kv_len + 457,574   (own cost)     per call 844,563 at seq 128, kv 64
                     fix   451*seq_len + 705*kv_len + 351,848                  per call 725,116
block                base  10,446*seq_len + 774,846    ->   fix  10,446*seq_len + 731,518
transformer_forward  base  66,353*frames + 5,591*seq_len + 713,696   ->   fix  66,419*frames + 5,592*seq_len + 681,243
denoise_step         base  514,271   ->   fix  505,353
```
The study's `ax_qkv`/`tf_cond` numbers came from fifty-five temporary probe
regions that anchor on the unfixed code (`videogen/mark_tax_probe.py`) and
cannot be inserted into the fixed tree, so the rerun reports the regions the
tree keeps: the cross-attention memo shows as `attn`'s constant 458k -> 352k
(-23%) and -14% per call, and the per-branch work moved out of `block` and
`transformer_forward`.  Latents and video identical.
