"""Equivalence check for the SECOND round of Wan 2.1 host-side fixes.

    .venv/bin/python equiv_more.py SRC_DIR

Complements equiv_wan.py (which hashes the denoise-loop latents and the decoded
video) with the paths round two touches or measures:

  * the rotary application inside `WanAttnProcessor`, driven directly at a
    spread of (seq_len, heads, head_dim) so the change is checked bit for bit
    rather than only through its effect on the latents;
  * `vae.encode(...)` -> mode / sample / std / var;
  * `vae.decode` and `vae.encode` under `enable_tiling()`;
  * `postprocess_video` for output_type "np" and "pil".

    .venv/bin/python equiv_more.py wan-src   > /tmp/base.txt
    .venv/bin/python equiv_more.py wan-more  > /tmp/fix.txt
    diff /tmp/base.txt /tmp/fix.txt
"""

import hashlib
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

SRC = os.path.abspath(sys.argv[1])
sys.path.insert(0, "/home/ubuntu/drperf/perfmark/python")
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

import diffusers  # noqa: E402
from diffusers import AutoencoderKLWan, WanTransformer3DModel  # noqa: E402
from diffusers.video_processor import VideoProcessor  # noqa: E402

got = os.path.dirname(os.path.abspath(diffusers.__file__))
if got != os.path.join(SRC, "diffusers"):
    sys.exit("equiv_more.py: diffusers imported from %s, not %s/diffusers" % (got, SRC))

SEED = 0
VAE_CONFIG = dict(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                  attn_scales=[], temperal_downsample=[False, True, True],
                  scale_factor_temporal=4, scale_factor_spatial=8)


def digest(t):
    if isinstance(t, (list, tuple)):
        return hashlib.sha256(b"".join(digest(x).encode() for x in t)).hexdigest()[:32] + " n=%d" % len(t)
    if hasattr(t, "tobytes") and not isinstance(t, torch.Tensor):        # PIL image
        if hasattr(t, "size"):
            a = np.asarray(t)
            return hashlib.sha256(a.tobytes()).hexdigest()[:32] + " shape=%s" % (a.shape,)
    if isinstance(t, np.ndarray):
        a = np.ascontiguousarray(t, dtype=np.float32) if t.dtype != np.uint8 else t
        return hashlib.sha256(a.tobytes()).hexdigest()[:32] + " shape=%s" % (a.shape,)
    a = t.detach().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:32] + " shape=%s" % (tuple(a.shape),)


# ------------------------------------------------------- 1. rotary application
# Driven through the real processor: build one WanAttention-shaped module and
# call the processor on inputs whose rotary tables have the exact structure
# WanRotaryPosEmbed produces (cos/sin repeat-interleaved by 2).
print("== rotary through WanAttnProcessor ==")
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor  # noqa: E402

for seq_len, heads, head_dim in [(64, 4, 16), (192, 4, 16), (320, 2, 32),
                                 (512, 8, 8), (1280, 4, 16), (720, 12, 16)]:
    dim = heads * head_dim
    torch.manual_seed(SEED)
    attn = WanAttention(dim=dim, heads=heads, dim_head=head_dim, eps=1e-6,
                        cross_attention_dim_head=None, processor=WanAttnProcessor()).eval()
    g = torch.Generator().manual_seed(SEED)
    hs = torch.randn(1, seq_len, dim, generator=g)
    half = torch.randn(1, seq_len, 1, head_dim // 2, generator=g)
    cos = half.repeat_interleave(2, dim=-1)
    sin = torch.randn(1, seq_len, 1, head_dim // 2, generator=g).repeat_interleave(2, dim=-1)
    out = attn(hs, None, None, (cos, sin))
    print("  seq=%-5d heads=%-3d head_dim=%-3d  %s" % (seq_len, heads, head_dim, digest(out)))

# ---------------------------------------------------------- 2. VAE encode path
print("== vae.encode ==")
torch.manual_seed(SEED)
vae = AutoencoderKLWan(**VAE_CONFIG).eval()
for frames, hw in [(1, 64), (5, 128), (9, 128), (17, 192), (33, 64)]:
    g = torch.Generator().manual_seed(SEED)
    video = torch.randn(1, 3, frames, hw, hw, generator=g)
    post = vae.encode(video).latent_dist
    smp = post.sample(generator=torch.Generator().manual_seed(SEED))
    print("  frames=%-3d hw=%-4d mode %s" % (frames, hw, digest(post.mode())))
    print("  frames=%-3d hw=%-4d samp %s" % (frames, hw, digest(smp)))
    print("  frames=%-3d hw=%-4d std  %s" % (frames, hw, digest(post.std)))
    print("  frames=%-3d hw=%-4d var  %s" % (frames, hw, digest(post.var)))
    print("  frames=%-3d hw=%-4d lvar %s" % (frames, hw, digest(post.logvar)))
    print("  frames=%-3d hw=%-4d kl   %s" % (frames, hw, digest(post.kl())))

# ------------------------------------------------------------- 3. tiled VAE
print("== tiled decode / encode ==")
for tmin, tstride, hw, frames in [(128, 96, 256, 1), (128, 80, 320, 5), (256, 192, 448, 5)]:
    torch.manual_seed(SEED)
    tvae = AutoencoderKLWan(**VAE_CONFIG).eval()
    tvae.enable_tiling(tile_sample_min_height=tmin, tile_sample_min_width=tmin,
                       tile_sample_stride_height=tstride, tile_sample_stride_width=tstride)
    g = torch.Generator().manual_seed(SEED)
    z = torch.randn(1, 16, (frames - 1) // 4 + 1, hw // 8, hw // 8, generator=g)
    dec = tvae.decode(z, return_dict=False)[0]
    video = torch.randn(1, 3, frames, hw, hw, generator=g)
    enc = tvae.encode(video).latent_dist.mode()
    print("  tmin=%-4d tstride=%-4d hw=%-4d frames=%-3d dec %s" % (tmin, tstride, hw, frames, digest(dec)))
    print("  tmin=%-4d tstride=%-4d hw=%-4d frames=%-3d enc %s" % (tmin, tstride, hw, frames, digest(enc)))

# ------------------------------------------------------------ 4. postprocess
print("== postprocess_video ==")
vp = VideoProcessor(vae_scale_factor=8)
for frames, hw, batch in [(9, 128, 1), (17, 192, 1), (5, 128, 2)]:
    g = torch.Generator().manual_seed(SEED)
    vid = torch.randn(batch, 3, frames, hw, hw, generator=g).clamp(-1, 1)
    for ot in ("np", "pt", "pil"):
        out = vp.postprocess_video(vid, output_type=ot)
        if ot == "pil":
            flat = [np.asarray(im) for b in out for im in b]
            d = digest(np.stack(flat))
        else:
            d = digest(out)
        print("  frames=%-3d hw=%-4d batch=%d %-4s %s" % (frames, hw, batch, ot, d))
