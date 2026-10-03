"""Equivalence check for the Wan 2.1 host-side fixes.

    .venv/bin/python equiv_wan.py SRC_DIR

Builds exactly the tiny pipeline run_tiny.py builds (same configs, same seeds)
from the diffusers package found under SRC_DIR, runs a handful of grid points
and prints a sha256 of

  * the returned LATENTS (output_type="latent") -- what the denoise loop produced
  * the decoded VIDEO tensor (output_type="pt") -- what the VAE produced

so a fixed tree can be compared against the pristine one byte for byte.

    .venv/bin/python equiv_wan.py diffusers/src        > /tmp/base.txt
    .venv/bin/python equiv_wan.py wan-src              > /tmp/fix.txt
    diff /tmp/base.txt /tmp/fix.txt
"""

import hashlib
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SRC = os.path.abspath(sys.argv[1])
sys.path.insert(0, "/home/ubuntu/drperf/perfmark/python")   # marked copies import perfmark
sys.path.insert(0, SRC)

import torch  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(8)

import diffusers  # noqa: E402
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline, WanTransformer3DModel  # noqa: E402

got = os.path.dirname(os.path.abspath(diffusers.__file__))
if got != os.path.join(SRC, "diffusers"):
    sys.exit("equiv_wan.py: diffusers imported from %s, not %s/diffusers" % (got, SRC))

TEXT_DIM = 32
SEED = 0

TRANSFORMER_CONFIG = dict(
    patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=16,
    in_channels=16, out_channels=16, text_dim=TEXT_DIM, freq_dim=32, ffn_dim=128,
    num_layers=2, cross_attn_norm=True, qk_norm="rms_norm_across_heads",
    eps=1e-6, rope_max_seq_len=1024,
)
VAE_CONFIG = dict(
    base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1, attn_scales=[],
    temperal_downsample=[False, True, True], scale_factor_temporal=4, scale_factor_spatial=8,
)

# (frames, steps, hw, batch, text) -- corners of the 27-point grid plus a batch/text point
POINTS = [
    (5, 4, 128, 1, 64),
    (9, 6, 192, 1, 64),
    (17, 8, 256, 1, 64),
    (17, 4, 128, 1, 64),
    (9, 4, 192, 2, 128),
]


def digest(t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:32] + " shape=%s" % (tuple(a.shape),)


def build():
    torch.manual_seed(SEED)
    transformer = WanTransformer3DModel(**TRANSFORMER_CONFIG).eval()
    vae = AutoencoderKLWan(**VAE_CONFIG).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae,
                       transformer=transformer, scheduler=scheduler)
    pipe.set_progress_bar_config(disable=True)
    return pipe


pipe = build()

for frames, steps, hw, batch, text_len in POINTS:
    g = torch.Generator().manual_seed(SEED)
    prompt_embeds = torch.randn(batch, text_len, TEXT_DIM, generator=g)
    negative_prompt_embeds = torch.randn(batch, text_len, TEXT_DIM, generator=g)
    label = "frames=%-3d steps=%d hw=%-4d batch=%d text=%-4d" % (frames, steps, hw, batch, text_len)
    kw = dict(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
              height=hw, width=hw, num_frames=frames, num_inference_steps=steps,
              guidance_scale=5.0)
    lat = pipe(generator=torch.Generator().manual_seed(SEED), output_type="latent", **kw).frames
    vid = pipe(generator=torch.Generator().manual_seed(SEED), output_type="pt", **kw).frames
    print("%s  latents %s" % (label, digest(lat)))
    print("%s  video   %s" % (label, digest(vid)))
