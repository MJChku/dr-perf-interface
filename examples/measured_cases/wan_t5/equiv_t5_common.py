"""Shared tiny-pipeline construction for equiv_t5.py / bench_t5.py."""
import torch

TEXT_DIM = 32
SEED = 0
TLAYERS = 4

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

PROMPT_WORDS = ("a cat and a dog baking a cake together in a cozy kitchen while "
                "sunlight streams through the window and flour drifts in the air "
                "over a wooden table covered in mixing bowls and measuring spoons ").split()
NEG = "blurry, low quality, static"


def make_prompt(words):
    body = " ".join(PROMPT_WORDS[i % len(PROMPT_WORDS)] for i in range(words))
    return "  " + body.replace(" a ", " a&amp; ", 1) + "  "


class DuckTokenizer:
    class Out(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

        @property
        def attention_mask(self):
            return self["attention_mask"]

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, prompt, padding=None, max_length=None, truncation=None,
                 add_special_tokens=None, return_attention_mask=None, return_tensors=None):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        b = len(prompts)
        ids = torch.zeros(b, max_length, dtype=torch.long)
        mask = torch.zeros(b, max_length, dtype=torch.long)
        for i, p in enumerate(prompts):
            n = min(len(p.split()) + 1, max_length)
            ids[i, :n] = torch.arange(1, n + 1) % self.vocab_size
            mask[i, :n] = 1
        return self.Out(input_ids=ids, attention_mask=mask)


def build(WanTransformer3DModel, AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanPipeline):
    from transformers import UMT5Config, UMT5EncoderModel
    torch.manual_seed(SEED)
    transformer = WanTransformer3DModel(**TRANSFORMER_CONFIG).eval()
    vae = AutoencoderKLWan(**VAE_CONFIG).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    cfg = UMT5Config(vocab_size=256, d_model=TEXT_DIM, d_kv=8, d_ff=64,
                     num_layers=TLAYERS, num_heads=4,
                     relative_attention_num_buckets=8, dropout_rate=0.0)
    torch.manual_seed(SEED)
    text_encoder = UMT5EncoderModel(cfg).eval()
    pipe = WanPipeline(tokenizer=DuckTokenizer(cfg.vocab_size), text_encoder=text_encoder,
                       vae=vae, transformer=transformer, scheduler=scheduler)
    pipe.set_progress_bar_config(disable=True)
    return pipe
