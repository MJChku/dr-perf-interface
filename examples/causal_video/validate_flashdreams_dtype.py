"""CPU parity check for FlashDreams' opt-in text-encoder loading dtype."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('tree', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
sys.path.insert(0, str(a.tree/'flashdreams'))
import torch
from transformers import UMT5Config, UMT5EncoderModel
from flashdreams.infra.encoder.text import umt5

torch.set_num_threads(1)

class Tokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.zeros(len(texts), 512, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, text in enumerate(texts):
            n = min(len(text), 16)
            ids[i, :n] = torch.arange(1, n+1)
            mask[i, :n] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)

results = []
with tempfile.TemporaryDirectory(prefix='flashdreams-dtype-check-') as directory:
    root = Path(directory)
    for checkpoint_dtype in [torch.float32, torch.bfloat16]:
        torch.manual_seed(42)
        model = UMT5EncoderModel(UMT5Config(vocab_size=32, d_model=32, d_ff=64,
            d_kv=8, num_heads=4, num_layers=2, dropout_rate=0.0)).to(checkpoint_dtype)
        model.save_pretrained(root/'text_encoder')
        del model
        for requested_dtype in [torch.float32, torch.bfloat16]:
            with patch.object(umt5, 'T5Tokenizer', SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())):
                before = umt5.UMT5TextEncoderConfig(model_id_or_local_path=str(root),
                    dtype=requested_dtype).setup().eval()
                after = umt5.UMT5TextEncoderConfig(model_id_or_local_path=str(root),
                    dtype=requested_dtype, load_in_requested_dtype=True).setup().eval()
            left, right = before.state_dict(), after.state_dict()
            assert left.keys() == right.keys()
            for name, value in left.items():
                assert value.dtype == right[name].dtype
                assert torch.equal(value, right[name]), name
            with torch.inference_mode():
                b = before(['cat', 'longer prompt'])
                c = after(['cat', 'longer prompt'])
            assert torch.equal(b, c), (b-c).abs().max()
            assert torch.isfinite(b).all()
            assert b.shape == (2, 512, 32)
            results.append({'checkpoint_dtype':str(checkpoint_dtype),
                'requested_dtype':str(requested_dtype), 'state_tensors':len(left),
                'state_equal':True, 'encoder_output_equal':True})
            del before, after, left, right, b, c
report = {'passed':True, 'scope':'Actual FlashDreams encoder constructor and forward; tiny UMT5 checkpoint. Tokenizer stub supplies deterministic CPU token IDs/masks. Does not validate real video, CUDA execution, or performance.', 'cases':results}
if a.output:
    a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
