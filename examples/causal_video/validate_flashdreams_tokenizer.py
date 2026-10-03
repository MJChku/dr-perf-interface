"""Check pipeline-local tokenizer retention with real tokenizer/tiny UMT5 on CPU."""
import argparse
import copy
import gc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import weakref

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('tree',type=Path)
p.add_argument('tokenizer',type=Path)
p.add_argument('--output',type=Path)
a=p.parse_args()
sys.path.insert(0,str(a.tree/'flashdreams'))
import torch
from transformers import UMT5Config,UMT5EncoderModel
from flashdreams.infra.encoder.text.umt5 import UMT5TextEncoderConfig
from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

torch.set_num_threads(1)
texts=['A cat walks on the grass, realistic style','中文，猫在草地上行走',
       'नमस्ते दुनिया','🫨\U0001fae9\U0001fac6','hello\nworld','[MASK] <extra_id_299>']

def base_init(self,config):
    torch.nn.Module.__init__(self)
    self.diffusion_model=SimpleNamespace(device=torch.device('cpu'))

def release(pipe):
    encoder=weakref.ref(pipe.text_encoder)
    weights=weakref.ref(pipe.text_encoder.text_encoder)
    pipe.release_oneshot_encoders()
    assert pipe.text_encoder is None
    assert encoder() is None and weights() is None
    pipe.release_oneshot_encoders()

results={}
with tempfile.TemporaryDirectory(prefix='flashdreams-tokenizer-check-') as temp:
    root=Path(temp)/'model';root.mkdir()
    (root/'tokenizer').symlink_to(a.tokenizer.resolve(),target_is_directory=True)
    torch.manual_seed(42)
    model=UMT5EncoderModel(UMT5Config(vocab_size=256384,d_model=8,d_ff=16,
            d_kv=4,num_heads=2,num_layers=1,dropout_rate=0.0)).to(torch.bfloat16)
    model.save_pretrained(root/'text_encoder');del model
    config=SimpleNamespace(text_encoder=UMT5TextEncoderConfig(
        model_id_or_local_path=str(root),load_in_requested_dtype=True),
        image_encoder=None,retain_cpu_tokenizer=True)
    with patch.object(StreamInferencePipeline,'__init__',base_init):
        pipe=WanInferencePipeline(config)
        other=WanInferencePipeline(copy.deepcopy(config))
    token=pipe.text_encoder.tokenizer
    assert other.text_encoder.tokenizer is not token
    before=pipe.text_encoder(texts)
    for _ in range(2):
        release(pipe)
        assert pipe._retained_text_tokenizer[1] is token
        pipe._ensure_oneshot_encoders_loaded()
        assert pipe.text_encoder.tokenizer is token
        assert torch.equal(before,pipe.text_encoder(texts))
    results['two_reloads_exact_embeddings']=True
    results['pipeline_instances_do_not_share_tokenizers']=True
    results['encoder_and_weights_collected_on_release']=True
    release(pipe)
    alias=Path(temp)/'alias';alias.symlink_to(root,target_is_directory=True)
    config.text_encoder.model_id_or_local_path=str(alias)
    pipe._ensure_oneshot_encoders_loaded()
    assert pipe.text_encoder.tokenizer is not token
    assert torch.equal(before,pipe.text_encoder(texts))
    results['changed_model_path_reloads_tokenizer']=True
    config.retain_cpu_tokenizer=False
    old=weakref.ref(pipe.text_encoder.tokenizer)
    release(pipe);assert pipe._retained_text_tokenizer is None and old() is None
    pipe._ensure_oneshot_encoders_loaded()
    assert torch.equal(before,pipe.text_encoder(texts))
    results['disabled_retention_releases_tokenizer']=True
    retained=weakref.ref(other.text_encoder.tokenizer)
    release(other)
    del pipe,other,token
    gc.collect()
    assert retained() is None
    results["deleting_pipeline_releases_retained_tokenizer"]=True
report={'passed':True,'scope':'Real tokenizer and tiny UMT5; actual Wan pipeline encoder release/reload methods. Heavy diffusion/decoder base constructor replaced by a CPU device stub. Does not validate full-model GPU output or performance.','prompts':texts,'checks':results}
if a.output:a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
