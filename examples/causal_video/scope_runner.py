"""Headless public Scope LongLive streaming pipeline, with audited GX control metadata."""
import argparse,ctypes,functools,hashlib,inspect,json,os,sys,time
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser()
p.add_argument('--tree',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','drperf','partial_sync','gpu_only_partial_sync'],default='emu')
p.add_argument('--chunks',type=int,default=6);p.add_argument('--switch-at',type=int,default=3)
p.add_argument('--height',type=int,default=320);p.add_argument('--width',type=int,default=576)
p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=1)
a=p.parse_args();assert a.chunks>0 and a.repetitions>0
for name in ['tree','base','checkpoint','output']:setattr(a,name,getattr(a,name).resolve())
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output);sys.path.insert(0,str(a.tree/'src'))
import torch,diffusers,transformers,peft
from omegaconf import OmegaConf
from scope.core.pipelines.longlive.pipeline import LongLivePipeline
from scope.core.pipelines.longlive.modules import causal_model
from scope.core.pipelines.wan2_1.components import WanTextEncoderWrapper,WanDiffusionWrapper
from scope.core.pipelines.wan2_1 import utils as wan_utils
from scope.core.pipelines import utils as pipeline_utils
torch.set_num_threads(1);torch.set_grad_enabled(False)
native=a.role in ['native','gpu-profile']
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==native
prompts=['A cat walks on the grass, realistic style','A cat runs through a field of flowers, realistic style']
report=dict(role=a.role,chunks=a.chunks,switch_at=a.switch_at,height=a.height,width=a.width,seed=42,
            prompts=prompts,warmup=a.warmup,repetitions=a.repetitions,torch=torch.__version__,
            diffusers=diffusers.__version__,transformers=transformers.__version__,peft=peft.__version__,
            text_dtype='BF16',quantization=None,performance_lora=True,vae_type='wan',
            generation_wall_seconds=[],warmup_wall_seconds=[],harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((a.tree/'src/scope/core/pipelines').rglob('*.py'))},
            adaptations=['Public BF16 text encoder checkpoint (supported constructor default), rather than test.py optional FP8 checkpoint.',
                         'Checkpoint tensors preconverted to final BF16 inference dtype, verified against originals; mmap loading outside measurement.',
                         'CPU-known tokenizer lengths and KV scalar metadata, retaining original GPU reductions, reads and writes; native asserts exact metadata.',
                         'Headless public pipeline calls, CPU output concatenation; exclude file encoding and UI.'])
def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save();counts=Counter();attention_shapes=Counter();model_calls=Counter()
# Bound checkpoint setup memory. Public modules imported these functions by value.
def load_state(path):
    if str(path).endswith(('.pt','.pth')):return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    return original_load(path)
original_load=pipeline_utils.load_state_dict
original_loaders=(original_load,wan_utils.load_state_dict)
for name,module in list(sys.modules.items()):
    if name.startswith('scope.core.pipelines') and getattr(module,'load_state_dict',None) in original_loaders:
        module.load_state_dict=load_state
class CacheScalar:
    def __init__(self,tensor):self.tensor=tensor;self.value=0
    def item(self):
        observed=self.tensor.item();counts['cache_reads']+=1
        if native:assert observed==self.value,(observed,self.value)
        elif observed!=self.value:counts['cache_mismatches']+=1
        return self.value
    def fill_(self,value):
        self.tensor.fill_(value);self.value=int(value);counts['cache_writes']+=1;return self
original_attn=causal_model.CausalWanSelfAttention.forward
@functools.wraps(original_attn)
def checked_attn(self,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache=None,current_start=0,cache_start=None,sink_recache_after_switch=False):
    if kv_cache is not None:
        for key in ['global_end_index','local_end_index']:
            if not isinstance(kv_cache[key],CacheScalar):kv_cache[key]=CacheScalar(kv_cache[key])
    return original_attn(self,x,seq_lens,grid_sizes,freqs,block_mask,kv_cache,current_start,cache_start,sink_recache_after_switch)
causal_model.CausalWanSelfAttention.forward=checked_attn
original_attention=causal_model.attention
@functools.wraps(original_attention)
def checked_attention(q,k,v,*args,**kwargs):
    assert k.shape==v.shape;attention_shapes[(q.shape[1],k.shape[1])]+=1
    return original_attention(q,k,v,*args,**kwargs)
causal_model.attention=checked_attention
original_text=WanTextEncoderWrapper.forward
@functools.wraps(original_text)
def checked_text(self,text_prompts):
    ids,mask=self.tokenizer(text_prompts,return_mask=True,add_special_tokens=True)
    lengths=mask.gt(0).sum(dim=1).long().tolist()
    ids=ids.to(self.device);mask=mask.to(self.device);seq_lens=mask.gt(0).sum(dim=1).long()
    context=self.text_encoder(ids,mask)
    for u,v,length in zip(context,seq_lens,lengths):
        observed=v.__index__();counts['text_reads']+=1
        if native:assert observed==length,(observed,length)
        elif observed!=length:counts['text_mismatches']+=1
        u[length:]=0.0
    if self.output_device!=self.device:context=context.to(self.output_device)
    return {'prompt_embeds':context}
WanTextEncoderWrapper.forward=checked_text

# This workload uses one positive-weight prompt per chunk and no smooth transition.
# Keep all public GPU reductions/comparisons/scalar reads, then replay their branches.
from scope.core.pipelines import blending
@functools.wraps(blending.normalize_weights)
def checked_weights(weights,dtype,device):
    assert weights==[100], weights
    wt=torch.tensor(weights,dtype=dtype,device=device);total=wt.sum()
    positive=bool(total>0);counts['weight_branch_reads']+=1
    if native:assert positive
    elif not positive:counts['weight_branch_mismatches']+=1
    return wt/total
blending.normalize_weights=checked_weights
@functools.wraps(blending.blend_embeddings)
def checked_blend(embeddings,weights,method,dtype,device):
    assert len(embeddings)==1 and weights==[100], (len(embeddings),weights)
    normalized=blending.normalize_weights(weights,dtype,device)
    target_norm=sum(embed.norm()*weight for embed,weight in zip(embeddings,normalized,strict=False))
    combined=torch.zeros_like(embeddings[0])
    for embed,weight in zip(embeddings,normalized,strict=False):combined+=weight*embed
    current_norm=combined.norm();positive=bool(current_norm>blending.EPSILON)
    counts['embedding_norm_branch_reads']+=1
    if native:assert positive, 'Native prompt embedding must be nonzero for this calibrated workload'
    elif not positive:counts['embedding_norm_branch_mismatches']+=1
    return combined*(target_norm/current_norm)
blending.blend_embeddings=checked_blend
report['adaptations'].append('Single positive-weight prompt normalization and nonzero embedding-norm branches replayed; original GPU arithmetic, comparison, and scalar reads retained; native asserts both branches.')
original_model=WanDiffusionWrapper.forward
@functools.wraps(original_model)
def checked_model(self,*args,**kwargs):
    video=kwargs.get('noisy_image_or_video',args[0] if args else None)
    model_calls[(video.shape[1],bool(kwargs.get('is_recache',False)),bool(kwargs.get('update_cache',False)))]+=1
    return original_model(self,*args,**kwargs)
WanDiffusionWrapper.forward=checked_model
models=a.output/'models';models.mkdir();(models/'Wan2.1-T2V-1.3B').symlink_to(a.base,target_is_directory=True)
config=OmegaConf.create(dict(model_dir=str(models),generator_path=str(a.checkpoint/'models/longlive_base.pt'),
                            lora_path=str(a.checkpoint/'models/lora.pt'),text_encoder_path=str(a.base/'models_t5_umt5-xxl-enc-bf16.pth'),
                            tokenizer_path=str(a.base/'google/umt5-xxl'),height=a.height,width=a.width,base_seed=42))
start=time.perf_counter();print('SCOPE_LOADING',flush=True)
pipe=LongLivePipeline(config,device=torch.device('cuda'),dtype=torch.bfloat16)
torch.cuda.synchronize();report['load_wall_seconds']=time.perf_counter()-start
report['config']=OmegaConf.to_container(config,resolve=True)
report['lora_layers']=sum(hasattr(m,'lora_A') for m in pipe.components.generator.modules())
assert report['lora_layers']>0
save();print('SCOPE_LOADED',report['load_wall_seconds'],flush=True)
def generate():
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    outputs=[];chunk_audit=[]
    for i in range(a.chunks):
        which=int(a.switch_at>=0 and i>=a.switch_at)
        start_frame=0 if i==0 else pipe.state.get('current_start_frame')
        output=pipe(prompts=[{'text':prompts[which],'weight':100}],init_cache=(i==0))['video']
        outputs.append(output.detach().cpu())
        chunk_audit.append(dict(chunk=i,prompt=which,start_frame=start_frame,end_frame=pipe.state.get('current_start_frame'),shape=list(output.shape)))
        print('SCOPE_CHUNK',i,list(output.shape),flush=True)
    video=torch.concat(outputs).contiguous();torch.cuda.synchronize()
    return video,chunk_audit

def install_marks():
    import perfmark
    def wrap(owner,method,name,features):
        original=getattr(owner,method);sig=inspect.signature(original)
        @functools.wraps(original)
        def marked(*args,**kwargs):
            b=sig.bind(*args,**kwargs);b.apply_defaults()
            with perfmark.region(name,**features(b.arguments)):return original(*args,**kwargs)
        setattr(owner,method,marked)
    wrap(LongLivePipeline,'_generate','scope_chunk',lambda b:{'start_frame':pipe.state.get('current_start_frame',0)})
    wrap(WanTextEncoderWrapper,'forward','scope_text',lambda b:{'chars':sum(map(len,b['text_prompts']))})
    wrap(WanDiffusionWrapper,'forward','scope_model',lambda b:{'frames':b['noisy_image_or_video'].shape[1],'start':int(b.get('current_start',0) or 0)})
    wrap(causal_model.CausalWanAttentionBlock,'forward','scope_block',lambda b:{'tokens':b['x'].shape[1],'start':int(b.get('current_start',0))})
    wrap(causal_model.CausalWanSelfAttention,'forward','scope_self_attn',lambda b:{'tokens':b['x'].shape[1],'start':int(b.get('current_start',0))})
    # Modular block entry costs include state/property processing outside model calls.
    from scope.core.pipelines.longlive.modular_blocks import ALL_BLOCKS
    for name,cls in ALL_BLOCKS.items():
        wrap(cls,'__call__','scope_'+name[:23],lambda b:{'start_frame':b['state'].get('current_start_frame',0),'changed':int(bool(b['state'].get('conditioning_embeds_updated',False)))})
    from peft.tuners.lora.layer import Linear
    wrap(Linear,'forward','scope_lora',lambda b:{'elements':b['x'].numel(),'out_features':b['self'].out_features})
with torch.inference_mode():
    for i in range(a.warmup):
        start=time.perf_counter();video,chunks=generate();report['warmup_wall_seconds'].append(time.perf_counter()-start);del video
        save();print('SCOPE_WARMUP',i,report['warmup_wall_seconds'][-1],flush=True)
    if a.role=='drperf':install_marks()
    runtime=ctypes.CDLL(None)
    if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
    elif a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
    elif a.role in ['partial_sync','gpu_only_partial_sync']:
        (runtime.gxvm_adopt_now if a.role=='partial_sync' else runtime.gxvm_start)()
        runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.repetitions):
        if a.role in ['partial_sync','gpu_only_partial_sync']:runtime.gxvm_timeline_mark(i+1,0)
        start=time.perf_counter();video,chunks=generate();report['generation_wall_seconds'].append(time.perf_counter()-start)
        if a.role in ['partial_sync','gpu_only_partial_sync']:runtime.gxvm_timeline_mark(i+1,1)
        save();print('SCOPE_GENERATED',i,report['generation_wall_seconds'][-1],flush=True)
    if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
    elif a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
report.update(output_shape=list(video.shape),chunk_audit=chunks,control_audit=dict(counts),
              attention_shapes=[dict(query_tokens=q,key_tokens=k,calls=n) for (q,k),n in sorted(attention_shapes.items())],
              model_calls=[dict(frames=f,recache=r,update_cache=u,calls=n) for (f,r,u),n in sorted(model_calls.items())],
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
if native:
    report['finite']=bool(torch.isfinite(video).all());assert report['finite']
    report['output_float32_sha256']=hashlib.sha256(video.float().numpy().tobytes()).hexdigest()
save();print('SCOPE_PASS',flush=True)
