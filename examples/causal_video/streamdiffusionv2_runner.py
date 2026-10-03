"""Public StreamDiffusionV2 staged API; native control tape for GX skipped arithmetic."""
import argparse,ctypes,functools,gzip,hashlib,inspect,json,logging,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['tree','base','checkpoint','input','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--role',choices=['emu','native','gpu-profile','cpu-profile','drperf','partial_sync','gpu_only_partial_sync'],default='emu')
p.add_argument('--tape',type=Path);p.add_argument('--frames',type=int,nargs='+',default=[17,33,65])
p.add_argument('--height',type=int,default=480);p.add_argument('--width',type=int,default=832)
p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=1)
a=p.parse_args()
for key in ['tree','base','checkpoint','input','output']:setattr(a,key,getattr(a,key).resolve())
a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
(a.output/'wan_models').mkdir();(a.output/'wan_models/Wan2.1-T2V-1.3B').symlink_to(a.base,target_is_directory=True)
os.environ['STREAMDIFFUSIONV2_ROOT']=str(a.output);sys.path.insert(0,str(a.tree))
import numpy as np
import torch,diffusers,transformers
torch.set_num_threads(1);torch.set_grad_enabled(False)
native=a.role in ['native','gpu-profile'];assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==native
report=dict(role=a.role,frames=a.frames,height=a.height,width=a.width,steps=2,mode='single',seed=0,
            prompt='A dog walks on the grass, realistic',torch=torch.__version__,diffusers=diffusers.__version__,transformers=transformers.__version__,
            warmup=a.warmup,repetitions=a.repetitions,generation_wall_seconds=[],warmup_wall_seconds=[],outputs=[],
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
            source_sha256={str(f.relative_to(a.tree)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['models','streamv2v','streamdiffusionv2'] for f in sorted((a.tree/folder).rglob('*.py'))},
            adaptations=['GX uses the CPU cache-position metadata patch; native validation can restore upstream metadata. Actual source hashes are recorded.',
                         'Native-recorded GPU scalar decisions replayed for the exact calibrated workload. Original scalar reads remain; no arbitrary-input GX control claim.',
                         'Meta/mmap BF16 T5 loading outside inference, equivalent stored inference tensors; generator strict state loading.',
                         'Public staged README API, CPU float32 input video and CPU outputs; input file decoding and output file encoding outside measurement.'])
def save(): (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save()
class ControlTape:
    def __init__(self):
        self.data={} if native else json.loads(gzip.decompress(a.tape.read_bytes()) if a.tape.suffix=='.gz' else a.tape.read_bytes())['phases'];self.phase='setup';self.index=0;self.busy=False;self.mismatches=0
        self.counts={};self.data.setdefault('setup',[])
    def change(self,phase):
        if not native:assert self.index==len(self.data[self.phase]),(self.phase,self.index,len(self.data[self.phase]))
        self.counts[self.phase]=self.index;self.phase=phase;self.index=0
        if native:self.data[phase]=[]
        else:assert phase in self.data,phase
    def wrap(self,name,original):
        @functools.wraps(original)
        def checked(tensor,*args,**kwargs):
            if self.busy or tensor.device.type!='cuda':return original(tensor,*args,**kwargs)
            assert tensor.numel()<=1024,(name,tensor.shape)
            self.busy=True
            try:
                observed=original(tensor,*args,**kwargs)
                caller=sys._getframe(1);site=Path(caller.f_code.co_filename).name+':'+caller.f_code.co_name
                key=[name,list(tensor.shape),str(tensor.dtype),site]
                if native:self.data[self.phase].append(dict(key=key,value=observed));value=observed
                else:
                    records=self.data[self.phase];assert self.index<len(records),(self.phase,self.index,key)
                    record=records[self.index];assert record['key']==key,(self.phase,self.index,record['key'],key)
                    value=record['value']
                    if observed!=value:self.mismatches+=1
                self.index+=1;return value
            finally:self.busy=False
        return checked
    def finish(self):
        if not native:assert self.index==len(self.data[self.phase]),(self.phase,self.index,len(self.data[self.phase]))
        self.counts[self.phase]=self.index
        report['control_tape']=dict(counts=self.counts,mismatches=self.mismatches)
        if native:(a.output/'control-tape.json').write_text(json.dumps(dict(phases=self.data),separators=(',',':'))+'\n')
tape=ControlTape()
for name in ['item','__bool__','__int__','__index__','tolist']:
    setattr(torch.Tensor,name,tape.wrap(name,getattr(torch.Tensor,name)))
# Memory-mapped setup loads only; no loader mutation in measured inference.
original_load=torch.load
def mmap_load(file,*args,**kwargs):
    if isinstance(file,(str,Path)) and str(file).endswith(('.pt','.pth')):
        kwargs['mmap']=True;kwargs['weights_only']=True
    return original_load(file,*args,**kwargs)
torch.load=mmap_load
from models.wan import wan_wrapper as wrapper
from models.wan.wan_base.modules.t5 import umt5_xxl
from models.wan.wan_base.modules.tokenizers import HuggingfaceTokenizer
def efficient_text_init(self,model_type='T2V-1.3B'):
    torch.nn.Module.__init__(self);assert model_type=='T2V-1.3B'
    self.text_encoder=umt5_xxl(encoder_only=True,return_tokenizer=False,dtype=torch.bfloat16,device='meta').eval().requires_grad_(False)
    state=torch.load(a.base/'models_t5_umt5-xxl-enc-bf16.pth',map_location='cpu',weights_only=True,mmap=True)
    self.text_encoder.load_state_dict(state,strict=True,assign=True)
    self.tokenizer=HuggingfaceTokenizer(name=str(a.base/'google/umt5-xxl'),seq_len=512,clean='whitespace')
wrapper.WanTextEncoder.__init__=efficient_text_init
from streamdiffusionv2 import StreamDiffusionV2Pipeline,load_video
from streamdiffusionv2 import pipeline as api
from streamv2v import inference,inference_common,inference_wo_batch
original_state_loader=inference_common.load_generator_state_dict
# Public manager retries strict=False. Reject missing tensors instead.
def strict_load_model(self,folder):
    _,state=original_state_loader(folder);self.pipeline.generator.load_state_dict(state,strict=True)
inference.SingleGPUInferencePipeline.load_model=strict_load_model
inference_wo_batch.SingleGPUInferencePipeline.load_model=strict_load_model
print('SDV2_LOADING',flush=True);start=time.perf_counter()
stream=StreamDiffusionV2Pipeline(str(a.checkpoint),height=a.height,width=a.width,step=2,seed=0,mode='single',device='cuda')
stream.pipeline_manager.logger.setLevel(logging.WARNING)
video=load_video(str(a.input),height=a.height,width=a.width)
assert video.device.type=='cpu' and video.dtype==torch.float32 and video.shape[1]>=max(a.frames),(video.device,video.dtype,video.shape)
inputs={frames:video[:,:frames].contiguous() for frames in a.frames};del video
torch.cuda.synchronize();report['load_wall_seconds']=time.perf_counter()-start
report['config']=__import__('omegaconf').OmegaConf.to_container(stream.config,resolve=True)
save();print('SDV2_LOADED',report['load_wall_seconds'],flush=True)
def generate(frames):
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);stream.prepare(report['prompt'])
    data=inputs[frames];noise_scale=stream.noise_scale;outputs=[];chunks=stream.chunk_video(data)
    for i,chunk in enumerate(chunks):
        encoded=stream.encode_chunk(data,chunk,previous_noise_scale=noise_scale,initial_noise_scale=stream.noise_scale)
        noise_scale=encoded.noise_scale;denoised=stream.denoise_chunk(encoded)
        if denoised is not None:outputs.append(stream.decode_chunk(denoised))
    output=np.concatenate(outputs,axis=0);torch.cuda.synchronize()
    return output,len(chunks)
def install_marks():
    import perfmark
    def wrap(owner,method,name,features):
        orig=getattr(owner,method);sig=inspect.signature(orig)
        @functools.wraps(orig)
        def marked(*args,**kwargs):
            b=sig.bind(*args,**kwargs);b.apply_defaults()
            with perfmark.region(name,**features(b.arguments)):return orig(*args,**kwargs)
        setattr(owner,method,marked)
    wrap(api,'_normalize_video_tensor','sdv2_normalize',lambda b:{'contiguous_elements':b['video'].numel() if b['video'].is_contiguous() else 0,'strided_elements':0 if b['video'].is_contiguous() else b['video'].numel()})
    wrap(StreamDiffusionV2Pipeline,'encode_chunk','sdv2_encode_chunk',lambda b:{'input_frames':b['input_video'].shape[1],'start':b['chunk'].start_idx})
    wrap(StreamDiffusionV2Pipeline,'denoise_chunk','sdv2_denoise',lambda b:{'start':b['chunk'].current_start})
    wrap(StreamDiffusionV2Pipeline,'decode_chunk','sdv2_decode',lambda b:{'frames':b['chunk'].denoised_pred.shape[1]})
    wrap(wrapper.WanTextEncoder,'forward','sdv2_text',lambda b:{'chars':sum(map(len,b['text_prompts']))})
    from models.wan import causal_model
    wrap(causal_model.CausalWanSelfAttention,'forward','sdv2_self_attn',lambda b:{'tokens':b['x'].shape[1],'batch':b['x'].shape[0]})
with torch.inference_mode():
    for repetition in range(a.warmup):
        for frames in a.frames:
            tape.change(f'warmup-{repetition}-{frames}');start=time.perf_counter();output,chunks=generate(frames)
            report['warmup_wall_seconds'].append(dict(frames=frames,seconds=time.perf_counter()-start));del output
            save();print('SDV2_WARMUP',frames,report['warmup_wall_seconds'][-1]['seconds'],flush=True)
    if a.role=='drperf':install_marks()
    runtime=ctypes.CDLL(None)
    if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
    elif a.role=='cpu-profile':runtime.gxvm_ipc_profile_start()
    elif a.role in ['partial_sync','gpu_only_partial_sync']:
        (runtime.gxvm_adopt_now if a.role=='partial_sync' else runtime.gxvm_start)()
        runtime.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
    torch.cuda.reset_peak_memory_stats();iteration=0
    for repetition in range(a.repetitions):
        for frames in a.frames:
            tape.change(f'measured-{repetition}-{frames}');iteration+=1
            if a.role in ['partial_sync','gpu_only_partial_sync']:runtime.gxvm_timeline_mark(iteration,0)
            start=time.perf_counter();output,chunks=generate(frames);seconds=time.perf_counter()-start
            if a.role in ['partial_sync','gpu_only_partial_sync']:runtime.gxvm_timeline_mark(iteration,1)
            report['generation_wall_seconds'].append(dict(frames=frames,seconds=seconds))
            result=dict(input_frames=frames,chunks=chunks,shape=list(output.shape))
            if native:
                result['finite']=bool(np.isfinite(output).all());assert result['finite']
                result['sha256']=hashlib.sha256(output.tobytes()).hexdigest()
            report['outputs'].append(result);save();print('SDV2_GENERATED',frames,seconds,flush=True)
    if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
    elif a.role=='cpu-profile':assert runtime.gxvm_ipc_profile_stop()==0
tape.finish();report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());save();print('SDV2_PASS',flush=True)
