"""Public Qwen3-TTS request; scoped replay of native GPU-to-host control reads."""
import argparse,ctypes,hashlib,json,os,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from importlib.metadata import version
import transformers.generation.logits_process as logits_process

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--role',choices=['emu','cpu-profile','gpu-profile','partial_sync','native'],required=True)
p.add_argument('--controls',type=Path);p.add_argument('--warmup',type=int,default=1)
p.add_argument('--repetitions',type=int,default=1)
p.add_argument('--text',default='Hello. This is a test of speech generation.')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output)
physical=a.role in ('native','gpu-profile')
assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==physical
assert physical or a.controls is not None
torch.set_num_threads(1);torch.set_num_interop_threads(1)
spec={'text':a.text,'language':'English','speaker':'Ryan','non_streaming_mode':True,
      'max_new_tokens':256,'do_sample':True,'subtalker_dosample':True}
reference=json.loads(a.controls.read_text()) if a.controls else None
if reference:assert reference['request']==spec

# This processor constructs arange(vocab_size) and a unique Python suppression
# list. GX cannot compute torch.unique's hidden GPU-to-host output count inside
# isin. Only this call uses assume_unique under GX; native keeps the original.
# It omits uniqueness work, which must be counted in the native/GX launch audit.
isin_checks={'gx_unique_adapter_calls':0,'native_original_calls':0}
original_suppress_init=logits_process.SuppressTokensLogitsProcessor.__init__
def suppress_init(self,suppress_tokens,*args,**kwargs):
    self._gx_unique_tokens=isinstance(suppress_tokens,(list,tuple)) and len(set(suppress_tokens))==len(suppress_tokens)
    return original_suppress_init(self,suppress_tokens,*args,**kwargs)
logits_process.SuppressTokensLogitsProcessor.__init__=suppress_init
original_isin=logits_process.isin_mps_friendly
def known_unique_isin(elements,test_elements):
    frame=sys._getframe(1);processor=frame.f_locals.get('self')
    if isinstance(processor,logits_process.SuppressTokensLogitsProcessor):
        assert processor._gx_unique_tokens and test_elements is processor.suppress_tokens
        assert elements.ndim==1 and elements.shape[0]==frame.f_locals['scores'].shape[-1]
        if not physical:
            isin_checks['gx_unique_adapter_calls']+=1
            return torch.isin(elements,test_elements,assume_unique=True)
        isin_checks['native_original_calls']+=1
    return original_isin(elements,test_elements)
logits_process.isin_mps_friendly=known_unique_isin

class ControlTape:
    """Retain original scalar transfer/wait; validate shape, dtype and call site."""
    def __init__(self):self.active=False;self.records=[];self.selected=None;self.cursor=0
    def start(self,index):
        self.events=[];self.cursor=0
        self.selected=reference['requests'][min(index,len(reference['requests'])-1)]['events'] if reference else None
        self.active=True
    def finish(self):
        self.active=False
        if self.selected is not None:assert self.cursor==len(self.selected),(self.cursor,len(self.selected))
        self.records.append({'event_count':self.cursor,'events':self.events})
tape=ControlTape()
def wrap(name,original):
    def read(tensor,*args,**kwargs):
        if not tape.active or tensor.device.type!='cuda':return original(tensor,*args,**kwargs)
        assert tensor.numel()<=4096,('Unexpected large control read',name,tensor.shape)
        frame=sys._getframe(1)
        tag=[name,Path(frame.f_code.co_filename).name,frame.f_code.co_name,list(tensor.shape),str(tensor.dtype)]
        observed=original(tensor,*args,**kwargs)
        if tape.selected is None:
            assert physical
            tape.events.append({'tag':tag,'line':frame.f_lineno,'value':observed})
            result=observed
        else:
            assert tape.cursor<len(tape.selected),('Extra GPU control read',tag,tape.cursor)
            expected=tape.selected[tape.cursor]
            assert expected['tag']==tag,('Control-path mismatch',tape.cursor,tag,expected['tag'])
            if physical:assert observed==expected['value'],('Native control-value mismatch',tag,observed,expected['value'])
            result=expected['value']
        tape.cursor+=1
        return result
    return read
for name in ('item','__bool__','__int__','__float__','__index__','tolist'):
    setattr(torch.Tensor,name,wrap(name,getattr(torch.Tensor,name)))

# Single-length batch has no trailing-text padding. The original boolean
# index_put internally reads a GPU nonzero count, bypassing Python scalar
# hooks. GX uses explicit empty indices; native retains the original operation.
empty_padding_checks={'gx_explicit_empty_indices':0,'native_original_assignments':0}
original_setitem=torch.Tensor.__setitem__
def known_empty_padding(tensor,key,value):
    if tape.active and tensor.device.type=='cuda' and isinstance(key,torch.Tensor) and key.dtype==torch.bool:
        frame=sys._getframe(1);local=frame.f_locals
        if frame.f_code.co_name=='generate' and Path(frame.f_code.co_filename).name=='modeling_qwen3_tts.py' and local.get('padded_hiddens') is tensor and local.get('padding_mask') is key:
            lengths=local['trailing_text_original_lengths']
            assert lengths and len(set(lengths))==1 and lengths[0]==tensor.shape[1]
            assert tuple(key.shape)==tuple(tensor.shape[:2])
            if not physical:
                empty=torch.empty(0,dtype=torch.long,device=tensor.device)
                empty_padding_checks['gx_explicit_empty_indices']+=1
                return original_setitem(tensor,(empty,empty),value)
            empty_padding_checks['native_original_assignments']+=1
    return original_setitem(tensor,key,value)
torch.Tensor.__setitem__=known_empty_padding


# The candidate's shape-only fast path removes a GPU scalar read present in
# the calibration tape. Consume exactly that known-false event, only where
# the original FlashAttention branch would have evaluated it.
import transformers.modeling_flash_attention_utils as fa_utils
packed_enabled=os.environ.get('QWEN_PACKED_SINGLE_TOKEN')=='1'
removed_packed_checks=0
if packed_enabled:
    original_packed=fa_utils._is_packed_sequence
    def checked_packed(position_ids,batch_size):
        global removed_packed_checks
        result=original_packed(position_ids,batch_size)
        if tape.active and result is False and position_ids is not None and batch_size==1 and position_ids.shape==(1,1) and position_ids.dtype in (torch.int32,torch.int64):
            frame=sys._getframe(1);local=frame.f_locals
            assert frame.f_code.co_name=='_flash_attention_forward'
            assert local['attention_mask'] is None
            assert not all(local.get(k) is not None for k in ('cu_seq_lens_q','cu_seq_lens_k','max_length_q','max_length_k'))
            assert tape.selected is not None and tape.cursor<len(tape.selected)
            expected=tape.selected[tape.cursor]
            assert expected['tag']==['__bool__','modeling_flash_attention_utils.py','_flash_attention_forward',[],'torch.bool']
            assert expected['value'] is False
            tape.cursor+=1;removed_packed_checks+=1
        return result
    fa_utils._is_packed_sequence=checked_packed

import qwen_tts
root=Path(qwen_tts.__file__).parent
report={'phase':'loading','packed_single_token':packed_enabled,'flash_attention_source_sha256':hashlib.sha256(Path(fa_utils.__file__).read_bytes()).hexdigest(),'role':a.role,'request':spec,'seed':42,
 'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()for f in root.rglob('*.py')},
 'versions':{n:version(n)for n in ('torch','torchaudio','transformers','tokenizers','accelerate','numpy','librosa','onnxruntime')},
 'control_reference_sha256':hashlib.sha256(a.controls.read_bytes()).hexdigest()if a.controls else None,
 'scope':'Original generate_custom_voice, including text tokenization, autoregressive talker and code predictor, speech decoder and CPU waveform return; no file encoding. BF16 resident parameters and FA2.'}
def save():(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
save();start=time.perf_counter()
model=Qwen3TTSModel.from_pretrained(str(a.model),device_map='cuda:0',dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',local_files_only=True)
report.update(phase='loaded',load_host_seconds=time.perf_counter()-start,
              parameters=sum(x.numel()for x in model.model.parameters()))
save();print('QWEN_TTS_LOADED',report['parameters'],flush=True)

def generate(index):
    tape.start(index)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    output=model.generate_custom_voice(**spec)
    torch.cuda.synchronize()
    return output
def checked_output(output):
    wavs,rate=output;assert len(wavs)==1
    w=np.asarray(wavs[0]);assert w.ndim==1 and len(w)>0
    row={'sample_rate':rate,'samples':len(w),'dtype':str(w.dtype)}
    if physical:
        assert np.isfinite(w).all()
        row.update(finite=True,output_sha256=hashlib.sha256(w.tobytes()).hexdigest(),minimum=float(w.min()),maximum=float(w.max()))
    return row
report['warmups']=[]
for i in range(a.warmup):
    output=generate(i);tape.finish();report['warmups'].append(checked_output(output))
print('QWEN_TTS_WARMUP_DONE',flush=True)
rt=ctypes.CDLL(None)
if a.role=='partial_sync':rt.gxvm_adopt_now();rt.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
if a.role=='cpu-profile':rt.gxvm_ipc_profile_start()
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
report['iterations']=[]
for i in range(a.repetitions):
    if a.role=='partial_sync':rt.gxvm_timeline_mark(i+1,0)
    start=time.perf_counter();output=generate(a.warmup+i);seconds=time.perf_counter()-start
    if a.role=='partial_sync':rt.gxvm_timeline_mark(i+1,1)
    tape.finish();row=checked_output(output);row.update(iteration=i+1,application_clock_seconds=seconds,control_reads=tape.records[-1]['event_count'])
    report['iterations'].append(row)
if a.role=='cpu-profile':assert rt.gxvm_ipc_profile_stop()==0
if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
if reference is None:
    (a.output/'controls.json').write_text(json.dumps({'request':spec,'requests':tape.records},indent=2)+'\n')
report.update(phase='completed',removed_packed_checks=removed_packed_checks,control_counts=[r['event_count']for r in tape.records],isin_adapter=isin_checks,empty_padding_adapter=empty_padding_checks,
              peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
              peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
save();print('QWEN_TTS_COMPLETE',json.dumps({k:v for k,v in report.items()if k!='source_sha256'}),flush=True)
