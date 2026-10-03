"""Run vLLM-Omni's public diffusion engine, including its multiprocess worker."""
import argparse, asyncio, ctypes, hashlib, json, os, resource, time
from pathlib import Path
import numpy as np
import torch
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler import WanEulerScheduler

# GX/DR currently keeps an environment pointer invalidated by setproctitle.
# Skip only the optional cosmetic worker title, before measurement, in timing
# runs. Keep the original late attachment and all model/IPC code unchanged.
import setproctitle
original_setproctitle=setproctitle.setproctitle
def gx_safe_process_title(*args,**kwargs):
    if os.environ.get('OMNI_GX_ROLE')=='partial_sync':
        print('OMNI_GX_SKIP_COSMETIC_PROCESS_TITLE',os.getpid(),flush=True)
        return None
    return original_setproctitle(*args,**kwargs)
setproctitle.setproctitle=gx_safe_process_title

# The ordinary Euler loop first passes schedule entry zero. GPU nonzero has a
# data-dependent output size and cannot run safely without GPU arithmetic.
# Native runs retain/verify the original lookup. GX retains its comparison but
# substitutes the known index, omitting nonzero and scalar-read work. Audits must
# report this difference; it is not an optimization of the framework.
old_index=WanEulerScheduler.index_for_timestep
control_checks=[]
def index_for_timestep(self,timestep):
    physical=os.environ.get('OMNI_GX_ROLE') in ('native','gpu-profile')
    if physical:
        observed=old_index(self,timestep)
        assert observed==0,('Unexpected noninitial Euler timestep',observed)
    else:
        comparison=self.timesteps == timestep
        assert comparison.shape == self.timesteps.shape
    # Preserve the original lookup's wait for preceding current-stream work.
    # Native already waited inside nonzero/item; this extra sync is then empty.
    torch.cuda.current_stream().synchronize()
    control_checks.append({'expected_initial_index':0,'native_checked':physical,'nonzero_and_index_read_omitted':not physical})
    return 0
WanEulerScheduler.index_for_timestep=index_for_timestep

# Use the selected Euler solver for the engine's built-in startup warmup too.
# Its default UniPC warmup otherwise requires a separate data-dependent adapter.
old_dummy_request=DiffusionEngine.add_req_and_wait_for_response
def add_req_and_wait_for_response(self,request):
    if request.is_dummy_run():
        request.sampling_params.extra_args.update(sample_solver='euler',flow_shift=3.0)
    return old_dummy_request(self,request)
DiffusionEngine.add_req_and_wait_for_response=add_req_and_wait_for_response

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
old_execute=DiffusionWorker.execute_model
worker_iteration=0
attached=False
def execute_model(self,*args,**kwargs):
    global worker_iteration,attached
    output=Path(os.environ['OMNI_GX_OUTPUT']);role=os.environ['OMNI_GX_ROLE']
    if not (output/'capture.ready').exists():return old_execute(self,*args,**kwargs)
    worker_iteration+=1;rt=ctypes.CDLL(None)
    if not attached:
        if role=='partial_sync':rt.gxvm_adopt_now()
        if role=='cpu-profile':rt.gxvm_ipc_profile_start()
        attached=True
    if role=='partial_sync':
        rt.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int];rt.gxvm_timeline_mark(worker_iteration,0)
    start=time.perf_counter();result=old_execute(self,*args,**kwargs);torch.cuda.synchronize();seconds=time.perf_counter()-start
    if role=='partial_sync':rt.gxvm_timeline_mark(worker_iteration,1)
    if role=='cpu-profile' and worker_iteration==int(os.environ['OMNI_GX_REPETITIONS']):assert rt.gxvm_ipc_profile_stop()==0
    r={'rank':self.rank,'pid':os.getpid(),'iteration':worker_iteration,'worker_clock_seconds':seconds,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_gpu_bytes':torch.cuda.max_memory_reserved(),'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'control_checks':list(control_checks)}
    save(output/f'worker-{os.getpid()}-{worker_iteration}.json',r);print('OMNI_WORKER_RESULT',json.dumps(r),flush=True)
    return result
DiffusionWorker.execute_model=execute_model

# DR-owned spawned-process exit did not run GX's C++ trace exporter. Explicitly
# drain/export during worker shutdown, after all client/worker request markers.
old_shutdown=DiffusionWorker.shutdown
def shutdown(self):
    result=old_shutdown(self)
    if os.environ.get('OMNI_GX_ROLE')=='partial_sync':
        os.chdir(os.environ['OMNI_GX_OUTPUT'])
        rt=ctypes.CDLL(None)
        rt.cleanup_all_stream_processors.argtypes=[]
        rt.cleanup_all_stream_processors.restype=None
        rt.cleanup_all_stream_processors()
        print('OMNI_GX_TRACE_FLUSHED',os.getpid(),flush=True)
    return result
DiffusionWorker.shutdown=shutdown


async def generate_all(a,report):
    rt=ctypes.CDLL(None);start=time.perf_counter()
    config=OmniDiffusionConfig.from_kwargs(model=str(a.model),model_class_name='WanPipeline',dtype=torch.bfloat16,num_gpus=1,enforce_eager=True,enable_cpu_offload=True,enable_layerwise_offload=False,cache_backend='none',flow_shift=3.0,diffusion_attention_backend='FLASH_ATTN')
    engine=DiffusionEngine(config)
    report.update(phase='loaded',load_seconds=time.perf_counter()-start);save(a.output/'report.json',report)
    async def generate(request_id):
        params=OmniDiffusionSamplingParams(height=a.height,width=a.width,num_frames=a.frames,num_inference_steps=a.steps,guidance_scale=5.0,seed=42,num_outputs_per_prompt=1,extra_args={'sample_solver':'euler','flow_shift':3.0})
        req=OmniDiffusionRequest(prompts=[{'prompt':'A cat walks on the grass, realistic style','negative_prompt':''}],sampling_params=params,request_id=request_id)
        result=await engine.step(req);assert len(result)==1;return result[0]
    try:
        for i in range(a.warmup):await generate(f'warmup-{i}')
        print('OMNI_WARMUP_DONE',flush=True);(a.output/'capture.ready').touch()
        if a.role=='partial_sync':rt.gxvm_adopt_now();rt.gxvm_timeline_mark.argtypes=[ctypes.c_uint,ctypes.c_int]
        if a.role=='cpu-profile':rt.gxvm_ipc_profile_start()
        if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_START_FILE']).touch(exist_ok=False)
        report['iterations']=[]
        for i in range(a.repetitions):
            if a.role=='partial_sync':rt.gxvm_timeline_mark(1001+i,0)
            start=time.perf_counter();result=await generate(f'measured-{i}');seconds=time.perf_counter()-start
            if a.role=='partial_sync':rt.gxvm_timeline_mark(1001+i,1)
            row={'iteration':i+1,'client_host_seconds':seconds,'metrics':result.metrics,'images_type':type(result.images).__name__}
            if a.role in ('native','gpu-profile'):
                value=np.asarray(result.images);assert np.isfinite(value).all();row.update(shape=list(value.shape),dtype=str(value.dtype),finite=True,output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),minimum=float(value.min()),maximum=float(value.max()))
            report['iterations'].append(row)
        if a.role=='gpu-profile':Path(os.environ['GX_PROFILE_STOP_FILE']).touch(exist_ok=False)
        if a.role=='cpu-profile':assert rt.gxvm_ipc_profile_stop()==0
        report.update(phase='completed',client_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(a.output/'report.json',report);print('OMNI_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='source_sha256'}),flush=True)
    finally:engine.close()

def main():
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=17);p.add_argument('--steps',type=int,default=8);p.add_argument('--height',type=int,default=256);p.add_argument('--width',type=int,default=448);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--role',choices=('emu','cpu-profile','partial_sync','native','gpu-profile'),default='emu');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);os.chdir(a.output);os.environ.update(OMNI_GX_OUTPUT=str(a.output),OMNI_GX_ROLE=a.role,OMNI_GX_REPETITIONS=str(a.repetitions));torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert ('gx_cuda.so' not in Path('/proc/self/maps').read_text())==(a.role in ('native','gpu-profile'))
    import vllm_omni
    from importlib.metadata import version
    root=Path(vllm_omni.__file__).parent
    report={'async_upload':os.environ.get('OMNI_ASYNC_UPLOAD','0')=='1','direct_pinned_d2h':os.environ.get('OMNI_DIRECT_PINNED_D2H','0')=='1','gx_skips_cosmetic_process_title':a.role=='partial_sync','phase':'loading','config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in root.rglob('*.py')},'versions':{n:version(n) for n in ('torch','vllm','transformers','diffusers','cache-dit','accelerate')},'scope':'Public DiffusionEngine.step through multiprocess worker and CPU numpy video postprocessing; component CPU offload retained, eager FA2, no file encoding.'}
    save(a.output/'report.json',report);asyncio.run(generate_all(a,report))
if __name__=='__main__':main()
