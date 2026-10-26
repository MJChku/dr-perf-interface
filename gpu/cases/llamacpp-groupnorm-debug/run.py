#!/usr/bin/env python3
"""CPU-only public llama.cpp normalization reproduction; raw artifacts are temporary."""
import difflib,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
CASE=Path(__file__).resolve().parent
ROOT=CASE.parents[2]
CUDA=Path(os.environ.get('CUDA_HOME','/home/ubuntu/GX/toolchains/cuda13'))
METRICS=['global_read_bytes','global_write_bytes','global_unique_read_bytes','global_unique_write_bytes','shared_read_bytes','shared_write_bytes','local_read_bytes','local_write_bytes','fp32_flops','special_function_ops']
def extract(source,needle):
 start=source.index(needle);depth=0;opened=False
 for end in range(start,len(source)):
  if source[end]=='{':depth+=1;opened=True
  if source[end]=='}':depth-=1
  if opened and depth==0:return source[start:end+1]
 raise ValueError(needle)
def main():
 results=CASE/'results';results.mkdir(exist_ok=True)
 record={'physical_gpu_used':False,'source_revision':'623c16393c7d53a37df8bb69c6ea8ed6e5171f62','commands':[],'runs':{},'modules':{}}
 started=time.monotonic()
 def run(cmd):
  begin=time.monotonic();p=subprocess.run(list(map(str,cmd)),cwd=ROOT,capture_output=True,text=True,timeout=600)
  record['commands'].append({'argv':list(map(str,cmd)),'seconds':time.monotonic()-begin,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  if p.returncode:raise RuntimeError(str(cmd)+'\n'+p.stdout+p.stderr)
  return p
 provenance=json.loads((CASE/'provenance.json').read_text())
 for original,local in [('ggml/src/ggml-cuda/norm.cu','upstream_norm.cu'),('ggml/src/ggml-cuda/common.cuh','upstream_common.cuh')]:assert hashlib.sha256((CASE/local).read_bytes()).hexdigest()==provenance['sources'][original]
 kernel='rms_norm_back_f32' if 'rmsnorm' in CASE.name else 'group_norm_f32'
 assert extract((CASE/'upstream_norm.cu').read_text(),'template <int block_size>\nstatic __global__ void '+kernel+'(') in (CASE/'before.cu').read_text()
 common=(CASE/'upstream_common.cuh').read_text();helper=(CASE/'common_extracted.cuh').read_text()
 for needle in ['static __device__ __forceinline__ void ggml_cuda_pdl_sync()','template<int width = WARP_SIZE>\nstatic __device__ __forceinline__ float warp_reduce_sum(float x)','template <block_reduce_method reduce_method_t, const unsigned int block_size_template = 0, typename T>']:
  assert extract(common,needle) in helper
 policy=common[common.index('enum class block_reduce_method {'):common.index('template <typename T> struct block_reduce_policy<block_reduce_method::MAX')]
 assert policy in helper
 record['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CASE.iterdir() if p.suffix in {'.py','.cc','.cu','.cuh'}}
 record['tool_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'bin/drperf',ROOT/'gpu/build/libdrperf_gpu.so',ROOT/'gpu/vendor/gxptx/Gxptx/prelude.h']}
 run([CUDA/'bin/nvcc','--version'])
 try:
  with tempfile.TemporaryDirectory(prefix='drperf-llama-') as temp:
   work=Path(temp);driver=work/'driver'
   run(['g++','-std=c++17','-O2','-I'+str(ROOT/'gpu/include'),CASE/'driver.cc','-L'+str(ROOT/'gpu/build'),'-ldrperf_gpu','-Wl,-rpath,'+str(ROOT/'gpu/build'),'-o',driver])
   for variant in ['before','after']:
    if not (CASE/(variant+'.cu')).exists():continue
    print('Compile/translate',CASE.name,variant,flush=True);ptx=work/(variant+'.ptx');module=work/(variant+'.so')
    run([CUDA/'bin/nvcc','-std=c++17','-lineinfo','-ptx','-arch=sm_80',CASE/(variant+'.cu'),'-o',ptx])
    assembly=run([CUDA/'bin/ptxas','-arch=sm_80','-v',ptx,'-o',work/(variant+'.cubin')])
    run([ROOT/'bin/drperf','--gpu','translate',ptx,'-o',module])
    meta=json.loads((work/(variant+'.so.gpu.json')).read_text());assert not meta['instructionCoverage']['unsupportedSites']
    record['modules'][variant]={'ptx_sha256':hashlib.sha256(ptx.read_bytes()).hexdigest(),'ptxas':assembly.stderr,'manifest':{k:v for k,v in meta.items() if k not in ['instructionCoverage','kernels']}}
    for suite in ['train','heldout']:
     print('Emulate/check',variant,suite,flush=True);report=work/(variant+'-'+suite)
     p=run([ROOT/'bin/drperf','--gpu','--output',report,'--',driver,module,variant,suite]);checks=json.loads(p.stdout.splitlines()[0]);assert checks['passed']
     record['runs'][variant+'-'+suite]={'numerical_checks':checks,'stdout':p.stdout,'stderr':p.stderr}
     model=json.loads((report/'profile.gpu.drperf.json').read_text());assert model['validity']=='complete';regions=[]
     for r in model['regions']:
      for obs in r['observations']:
       v,m=obs['pcvs'],obs['metrics'];n,cached=v['elements'],v['cached_elements']
       if 'rmsnorm' in CASE.name:
        assert m['global_read_bytes']==16*n-(8*cached if variant=='after' else 0)
        assert m['global_write_bytes']==4*n
       else:
        assert m['global_read_bytes']==12*n-(8*cached if variant=='after' else 0)
        assert m['global_write_bytes']==8*n-(4*cached if variant=='after' else 0)
       assert m.get('local_read_bytes',0)==m.get('local_write_bytes',0)==0
      regions.append({'name':r['name'],'state_count':r['stateCount'],'pcvs':r['pcvs'],'fits':{m:r['fits'].get(m) for m in METRICS},'interfaces':{m:r['interfaces'].get(m) for m in METRICS},'states':[{'pcvs':s['pcvs'],'invocations':s['invocations'],'metrics':{m:v for m,v in s['metrics'].items() if m in METRICS}} for s in r['stateStatistics']], 'global_memory_sites':{s:{m:f[m] for m in ['global_read_bytes','global_write_bytes'] if m in f} for s,f in r['sites'].items() if any(m in f for m in ['global_read_bytes','global_write_bytes'])}})
     compact={'validity':model['validity'],'regions':regions}
     (results/(variant+'-'+suite+'.json')).write_text(json.dumps(compact,indent=2)+'\n')
     record['runs'][variant+'-'+suite]['resource_assertions']='passed'
     if suite=='train':(results/(variant+'-report.txt')).write_text((report/'report.txt').read_text())
   if (CASE/'after.cu').exists():(results/'before-to-after.patch').write_text(''.join(difflib.unified_diff((CASE/'before.cu').read_text().splitlines(True),(CASE/'after.cu').read_text().splitlines(True),fromfile='before.cu',tofile='after.cu')))
 finally:
  record['elapsed_seconds']=time.monotonic()-started;(results/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
 print('Evidence:',results,flush=True)
if __name__=='__main__':main()
