#!/usr/bin/env python3
"""CPU-only public-kernel reproduction. Generated code and raw captures are temporary."""
import ast,difflib,hashlib,json,os,subprocess,tempfile,time
from pathlib import Path
CASE=Path(__file__).resolve().parent
ROOT=CASE.parents[2]
CUDA=Path(os.environ.get('CUDA_HOME','/home/ubuntu/GX/toolchains/cuda13'))
METRICS=['global_read_bytes','global_write_bytes','global_unique_read_bytes','global_unique_write_bytes','shared_read_bytes','shared_write_bytes','local_read_bytes','local_write_bytes','fp32_flops','tensor_float_flops','tensor_flops','mma_flops','special_function_ops','atomic_ops','global_atomic_bytes','atomic_memory_operations']
def main():
 results=CASE/'results';results.mkdir(exist_ok=True)
 record={'physical_gpu_used':False,'source_revision':'5c94620c59ab9839a4ad4ccad309f3d91752d042','commands':[],'runs':{},'modules':{}}
 start=time.monotonic()
 def run(cmd,allowed=(0,)):
  begin=time.monotonic();p=subprocess.run(list(map(str,cmd)),cwd=ROOT,capture_output=True,text=True,timeout=600)
  record['commands'].append({'argv':list(map(str,cmd)),'seconds':time.monotonic()-begin,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  if p.returncode not in allowed:raise RuntimeError(str(cmd)+'\n'+p.stdout+p.stderr)
  return p
 tree=ast.parse((CASE/'upstream_src.py').read_text());src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and '__global__' in n.value.value)
 src=src[:src.index('torch::Tensor')];src='\n'.join(l for l in src.split('\n') if not l.startswith('#include <torch/') and not l.startswith('#include <ATen/'))
 assert src==(CASE/'before.cu').read_text()
 record['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CASE.iterdir() if p.suffix in {'.py','.cc','.cu'}}
 record['tool_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'bin/drperf',ROOT/'gpu/build/libdrperf_gpu.so',ROOT/'gpu/vendor/gxptx/Gxptx/prelude.h']}
 run([CUDA/'bin/nvcc','--version'])
 try:
  with tempfile.TemporaryDirectory(prefix='drperf-public-') as temp:
   work=Path(temp);driver=work/'driver'
   run(['g++','-std=c++17','-O2','-I'+str(ROOT/'gpu/include'),CASE/'driver.cc','-L'+str(ROOT/'gpu/build'),'-ldrperf_gpu','-Wl,-rpath,'+str(ROOT/'gpu/build'),'-o',driver])
   for variant in (['before','after'] if 'matmul' in CASE.name else ['compatible','after']):
    if not (CASE/(variant+'.cu')).exists():continue
    print('Compile/translate',CASE.name,variant,flush=True);ptx=work/(variant+'.ptx');module=work/(variant+'.so')
    flags=['-std=c++17','-lineinfo']
    if 'conv2d' in CASE.name:flags+=['--use_fast_math']
    run([CUDA/'bin/nvcc','-ptx','-arch=sm_80',*flags,CASE/(variant+'.cu'),'-o',ptx])
    assembly=run([CUDA/'bin/ptxas','-arch=sm_80','-v',ptx,'-o',work/(variant+'.cubin')])
    run([ROOT/'bin/drperf','--gpu','translate',ptx,'-o',module])
    meta=json.loads((work/(variant+'.so.gpu.json')).read_text());unsupported=meta['instructionCoverage']['unsupportedSites']
    record['modules'][variant]={'ptx_sha256':hashlib.sha256(ptx.read_bytes()).hexdigest(),'ptxas':assembly.stderr,'unsupportedSites':unsupported,'manifest':{k:v for k,v in meta.items() if k not in ['instructionCoverage','kernels']}}
    if unsupported:print('Static unsupported sites',unsupported[:2],flush=True)
    for suite in ['train','heldout']:
     print('Emulate/check',variant,suite,flush=True);report=work/(variant+'-'+suite)
     p=run([ROOT/'bin/drperf','--gpu','--output',report,'--',driver,module,variant,suite],allowed=(0,1,2))
     record['runs'][variant+'-'+suite]={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
     expected=1 if 'matmul' in CASE.name and variant=='after' and suite=='heldout' else 0
     assert p.returncode==expected, p.stdout+p.stderr
     prof=report/'profile.gpu.drperf.json'
     if not prof.exists():continue
     model=json.loads(prof.read_text());r=model['regions'][0]
     assert model['validity']=='complete'
     for obs in r['observations']:
      v=obs['pcvs'];m=obs['metrics']
      if 'matmul' in CASE.name:
       assert m['tensor_float_flops']==(3 if variant=='after' else 4)*2*128*128*16*v['tile_products']
       assert m['global_read_bytes']==8*v['input_elements_per_tile_grid']
       assert m['global_write_bytes']==4*v['outputs']
      else:
       assert m['tensor_float_flops']==2*128*128*16*v['tile_products']
       assert m.get('global_write_bytes',0)==4*v['positive_updates' if variant=='after' else 'pool_updates']
      assert m.get('local_read_bytes',0)==m.get('local_write_bytes',0)==0
     record['runs'][variant+'-'+suite]['resource_assertions']='passed'
     compact={'validity':model['validity'],'state_count':r['stateCount'],'pcvs':r['pcvs'],'fits':{m:r['fits'].get(m) for m in METRICS},'interfaces':{m:r['interfaces'].get(m) for m in METRICS},'states':[{'pcvs':s['pcvs'],'invocations':s['invocations'],'metrics':{m:v for m,v in s['metrics'].items() if m in METRICS or 'flop' in m or 'atomic' in m}} for s in r['stateStatistics']], 'global_memory_sites':{s:{m:f[m] for m in ['global_read_bytes','global_write_bytes'] if m in f} for s,f in r['sites'].items() if any(m in f for m in ['global_read_bytes','global_write_bytes'])}}
     (results/(variant+'-'+suite+'.json')).write_text(json.dumps(compact,indent=2)+'\n')
     if suite=='train':(results/(variant+'-report.txt')).write_text((report/'report.txt').read_text())
   for a,b in [('before','compatible'),('compatible','after')]:
    if (CASE/(b+'.cu')).exists():(results/(a+'-to-'+b+'.patch')).write_text(''.join(difflib.unified_diff((CASE/(a+'.cu')).read_text().splitlines(True),(CASE/(b+'.cu')).read_text().splitlines(True),fromfile=a+'.cu',tofile=b+'.cu')))
 finally:
  record['elapsed_seconds']=time.monotonic()-start;(results/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
 print('Evidence:',results,flush=True)
if __name__=='__main__':main()
