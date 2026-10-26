#!/usr/bin/env python3
"""Build PTX, translate, capture concurrent ranks, and fit only small states."""
import argparse,datetime,gzip,hashlib,json,os,pathlib,shutil,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCRATCH=ROOT/'gpu/build/ext-work2'
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=HERE/'results/reproduction');p.add_argument('--skip-build',action='store_true');a=p.parse_args()
out=a.output.resolve()
if not out.is_relative_to(HERE):p.error('output must be within the example directory')
out.mkdir(parents=True,exist_ok=False)
(out/'tool_snapshot').mkdir();(SCRATCH/'tmp').mkdir(exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(SCRATCH/'tmp'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def call(label,argv,timeout=120):
 argv=list(map(str,argv));rec={'argv':argv,'cwd':str(ROOT),'start_utc':utc(),'physical_gpu_seconds':0,'status':'running'}
 path=out/(label+'.record.json')
 def save():path.write_text(json.dumps(rec,indent=2)+'\n')
 save();start=time.monotonic()
 try:
  with (out/(label+'.stdout.log')).open('w') as so,(out/(label+'.stderr.log')).open('w') as se:
   cp=subprocess.run(argv,cwd=ROOT,env=env,stdout=so,stderr=se,timeout=timeout)
  rec.update(exit_code=cp.returncode,status='complete' if cp.returncode==0 else 'failed')
 except BaseException as e:rec.update(status='interrupted',error=repr(e));raise
 finally:rec.update(end_utc=utc(),elapsed_seconds=time.monotonic()-start);save()
 if cp.returncode:raise RuntimeError(label+' failed; see '+str(out/(label+'.stderr.log')))
 print(label+': pass',flush=True)
runtime=ROOT/'gpu/build/libdrperf_gpu.so'
shutil.copyfile(runtime,out/'tool_snapshot/libdrperf_gpu.so')
env['LD_LIBRARY_PATH']=str(out/'tool_snapshot')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
inputs=[HERE/'kernels.cu',HERE/'driver.cc',HERE/'run.py',HERE/'analyze.py',HERE/'visibility.cu',HERE/'visibility_driver.cc',ROOT/'gpu/vendor/gxptx/.lake/build/bin/gxptx',ROOT/'gpu/vendor/gxptx/Gxptx/prelude.h',runtime]
before={str(f):sha(f) for f in inputs if f.is_file()}
if a.skip_build:
 for name in ['kernels.ptx','kernels.so','kernels.so.cpp','kernels.so.gpu.json','driver']:
  shutil.copyfile(HERE/'results'/name,out/name)
else:
 call('ptx',['/home/ubuntu/GX/toolchains/cuda13/bin/nvcc','-std=c++17','-O2','-arch=compute_103a','-ptx',HERE/'kernels.cu','-o',out/'kernels.ptx'])
 call('translate',[ROOT/'bin/drperf','--gpu','translate',out/'kernels.ptx','-o',out/'kernels.so'])
 call('driver',['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT/'gpu/include'),HERE/'driver.cc','-L'+str(out/'tool_snapshot'),'-ldrperf_gpu','-Wl,-rpath,'+str(out/'tool_snapshot'),'-o',out/'driver'])
call('visibility-ptx',['/home/ubuntu/GX/toolchains/cuda13/bin/nvcc','-std=c++17','-O2','-arch=compute_103a','-ptx',HERE/'visibility.cu','-o',out/'visibility.ptx'])
call('visibility-translate',[ROOT/'bin/drperf','--gpu','translate',out/'visibility.ptx','-o',out/'visibility.so'])
call('visibility-driver',['g++','-std=c++17','-O2','-pthread','-I'+str(ROOT/'gpu/include'),HERE/'visibility_driver.cc','-L'+str(out/'tool_snapshot'),'-ldrperf_gpu','-Wl,-rpath,'+str(out/'tool_snapshot'),'-o',out/'visibility-driver'])
call('visibility',[ROOT/'bin/drperf','--gpu','--output',out/'visibility','--timeout','30','--',out/'visibility-driver',out/'visibility.so'])
def archive_capture(label):
 for f in (out/label).rglob('*'):
  if f.is_file() and (f.suffix=='.jsonl' or f.name in ['profile.gpu.drperf.json','report.txt']):
   with f.open('rb') as src,gzip.open(str(f)+'.gz','wb',compresslevel=6) as dst:shutil.copyfileobj(src,dst)
   f.unlink()
archive_capture('visibility')
# Each case includes 12 consecutive calls per width and rank with triple-buffer reuse.
for label,mode,ranks,cluster,skew in [('r2','all',2,0,0),('r2-skew','all',2,0,1000),
 ('r2-cluster','norm',2,2,0),('r4','all',4,0,0),('r4-cluster','norm',4,2,0)]:
 call(label,[ROOT/'bin/drperf','--gpu','--output',out/label,'--timeout','120','--',out/'driver',out/'kernels.so',mode,ranks,0,cluster,skew],180)
 archive_capture(label)
# Intermediate widths validate fitted coefficients without joining training.
for hidden in [96,192,384]:
 for ranks in [2,4]:
  label='heldout-h'+str(hidden)+'-r'+str(ranks)
  call(label,[ROOT/'bin/drperf','--gpu','--output',out/label,'--timeout','120','--',out/'driver',out/'kernels.so','all',ranks,0,0,0,hidden],180)
  archive_capture(label)
# The production launch rule first selects a nontrivial cluster at hidden=2048.
# This is a numerical/protocol validation only, entirely excluded from fitting.
for ranks in [2,4]:
 label='launcher-cluster-h2048-r'+str(ranks)
 call(label,[ROOT/'bin/drperf','--gpu','--output',out/label,'--timeout','120','--',out/'driver',out/'kernels.so','norm',ranks,1,0,0,2048],180)
 archive_capture(label)
call('analyze',[sys.executable,'-B',HERE/'analyze.py','--run',out])
after={str(f):sha(f) for f in inputs if f.is_file()}
(out/'manifest.json').write_text(json.dumps({'completed_utc':utc(),'physical_gpu_seconds':0,'no_git_commands':True,
 'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','TMPDIR','PYTHONDONTWRITEBYTECODE','LD_LIBRARY_PATH']},
 'input_hashes_before':before,'input_hashes_after':after,'external_tool_drift':{k:[v,after.get(k)] for k,v in before.items() if after.get(k)!=v},
 'runtime_snapshot_sha256':sha(out/'tool_snapshot/libdrperf_gpu.so'),'artifacts':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.suffix in ['.ptx','.so','.cc','.json'] and f.name!='manifest.json'}},indent=2)+'\n')
print('Results: '+str(out))
