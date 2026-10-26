#!/usr/bin/env python3
"""Reproduce bounded GPU-resource scaling on CPUs; records every command."""
import argparse,datetime,hashlib,json,os,pathlib,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=HERE/'results');p.add_argument('--build',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def call(label,cmd,timeout=180):
 rec={'label':label,'argv':[str(x) for x in cmd],'cwd':str(ROOT),'start_utc':utc(),'status':'running','physical_gpu_seconds':0,'category':'cpu_build' if label in ['build','driver'] else 'drperf_gpu_ptx'}
 def save():(out/(label+'.record.json')).write_text(json.dumps(rec,indent=2)+'\n')
 save();start=time.monotonic()
 try:
  with (out/(label+'.stdout.log')).open('w') as stdout,(out/(label+'.stderr.log')).open('w') as stderr:
   r=subprocess.run(rec['argv'],cwd=ROOT,stdout=stdout,stderr=stderr,timeout=timeout)
  rec.update(exit_code=r.returncode,status='completed' if r.returncode==0 else 'failed')
 except BaseException as e:rec.update(status='interrupted',error=repr(e));raise
 finally:rec.update(end_utc=utc(),elapsed_seconds=time.monotonic()-start);save()
 if r.returncode:raise SystemExit(label+' failed; see '+str(out))
if a.build:call('build',[ROOT/'bin/drperf','--gpu','build'],360)
call('translate',[ROOT/'bin/drperf','--gpu','translate',HERE/'kernels.ptx','-o',out/'kernels.so'])
call('driver',[os.environ.get('CXX','g++'),'-std=c++17','-O2','-I'+str(ROOT/'gpu/include'),HERE/'driver.cc','-L'+str(ROOT/'gpu/build'),'-ldrperf_gpu','-Wl,-rpath,'+str(ROOT/'gpu/build'),'-o',out/'driver'])
call('capture',[ROOT/'bin/drperf','--gpu','--output',out/'capture-report','--',out/'driver',out/'kernels.so'],240)
call('analysis',[sys.executable,HERE/'analyze.py','--profile',out/'capture-report/profile.gpu.drperf.json','--output',out])
files=[HERE/'kernels.cu',HERE/'kernels.ptx',HERE/'driver.cc',HERE/'analyze.py',HERE/'run.py',HERE/'sources.json',ROOT/'gpu/runtime/runtime.cc',ROOT/'gpu/vendor/gx/tcgen05.cc',ROOT/'gpu/build/libdrperf_gpu.so',ROOT/'gpu/vendor/gxptx/.lake/build/bin/gxptx']
(out/'manifest.json').write_text(json.dumps({'completed_utc':utc(),'physical_gpu_seconds':0,'drperf_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'ptx_build':'CUDA13 nvcc -std=c++17 -O2 -arch=compute_103a -ptx gpu/examples/glm53_scaling/kernels.cu; checked-in PTX permits GPU-free/toolkit-free reproduction'},indent=2)+'\n')
print('Results: '+str(out))
