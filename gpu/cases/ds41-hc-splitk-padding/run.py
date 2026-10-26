#!/usr/bin/env python3
"""Run from this case's directory. CPU-only; byte-pinned DrPerf; no CUDA GPU."""
import argparse,gzip,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else p.open()) as f:return json.load(f)
def compact(p):
 d=load(p); regions=[]
 for r in d['regions']:
  fits={k:{**v,'coefficients':{n:x for n,x in v.get('coefficients',{}).items() if abs(x)>1e-9}} for k,v in r.get('fits',{}).items()}
  regions.append(dict(name=r['name'],pcvs=r['pcvs'],stateCount=r['stateCount'],invocations=r['invocations'],singleState=r.get('singleState'),fits=fits,observations=[dict(pcvs=o['pcvs'],metrics=o['metrics']) for o in r['observations']]))
 return dict(validity=d['validity'],semantics=d['semantics'],provenance=d['provenance'],regions=regions)
def check(config,results):
 checks=[]
 for expectation in config.get('expect',[]):
  found=[]
  for result in results:
   if result.get('status')!='pass':continue
   for region in result['summary']['regions']:
    if region['name']!=expectation['region']:continue
    for o in region['observations']:
     metric=o['metrics'].get(expectation['metric'],0);want=expectation['factor']*o['pcvs'][expectation['pcv']]
     if abs(metric-want)>max(1e-5,abs(want)*1e-9):raise AssertionError((expectation,metric,want,o['pcvs']))
     found.append(metric)
  if not found:raise AssertionError('No observations for '+str(expectation))
  checks.append(dict(expectation=expectation,observations=len(found),status='pass'))
 return checks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--toolchain',type=Path,default=Path(os.environ.get('DRPERF_HOME','/home/ubuntu/gen-deepseek41/build/drperf-gpu-toolchain')));ap.add_argument('--keep-work',action='store_true');a=ap.parse_args();os.chdir(HERE)
 cfg=load(HERE/'case.json');pin=load(HERE/'toolchain.json');tool=a.toolchain.resolve()
 for name,digest in pin['files'].items():
  if sha(tool/name)!=digest:raise RuntimeError('Pinned toolchain changed: '+name)
 for name,digest in cfg['input_hashes'].items():
  if sha(HERE/name)!=digest:raise RuntimeError('Case input changed: '+name)
 out=HERE/'results';out.mkdir(exist_ok=True);work=HERE/'.work';work.mkdir(exist_ok=True);results=[];commands=[]
 def run(argv,log,env=None):
  commands.append(dict(argv=list(map(str,argv)),environment=env or {}))
  with log.open('w') as f:return subprocess.run(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,**(env or {})),timeout=cfg.get('command_timeout',300)).returncode
 try:
  for v in cfg.get('variants',[]):
   name=v['name'];dest=out/name;dest.mkdir(exist_ok=True);module=work/(v.get('module_name',name)+'.so');binary=work/(name+'-oracle')
   code=run([tool/'bin/drperf','--gpu','translate',HERE/v['ptx'],'-o',module],dest/'translation.log')
   if code:raise RuntimeError('PTX translation failed: '+name)
   flags=['-D'+x for x in v.get('defines',[])];cmd=['g++','-std=c++20','-O2','-DDS41_DRPERF_GPU','-I'+str(HERE/'support'),'-I'+str(tool/'gpu/include'),*flags,HERE/v.get('driver','driver.cpp'),'-L'+str(tool/'gpu/build'),'-ldrperf_gpu','-Wl,-rpath,'+str(tool/'gpu/build'),'-o',binary]
   if run(cmd,dest/'build.log'):raise RuntimeError('CPU oracle compilation failed: '+name)
   child=[binary,module]
   if v.get('grid_env'):child=[sys.executable,HERE/'run.py','--child',binary,module,json.dumps(v['grid_env'])]
   argv=[tool/'bin/drperf','--gpu','--timeout',str(cfg.get('execution_timeout',240)),'--output',dest/'capture','--',*child]
   rc=run(argv,dest/'execution.log',v.get('env',{}));text=(dest/'execution.log').read_text()
   if v.get('expected_blocker'):
    if rc==0 or v['expected_blocker'] not in text:raise RuntimeError('Expected coverage blocker not reproduced: '+name+' rc='+str(rc))
    results.append(dict(name=name,status='blocked',reason=v['expected_blocker'],numerical_status='not_assessed'));continue
   if rc:raise RuntimeError('Numerical/resource execution failed: '+name+'; see '+str(dest/'execution.log'))
   p=dest/'capture/profile.gpu.drperf.json';summary=compact(p)
   if summary['validity']!='complete':raise RuntimeError('Incomplete resource capture '+name)
   (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copy2(dest/'capture/report.txt',dest/'report.txt')
   # Keep reports + compact state/fits, not site-level multi-MB traces; frozen PTX regenerates those.
   shutil.rmtree(dest/'capture')
   results.append(dict(name=name,status='pass',summary=summary,numerical_status='pass'))
  checks=check(cfg,results)
  public=[{k:v for k,v in r.items() if k!='summary'} for r in results]
  status='blocked' if any(r['status']=='blocked' for r in public) else ('proposed' if not public else 'pass')
  receipt=dict(case=cfg['case'],status=status,toolchain_commit=pin['upstream_commit'],toolchain_runtime_sha256=pin['files']['gpu/build/libdrperf_gpu.so'],variants=public,interface_checks=checks,physical_gpu_used=False,commands=commands)
  (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
  import check_interfaces
  receipt['structural_interface_checks']=check_interfaces.check()['checks']
  (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
  (HERE/'commands.txt').write_text('# Exact commands; run from '+str(HERE)+'\n'+''.join(shlex.join([k+'='+v for k,v in c['environment'].items()]+c['argv'])+'\n' for c in commands))
  for p in out.rglob('*'):
   if p.is_file() and p.stat().st_size>1000000 and p.suffix!='.gz':
    with p.open('rb') as src,gzip.open(str(p)+'.gz','wb',compresslevel=9) as dst:shutil.copyfileobj(src,dst)
    p.unlink()
  print(json.dumps({k:v for k,v in receipt.items() if k!='commands'}));return 0
 finally:
  if not a.keep_work:shutil.rmtree(work,ignore_errors=True)
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--child':
  for env in json.loads(sys.argv[4]):subprocess.run(sys.argv[2:4],env=dict(os.environ,**env),check=True)
 else:sys.exit(main())
