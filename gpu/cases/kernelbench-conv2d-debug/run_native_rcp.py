#!/usr/bin/env python3
"""Validate two frozen public kernels in one bounded RCP GPU allocation."""
import datetime,hashlib,json,subprocess,tempfile,time
from pathlib import Path
CASE=Path(__file__).resolve().parent
CASES={'layernorm':CASE.parent/'kernelbench-layernorm-debug','conv2d':CASE}
NS='runai-vca-jma'
CONTEXT='rcp-caas-prod'
IMAGE='vllm/vllm-openai@sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86'
def main():
 job='drperf-public-'+datetime.datetime.now(datetime.timezone.utc).strftime('%m%d-%H%M%S')
 output=CASE/'results'/job;output.mkdir()
 record={'job':job,'namespace':NS,'context':CONTEXT,'image':IMAGE,'pool':'b300','gpu_count':1,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':[]}
 def run(cmd,timeout=60,check=True):
  cmd=list(map(str,cmd));start=time.monotonic();p=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout);saved=p.stdout+p.stderr
  if cmd[:3]==['kubectl','--context',CONTEXT] and 'pods' in cmd and p.returncode==0:
   saved=json.dumps([{'name':x['metadata']['name'],'phase':x['status']['phase'],'node':x['spec'].get('nodeName')} for x in json.loads(p.stdout)['items'] if x['metadata']['name'].startswith(job+'-')])
  record['commands'].append({'argv':cmd,'seconds':time.monotonic()-start,'exit_code':p.returncode,'output':saved})
  if check and p.returncode:raise RuntimeError(p.stdout+p.stderr)
  return p.stdout
 def kube(args,**kw):return run(['kubectl','--context',CONTEXT,*args],**kw)
 def pods():return [x for x in json.loads(kube(['get','pods','-n',NS,'-o','json']))['items'] if x['metadata']['name'].startswith(job+'-')]
 try:
  assert run(['kubectl','config','current-context']).strip()==CONTEXT
  with tempfile.TemporaryDirectory(prefix='drperf-public-native-') as tmp:
   bundle=Path(tmp)/'bundle';bundle.mkdir();record['sources']={}
   for name,case in CASES.items():
    sub=bundle/name;sub.mkdir()
    for filename in (['before.cu','after.cu','native_benchmark.cu'] if name=='layernorm' else ['compatible.cu','after.cu','native_benchmark.cu']):
     data=(case/filename).read_bytes();(sub/filename).write_bytes(data);record['sources'][name+'/'+filename]=hashlib.sha256(data).hexdigest()
   (bundle/'run.sh').write_text('''#!/bin/bash
set -euo pipefail
cd /tmp/drperf-public-native
mkdir -p results/layernorm results/conv2d
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit --format=csv > results/gpu-before.txt
nvcc --version > results/compiler.txt
nvcc -std=c++17 -O3 -lineinfo -arch=native -Xcompiler=-fopenmp -Xptxas=-v layernorm/native_benchmark.cu -o layernorm/benchmark > results/layernorm/compile.log 2>&1
nvcc -std=c++17 -O3 --use_fast_math -lineinfo -arch=native -Xcompiler=-fopenmp -Xptxas=-v conv2d/native_benchmark.cu -o conv2d/benchmark > results/conv2d/compile.log 2>&1
OMP_NUM_THREADS=8 timeout 180 layernorm/benchmark > results/layernorm/native.json 2>results/layernorm/progress.log
OMP_NUM_THREADS=8 timeout 300 conv2d/benchmark > results/conv2d/native.json 2>results/conv2d/progress.log
nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > results/gpu-after.txt
''')
   print('Submitting one B300:',job,flush=True)
   run(['runai','submit',job,'-p','vca-jma','--image',IMAGE,'--gpu','1','--cpu','8','--cpu-limit','8','--memory','16Gi','--memory-limit','16Gi','--node-pools','b300','--interactive','--preemptible','--run-as-uid','228407','--run-as-gid','83691','--backoff-limit','0','--auto-deletion-time-after-completion','1m','--command','--','/usr/bin/timeout','--signal=TERM','--kill-after=10','660','sleep','650'])
   start=time.monotonic();pod=None
   while time.monotonic()-start<300:
    selected=pods()
    if selected:
     p=selected[0]
     if p['status']['phase'] in ['Failed','Succeeded']:raise RuntimeError('Owned pod ended early')
     if p['status']['phase']=='Running' and all(c.get('ready') for c in p['status'].get('containerStatuses',[])):
      pod=p['metadata']['name'];record['node']=p['spec'].get('nodeName');break
    time.sleep(5)
   if pod is None:raise RuntimeError('No ready GPU within five minutes')
   record['pod']=pod;print('Measuring on',record['node'],pod,flush=True)
   kube(['cp','-n',NS,bundle,pod+':/tmp/drperf-public-native'])
   try:kube(['exec','-n',NS,pod,'--','bash','/tmp/drperf-public-native/run.sh'],timeout=540)
   finally:kube(['cp','-n',NS,pod+':/tmp/drperf-public-native/results',output/'native'],check=False)
   for name in CASES:
    result=json.loads((output/'native'/name/'native.json').read_text());assert len(result['cases'])==3 and all(c['passed'] for c in result['cases']);print(name,json.dumps(result),flush=True)
   record['result']='passed'
 finally:
  print('Deleting owned job',job,flush=True)
  run(['runai','delete','job',job,'-p','vca-jma'],check=False)
  for _ in range(12):
   if not pods():break
   time.sleep(5)
  record['job_removed']=not bool(pods());record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(output/'allocation.json').write_text(json.dumps(record,indent=2)+'\n')
  if not record['job_removed']:raise RuntimeError('Owned GPU pod remains after cleanup')
 print('Evidence:',output,flush=True)
if __name__=='__main__':main()
