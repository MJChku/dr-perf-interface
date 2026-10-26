#!/usr/bin/env python3
import argparse,concurrent.futures,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--jobs',type=int,default=2);p.add_argument('--toolchain',default='/home/ubuntu/gen-deepseek41/build/drperf-gpu-toolchain');a=p.parse_args();root=Path(__file__).resolve().parent

def one(d):
 with (d/'rerun.log').open('w') as f:r=subprocess.run([sys.executable,'run.py','--toolchain',a.toolchain],cwd=d,stdout=f,stderr=subprocess.STDOUT)
 return dict(case=d.name,exit_status=r.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:result=list(ex.map(one,sorted(d.parent for d in root.glob('*/case.json'))))
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));sys.exit(any(x['exit_status'] for x in result))
