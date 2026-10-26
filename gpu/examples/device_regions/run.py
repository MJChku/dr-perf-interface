#!/usr/bin/env python3
"""Reproduce inner-kernel PCV refinement using checked-in PTX; CPU only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    steps=[]
    def run(label,argv):
        command=[str(x) for x in argv]
        print(label,shlex.join(command),flush=True)
        started=datetime.now(timezone.utc).isoformat();t=time.monotonic()
        result=subprocess.run(command,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(label+'.log')).write_text(result.stdout)
        steps.append({'label':label,'argv':command,'cwd':str(ROOT),'startedUtc':started,
                      'finishedUtc':datetime.now(timezone.utc).isoformat(),
                      'durationSeconds':time.monotonic()-t,'exitCode':result.returncode})
        (out/'commands.json').write_text(json.dumps(steps,indent=2)+'\n')
        if result.returncode:raise RuntimeError(f'{label} failed; see {out/(label+".log")}')
    driver=out/'driver';module=out/'kernels.so'
    run('build',[os.environ.get('CXX','g++'),'-std=c++17','-O2','-I'+str(ROOT/'gpu/include'),
                 ROOT/'gpu/tests/device_regions_driver.cc','-L'+str(ROOT/'gpu/build'),'-ldrperf_gpu',
                 '-Wl,-rpath,'+str(ROOT/'gpu/build'),'-o',driver])
    run('translate',[ROOT/'bin/drperf','--gpu','translate',ROOT/'gpu/tests/device_regions.ptx','-o',module])
    models={}
    for name,mode in [('before','sparse'),('refined','sparse-refined')]:
        run(name,[ROOT/'bin/drperf','--gpu','--output',out/name,'--',driver,module,mode])
        models[name]=json.loads((out/name/'profile.gpu.drperf.json').read_text())
    before={r['name']:r for r in models['before']['regions']}
    refined={r['name']:r for r in models['refined']['regions']}
    clear='device.block.dense_clear';math='device.block.selected_math'
    assert not before[clear]['fits']['global_write_bytes']['accepted']
    assert before[math]['fits']['fp32_flops']['accepted']
    fit=refined[clear]['fits']['global_write_bytes']
    assert fit['accepted'] and fit['coefficients']['history']==4
    assert fit['coefficients']['selected']==0 and fit['constant']==0
    points=[{'history':r['pcvs']['history'],'selected':r['pcvs']['selected'],
             'globalWriteBytes':r['metrics']['global_write_bytes']}
            for r in refined[clear]['observations']]
    assert all(p['globalWriteBytes']==4*p['history'] for p in points)
    summary={'execution':'standalone CPU/PTX; no physical GPU',
             'before':{'selectedMathExplained':True,'denseClearExplained':False},
             'refined':{'denseClearWriteBytes':'4 * history','selectedMathFlops':'selected'},
             'points':points,'writeGrowth2048To100000':100000/2048,
             'meaning':'requested resource counts; not a GPU latency prediction'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    files=['gpu/include/drperf_gpu_device.cuh','gpu/tests/device_regions.cu',
           'gpu/tests/device_regions.ptx','gpu/tests/device_regions_driver.cc',
           'gpu/runtime/runtime.cc','gpu/build/libdrperf_gpu.so',
           'gpu/vendor/gxptx/.lake/build/bin/gxptx','lib/gpu_backend.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    (out/'manifest.json').write_text(json.dumps({'sha256':hashes,'commands':'commands.json'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
