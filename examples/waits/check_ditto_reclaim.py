#!/usr/bin/env python3
"""Compare Ditto's nonblocking reclaim with an isolated forced-GC-wait variant."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import event_model
import explorer
import runner

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ditto',type=Path,default=Path('/home/ubuntu/compression/ditto_kv'))
args=parser.parse_args()
out=ROOT/'out/waits/ditto-reclaim';source=out/'source';source.mkdir(parents=True,exist_ok=True)
files={'compaction.py':'src/cache/compaction.py','runtime.py':'src/cache/runtime.py',
       'worker.py':'src/integration/vllm/worker.py'}
hashes={}
for name,path in files.items():
    original=args.ditto/path;shutil.copyfile(original,source/name)
    hashes[path]=hashlib.sha256(original.read_bytes()).hexdigest()
original=(source/'compaction.py').read_text()
needle='        if aborted:\n            if newly:'
assert original.count(needle)==1,'Lease-flush source changed; review before patching'
changed=original.replace(needle,'        if aborted:\n            self.compaction_stream.synchronize()  # Deliberately restored blocking fence.\n            if newly:')
(source/'compaction_forced.py').write_text(changed)
(out/'forced-wait.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),
    fromfile='src/cache/compaction.py',tofile='isolated/compaction_forced.py')))
for name in ('ditto_reclaim.py','ditto_reclaim_completion.c'):
    shutil.copyfile(Path(__file__).with_name(name),source/name)
subprocess.run(['gcc','-O2','-shared','-fPIC','-pthread',str(source/'ditto_reclaim_completion.c'),
    '-I'+str(ROOT/'perfmark'),'-L'+str(ROOT/'build'),'-lperfmark',
    '-Wl,-rpath,'+str(ROOT/'build'),'-o',str(out/'completion.so')],check=True)
results={'scope':'Actual Ditto worker.wait, CacheRuntime and CompactionEngine methods; upstream CPU GC harness and real FTL; controlled completion, no GPU compression or full vLLM run.',
         'sourceHashes':hashes,'runs':{}}
for mode,delay in [('fixed',0),('forced',0),('omitted',0),('fixed',400),('forced',400),('false-claim',400)]:
    label=mode+('-probe' if delay else '')
    raw=out/label
    for p in raw.glob('run.*.json*'):p.unlink()
    with patch.dict(os.environ,{'DRPERF_WAITS':'1','DRPERF_FOLLOW_THREADS':'0',
        'DRPERF_WAIT_DELAY_REGION':'ditto.compression.complete',
        'DRPERF_WAIT_DELAY_KIND':'event','DRPERF_WAIT_DELAY_MS':str(delay)}):
        rc,log,_=runner.run([sys.executable,str(source/'ditto_reclaim.py'),str(args.ditto),str(out),mode],str(raw),timeout=60)
    (out/(label+'.log')).write_text(log)
    assert rc==0,log
    line=next(l for l in log.splitlines() if l.startswith('RECLAIM_RESULT='))
    case=json.loads(line.split('=',1)[1])
    comp='compaction_forced.py' if mode in ('forced','omitted') else 'compaction.py'
    paths=['source/'+name for name in (comp,'runtime.py','worker.py','ditto_reclaim.py','ditto_reclaim_completion.c')]
    m=explorer.build_model(raw,out,paths,discover=False)
    m['title']='Ditto reclaim control-path regression: '+label
    m['experiment']={'scope':results['scope'],'sourceHashes':hashes,'case':case,'delayMs':delay}
    report=m['eventModel']
    assert not report['unverified'],report
    wanted=[e for e in report['edges'] if e['consumer']=='vllm.reclaim_pages' and e['producer']=='ditto.compression.complete']
    if mode=='forced':
        assert len(wanted)==3 and all(e['status']=='ordered' for e in wanted),report
    elif mode=='false-claim':
        assert len(wanted)==3 and all(e['status']=='violation' for e in wanted),report
    else: assert not wanted,report
    if mode=='fixed' and delay: assert all(not c['readyAtReclaimReturn'] for c in case['cases']),case
    explorer.write_model(m,out/(label+'.drperf.json'))
    (out/(label+'.report.txt')).write_text('\n'.join(event_model.lines(report))+'\n')
    results['runs'][label]=dict(case=case,edges=wanted,coverage=report['coverage'],violations=report['violations'])
    print(label,'edges',len(wanted),'violations',len(report['violations']),
        'coverage',report['coverage'].get('covered'), '/',report['coverage'].get('obligations'),flush=True)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(hashlib.sha256((args.ditto/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print(out/'results.json')
