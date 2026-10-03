#!/usr/bin/env python3
"""Execute nested, retrying, shared, and parallel native coverage examples."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import explorer
import runner


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    program=output/'coverage'
    subprocess.run(['gcc','-O2','-pthread',str(Path(__file__).with_name('coverage.c')),
        '-I'+str(ROOT/'perfmark'),'-L'+str(ROOT/'build'),'-lperfmark',
        '-Wl,-rpath,'+str(ROOT/'build'),'-o',str(program)],check=True)
    expectations={'deep':(3,1),'retries':(3,1),'two':(2,2),'missing':(2,1),
        'direct_missing':(2,1),'direct_two':(2,2),'early':(1,0),'own':(2,2),'sibling':(1,0),'none':(2,0),
        'shared':(2,0),'relocated':(2,2),'parallel':(2,2),'branch':(9,3),
        'uncontended':(1,0), 'null_shared':(2,2), 'null_parent':(2,2), 'child_publisher':(1,0)}
    results={}
    for mode,expected in expectations.items():
        raw=output/mode
        for previous in raw.glob('run.*.json*'): previous.unlink()
        with patch.dict(os.environ,{'DRPERF_WAITS':'1','DRPERF_FOLLOW_THREADS':'0',
            'DRPERF_WAIT_DELAY_MS':'0','DRPERF_WAIT_DELAY_REGION':'','DRPERF_WAIT_DELAY_KIND':'event'}):
            rc,log,_=runner.run([str(program),mode],str(raw),timeout=30)
        assert rc==0,(mode,log)
        # Explicit contracts for the branches in coverage.c, not learned guards.
        pairs = [('A', '1', 'P', 'need == 1' if mode == 'branch' else 'True')]
        if mode in ('parallel', 'relocated'): pairs.append(('C', '2', 'Q', 'True'))
        if mode == 'sibling': pairs = [('C', '1', 'P', 'True')]
        if mode == 'shared': pairs = [('B', '1', 'P', 'True')]
        if mode == 'child_publisher': pairs = [('A', '1', 'B', 'True')]
        if mode in ('none', 'uncontended'): pairs = []
        declarations = dict(version=1, claims=[dict(id=f'wait-{i}', region=r, event=e,
            producer=p, indicator=guard) for i, (r,e,p,guard) in enumerate(pairs)])
        if mode.startswith('null_'):
            declarations = dict(version=1, claims=[dict(id='counter-'+region, region=region,
                event=None, producer=None, indicator='need == 1',
                reason='Protect a local bookkeeping counter; no cross-region publisher is claimed.')
                for region in (('B',) if mode=='null_shared' else ('A','C'))])
        model=explorer.build_model(raw,ROOT,['examples/waits/coverage.c'],discover=False,
                                   wait_declarations=declarations)
        report=model['eventModel'];coverage=report['coverage']
        assert not report['unverified'],(mode,report)
        assert (coverage['obligations'],coverage['covered'])==expected,(mode,coverage)
        assert bool(report['violations'])==(mode in ('shared','child_publisher')),(mode,report)
        path=output/(mode+'.drperf.json');explorer.write_model(model,path)
        checked=subprocess.run([str(ROOT/'tools/drperf-check-events'),str(path)],capture_output=True,text=True)
        assert checked.returncode==(1 if mode in ('shared','child_publisher') else 2 if coverage['uncovered'] else 0),(mode,checked.stdout,checked.stderr)
        if mode=='retries': assert coverage['nativeCalls']>=9,coverage
        if mode=='uncontended':
            assert coverage['nativeCalls']==1 and not report['edges'],report
            assert report['unexplained'][0]['region']=='A',report
        if mode.startswith('null_'):
            assert not report['edges'] and not report['eventPairs'], report
            assert len(report['nullWaits'])==2 and report['interfaceChecks']['status']=='checked',report
            assert all(e['instructionExcluded'] for e in model['waits']['events']
                       if e['kind']=='declared_waited_null'), model['waits']
        if mode=='branch':
            interface=next(i for i in report['interfaces'] if i['region']=='A')
            assert (interface['present'],interface['absent'])==(3,3),interface
        results[mode]=dict(order=report['status'],coverage=coverage)
        print(f"{mode}: {coverage['covered']}/{coverage['obligations']} obligations covered, "
              f"{coverage['nativeCalls']} native calls, order={report['status']}",flush=True)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

if __name__=='__main__':
    run(ROOT/'out/waits/coverage')
