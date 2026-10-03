#!/usr/bin/env python3
"""Integration regression: actual coroutine suspension under DynamoRIO."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'lib'))
import explorer
import runner


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []
    for parent in (False, True):
        for credits in (0, 1, 2):
            label = ('parent' if parent else 'local') + '-' + str(credits)
            raw = output/label
            for old in raw.glob('run.*.json*'): old.unlink()
            command = [sys.executable, str(Path(__file__).with_name('async_operations.py')),
                       '--checkpoints', str(credits)] + (['--parent'] if parent else [])
            with patch.dict(os.environ, {'DRPERF_WAITS':'1', 'DRPERF_FOLLOW_THREADS':'0'}):
                rc, log, _ = runner.run(command, str(raw), timeout=60)
            assert rc == 0, log
            consumer = 'request' if parent else 'consumer'
            claims = dict(version=1, claims=[dict(id='dependency-'+str(i), region=consumer,
                event=str(i), producer='producer.'+str(i), indicator='need == 1') for i in (1,2)])
            model = explorer.build_model(raw, ROOT, ['examples/waits/async_operations.py'],
                                         discover=False, wait_declarations=claims)
            report = model['eventModel']
            assert not report['violations'] and not report['unverified'], report
            coverage = report['coverage']
            assert (coverage['obligations'],coverage['covered']) == (2,credits), coverage
            assert sum(c['unexpectedAbsence'] for c in report['interfaceChecks']['claims']) == 2-credits
            assert all(e['instructionExcluded'] for e in model['waits']['events']
                       if e['kind'].startswith(('runtime_', 'declared_')))
            assert len([o for o in model['waits']['operations'] if o['kind']=='runtime_wait']) == 2
            explorer.write_model(model, output/(label+'.drperf.json'))
            rows.append(dict(case=label,obligations=2,covered=credits,
                             missingCheckpoints=2-credits,status=report['interfaceChecks']['status']))
            print(rows[-1], flush=True)
    (output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows

if __name__ == '__main__': run(ROOT/'out/waits/async-operations')
