"""Run the LMCache prefetch study in an environment with LMCache installed.

Uses real disk-backend/serializer code, then checks declared dependencies and
deliberately false alternatives. Native timing is kept separate from captures.
"""
import argparse
import collections
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--drperf-root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root = args.drperf_root.resolve()
sys.path.insert(0, str(root / 'lib'))
import runner
import explorer
import event_model

driver = Path(__file__).with_name('lmcache_prefetch_admission.py')
declarations = json.loads(driver.with_suffix('.interfaces.json').read_text())
args.out.mkdir(parents=True, exist_ok=True)
os.environ['LD_LIBRARY_PATH'] = str(root / 'build') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ.update(DRPERF_WAITS='1', DRPERF_FOLLOW_THREADS='0', DRPERF_MAX_WAIT_RECORDS='200000')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def capture(mode, delay=0):
    folder = args.out / mode
    folder.mkdir(exist_ok=True)
    # Only replace this driver's own previous capture files, never merge runs.
    for path in (folder / 'raw').glob('run.*'):
        path.unlink()
    os.environ.update(DRPERF_WAIT_DELAY_MS=str(delay), DRPERF_WAIT_DELAY_KIND='event',
                      DRPERF_WAIT_DELAY_REGION='lmc.disk.read.large')
    extra = {'false-peer': ['--force-peer'], 'omit-peer': ['--omit-peer']}.get(mode, [])
    command = [sys.executable, str(driver), '--output', str(folder / 'timing.json'),
               '--chunks', '4,16', '--repeats', '2', '--capture'] + extra
    (folder / driver.name).write_bytes(driver.read_bytes())
    rc, log, _ = runner.run(command, str(folder / 'raw'), timeout=240)
    (folder / 'capture.log').write_text(log)
    if rc:
        raise RuntimeError(f'{mode}: capture failed ({rc}); see {folder / "capture.log"}')
    declared = copy.deepcopy(declarations)
    if mode == 'false-peer':
        declared['claims'][1]['indicator'] = 'True'
    model = explorer.build_model(folder / 'raw', discover=False, wait_declarations=declared)
    explorer.write_model(model, folder / 'profile.drperf.json')
    report = model['eventModel']
    save(folder / 'checks.json', report)
    print(mode, '\n'.join(event_model.lines(report)), flush=True)
    return model


with (args.out / 'native.log').open('w') as log:
    subprocess.run([sys.executable, str(driver), '--output', str(args.out / 'native.json'),
                    '--chunks', '4,16,64', '--repeats', '11'],
                   stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
baseline = capture('baseline')
b = baseline['eventModel']
assert b['status'] == 'ordered'
assert all(c['status'] == 'checked' for c in b['interfaceChecks']['claims'])
assert b['interfaceChecks']['claims'][1]['present'] > 0
assert b['interfaceChecks']['claims'][1]['absent'] > 0
plan = event_model.probe_plan(b)
delay = next(p['delayMs'] for p in plan if p['region'] == 'lmc.disk.read.large')
save(args.out / 'probe-plan.json', plan)
probe = capture('probe', delay)['eventModel']
assert probe['probe'] and probe['status'] == 'ordered'
assert all(c['status'] == 'checked' for c in probe['interfaceChecks']['claims'])
false_peer = capture('false-peer', delay)['eventModel']
assert false_peer['status'] == 'violation'
assert any('before declared publication' in v['reason'] for v in false_peer['violations'])
omitted = capture('omit-peer')
assert omitted['eventModel']['interfaceChecks']['claims'][1]['status'] == 'invalid'

controls = {}
for case in ('wrong-indicator', 'wrong-producer', 'missing-declaration', 'both-omitted'):
    altered = copy.deepcopy(omitted if case == 'both-omitted' else baseline)
    claims = altered['waitDeclarations']['claims']
    if case == 'wrong-indicator':
        claims[1]['indicator'] = 'True'
    elif case == 'wrong-producer':
        claims[1]['producer'] = 'lmc.disk.read.small'
    else:
        claims.pop()
    checked = event_model.check(altered)
    controls[case] = dict(interfaces=checked['interfaceChecks'], coverage=checked['coverage'])
assert controls['wrong-indicator']['interfaces']['status'] == 'invalid'
assert controls['wrong-producer']['interfaces']['status'] == 'invalid'
assert any(x['term'] == 'unexplained(Wait[lmc.disk.read.large])'
           for x in controls['missing-declaration']['interfaces']['unexplained'])
# Record the coverage limitation, rather than pretending to detect this edge.
assert not any('lmc.disk.read.large]' in x['term']
               for x in controls['both-omitted']['interfaces']['unexplained'])
save(args.out / 'controls.json', controls)

groups = collections.defaultdict(list)
native = json.loads((args.out / 'native.json').read_text())
for row in native['rows']:
    groups[row['mode'], row['chunks'], row['roomy']].append(row)
summary = []
for (mode, chunks, roomy), rows in sorted(groups.items()):
    summary.append(dict(mode=mode, chunks=chunks, roomy=roomy, trials=len(rows),
        requestMedianMs=statistics.median(r['requestLatencyMs'] for r in rows),
        peerWaits=sum(r['admissionWait'] for r in rows),
        smallFirst=sum(r['smallBeforeLarge'] for r in rows)))
save(args.out / 'summary.json', dict(native=summary,
    payloadsChecked=sum(r['payloadsChecked'] for r in native['rows']),
    plannedDelayMs=delay, orderViolations=len(false_peer['violations']),
    sources=native['sources'], scope=native['scope']))
print('All dependency checks and negative controls passed:', args.out, flush=True)
