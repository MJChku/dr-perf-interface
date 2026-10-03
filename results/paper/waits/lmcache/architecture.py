"""Export the two complete serving captures without merging their timelines.

Run from the repository root after copying each capture to architecture-ARM/.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'lib'))
import explorer
import event_interfaces
import event_model
import execution

arm = sys.argv[1]
assert arm in ('remote', 'cpu', 'pressure')
capture = BASE / ('architecture-' + arm)
source = BASE / 'overlay'
declarations = json.loads((BASE / ('wait-interfaces.json' if arm == 'remote'
                                   else 'wait-interfaces-cpu.json')).read_text())
model = explorer.build_model(capture / 'raw', source, None, True, 400_000,
                             200_000, declarations)
# Only exercised annotations can have their indicator checked. Keep the
# unexercised, source-declared refinements explicit in provenance, rather than
# passing unknown regions as claims or pretending they were validated.
refinements = json.loads((BASE / 'null-refinements.json').read_text())
observed = {r['id'] for r in model['regions']}
declarations['claims'] = [c for c in declarations['claims']
                          if not (c['event'] is None and c['region'] in refinements)]
declarations['claims'] += [dict(id='scope-null-' + name, region=name, event=None,
    producer=None, indicator='True', reason=reason)
    for name, reason in refinements.items() if name in observed]
model['waitDeclarations'] = declarations
model['eventModel'] = event_model.check(model)
model['executionGraph'] = execution.build(model)
model['provenance']['unexercisedNullRefinements'] = sorted(set(refinements) - observed)
assert not model['validity']['errors'], model['validity']
assert not model['validity']['traceErrors'], model['validity']
assert model['trace']['complete']
model['title'] = 'LMCache complete serving workflow: ' + arm
workload = json.loads((capture / 'client.json').read_text())
assert workload['summary']['failed'] == 0, workload['summary']
model['provenance']['experiment'] = {
    'application': 'Qwen2.5-0.5B-Instruct / vLLM / LMCache',
    'weights': 'dummy', 'execution': 'GX functional emulation, no GPU timing',
    'sessions': workload['args']['sessions'], 'turns': workload['args']['turns'],
    'requests': workload['summary']['requests'],
    'backend': 'remote_naive' if arm == 'remote' else 'cpu',
    'deviceKvBytes': 40_000_000, 'chunkTokens': 256,
    'localCpuGiB': {'remote': 2, 'cpu': 0.4, 'pressure': 0.12}[arm],
    'scope': 'Observed client serving paths, not all LMCache backends or schedules',
    'runtime': json.loads((capture / 'runtime.json').read_text()) if (capture / 'runtime.json').exists() else {},
    'captureHarness': dict(path=str(BASE / 'capture_workload.py'),
        sha256=hashlib.sha256((BASE / 'capture_workload.py').read_bytes()).hexdigest(),
        shutdown='Close the actual LMCache lookup server after all requests; require its receive thread to finish.'),
    'nullRefinements': dict(path=str(BASE / 'null-refinements.json'),
        sha256=hashlib.sha256((BASE / 'null-refinements.json').read_bytes()).hexdigest(),
        review='Manual source review; not automatic producer discovery or a proof of nonblocking execution.'),
}
# These names are selected by the active-coroutine-step adapter. Point to the
# actual operation, not an unrelated marker-only helper.
connector = source / 'lmcache/v1/storage_backend/connector/lm_connector.py'
data = connector.read_bytes()
regions = {r['id']: r for r in model['regions']}
for path in source.rglob('*.py'):
    content = path.read_bytes()
    for node in ast.walk(ast.parse(content)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == 'active_marked' and decorator.args):
                name = ast.literal_eval(decorator.args[0])
                if name in regions:
                    regions[name]['sources'] = [dict(path=str(path.relative_to(source)),
                        line=node.lineno, endLine=node.end_lineno,
                        sha256=hashlib.sha256(content).hexdigest(), expressions={})]
for node in ast.walk(ast.parse(data)):
    if isinstance(node, ast.AsyncFunctionDef) and node.name in ('exists', 'get', 'put'):
        name = 'lmc.remote.' + node.name + '_operation'
        if name in regions:
            regions[name]['sources'] = [dict(path=str(connector.relative_to(source)),
                line=node.lineno, endLine=node.end_lineno,
                sha256=hashlib.sha256(data).hexdigest(), expressions={})]
event = model['eventModel']
print('CHECK', arm, 'coverage', event['coverage']['covered'], '/', event['coverage']['obligations'],
      'remaining', [(r['region'], r['uncovered']) for r in event['coverage']['regions'] if r['uncovered']],
      'invalid claims', [r for r in event['interfaceChecks']['claims'] if r['status'] != 'checked'], flush=True)
assert event['status'] == 'ordered', 'Declared-event order did not pass'
assert not event['violations'] and not event['unverified'], 'Incomplete or invalid declaration evidence'
assert all(row['status'] == 'checked' for row in event['interfaceChecks']['claims']), 'A supplied wait interface did not check'
# Reject accidental declarations for metadata-only empty tensor conversions.
# Compare invocation identities, not just aggregate totals. This is evidence
# for these blocking Tensor.to paths in this capture, not a generic CUDA rule.
slot_checks = []
for direction in ('load', 'store'):
    name = 'lmc.gpu.' + direction + '_slot_mapping'
    def identity(row, trace=False):
        return (row['group'], str(row['thread' if trace else 'tid']),
                int(row['seq' if trace else 'regionSeq']))
    calls = {identity(t, True): t for t in model['trace']['events'] if t['region'] == name}
    consumer = 'lmc.vllm.' + ('start_load_kv' if direction == 'load' else 'wait_for_save')
    markers = [e for e in model['waits']['events']
               if e['region'] == consumer and e['kind'] == 'declared_waited_null']
    # One explicit scope-exit refinement follows the upload checkpoints in
    # every completed adapter invocation. Preserve the independent check of
    # uploads, including the no-transfer empty-tensor control.
    by_call = {}
    for marker in markers:
        by_call.setdefault(identity(marker), []).append(marker)
    parent_calls = [t for t in model['trace']['events'] if t['region'] == consumer]
    assert len(by_call) == len(parent_calls), (consumer, 'missing scope-exit checkpoint')
    markers = [e for group in by_call.values()
               for e in sorted(group, key=lambda e: e['start'])[:-1]]
    syncs = [o for o in model['waits']['operations']
             if o['region'] == name and o['api'] == 'cudaStreamSynchronize'
             and o.get('returned') and o['result'] == 0]
    assert len(markers) == len(syncs), (name, 'null checkpoints differ from completed transfer waits')
    assert all(calls[identity(o)]['values']['bytes'] > 0 for o in syncs), name
    # Source placement is separately tested. These observations must pair in
    # thread order, without skipping or reusing any completed slot transfer.
    for marker, sync in zip(sorted(markers, key=lambda x:(x['group'],str(x['tid']),x['start'])),
                            sorted(syncs, key=lambda x:(x['group'],str(x['tid']),x['start']))):
        assert (marker['group'],str(marker['tid'])) == (sync['group'],str(sync['tid']))
        assert sync['end'] < marker['start'], name
    slot_checks.append(dict(region=name, calls=len(calls), empty=sum(
        t['values']['bytes'] == 0 for t in calls.values()),
        nullCheckpoints=len(markers), completedNativeSyncs=len(syncs)))
# Negative controls: constant false must contradict every present wait, and
# constant true must contradict every absent wait. Do not modify saved claims.
indicator_controls = []
for value, field, expected in [('False', 'unexpectedPresence', 'present'),
                               ('True', 'unexpectedAbsence', 'absent')]:
    altered = dict(model, waitDeclarations=dict(version=1, claims=[
        dict(c, indicator=value) for c in declarations['claims']]))
    checked = event_interfaces.check(altered, event)
    original = {r['id']: r for r in event['interfaceChecks']['claims']}
    for row in checked['claims']:
        count = original[row['id']][expected]
        assert row[field] == count, (row['id'], value, row)
        assert not count or row['status'] == 'invalid', (row['id'], value, row)
    indicator_controls.append(dict(indicator=value, rejectedInvocations=sum(
        r[field] for r in checked['claims'])))
# Negative control on the same capture: reasons/declarations alone cannot
# discharge native observations. Removing null checkpoints must expose gaps.
without_null = dict(model, waits=dict(model['waits'], events=[e for e in model['waits']['events']
    if e['kind'] != 'declared_waited_null']))
# This is a synthetic missing-annotation control, not a truncated capture:
# keep its record-count metadata consistent. The real model is unchanged.
control_counts = Counter(e['group'] for e in without_null['waits']['events'])
without_null['provenance'] = dict(model['provenance'], runs=[
    dict(run, measurement=dict(run['measurement'], wait_records=control_counts[
        str(i) + ':' + str(run['measurement']['pid'])]))
    for i, run in enumerate(model['provenance']['runs'])])
missing_null = event_model.check(without_null)
assert missing_null['coverage']['uncovered'] > event['coverage']['uncovered']
assert missing_null['interfaceChecks']['status'] == 'invalid'
null_control = dict(removed=len(event.get('nullWaits', [])),
    uncoveredWithoutMarkers=missing_null['coverage']['uncovered'],
    interfaceStatus=missing_null['interfaceChecks']['status'])
model['id'] = hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
destination = BASE / ('architecture-' + arm + '.drperf.json')
explorer.write_model(model, destination)
summary = {
    'profile': str(destination), 'regionCount': len(regions),
    'workload': model['provenance']['experiment'],
    'validity': model['validity'], 'traceComplete': model['trace']['complete'],
    'eventStatus': event['status'], 'declaredOccurrences': len(event['edges']),
    'nullOccurrences': len(event.get('nullWaits', [])),
    'violations': event['violations'], 'unverified': event['unverified'],
    'coverage': event['coverage'], 'interfaceChecks': event['interfaceChecks'],
    'waitEdges': [e for e in model['executionGraph']['edges'] if e['kind'] == 'wait'],
    'slotTransferChecks': slot_checks, 'indicatorNegativeControls': indicator_controls,
    'nullCheckpointNegativeControl': null_control,
    'regions': [dict(id=r['id'], calls=r['calls'], states=r['states'], sources=r['sources'])
                for r in model['regions']],
}
calls = {(t['group'], str(t['thread']), int(t['seq'])) for t in model['trace']['events']}
covered = {(c['group'], str(c['thread']), int(c['invocation']))
           for edge in event['edges'] + event.get('nullWaits', []) if (c := edge.get('coverageCredit'))}
unknown, by_module, by_api = set(), Counter(), Counter()
for op in model['waits']['operations']:
    key = op['group'], str(op['tid']), int(op['regionSeq'])
    if key in calls and key not in covered:
        unknown.add(key)
        by_module[op.get('callerModule', 'unknown')] += 1
        by_api[op['api']] += 1
assert len(unknown) == event['coverage']['uncovered']
summary['uncoveredCallerAudit'] = dict(invocations=len(unknown),
    byModule=dict(by_module), byApi=dict(by_api),
    contract='API caller attribution, not producer identity or proof of blocking.')
summary['implementationSynchronization'] = model['waits'].get('implementationSynchronization', {})
inventory = json.loads((BASE / 'annotation-inventory.json').read_text())
summary['annotationInventory'] = [dict(row, observed=row.get('region') in regions)
                                  for row in inventory]
(capture / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
subprocess.run(['/home/ubuntu/.local/node22/bin/node', '-e', '''
const fs = require('fs');
const {graphHtml} = require('./extensions/vscode/graph-export');
const G = require('./extensions/vscode/media/execution-graph');
const report = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
const expanded = [...G.hierarchy(report).nodes.values()].filter(n => n.children.length).map(n => n.key);
fs.writeFileSync(process.argv[2], graphHtml(report, {mode:'top', expanded, selected:'lmc.retrieve'}));
''', str(destination), str(destination.with_suffix('.html'))], cwd=ROOT, check=True)
print(destination, len(regions), 'regions;', len(event['edges']), 'declarations;',
      event['status'], event['interfaceChecks']['status'], flush=True)
for row in event['interfaceChecks']['claims']:
    print(row['region'], row['term'], row['status'], row['present'], row['absent'])
print('coverage', {k:v for k,v in event['coverage'].items() if k != 'regions'})
