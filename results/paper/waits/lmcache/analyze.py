"""Refresh wait analysis from complete captures and emit reviewable evidence."""
import ast
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'lib'))
import composition
import event_model
import execution
import explorer
import waits

BASE = Path(__file__).resolve().parent
arm = sys.argv[1]
path = BASE / (arm + '.drperf.json')
if path.exists():
    m = explorer.load_model(path, wait_evidence=True)
else:
    with gzip.open(str(path)+'.gz', 'rt') as f:
        m = json.load(f)
raw = BASE / arm / 'raw'
if '--existing' not in sys.argv:
    m['waits'] = waits.build(explorer.load_raw_runs(raw), m['regions'], m['trace']['events'],
                            m['validity']['errors'] + m['validity']['traceErrors'])
declared = event_model.check(m)
if declared:
    m['eventModel'] = declared
source = (BASE / 'overlay' if arm != 'baseline' else
          Path('/home/ubuntu/compression/ditto_kv/example/qwen2.5B/lmcache/lmcache_overlay'))
m['provenance']['sourceRoot'] = str(source)
locations = explorer.source_locations(source, [r['id'] for r in m['regions']])
for r in m['regions']:
    r['sources'] = locations.get(r['id'], [])
# Operation regions execute connector coroutines one active step at a time.
# Their name is selected dynamically; map them to the actual source method.
connector = source / 'lmcache/v1/storage_backend/connector/lm_connector.py'
raw_source = connector.read_bytes()
for node in ast.walk(ast.parse(raw_source)):
    if isinstance(node, ast.AsyncFunctionDef) and node.name in ('exists', 'get', 'put'):
        name = 'lmc.remote.' + node.name + '_operation'
        for r in m['regions']:
            if r['id'] == name:
                r['sources'] = [dict(path=str(connector.relative_to(source)),
                    line=node.lineno, endLine=node.end_lineno,
                    sha256=hashlib.sha256(raw_source).hexdigest(), expressions={})]
if '--existing' not in sys.argv:
    m['executionGraph'] = execution.build(m)
    m = explorer.portable(m)
    m['id'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    explorer.write_model(m, path)

traces = m['trace']['events']
by_region = collections.defaultdict(list)
for t in traces:
    by_region[t['region']].append(t)
regions = {r['id']:r for r in m['regions']}
def counts_in(region, target, label):
    calls = by_region[region]
    values = [sum(t['group'] == x['group'] and str(t['thread']) == str(x.get('thread', x.get('tid')))
                  and int(t['seq']) <= int(x.get('seq', x.get('regionSeq'))) < int(t['end'])
                  for x in target) for t in calls]
    states = regions[region]['states']
    fit = composition.affine([[int(t['values'][n]) for n in states] for t in calls], values, states)
    return dict(region=region, target=label, calls=len(calls), occurrences=sum(values), states=states,
                countFormula=fit, text=composition.affine_text(fit, states) if fit else 'unexplained',
                observations=[dict(values=t['values'], count=v) for t,v in zip(calls,values)])

ops = m['waits']['operations']
sync = [o for o in ops if o['kind']=='stream_wait']
facts = []
for name in ['lmc.gpu.from_gpu', 'lmc.gpu.to_gpu']:
    facts.append(counts_in(name, sync, 'cudaStreamSynchronize'))
for parent, child in [('lmc.remote.get','lmc.remote.wait_get'),
                      ('lmc.remote.contains','lmc.remote.wait_exists')]:
    if child in regions:
        facts.append(counts_in(parent, by_region[child], child))
    elif parent in regions:
        event_id = '2002' if parent == 'lmc.remote.get' else '2001'
        completed = [e for e in m['waits']['events']
                     if e['kind'] == 'declared_waited' and str(e['object']) == event_id]
        facts.append(counts_in(parent, completed, 'event_waited(' + event_id + ')'))
if 'lmc.remote.put_callback' in regions:
    completed = [e for e in m['waits']['events']
                 if e['kind'] == 'declared_waited' and str(e['object']) == '2003']
    facts.append(counts_in('lmc.remote.put_callback', completed, 'event_waited(2003)'))
for parent in ['lmc.vllm.start_load_kv','lmc.vllm.wait_for_save','lmc.vllm.build_connector_meta']:
    if parent in regions:
        facts.append(counts_in(parent, sync, 'cudaStreamSynchronize'))

summary = dict(arm=arm, validity=m['validity'], traceComplete=m['trace']['complete'],
    regionCount=len(regions), waitStatus=m['waits']['status'], warnings=m['waits']['warnings'],
    pendingCalls=m['waits']['pendingCalls'], nativeWaits=m['waits']['regions'],
    eventChecks=None if not declared else dict(status=declared['status'],probe=declared['probe'],
        edges=dict(collections.Counter((e['consumer']+' -> '+str(e['producer'])) for e in declared['edges'])),
        violations=declared['violations'],unverified=declared['unverified']),
    graphWaits=[e for e in m['executionGraph']['edges'] if e['kind']=='wait'], counts=facts,
    instructions=[dict(region=r['id'], calls=r['calls'], states=r['states'],
                      ownMean=round(sum(float(p['observed'])*p['calls'] for p in r['points'])/r['calls']),
                      regimes=[{k:f[k] for k in ['coefficients','constant','unexplainedShare']}
                               for f in r['regimes']]) for r in m['regions']])
(BASE/(arm+'-analysis.json')).write_text(json.dumps(summary,indent=2)+'\n')
print(arm, 'validity', summary['validity'], 'waits', summary['waitStatus'], flush=True)
print('event checks', summary['eventChecks'], flush=True)
for f in facts:
    print(f['region'], f['target'], f['text'], f['occurrences'], 'in',f['calls'], 'calls', flush=True)
if declared and not declared['probe']:
    (BASE/'probe-plan.json').write_text(json.dumps(event_model.probe_plan(declared),indent=2)+'\n')
