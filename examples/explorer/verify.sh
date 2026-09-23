#!/bin/sh
# Reproduce the proposed-interface -> changed-program -> refined-interface loop.
set -eu
cd "$(dirname "$0")/../.."
OUT=${DRPERF_EXPLORER_VALIDATION_OUT:-out/explorer-validation}
for CASE in baseline changed refined refined-changed; do
    SCALE=1
    REFINED=0
    case "$CASE" in *changed) SCALE=2;; esac
    case "$CASE" in refined*) REFINED=1;; esac
    DRPERF_EXPLORER_OUT="$OUT/$CASE" DRPERF_EXPLORER_SCALE="$SCALE" \
        DRPERF_EXPLORER_REFINED="$REFINED" examples/explorer/run.sh
done
DRPERF_EXPLORER_OUT="$OUT/expansion-changed" DRPERF_EXPLORER_REFINED=1 \
    DRPERF_EXPLORER_TOKEN_MULTIPLIER=4 examples/explorer/run.sh
DRPERF_EXPLORER_OUT="$OUT/joint-changed" DRPERF_EXPLORER_REFINED=1 \
    DRPERF_EXPLORER_SCALE=2 DRPERF_EXPLORER_TOKEN_MULTIPLIER=3 examples/explorer/run.sh
bin/drperf-explore "$OUT/baseline/pipeline.drperf.json" \
    --edit enqueue items scale 2 --assume-first \
    --validate "$OUT/changed/pipeline.drperf.json" -o "$OUT/initial-scenario.json"
bin/drperf-explore "$OUT/refined/pipeline.drperf.json" \
    --edit enqueue items scale 2 --assume-first \
    --propose lookup pairs 'last("dequeue", "items") ** 2' \
    --validate "$OUT/refined-changed/pipeline.drperf.json" -o "$OUT/refined-scenario.json"
bin/drperf-explore "$OUT/refined/pipeline.drperf.json" \
    --edit decode tokens scale 2 --assume-first --audit-alternatives \
    --propose lookup pairs 'last("dequeue", "items") ** 2' \
    --validate "$OUT/expansion-changed/pipeline.drperf.json" -o "$OUT/expansion-first.json"
bin/drperf-explore "$OUT/refined/pipeline.drperf.json" \
    --edit decode tokens scale 2 --assume-first --audit-alternatives \
    --propose lookup pairs 'last("dequeue", "items") ** 2' \
    --propose dispatch items 'last("dequeue", "items")' \
    --validate "$OUT/expansion-changed/pipeline.drperf.json" -o "$OUT/expansion-checked.json"
bin/drperf-explore "$OUT/refined/pipeline.drperf.json" \
    --edit enqueue items scale 2 --edit decode tokens scale 1.5 --assume-first \
    --propose lookup pairs 'last("dequeue", "items") ** 2' \
    --propose dispatch items 'last("dequeue", "items")' \
    --validate "$OUT/joint-changed/pipeline.drperf.json" -o "$OUT/joint-scenario.json"
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
for name in ('initial','refined'):
    result=json.loads((root/(name+'-scenario.json')).read_text())['validation']
    assert result['structureMatches'], result
    print(name+' interface:')
    for row in result['regions']:
        print('  {region}: {stateMatches}/{calls} states, {costChecks}/{calls} cost checks, error={relativeAbsoluteError}'.format(**row))
        if row['region'] in ('enqueue','dequeue','decode','copy') or (name=='refined' and row['region']=='lookup'):
            assert row['stateMatches']==row['calls'] and row['costChecks']==row['calls'],row
            assert (row['relativeAbsoluteError']<.01 if row['region']=='enqueue' else row['meanAbsoluteError']<1e-5),row
first=json.loads((root/'expansion-first.json').read_text())['validation']
checked=json.loads((root/'expansion-checked.json').read_text())['validation']
assert next(r for r in first['regions'] if r['region']=='dispatch')['stateMatches']==0
assert all(r['stateMatches']==r['calls'] for r in checked['regions'])
joint=json.loads((root/'joint-scenario.json').read_text())['validation']
assert joint['structureMatches'] and all(r['stateMatches']==r['calls'] for r in joint['regions']),joint
for row in joint['regions']:
    if row['region'] not in ('startup','dispatch'):
        assert row['costChecks']==row['calls'],row
        assert (row['relativeAbsoluteError']<.01 if row['region']=='enqueue' else row['meanAbsoluteError']<1e-5),row
print('Expansion intervention: first dispatch relationship fails 24/24 calls; revised relationship matches all calls.')
print('Joint producer/expansion intervention: every PCV matches; all non-branch affine costs match within tolerance.')
print('PASS: propagated states and affine regions match the changed program; refined lookup matches every call.')
PY

# A changed call structure must not be presented as a validated fixed-trace
# counterfactual. Shared-state interface comparison remains useful, however.
gcc -O2 examples/explorer/call_structure.c -Lbuild -lperfmark \
    -Wl,-rpath,"$PWD/build" -o "$OUT/call-structure"
PYTHONPATH="$PWD/lib" python3 - "$OUT" <<'PY'
import os, pathlib, sys, tempfile
import explorer, runner
root=pathlib.Path(sys.argv[1]).resolve()
os.environ['DRPERF_FOLLOW_THREADS']='0'
for scale, name in ((1,'call-structure-before'),(2,'call-structure-after')):
    with tempfile.TemporaryDirectory(prefix='.structure-raw-', dir=root) as raw:
        rc, log, files=runner.run([str(root/'call-structure'),str(scale)],raw,timeout=60)
        if rc or not files: raise RuntimeError(log)
        model=explorer.build_model(raw,source_root='.',source_paths=['examples/explorer/call_structure.c'])
        explorer.write_model(model,root/(name+'.drperf.json'))
PY
bin/drperf-explore "$OUT/call-structure-before.drperf.json" --recorded \
    --edit batch items scale 2 --validate "$OUT/call-structure-after.drperf.json" \
    -o "$OUT/call-structure-scenario.json"
bin/drperf-explore "$OUT/call-structure-before.drperf.json" --recorded \
    --compare "$OUT/call-structure-after.drperf.json" -o "$OUT/call-structure-comparison.json"
bin/drperf-explore "$OUT/call-structure-before.drperf.json" \
    --propose batch items 'count("batch") + 1' \
    --check-relations "$OUT/call-structure-after.drperf.json" -o "$OUT/call-structure-relations.json"
python3 - "$OUT" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
scenario=json.loads((root/'call-structure-scenario.json').read_text())
assert not scenario['validation']['structureMatches'],scenario
assert not scenario['validation']['regions'],scenario
comparison=json.loads((root/'call-structure-comparison.json').read_text())
item=next(r for r in comparison['regions'] if r['region']=='item')
assert item['pairedStates']>=5 and abs(item['deltaMean'])<1e-5,item
relations=json.loads((root/'call-structure-relations.json').read_text())
assert relations['summary']['fails']==1,relations
assert relations['checks'][0]['measured']['mismatches']==8,relations
print('PASS: added calls reject fixed-trace validation; item costs match at shared PCV states.')
print('PASS: relationship checking still runs on the changed call structure and rejects the old batch-count equation.')
PY
