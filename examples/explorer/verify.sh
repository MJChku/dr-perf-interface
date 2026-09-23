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
print('Expansion intervention: first dispatch relationship fails 24/24 calls; revised relationship matches all calls.')
print('PASS: propagated states and affine regions match the changed program; refined lookup matches every call.')
PY
