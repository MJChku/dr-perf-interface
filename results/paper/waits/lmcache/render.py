"""Small standalone graph snapshot; raw evidence stays in the capture archive."""
import json
import gzip
from pathlib import Path
import subprocess
import sys

base=Path(__file__).resolve().parent
arm=sys.argv[1] if len(sys.argv)>1 else 'futures'
path=base/(arm+'.drperf.json')
with (path.open() if path.exists() else gzip.open(str(path)+'.gz','rt')) as f:
    m=json.load(f)
w=m['waits']
ops=[o for o in w['operations'] if o['region'] and o['producers']]
keep={o['id'] for o in ops} | {p for o in ops for p in o['producers']}
keep.update(e['id'] for e in w['events'] if e['kind'].startswith('declared_'))
keep.update(e['id'] for e in w.get('pendingCalls',[]))
m['provenance']['visualizationSnapshot']={
    'purpose':'Display existing checked graph and formulas; not a replacement for raw evidence.',
    'omittedNativeDetails':w.get('evidence', {}).get('eventCount', len(w['events']))-len(keep),
    'rawDirectory':m['provenance']['rawDirectory'],
    'retained':'All region traces, costs, wait summaries, declared checkpoints, and resolved in-region native edges.'}
w['events']=[e for e in w['events'] if e['id'] in keep]
w['operations']=ops
m['title']='LMCache: checked wait relationships and instruction counts'
m['id']+='-graph-snapshot'
temp=base/'graph-snapshot.json'
temp.write_text(json.dumps(m,separators=(',',':')))
try:
    subprocess.run(['node','-e','''
const fs=require('fs');
const {graphHtml}=require('./extensions/vscode/graph-export');
const G=require('./extensions/vscode/media/execution-graph');
const m=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const tree=G.hierarchy(m);
const expanded=[...tree.nodes.values()].filter(n=>n.children.length).map(n=>n.key);
fs.writeFileSync(process.argv[2],graphHtml(m,{mode:'top',selected:'lmc.remote.get',expanded}));
console.log(m.regions.length+' regions; '+G.waits(G.build(m)).edges.length+' wait edges');
''',str(temp),str(base/'lmcache-waits.html')],check=True,cwd=base.parents[2])
finally:
    temp.unlink()
