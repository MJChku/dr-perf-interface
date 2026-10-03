import csv,json
from pathlib import Path
rows=[]
for p in sorted(Path('measurements/round-01').glob('point-*/raw/*.json')):
 d=json.loads(p.read_text())
 for r in d['regions']:
  rows.append(dict(point=p.parent.parent.name,region=r['region'],parent=r['parent'],states=json.dumps(r['states'],sort_keys=True),count=r['count'],own_sum=r['self'],own_mean=r['self']/r['count'],own_min=r['self_min'],own_max=r['self_max'],inclusive_sum=r['incl']['sum'],other_threads=r['other_threads']))
with open('evidence.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved',len(rows),'observed region/state rows')
