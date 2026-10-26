#!/usr/bin/env python3
"""Check matched compact baseline/candidate aggregates without fitting a new model."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent/'results'
def main():
 out={'assertions':['baseline global writes =4*pool_updates','candidate global writes =4*positive_updates','tensor work unchanged in every matched state'],'states':{}}
 for suite in ['train','heldout']:
  a=json.loads((P/f'after-{suite}.json').read_text());b=json.loads((P/f'compatible-{suite}.json').read_text());lk={tuple(sorted(s['pcvs'].items())):s for s in b['states']};rows=[]
  for s in a['states']:
   v=s['pcvs'];am=s['metrics'];bm=lk[tuple(sorted(v.items()))]['metrics'];base=lambda m:bm.get(m,{'mean':0})['mean'];after=lambda m:am.get(m,{'mean':0})['mean']
   assert base('global_write_bytes')==4*v['pool_updates'];assert after('global_write_bytes')==4*v['positive_updates'];assert base('tensor_float_flops')==after('tensor_float_flops')
   rows.append({'pcvs':v,'baseline_read_bytes':base('global_read_bytes'),'after_read_bytes':after('global_read_bytes'),'baseline_write_bytes':base('global_write_bytes'),'after_write_bytes':after('global_write_bytes'),'extra_nonatomic_read_bytes':after('global_read_bytes')-base('global_read_bytes')+4*(v['pool_updates']-v['positive_updates']),'baseline_fp32_flops':base('fp32_flops'),'after_fp32_flops':after('fp32_flops')})
  out['states'][suite]=rows
 (P/'matched-resource-checks.json').write_text(json.dumps(out,indent=2)+'\n')
 print('All matched resource assertions passed.')
if __name__=='__main__':main()
