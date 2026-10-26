#!/usr/bin/env python3
"""Verify that the expected cancellation case is the sole numerical rejection."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def main():
 x=json.loads((P/'results/verification.json').read_text())
 assert all(hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in x['source_hashes'].items())
 checked=0
 for key,r in x['runs'].items():
  assert r['exit_code']==0 and r['resource_assertions']=='passed'
  lines=[json.loads(l) for l in r['stdout'].splitlines() if l.startswith('{')]
  for c in lines[:-1]:
   reject=key=='after-heldout' and c['N']==128 and c['pattern']==3
   assert c['failures']==(128*128 if reject else 0),(key,c)
   if reject:assert abs(c['max_error']-3.814697265625e-6)<1e-11
   checked+=1
  z=json.loads((P/f'results/{key}.json').read_text());assert z['validity']=='complete'
  for s in z['states']:
   v=s['pcvs'];m=s['metrics'];f=lambda k:m[k]['mean']
   assert f('tensor_float_flops')==(3 if key.startswith('after') else 4)*2*128*128*16*v['tile_products']
   assert f('global_read_bytes')==8*v['input_elements_per_tile_grid']
   assert f('global_write_bytes')==4*v['outputs']
 assert checked==16
 out={'decision':'reject candidate','numerical_cases_checked':checked,'baseline_passed':8,'candidate_passed':7,'candidate_rejected':1,'sole_rejected_state':{'N':128,'pattern':3,'failed_outputs':16384,'max_absolute_error':3.814697265625e-6},'all_four_resource_extracts_complete':True,'source_hashes_match':True}
 (P/'results/rejection.json').write_text(json.dumps(out,indent=2)+'\n')
 print('Expected cancellation rejection, all other numerical cases, guards and resource evidence verified.')
if __name__=='__main__':main()
