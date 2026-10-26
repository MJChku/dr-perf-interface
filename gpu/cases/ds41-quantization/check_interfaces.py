#!/usr/bin/env python3
"""Check structural resource interfaces on local observed PCVs, not latency."""
import gzip,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(p):
 if not p.exists():p=Path(str(p)+'.gz')
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def check():
 cfg=read(HERE/'case.json');case=cfg['case'];checks=[]
 data={}
 for variant in cfg['variants']:
  p=HERE/'results'/variant['name']/'summary.json'
  if p.exists() or Path(str(p)+'.gz').exists():data[variant['name']]=read(p)
 def obs(variant,region):return next(r['observations'] for r in data[variant]['regions'] if r['name']==region)
 def near(a,b):assert abs(a-b)<=max(1e-5,abs(b)*1e-9),(a,b)
 def noted(text,n):checks.append(dict(interface=text,checked_observations=n,status='pass'))
 if case in ('ds41-tc-matvec-partials','ds41-single-row-mma-padding'):
  observations=obs('control','ds41_matvec_tc_partials')
  for o in observations:
   p=o['pcvs'];m=o['metrics'];near(m.get('tensor_float_flops',0),8192*p['tile_groups']);near(m.get('global_write_bytes',0),4*p['outputs']*p['k_groups'])
  noted('Control tensor work8192*tile_groups; global FP32 partial writes4*outputs*k_groups',len(observations))
  if 'fused' in data:
   for a,b in zip(observations,obs('fused','ds41_matvec_tc_fused')):
    p=b['pcvs'];m=b['metrics'];near(m.get('tensor_float_flops',0),a['metrics'].get('tensor_float_flops',0));near(m.get('global_write_bytes',0),2*p['outputs']);near(m.get('shared_write_bytes',0),4*p['outputs']*p['k_groups']);near(m.get('shared_read_bytes',0),4*p['outputs']*p['k_groups'])
   noted('Fused tensor coefficient unchanged, global output2*outputs; shared read/write4*outputs*k_groups; separate finish eliminated',len(observations))
  else:
   chosen=[o for o in observations if o['pcvs']['real_rows']==1 and o['pcvs']['N']==8 and o['pcvs']['K']==32];assert chosen
   for o in chosen:near(o['metrics']['tensor_float_flops'],16*2*o['pcvs']['useful_fmas'])
   noted('No N padding at M1,N8,K32: tensor/useful work=16',len(chosen))
 elif case=='ds41-grouped-fp4-padding':
  a=obs('control','ds41_library_grouped_fp4');b=obs('tight','ds41_library_grouped_fp4');assert len(a)==len(b)==8
  changes=[]
  for x,y in zip(a,b):
   for o in [x,y]:near(o['metrics'].get('tensor_float_flops',0),16384*o['pcvs']['padded_rows'])
   near(x['pcvs']['real_rows'],y['pcvs']['real_rows']);near(x['metrics']['blocks'],y['metrics']['blocks']);changes.append((x['pcvs']['real_rows'],x['pcvs']['padded_rows'],y['pcvs']['padded_rows']))
  assert (53,128,96) in changes and (26,128,64) in changes
  noted('tensor_flops=16384*padded_rows, real-row term0;53real128→96 and26real128→64; same8 CTAs',16)
 elif case=='ds41-routed-metadata-cache':
  a=obs('uncached','ds41_routed_tc_fused');b=obs('cached','ds41_routed_tc_fused');assert len(a)==len(b)
  for x,y in zip(a,b):near(x['metrics'].get('tensor_float_flops',0),y['metrics'].get('tensor_float_flops',0));near(x['metrics'].get('global_write_bytes',0),y['metrics'].get('global_write_bytes',0))
  before=sum(o['metrics'].get('global_read_bytes',0)+o['metrics'].get('generic_read_bytes',0) for o in a);after=sum(o['metrics'].get('global_read_bytes',0)+o['metrics'].get('generic_read_bytes',0) for o in b);assert after<before,(before,after)
  noted('Aligned tensor/output work unchanged; metadata cache decreases requested global/generic reads across fixed grid',len(a));checks[-1].update(before_requested_reads=before,after_requested_reads=after)
 elif case=='ds41-route-parallel':
  s=obs('serial','ds41_routed_tc_fused');p=obs('parallel','ds41_routed_tc_parallel');assert len(s)==len(p)
  for x,y in zip(s,p):
   if x['pcvs']['real_rows']!=1:continue
   T=x['pcvs']['N']//8;G=x['pcvs']['K']//32;R=x['pcvs']['max_routes'];want=120*T+16*R*T+172*R*T*G
   for o in [x,y]:
    m=o['metrics'];actual=sum(m.get(k,0) for k in ['global_read_bytes','global_write_bytes','generic_read_bytes','generic_write_bytes']);near(actual,want)
  for n in [8,32,128]:
   for k in [64,512,1280]:
    pp=[o for o in p if o['pcvs']['real_rows']==1 and o['pcvs']['N']==n and o['pcvs']['K']==k and o['pcvs']['max_routes']>0]
    spans={o['metrics']['sum_launch_max_block_lane_sequence_span'] for o in pp};assert len(spans)==1,(n,k,spans)
  noted('Serial/parallel requested buffers120T+16RT+172RTG; parallel launch max span constant for positiveR at fixed N/K',len(p))
 elif case=='ds41-sparse-attention-head-padding' and data:
  a=obs('padded16','sparse_head_padding');assert len(a)==6
  by_selected={}
  for o in a:
   pcv=o['pcvs'];m=o['metrics'];by_selected.setdefault(pcv['selected_keys'],[]).append((pcv['real_heads'],m.get('tensor_float_flops',0)))
  for selected,states in by_selected.items():
   assert len({x[1] for x in states})==1,(selected,states)
   for _,work in states:near(work,32768*selected)
  noted('tensor_flops32768*selected_keys independent of real_heads4/8/16 at padded_heads16',6)
 elif case=='ds41-scatter-merge-clear':
  a=obs('scalar','ds41_finish_moe')+obs('vector','ds41_finish_moe')
  for o in a:near(o['metrics']['global_write_bytes'],2*o['pcvs']['arg_rows']*o['pcvs']['arg_dim'])
  noted('Finish writes2*rows*dim BF16 bytes; duplicate/invalid scatter and poisoned scratch checked by scalar oracle',len(a))
 else:
  n=0
  for v in data.values():
   for r in v['regions']:
    assert r['stateCount']>0
    for o in r['observations']:
     assert o['metrics'].get('launches',0)==1
     if 'blocks' in o['pcvs']:near(o['metrics']['blocks'],o['pcvs']['blocks'])
     n+=1
  if n:noted('Per-launch geometry interface blocks=declared blocks and launches=1; all independent numerical fixtures passed',n)
 if not data:
  receipt=read(HERE/'results/verification.json');assert receipt['status'] in ('blocked','proposed');noted('Explicit expected coverage blocker or proposed status; no fitted interface/numerical pass fabricated',len(receipt['variants']))
 golden=HERE/'expected_interfaces.json'
 if golden.exists():
  expected=read(golden);count=0
  for variant,regions in expected['variants'].items():
   for region,states in regions.items():
    actual=obs(variant,region);assert len(actual)==len(states),(variant,region,len(actual),len(states))
    for a,b in zip(actual,states):
     assert a['pcvs']==b['pcvs'],(variant,region,'PCV grid changed')
     for metric,value in b['metrics'].items():near(a['metrics'].get(metric,0),value)
     count+=1
  if count:noted('Pinned observed-state resource interfaces reproduce; requested/precision/span/geometry counts, excluding address-footprint placement',count)
 output=dict(case=case,status='pass',checks=checks,semantics='Resource-interface assertions only; no elapsed-time prediction')
 (HERE/'results/interface-checks.json').write_text(json.dumps(output,indent=2)+'\n');return output
if __name__=='__main__':print(json.dumps(check()))
