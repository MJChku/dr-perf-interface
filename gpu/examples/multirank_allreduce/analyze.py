#!/usr/bin/env python3
"""Recover polling loops from emitted C++, retain retries, fit compulsory work."""
import argparse,collections,gzip,hashlib,json,pathlib,re,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import gpu_backend
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();run=a.run.resolve()
def load(path):
 return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())
def dump(name,obj):
 data=(json.dumps(obj,indent=2)+'\n').encode();(run/name).write_bytes(gzip.compress(data,mtime=0) if name.endswith('.gz') else data)
def additive(m):return {k:v for k,v in m.items() if 'unique_' not in k}
def add(out,m):
 for k,v in m.items():out[k]+=v
# Site IDs are the __LINE__ at the original PTX counter. No source instrumentation,
# replacement load, removed wait, or patched generated C++ is used.
lines=(run/'kernels.so.cpp').read_text().splitlines();fn=None;start=0;functions={};ops={}
for i,line in enumerate(lines,1):
 m=re.match(r'extern "C" void (\w+__gxptx)\(void\*\* args\) \{',line)
 if m:
  if fn:functions[fn]=(start,i-1)
  fn=m[1];start=i
 if fn:
  op=re.search(r'drperf_gpu_instruction\(__func__, "([^"]+)"',line)
  if op:ops[fn+':cpp'+str(i)+':'+op[1]]=(i,op[1])
if fn:functions[fn]=(start,len(lines))
loops={};site_groups={}
for fn,(start,end) in functions.items():
 labels={};candidates=[]
 for i in range(start,end+1):
  line=lines[i-1];m=re.match(r'\s*(L_\w+): ;',line)
  if m:labels[m[1]]=i
  m=re.search(r'goto (L_\w+);',line)
  if m and m[1] in labels:
   first=labels[m[1]];body='\n'.join(lines[first-1:i])
   if '"ld.volatile' in body:candidates.append((first,i))
 # The innermost backward interval containing the load is precisely its retry
 # loop. Outer payload/clear loops also contain it, so discard those supersets.
 intervals=[x for x in candidates if not any(x[0]<=y[0] and y[1]<=x[1] and x!=y for y in candidates)]
 assert len(intervals)==2,(fn,intervals)
 groups=[]
 for first,last in intervals:
  group='vector_poll' if '"ld.volatile.global.v4.' in '\n'.join(lines[first-1:last]) else 'counter_poll'
  keys=[key for key,(i,op) in ops.items() if key.startswith(fn+':') and first<=i<=last]
  assert keys and all(op not in ['ret','st.u32'] for key in keys for i,op in [ops[key]])
  groups.append({'group':group,'cpp_first_line':first,'cpp_last_line':last,'sites':keys})
  for key in keys:site_groups[key]=group
 loops[fn]=groups
dump('poll-site-groups.json',{'generated_cpp_sha256':hashlib.sha256((run/'kernels.so.cpp').read_bytes()).hexdigest(),'loops':loops})
raw_rows=[];source_profiles={}
for label in ['r2','r2-skew','r2-cluster','r4','r4-cluster']:
 path=run/label/'profile.gpu.drperf.json'
 if not path.exists():path=path.with_suffix(path.suffix+'.gz')
 model=load(path);assert model['validity']=='complete'
 source_profiles[label]={'profile':str(path.relative_to(run)),'invocations':sum(r['invocations'] for r in model['regions']),
  'rejected_metric_fits':{r['name']:[k for k,v in r['fits'].items() if not v['accepted']] for r in model['regions']}}
 for region in model['regions']:
  for row in region['observations']:row=dict(row);row['capture_label']=label;raw_rows.append(row)
normalized=[];spins=[];checks=[];evidence=collections.defaultdict(list)
for row in raw_rows:
 pc=row['pcvs'];n,c,ranks=pc['N'],pc['C'],pc['ranks'];assert 32<=pc['hidden']<=512 and 1<=pc['w']<=4
 mode=row['region'].split('.')[0];counts=collections.defaultdict(int);polls=collections.defaultdict(int);mandatory=collections.defaultdict(int);payload=collections.defaultdict(int)
 round_counts={};normalized_sites={}
 for group in loops[row['launches'][0]['kernel']+'__gxptx']:
  if group['group']=='vector_poll':
   vector_keys=[k for k in group['sites'] if ops[k][1].startswith('ld.volatile.global.v4.')]
   assert len(vector_keys)==ranks
   rounds=[row['sites'][k]['ptx_lane_instructions'] for k in vector_keys];assert len(set(rounds))==1
   assert rounds[0]>=n//8;round_counts['vector_poll']=rounds[0]
  else:
   key=next(k for k in group['sites'] if ops[k][1].startswith('ld.volatile'))
   round_counts['counter_poll']=row['sites'][key]['ptx_lane_instructions'];assert round_counts['counter_poll']>=1
 for site,m in row['sites'].items():
  am=additive(m);group=site_groups.get(site)
  if group:
   add(polls,am);baseline=n//8 if group=='vector_poll' else 1
   # NVCC's retry backedge is taken only on unsuccessful rounds.
   observed_rounds=round_counts[group]
   expect=observed_rounds-baseline if ops[site][1]=='bra' else observed_rounds
   assert am['ptx_lane_instructions']==expect,(site,am['ptx_lane_instructions'],expect)
   base=0 if ops[site][1]=='bra' else baseline
   norm={k:(v//am['ptx_lane_instructions'])*base for k,v in am.items()}
   assert all(v%am['ptx_lane_instructions']==0 for v in am.values())
   add(mandatory,norm)
  else:norm=am
  normalized_sites[site]=norm;add(counts,norm)
  op=site.rsplit(':',1)[1]
  if '.v4.' in op:
   payload['read_bytes']+=norm.get('global_read_bytes',0)+norm.get('generic_read_bytes',0)
   payload['write_bytes']+=norm.get('global_write_bytes',0)+norm.get('generic_write_bytes',0)
 payload['bytes']=payload['read_bytes']+payload['write_bytes']
 reads=counts['global_read_bytes']+counts['generic_read_bytes'];writes=counts['global_write_bytes']+counts['generic_write_bytes']
 expected_read=(2 if mode=='allreduce' else 6)*n+2*ranks*n
 expected_write=(2 if mode=='allreduce' else 4)*n+2*ranks*n+2*c
 assert payload['read_bytes']==expected_read,(row['region'],pc,payload,expected_read)
 assert payload['write_bytes']==expected_write,(row['region'],pc,payload,expected_write)
 control_read=(8*ranks+28)*(n//8)+4*pc['blocks']+4
 control_write=4*pc['blocks']+12
 assert reads-payload['read_bytes']==control_read,(row['region'],pc,reads,payload,control_read)
 assert writes-payload['write_bytes']==control_write
 retry_bytes=sum(polls[k]-mandatory[k] for k in ['global_read_bytes','generic_read_bytes'])
 raw_bytes=sum(row['metrics'].get(k,0) for k in ['global_read_bytes','generic_read_bytes','global_write_bytes','generic_write_bytes'])
 assert raw_bytes==payload['bytes']+control_read+control_write+retry_bytes
 metrics={'payload_read_bytes':payload['read_bytes'],'payload_write_bytes':payload['write_bytes'],'payload_bytes':payload['bytes'],
  'control_read_bytes':control_read,'control_write_bytes':control_write,'compulsory_requested_bytes':reads+writes,
  **{k:counts[k] for k in ['fp32_flops','shared_read_bytes','shared_write_bytes','special_function_ops','ptx_lane_instructions','integer_ops','conversion_ops','param_read_bytes','local_read_bytes','local_write_bytes','block_barrier_lane_arrivals','collective_lane_arrivals']}}
 # Preserve the original count categories in the normalized observations.
 normrow={'region':row['region'].rsplit('.r',1)[0], 'pcvs':dict(pc,lanes=n//8,block_warps=pc['blocks']*((pc['threads']+31)//32)), 'metrics':metrics,'sites':{},'scope':'host',
  'capture_label':row['capture_label'],'original_token':row['token']}
 normalized.append(normrow)
 spin={'region':row['region'],'capture_label':row['capture_label'],'token':row['token'],'pcvs':pc,
  'vector_poll_rounds':round_counts['vector_poll'],'vector_retry_rounds':round_counts['vector_poll']-n//8,
  'counter_polls':round_counts['counter_poll'],'counter_retry_polls':round_counts['counter_poll']-1,
  'retry_read_bytes':retry_bytes,'poll_metrics_including_success':dict(polls),'mandatory_poll_metrics':dict(mandatory),
  'retry_metrics':{k:polls[k]-mandatory[k] for k in polls},'raw_requested_bytes':raw_bytes}
 spins.append(spin);evidence[(normrow['region'],ranks)].append(normrow)
model=gpu_backend.build_model(normalized,absolute=1e-7,relative=1e-10)
dump('compulsory-profile.json.gz',model);dump('spin-observations.json.gz',spins)
# Fit a deliberately insufficient current-size-only model, excluding C.
wrong=[]
for row in normalized:
 if row['region']=='residual.launcher':wrong.append(dict(row,pcvs={k:row['pcvs'][k] for k in ['N','hidden','w','ranks']}))
wrongfit=gpu_backend.build_model(wrong)['regions'][0]['fits']['payload_bytes'];assert not wrongfit['accepted']
# Learn interfaces from executed small states, rather than only checking the
# source formulas above. Fixed-rank subsets make the familiar N coefficient
# directly readable; a combined fit additionally learns ranks*N.
contracts={}
for (name,ranks),rows in sorted(evidence.items()):
 narrow=[dict(row,pcvs={k:row['pcvs'][k] for k in ['N','C','blocks']}) for row in rows]
 fitted=gpu_backend.build_model(narrow)['regions'][0]['fits']
 for metric in ['payload_read_bytes','payload_write_bytes','payload_bytes','control_read_bytes','control_write_bytes','compulsory_requested_bytes']:
  assert fitted[metric]['accepted'],(name,ranks,metric,fitted[metric])
 contracts[name+'.r'+str(ranks)]={metric:fitted[metric] for metric in ['payload_read_bytes','payload_write_bytes','payload_bytes','control_read_bytes','control_write_bytes','compulsory_requested_bytes','fp32_flops','special_function_ops']}
# Report same-state timing variation without explaining it away with shape PCVs.
states=collections.defaultdict(list)
for spin in spins:
 states[(spin['region'],tuple(sorted(spin['pcvs'].items())))].append(spin['retry_read_bytes'])
variation=[{'region':key[0],'pcvs':dict(key[1]),'observations':len(values),'min_retry_bytes':min(values),'max_retry_bytes':max(values)} for key,values in states.items() if min(values)!=max(values)]
scenario_checks=[]
for label in source_profiles:
 for line in (run/(label+'.stdout.log')).read_text().splitlines():
  if line.startswith('{'):scenario_checks.append(json.loads(line))
# Held-out cases never enter either fitting matrix. Reconstruct compulsory
# payload/control from raw sites and predict with the learned fixed-rank fits.
heldout=[];extra_scenarios=[]
for directory in sorted(run.iterdir()):
 if not directory.is_dir() or not directory.name.startswith(('heldout-','launcher-cluster-')):continue
 path=directory/'profile.gpu.drperf.json.gz'
 m=load(path);assert m['validity']=='complete'
 for region in m['regions']:
  for row in region['observations']:
   pc=row['pcvs'];n=pc['N'];ranks=pc['ranks'];pr=pw=vr=cr=0
   for site,metrics in row['sites'].items():
    op=site.rsplit(':',1)[1]
    if '.v4.' in op:
     pr+=metrics.get('global_read_bytes',0)+metrics.get('generic_read_bytes',0)
     pw+=metrics.get('global_write_bytes',0)+metrics.get('generic_write_bytes',0)
    if op.startswith('ld.volatile.global.v4.'):vr+=metrics['global_read_bytes']
    elif op.startswith('ld.volatile'):cr+=metrics.get('generic_read_bytes',0)+metrics.get('global_read_bytes',0)
   assert vr>=2*ranks*n and cr>=4
   raw_bytes=sum(row['metrics'].get(k,0) for k in ['global_read_bytes','generic_read_bytes','global_write_bytes','generic_write_bytes'])
   observed={'payload_read_bytes':pr-vr+2*ranks*n,'payload_write_bytes':pw,'payload_bytes':pr-vr+2*ranks*n+pw,
    'compulsory_requested_bytes':raw_bytes-(vr-2*ranks*n)-(cr-4)}
   fitted=contracts[region['name']]
   for metric,value in observed.items():
    f=fitted[metric];prediction=f['constant']+sum(v*pc[k] for k,v in f['coefficients'].items());err=abs(value-prediction)
    assert err<1e-6,(directory.name,metric,pc,prediction,value)
    heldout.append({'capture':directory.name,'region':region['name'],'pcvs':pc,'metric':metric,'observed':value,'predicted':prediction,'absolute_error':err})
 for line in (run/(directory.name+'.stdout.log')).read_text().splitlines():
  if line.startswith('{'):extra_scenarios.append(json.loads(line))
dump('heldout-checks.json.gz',heldout)
visibility_lines=(run/'visibility.stdout.log').read_text().splitlines() if (run/'visibility.stdout.log').exists() else []
visibility=[json.loads(x) for x in visibility_lines if x.startswith('{')]
assert not visibility or visibility[0]['status']=='pass'
summary={'status':'pass','physical_gpu_seconds':0,'small_state_rank_launches':len(raw_rows),'collective_calls':sum(x['calls'] for x in scenario_checks),
 'checked_elements':sum(x['checked_elements'] for x in scenario_checks),'norm_ulp1':sum(x['norm_ulp1'] for x in scenario_checks),
 'widths':[32,64,128,256,512],'rows':[1,2,3,4],'ranks':[2,4],'clusters':[1,2],
 'consecutive_rows':[1,4,2,1,3,4,1,2,4,3,1,4],
 'heldout':{'checks':len(heldout),'max_absolute_error':max((x['absolute_error'] for x in heldout),default=0),'intermediate_widths':[96,192,384],'actual_launcher_cluster_width':2048,'scenarios':extra_scenarios},
 'visibility_probe':visibility,
 'source_payloads_at_ranks2':{'allreduce':'12*N + 2*C','residual':'18*N + 2*C','norm':'18*N + 2*C'},
 'learned_interfaces':contracts,'current_shape_only_payload_fit':wrongfit,
 'spin':{'vector_retry_rounds':sum(x['vector_retry_rounds'] for x in spins),'counter_retry_polls':sum(x['counter_retry_polls'] for x in spins),
  'retry_read_bytes':sum(x['retry_read_bytes'] for x in spins),'same_state_variation_groups':len(variation),'same_state_variation_examples':variation[:8],
  'max_vector_rounds_per_launch':max(x['vector_poll_rounds'] for x in spins),'max_counter_polls_per_launch':max(x['counter_polls'] for x in spins)},
 'raw_profiles':source_profiles,'numerical_scenarios':scenario_checks,
 'normalized_rejected_fits':{r['name']:[k for k,f in r['fits'].items() if not f['accepted']] for r in model['regions']},
 'normalization':'Only retry-loop work beyond one successful vector round per N/8 and one successful counter read is removed. Original raw reports retain every iteration. Payload uses executed 16-byte vector load/store sites; scalar workspace/flag traffic stays in control metrics.'}
dump('summary.json',summary)
print(json.dumps({k:summary[k] for k in ['status','small_state_rank_launches','collective_calls','checked_elements','norm_ulp1','spin']}))
