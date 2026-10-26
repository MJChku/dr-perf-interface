#!/usr/bin/env python3
"""Fit only small states; verify formulas at held-out 2048 and 100000 histories."""
import argparse, gzip, hashlib, json, pathlib, sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import gpu_backend
p=argparse.ArgumentParser();p.add_argument('--profile',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
model=json.loads(a.profile.read_text());assert model['validity']=='complete'
regions={r['name']:r for r in model['regions']}
cases=[(32,16),(64,16),(96,32),(128,16),(128,32),(256,32),(512,64),(2048,2048),(100000,2048)]
train=[row for r in model['regions'] for row in r['observations'][:7]]
trained=gpu_backend.build_model(train)
fits={r['name']:r for r in trained['regions']}
assert not fits['history-pack-initial']['fits']['global_write_bytes']['accepted']
assert not fits['selected-softmax-initial']['fits']['global_write_bytes']['accepted']
contracts={'history-pack-refined':{'global_read_bytes':{'history':12},'global_write_bytes':{'history':12}},
 'selected-softmax-refined':{'global_write_bytes':{'history':2,'selected':2},'global_read_bytes':{'selected':26},'special_function_ops':{'selected':1}},
 'selected-softmax-compact':{'global_write_bytes':{'selected':2},'special_function_ops':{'selected':1}}}
verified=[]
for name,metrics in contracts.items():
 r=regions[name]
 assert len(r['observations'])==len(cases)
 for metric,coefs in metrics.items():
  f=fits[name]['fits'][metric];assert f['accepted'] and f['constant']==0,(name,metric,f)
  for k,v in coefs.items():assert f['coefficients'][k]==v,(name,metric,f)
  for row in r['observations'][7:]:
   expected=f['constant']+sum(v*row['pcvs'][k] for k,v in f['coefficients'].items())
   observed=row['metrics'].get(metric,0);assert expected==observed,(name,metric,expected,observed)
   verified.append({'region':name,'metric':metric,'pcvs':row['pcvs'],'predicted':expected,'observed':observed,'absolute_error':abs(expected-observed)})
points=[]
for i,(t,k) in enumerate(cases):
 pack=regions['history-pack-refined']['observations'][i]['metrics'];sparse=regions['selected-softmax-refined']['observations'][i]['metrics'];compact=regions['selected-softmax-compact']['observations'][i]['metrics']
 assert pack['global_write_bytes']==12*t and pack['global_read_bytes']==12*t
 assert sparse['global_write_bytes']==2*(t+k) and compact['global_write_bytes']==2*k
 points.append({'history':t,'selected':k,'materialize_read_bytes':pack['global_read_bytes'],'materialize_write_bytes':pack['global_write_bytes'],'sparse_write_bytes':sparse['global_write_bytes'],'compact_write_bytes':compact['global_write_bytes'],'sparse_fp32_flops':sparse['fp32_flops'],'compact_fp32_flops':compact['fp32_flops'],'sparse_special_ops':sparse['special_function_ops'],'held_out':i>=7})
summary={'scope':'Standalone PTX resource-count experiments on frozen GLM kernels and a compact-output demonstration; not GPU timing or serving integration.',
 'physical_gpu_use_seconds':0,'training_histories':[t for t,k in cases[:7]],'heldout_histories':[2048,100000],
 'dimensions':{'queries':1,'heads':1,'materialize_nope':1,'materialize_rope':1,'materialize_value':1},
 'points':points,'heldout_checks':verified,'training_fits':{name:{metric:fits[name]['fits'][metric] for metric in metrics} for name,metrics in contracts.items()},
 'initial_models_rejected':{'pack_queries_only':True,'softmax_selected_only_write_bytes':True},
 'refined_site_residuals':{name:sum(not f['accepted'] for site in fits[name]['sites'].values() for f in site.values()) for name in contracts},
 'ratios_100k_over_2k':{'materialize_write_bytes':points[-1]['materialize_write_bytes']/points[-2]['materialize_write_bytes'],'sparse_write_bytes':points[-1]['sparse_write_bytes']/points[-2]['sparse_write_bytes'],'compact_write_bytes':points[-1]['compact_write_bytes']/points[-2]['compact_write_bytes'],'sparse_fp32_flops':points[-1]['sparse_fp32_flops']/points[-2]['sparse_fp32_flops']},
 'limitations':['Requested PTX bytes are not cache transactions or HBM traffic.','Reduced width and one query/head isolate history scaling; full production width not emulated.','Softmax fixtures use uniform finite scores and valid unique IDs; acceptance is empirical, not all-input proof.','The compact output needs a compatible downstream selected-ID consumer; it is not a drop-in replacement for dense PV.','No native timing, E2E speedup, or attribution of the complete 2K/100K serving gap is claimed.','These historical materialization/softmax paths do not assert that the latest FlashMLA integration still uses them.','Some statement/integer/parallelism fits retain residuals; accepted byte fits are not acceptance of every metric.'],
 'provenance':{'sources':str(HERE/'sources.json'),'profile':str(a.profile.resolve()),'profile_sha256':hashlib.sha256(a.profile.read_bytes()).hexdigest()}}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.output/'training-profile.json.gz').write_bytes(gzip.compress(json.dumps(trained).encode(),mtime=0))
# Keep compact complete source-level evidence with every invocation and failed fit.
(a.output/'profile.gpu.drperf.json.gz').write_bytes(gzip.compress(a.profile.read_bytes(),mtime=0))
report='''# GLM kernel scaling: measured resources, not predicted latency

Two original kernel bodies from gen-glm53 are retained with source hashes. A separate compact-output softmax changes only the output representation and is a demonstration candidate, not installed in serving. Every output/guard passed independent numerical checks.

Train the resource interfaces on histories32–512, varying selected counts16–64. Hold out both2048 and100000 histories, selected count2048. The small-state formulas predict every reported held-out byte/SFU count exactly.

| Resource per one reduced-width row | 2,048 history | 100,000 history | Growth |
|---|---:|---:|---:|
| Original materialization writes |24,576bytes|1,200,000bytes|48.828x|
| Original sparse-softmax FP32 work |31,230FLOPs|31,230FLOPs|1x|
| Original sparse-softmax special ops |2,048|2,048|1x|
| Original sparse-softmax writes |8,192bytes|204,096bytes|24.914x|
| Compact-output demonstration writes |4,096bytes|4,096bytes|1x|

## Discovery and refinement

1. Propose that one-query materialization depends only on query count. DrPerf rejects its byte fits because history length changes behind the same PCV state. Add history: reads=12T and writes=12T for H=N=R=V=1. This pure-copy work grows48.828x from2K to100K.
2. Propose that sparse softmax depends only on selected count K. Its arithmetic and input reads fit, but writes do not: the implementation clears a dense T-sized row. Add history: writes=2T+2K, reads=26K, special_ops=K for these fixtures. Merely counting FLOPs would miss this scaling problem.
3. Change the demonstration's output to K probability slots aligned with the selected IDs, explicitly writing zero to invalid slots. DrPerf now accepts writes=2K without T. All selected probabilities match the original dense row and the independent oracle. A dense PV consumer cannot use this representation unchanged; a compact/sparse consumer and whole-sequence timing are required before claiming an optimization.

The lesson is to challenge an explained dependency: adding T explains the dense clear, while changing the producer-consumer interface can remove it. A good fit alone does not certify efficiency.

## Scope

These are actual standalone PTX executions at2K and100K history with reduced widths, not extrapolated observations or physical GPU timings. The illustrative full-width generalization of materialization is4*H*(N+R+V)*T requested read bytes and the same write bytes; that general expression is source analysis, not a measured full-width run. No full-model numerical run was performed.

These components were concrete bottlenecks in the GLM study. A historical native softmax comparison regressed at2K and improved3.58–3.66x at16K, including dense zeroing; see gen-glm53/docs/moe-helper-profile.md and its archived receipts. That historical comparison uses a different fixture/measurement and is not a native validation of this compact-output prototype or a100K timing claim. Full request benchmarks also change concurrency/output count, so their ratio is not assigned to this kernel alone.

The tcgen05 extension and this example are complementary: Blackwell instruction accounting is tested with separate BF16/NVFP4 matrix fixtures; these two scaling kernels do not execute tcgen05. No claim is made that tcgen05 itself causes the history-scaling issue. Current standalone support is CTA-group1; cluster/TMA-dependent library kernels still require additional support.

See summary.json, training-profile.json.gz and profile.gpu.drperf.json.gz for formulas, residuals, held-out checks, scope and raw provenance. All original emitted captures remain in the run directory.
'''
(a.output/'REPORT.md').write_text(report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
ax.plot([x['history'] for x in points],[x['materialize_write_bytes'] for x in points],'o-');ax.set(xscale='log',yscale='log',xlabel='History tokens',ylabel='Requested write bytes',title='Original history materialization')
held=points[-2:];ts=[x['history'] for x in held]
bx.plot(ts,[x['sparse_write_bytes'] for x in held],'o-',label='Original dense output');bx.plot(ts,[x['compact_write_bytes'] for x in held],'s-',label='Compact output demo');bx.set(xscale='log',yscale='log',xlabel='History tokens (K=2048 fixed)',ylabel='Requested write bytes',title='Sparse softmax: same arithmetic, growing writes');bx.legend()
for x in (ax,bx):x.grid(alpha=.25,which='both')
fig.suptitle('DrPerf GPU: executed PTX resource counts; reduced-width fixtures; no GPU latency claim',fontsize=10)
fig.savefig(a.output/'scaling.svg');fig.savefig(a.output/'scaling.png',dpi=160);plt.close(fig)
svg=(a.output/'scaling.svg').read_text();(a.output/'scaling.html').write_text('<!doctype html><meta charset="utf-8"><title>GLM kernel scaling</title><style>body{font:16px system-ui;margin:2em}svg{max-width:1100px;width:100%;height:auto}</style><h1>GLM kernel scaling</h1>'+svg+'<p>Executed requested bytes, not hardware memory traffic or latency. Model fitted only on histories32–512;2K and100K are held-out executions. Compact output requires a compatible consumer.</p><script type="application/json" id="scaling-data">'+json.dumps(summary).replace('</','<\\/')+'</script>')
print(json.dumps({'heldout_checks':len(verified),'max_absolute_error':max(x['absolute_error'] for x in verified),'ratios':summary['ratios_100k_over_2k'],'report':str(a.output/'REPORT.md')}))
