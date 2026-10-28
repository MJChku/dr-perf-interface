"""Read-only resource queries. No fitting, latency model, or mixed-unit score."""
import copy
import json
from collections import Counter


def metrics(region):
    return set(region.get('fits', {})) | set(region.get('interfaces', {}))


def select(model, *, region=None, metric=None):
    regions=model['regions']
    if region is not None:
        regions=[r for r in regions if r['name']==region]
        if not regions:raise ValueError('unknown GPU region '+region)
    available=sorted(set().union(*(metrics(r) for r in regions)))
    if metric is not None and metric not in available:
        raise ValueError('unknown GPU metric '+metric+'; available: '+', '.join(available))
    if region is None and metric is None:return model
    result=dict(model,regions=copy.deepcopy(regions),selection={'region':region,'metric':metric})
    if metric is None:return result
    for r in result['regions']:
        for field in ('fits','interfaces'):
            r[field]={m:v for m,v in r.get(field,{}).items() if m==metric}
        r['sites']={site:{metric:fs[metric]} for site,fs in r.get('sites',{}).items() if metric in fs}
        r['basicBlockMembers']={k:v for k,v in r.get('basicBlockMembers',{}).items() if k in r['sites']}
        for s in r.get('stateStatistics',[]):
            s['metrics']={m:v for m,v in s['metrics'].items() if m==metric}
        for o in r.get('observations',[]):
            o['metrics']={m:v for m,v in o['metrics'].items() if m==metric}
            o['sites']={site:{metric:ms[metric]} for site,ms in o.get('sites',{}).items() if metric in ms}
        children=r.get('composition',{}).get('children',[])
        for child in children:
            child['metrics']=[m for m in child['metrics'] if m==metric]
            for s in child.get('unexplainedByState',[]):
                s['metrics']={m:v for m,v in s['metrics'].items() if m==metric}
        if 'composition' in r:
            r['composition']['children']=[c for c in children if c['metrics']]
    return result


def rank(model, *, metric, top, topk=10, region=None):
    if not metric:raise ValueError('GPU ranking requires --metric; bytes and operations have separate units')
    if top not in ('costly','unexplained'):raise ValueError('GPU ranking must be costly or unexplained')
    if topk<1:raise ValueError('--topk must be positive')
    selected=select(model,region=region)
    available=sorted(set().union(*(metrics(r) for r in selected['regions'])))
    if metric not in available:
        raise ValueError('unknown GPU metric '+metric+'; available: '+', '.join(available))
    rows=[]
    for r in selected['regions']:
        if metric not in metrics(r):continue
        observations=r.get('observations')
        if observations is None:raise ValueError('observations missing for GPU ranking: '+r['name'])
        counts=Counter(tuple(sorted(o['pcvs'].items())) for o in observations)
        own_total=sum(o['metrics'].get(metric,0) for o in observations)
        interface=r.get('interfaces',{}).get(metric)
        fit=r.get('fits',{}).get(metric)
        unexplained=0
        if interface is not None:
            for s in interface['unexplainedByState']:
                unexplained += counts[tuple(sorted(s['pcvs'].items()))]*s['total']
            expression=interface['formula']
        else:
            from gpu_backend import formula
            for s in (fit or {}).get('unexplainedByState',[]):
                unexplained += counts[tuple(sorted(s['pcvs'].items()))]*s['mean']
            expression=formula(fit)+(' + unexplained' if not fit['accepted'] else '')
        score=own_total if top=='costly' else unexplained
        if score<=0:continue
        rows.append({'name':r['name'],'pcvs':r['pcvs'],'invocations':r['invocations'],
            'stateCount':r['stateCount'],'formula':expression,'score':score,
            'measuredOwnTotal':own_total,'unexplainedInterfaceTotal':unexplained,
            'ownValidity':r.get('validity'),'inclusiveValidity':r.get('inclusiveValidity','unavailable'),
            'unmodeledWorkEvents':r.get('unmodeledWorkEvents',0),
            'inclusiveUnmodeledWorkEvents':r.get('inclusiveUnmodeledWorkEvents')})
    rows.sort(key=lambda r:(-r['score'],r['name']))
    return {'schema':'drperf.gpu.query.v1','captureValidity':model.get('validity'),
        'metric':metric,'ranking':top,'topk':topk,'matchingRegions':len(rows),
        'scoreScope':('sum of measured exclusive invocation values' if top=='costly' else
                      'observed-invocation-weighted unexplained interface counts, including unfitted child terms'),
        'limits':'Observed states only; unknown work is not estimated; parent/child unexplained entries may overlap and must not be summed.',
        'regions':rows[:topk]}


def ranking_text(result):
    lines=[f'GPU {result["ranking"]}: {result["metric"]}',result['scoreScope']+'.']
    if result['captureValidity']!='complete':
        lines.append('INCOMPLETE CAPTURE: ranking covers measured contributions only; unknown cost is not estimated.')
    if not result['regions']:lines.append('No positive measured contributions for this query.')
    for index,r in enumerate(result['regions'],1):
        lines.append(f'{index}. {r["name"]}: {r["score"]:,.0f}; {r["invocations"]} invocations, {r["stateCount"]} states')
        lines.append('   '+r['formula'])
        if r['unmodeledWorkEvents'] or r['inclusiveUnmodeledWorkEvents']:
            lines.append('   INCOMPLETE resource capture in this region or its descendants.')
    lines.append(result['limits'])
    return '\n'.join(lines)+'\n'


def render(model, *, region=None, metric=None, top=None, topk=10, as_json=False):
    from gpu_backend import text_report
    result=rank(model,metric=metric,top=top,topk=topk,region=region) if top else select(model,region=region,metric=metric)
    return json.dumps(result,indent=2)+'\n' if as_json else ranking_text(result) if top else text_report(result)
