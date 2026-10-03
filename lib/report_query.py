"""Read-only agent queries over saved reports; never rerun or refit the program."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import composition
import event_interfaces
import explorer


def number(value):
    return int(value) if isinstance(value, str) else value


def key(state):
    return tuple(int(x) for x in state)


def child_costs(model):
    """Replay validated call nesting using recorded state means, including recursion."""
    if model.get('composition', {}).get('status') != 'observed':
        return {}, 'Composition is unavailable.'
    if not model.get('trace', {}).get('complete'):
        return {}, 'Complete invocation trace is unavailable.'
    regions = {r['id']:r for r in model['regions']}
    edges = {r['id']:{e['child']:e for e in r['children']}
             for r in model['composition']['regions']}
    try:
        nodes = composition._forest(regions, model['trace']['events'])
        means = {(r['id'],key(p['state'])):number(p['observed'])
                 for r in regions.values() for p in r['points']}
        result = {r:dict(inclusive=0, unresolved=0, children=Counter()) for r in regions}
        for node in reversed(nodes):
            own = means[node['region'],node['state']]
            if own < 0:
                raise ValueError('Negative own-work observation.')
            node['cost'] = own + sum(c['cost'] for c in node['children'])
            row = result[node['region']]
            row['inclusive'] += node['cost']
            for child in node['children']:
                edge = edges.get(node['region'],{}).get(child['region'])
                if edge is None:
                    raise ValueError('Composition does not match invocation nesting.')
                if edge['multiplicity'] is None:
                    row['unresolved'] += child['cost']
                    row['children'][child['region']] += child['cost']
        return result, None
    except (ValueError, KeyError, TypeError) as error:
        return {}, str(error)


def metrics(model):
    children, reason = child_costs(model)
    parents = {r['id']:r for r in model.get('composition',{}).get('regions',[])}
    result = []
    for region in model['regions']:
        points, fits = region['points'], region['regimes']
        own = sum(number(p['observed'])*number(p['calls']) for p in points)
        residuals = {key(p['state']):number(p['unexplained']) for f in fits for p in f['points']}
        unexplained = sum(residuals.get(key(p['state']), number(p['observed']))*number(p['calls']) for p in points)
        has_children = bool(parents.get(region['id'],{}).get('children'))
        # If composition is missing, do not silently report a parent as a leaf.
        has_children |= bool(region.get('nested'))
        child = children.get(region['id'])
        unresolved = child['unresolved'] if child else None if has_children else 0
        inclusive = child['inclusive'] if child else None if has_children else own
        interface = unexplained + unresolved if unresolved is not None else None
        result.append(dict(region=region['id'], calls=number(region['calls']), states=len(points),
            ownInstructions=own, ownUnexplainedInstructions=unexplained,
            inclusiveInstructionsEstimate=inclusive, unresolvedChildInstructionsEstimate=unresolved,
            interfaceUnexplainedInstructionsEstimate=interface,
            interfaceUnexplainedShare=interface/inclusive if inclusive and interface is not None else 0 if inclusive == 0 else None,
            unresolvedChildren=dict(child['children']) if child else {},
            compositionUnavailable=reason if has_children and child is None else None,
            singleState=len(points)==1,
            meanInstructionsPerCall=own/number(region['calls']) if number(region['calls']) else None))
    return result


def summary(model, rows):
    event = model.get('eventModel',{})
    coverage = event.get('coverage',{})
    channels = {(e['consumer'],e.get('producer'),e['event']) for e in event.get('edges',[])}
    source = model.get('codeCoverage') or {}
    coverage_summary = {k:v for k,v in source.items() if k not in ('regions','withoutSource','lines')}
    coverage_summary['lines'] = {k:v for k,v in source.get('lines',{}).items() if k not in ('files','skipped')}
    coverage_summary['lines']['fileCount'] = len(source.get('lines',{}).get('files',[]))
    coverage_summary['lines']['skippedFileCount'] = len(source.get('lines',{}).get('skipped',[]))
    return dict(regions=len(rows), singleStateRegions=sum(r['singleState'] for r in rows),
        zeroOwnResidualRegions=sum(r['ownUnexplainedInstructions']==0 for r in rows),
        nonzeroInterfaceResidualRegions=sum((r['interfaceUnexplainedInstructionsEstimate'] or 0)>0 for r in rows),
        unavailableInterfaceResidualRegions=sum(r['interfaceUnexplainedInstructionsEstimate'] is None for r in rows),
        dependencyChannels=len(channels), nullRefinements=sum(c.get('event') is None for c in event.get('interfaceChecks',{}).get('claims',[])),
        waitObligations={k:coverage.get(k) for k in ('obligations','covered','uncovered')},
        codeCoverage=coverage_summary, validity=model.get('validity',{}),
        captureNote=model.get('provenance',{}).get('reportNote'),
        contract='Costs are CPU instructions, not elapsed time. Totals weight state means by invocation count. '
        'Cost ranking uses own work, excluding children. Unexplained ranking includes full unresolved F[child] '
        'contributions, but not residuals inside fitted child terms. Child totals are estimates from state means '
        'and observed nesting. Regions can overlap: do not sum inclusive or interface-unexplained totals. '
        'Zero residual at one state is a constant fit of that state mean, not a growth model or per-call guarantee.')


def region_detail(model, name, rows):
    region = next((r for r in model['regions'] if r['id']==name), None)
    if region is None:
        candidates = [r['id'] for r in model['regions'] if name.casefold() in r['id'].casefold()]
        raise ValueError('Unknown region: '+name+('. Matches: '+', '.join(candidates[:20]) if candidates else ''))
    parent = next((r for r in model.get('composition',{}).get('regions',[]) if r['id']==name),None)
    waits = event_interfaces.terms(model.get('eventModel',{}), name)
    formulas = []
    for fit in region['regimes']:
        terms = [f'{round(number(a))}*{s}' for a,s in zip(fit['coefficients'],region['states']) if a]
        terms.append(str(round(number(fit['constant']))))
        if any(number(p['unexplained'])>0 for p in fit['points']): terms.append('unexplained(own)')
        if parent: terms.extend(composition.edge_text(e,region['states']) for e in parent['children'])
        terms.extend(waits)
        formulas.append(' + '.join(terms).replace('+ -','- '))
    event = model.get('eventModel',{})
    return dict(region=region['id'], pcvs=region['states'], sources=region.get('sources',[]),
        metrics=next(r for r in rows if r['region']==name), formulas=formulas,
        regimes=region['regimes'], points=region['points'], composition=parent,
        waitClaims=[c for c in event.get('interfaceChecks',{}).get('claims',[]) if c['region']==name],
        unexplainedWaits=[r for r in event.get('unexplained',[]) if r['region']==name],
        diagnostics=region.get('diagnostics',[]),
        breakdownNote='Function attribution is grouped by affine coefficient, constant, and unexplained own work. '
        'Coefficients are instructions per PCV unit; unexplained values are equally weighted state means, '
        'not invocation-weighted totals. Older reports retain only the top 8 functions per group; '
        'omitted symbols cannot be recovered without rebuilding from raw counts.')


def integer(value):
    if value is None: return 'unavailable'
    if 0 < abs(value) < 1: return '<1' if value > 0 else '>-1'
    return f'{round(value):,}'


def percent(value):
    if value is None: return 'unavailable'
    if 0 < value < .01: return '<1%'
    return f'{value*100:.0f}%'


def render_summary(data):
    lines = [f'Regions: {data["regions"]}; one observed state: {data["singleStateRegions"]}',
             f'Zero own-work residual: {data["zeroOwnResidualRegions"]}; nonzero interface residual: {data["nonzeroInterfaceResidualRegions"]}',
             f'Declared dependency channels: {data["dependencyChannels"]}; null refinements: {data["nullRefinements"]}',
             'Wait obligations: '+json.dumps(data['waitObligations']),
             'Zero residual at one state describes its mean, not growth or individual calls.']
    c = data.get('codeCoverage') or {}
    source = c.get('lines',{})
    if source.get('kind') == 'marked-source-lines':
        lines.append(f'Marked source lines: {source["marked"]}/{source["total"]} ({percent(source["share"])}); scope: {source["sourceRoot"]}; {source["status"]}')
    else: lines.append('Marked source-line coverage: unavailable in this saved report.')
    if 'declared' in c: lines.append(f'Region names exercised: {c["observed"]}/{c["declared"]} (separate from source coverage)')
    if data.get('captureNote'): lines.append('Capture: '+data['captureNote'])
    if data['validity'].get('errors') or data['validity'].get('traceErrors'):
        lines.append('MEASUREMENT ERRORS: '+json.dumps(data['validity']))
    return lines


def render_region(data):
    r=data['metrics']
    lines=[f'REGION {data["region"]}', 'PCVs: '+(', '.join(data['pcvs']) or 'none'),
           f'Calls: {r["calls"]}; distinct states: {r["states"]}',
           f'Own instructions: {integer(r["ownInstructions"])}; mean/call: {integer(r["meanInstructionsPerCall"])}',
           f'Unexplained own: {integer(r["ownUnexplainedInstructions"])}; unresolved children (estimate): {integer(r["unresolvedChildInstructionsEstimate"])}',
           f'Interface unexplained (estimate): {integer(r["interfaceUnexplainedInstructionsEstimate"])} ({percent(r["interfaceUnexplainedShare"])})']
    if r['singleState']: lines.append('ONE STATE: a constant fits the observed mean; growth and per-call variation are not established.')
    for s in data['sources']: lines.append(f'Source: {s["path"]}:{s["line"]}')
    for i,formula in enumerate(data['formulas'],1): lines.append(f'Formula {i} (rounded display; full precision in JSON regimes): {formula}')
    lines += ['FUNCTION BREAKDOWN (retained functions; coefficients/unit or equal-state means)']
    for i,fit in enumerate(data['regimes'],1):
        groups=list(zip(data['pcvs'],fit['attribution']['coefficients']))
        groups += [('constant',fit['attribution']['constant']),('unexplained own',fit['attribution']['unexplained'])]
        for label,functions in groups:
            if not functions:
                continue
            lines.append(f'  Regime {i} / {label}:')
            lines.extend(f'    {integer(f["instructions"])}  {f["function"]} [{f["module"]}]' for f in functions)
    for name,cost in r['unresolvedChildren'].items(): lines.append(f'  unexplained(F[{name}]): {integer(cost)} instructions (estimate)')
    lines.append('OBSERVED STATES (own instructions per call; not individual-call measurements)')
    residuals={key(p['state']):p['unexplained'] for f in data['regimes'] for p in f['points']}
    for p in data['points']:
        lines.append(f'  {dict(zip(data["pcvs"],p["state"]))}: calls={p["calls"]}; observed={integer(p["observed"])}; unexplained={integer(residuals.get(key(p["state"]),p["observed"]))}')
    for claim in data['waitClaims']:
        lines.append(f'Wait: {claim["term"]} [{claim["status"]}]')
        if claim.get('reason'): lines.append('  Reason for manual review: '+claim['reason'])
        for example in claim.get('counterexamples',[]): lines.append('  Counterexample: '+json.dumps(example))
    for row in data['unexplainedWaits']: lines.append('Unexplained wait: '+json.dumps(row))
    lines.extend('Note: '+s for s in data['diagnostics'])
    return lines


def main(argv):
    parser=argparse.ArgumentParser(prog='drperf --report',description=__doc__)
    parser.add_argument('path',nargs='?',help='saved JSON or report directory (default: drperf-report)')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--stats',action='store_true',help='summary (default)')
    mode.add_argument('--region',help='exact region ID; includes formula, breakdown, states and waits')
    mode.add_argument('--top',choices=['costly','unexplained'],help='rank by invocation-weighted instruction work')
    mode.add_argument('--full',action='store_true',help='render the complete text report')
    parser.add_argument('--topk',type=int,default=10,help='number of ranked regions (default: 10)')
    parser.add_argument('--json',action='store_true',help='structured result; numeric values retain precision')
    args=parser.parse_args(argv)
    if args.topk < 1: parser.error('--topk must be positive')
    path=Path(args.path or os.environ.get('DRPERF_REPORT',str(Path(os.environ.get('DRPERF_REPORT_DIR','drperf-report'))/'profile.drperf.json')))
    if path.is_dir(): path=path/'profile.drperf.json'
    try:
        model=explorer.load_model(path)  # Does not read or require raw wait sidecars.
        if model.get('schema') != explorer.SCHEMA or not isinstance(model.get('regions'),list):
            raise ValueError('Not a DrPerf report JSON.')
        if args.full:
            if args.json: print(json.dumps(model,ensure_ascii=False,indent=2)); return 0
            import report_bundle
            print(report_bundle.text_report(model),end=''); return 0
        rows=metrics(model)
        if args.region:
            result=region_detail(model,args.region,rows)
            lines=render_region(result)
        elif args.top:
            metric='ownInstructions' if args.top=='costly' else 'interfaceUnexplainedInstructionsEstimate'
            ranked=sorted((r for r in rows if r[metric] is not None),key=lambda r:(-r[metric],r['region']))
            if args.top=='unexplained': ranked=[r for r in ranked if r[metric]>0]
            result=dict(metric=metric,weighting='invocation-count',regions=ranked[:args.topk],
                        unavailable=[r['region'] for r in rows if r[metric] is None],
                        contract=summary(model,rows)['contract'])
            lines=[f'TOP {args.topk} {args.top.upper()} — {metric}',
                   ('Own CPU instructions, weighted by calls; excludes children.' if args.top=='costly' else
                    'Own residual + full unresolved child cost estimates, weighted by calls; overlapping regions must not be summed.'),
                   'instructions | calls | states | region']
            lines += [f'{integer(r[metric])} | {r["calls"]} | {r["states"]} | {r["region"]}' for r in result['regions']]
            if not result['regions']: lines.append('No regions with a positive available value.')
            if result['unavailable']: lines.append('Unavailable: '+', '.join(result['unavailable']))
        else:
            result=summary(model,rows)
            lines=render_summary(result)
        print(json.dumps(explorer.portable(result),ensure_ascii=False,indent=2) if args.json else '\n'.join(lines))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('drperf --report: '+str(error),file=sys.stderr)
        return 2
