#!/usr/bin/env python3
"""Compare fitted region and statement-site resource interfaces in two reports."""
import argparse
import fnmatch
import json
import math
import sys

from report_io import read_report


def _regions(report):
    result = {}
    for region in report['regions']:
        if not isinstance(region, dict) or not isinstance(region.get('name'), str):
            raise ValueError('regions must have string names')
        if not isinstance(region.get('sites', {}), dict):
            raise ValueError('statement sites must be an object')
        pcvs = region.get('pcvs', [])
        if not isinstance(pcvs, list) or any(not isinstance(p, str) for p in pcvs):
            raise ValueError('region PCVs must be a list of names')
        for fits in [region.get('fits', {}), *region.get('sites', {}).values()]:
            if not isinstance(fits, dict):
                raise ValueError('metric interfaces must be objects')
            for metric, fit in fits.items():
                if not isinstance(metric, str) or not isinstance(fit, dict):
                    raise ValueError('metric interfaces must contain named fit objects')
                coefficients = fit.get('coefficients', {})
                if not isinstance(coefficients, dict) or any(not isinstance(k, str) for k in coefficients):
                    raise ValueError('fit coefficients must be named numbers')
                for value in [*coefficients.values(), fit.get('constant', 0)]:
                    _number(value)
        if region['name'] in result:
            raise ValueError('duplicate region name: ' + region['name'])
        result[region['name']] = region
    return result


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('fit values must be finite numbers')
    return value


def diff_reports(before, after, *, absolute=0, relative=0, site_map=None,
                 include_unchanged=False, region_pattern='*', metric_pattern='*'):
    if not math.isfinite(absolute) or not math.isfinite(relative) or min(absolute, relative) < 0:
        raise ValueError('tolerances must be finite and nonnegative')
    site_map = {} if site_map is None else site_map
    if (not isinstance(site_map, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in site_map.items())
            or len(set(site_map.values())) != len(site_map)):
        raise ValueError('site map must be a one-to-one JSON object of old-site: new-site strings')
    old, new = _regions(before), _regions(after)
    all_old_sites = {s for r in old.values() for s in r.get('sites', {})}
    all_new_sites = {s for r in new.values() for s in r.get('sites', {})}
    if set(site_map) - all_old_sites or set(site_map.values()) - all_new_sites:
        raise ValueError('site map contains an unknown source or destination site')
    entries, metadata = [], []

    def close(a, b):
        return abs(_number(a) - _number(b)) <= max(absolute, relative * max(abs(a), abs(b)))

    for name in sorted(old.keys() | new.keys()):
        if not fnmatch.fnmatchcase(name, region_pattern):
            continue
        left, right = old.get(name), new.get(name)
        context_changes = {}
        for key in ('scope', 'metricScope', 'pcvs', 'validity', 'fitSemantics'):
            a, b = left.get(key) if left else None, right.get(key) if right else None
            if a != b:
                context_changes[key] = {'before': a, 'after': b}
        # Compare the symbolic child interface, not run-dependent invocation
        # IDs/histograms. Multiplicity changes matter even with zero own cost.
        def child_terms(region):
            return [{k: t.get(k) for k in ('child', 'accepted', 'term', 'metrics')}
                    for t in (region or {}).get('composition', {}).get('children', [])]
        if child_terms(left) != child_terms(right):
            context_changes['childInterfaces'] = {'before': child_terms(left), 'after': child_terms(right)}
        if context_changes or left is None or right is None:
            metadata.append({'region': name, 'status': 'added' if left is None else 'removed' if right is None else 'changed',
                             'changes': context_changes})

        def compare(a, b, site=None, after_site=None):
            for metric in sorted(a.keys() | b.keys()):
                if not fnmatch.fnmatchcase(metric, metric_pattern):
                    continue
                x, y = a.get(metric), b.get(metric)
                changes = {}
                for pcv in sorted((x or {}).get('coefficients', {}).keys() | (y or {}).get('coefficients', {}).keys()):
                    av = x.get('coefficients', {}).get(pcv, 0) if x else None
                    bv = y.get('coefficients', {}).get(pcv, 0) if y else None
                    if av is None or bv is None or not close(av, bv):
                        delta = _number(bv) - _number(av) if av is not None and bv is not None else None
                        changes[pcv] = {'before': av, 'after': bv, 'delta': delta,
                                        'relativeChange': delta / abs(av) if delta is not None and av else None}
                constant_a, constant_b = x.get('constant', 0) if x else None, y.get('constant', 0) if y else None
                constant_changed = (constant_a is None or constant_b is None or not close(constant_a, constant_b))
                status_changes = {key: {'before': (x or {}).get(key), 'after': (y or {}).get(key)}
                                  for key in ('accepted', 'dependentPCVs') if (x or {}).get(key) != (y or {}).get(key)}
                changed = bool(changes or constant_changed or status_changes)
                if not changed and not include_unchanged:
                    continue
                entries.append({'region': name, 'site': site, 'afterSite': after_site, 'metric': metric,
                                'status': 'added' if x is None else 'removed' if y is None else 'changed' if changed else 'unchanged',
                                'before': x, 'after': y, 'coefficientChanges': changes,
                                'constantChange': {'before': constant_a, 'after': constant_b,
                                                   'delta': _number(constant_b) - _number(constant_a)
                                                   if constant_a is not None and constant_b is not None else None}
                                                  if constant_changed else None,
                                'fitStatusChanges': status_changes,
                                'comparable': bool(left and right and all(left.get(k) == right.get(k)
                                                   for k in ('scope', 'metricScope', 'pcvs', 'fitSemantics'))),
                                'dependentPCVs': sorted(set((x or {}).get('dependentPCVs', [])) |
                                                        set((y or {}).get('dependentPCVs', []))),
                                'bothAccepted': bool(x and y and x.get('accepted') and y.get('accepted'))})

        compare((left or {}).get('fits', {}), (right or {}).get('fits', {}))
        left_sites, right_sites = (left or {}).get('sites', {}), (right or {}).get('sites', {})
        matched = set()
        for site in sorted(left_sites):
            target = site_map.get(site, site)
            if target in right_sites:
                if target in matched:
                    raise ValueError(f'{name}: site map collides with another site: {target}')
                matched.add(target)
            compare(left_sites[site], right_sites.get(target, {}), site, target if target in right_sites else None)
        for site in sorted(right_sites.keys() - matched):
            compare({}, right_sites[site], None, site)
    return {'schema': 'drperf.gpu.diff.v1',
            'beforeValidity': before.get('validity'), 'afterValidity': after.get('validity'),
            'tolerances': {'absolute': absolute, 'relative': relative},
            'semantics': 'after minus before; rejected fits are diagnostic; scope/PCV changes are not comparable. '
                         'Sites match exact IDs unless explicitly mapped. No latency or speedup is inferred.',
            'metadataChanges': metadata, 'changes': entries,
            'changedMetrics': sum(e['status'] != 'unchanged' for e in entries)}


def format_text(diff):
    lines = ['GPU INTERFACE DIFF (after - before)', diff['semantics'],
             f'Capture validity: {diff["beforeValidity"]} -> {diff["afterValidity"]}',
             f'{diff["changedMetrics"]} changed metric interfaces']
    for item in diff['metadataChanges']:
        lines.append(f'REGION {json.dumps(item["region"])} {item["status"]}: {json.dumps(item["changes"], sort_keys=True)}')
    for item in diff['changes']:
        site = item['site'] or item['afterSite']
        label = 'region total' if site is None else 'site ' + site
        flags = '' if item['comparable'] else ' [scope/PCVs differ]'
        if not item['bothAccepted']:
            flags += ' [added/removed or rejected fit]'
        if item['dependentPCVs']:
            flags += ' [dependent PCVs: ' + ', '.join(item['dependentPCVs']) + ']'
        lines.append(f'{item["region"]} | {label} | {item["metric"]}: {item["status"]}{flags}')
        if item['site'] and item['afterSite'] and item['site'] != item['afterSite']:
            lines.append('  mapped to ' + item['afterSite'])
        for pcv, change in item['coefficientChanges'].items():
            delta = change['delta']
            relative = change['relativeChange']
            lines.append(f'  {pcv}: {change["before"]} -> {change["after"]}; delta {delta}' +
                         ('' if relative is None else f' ({100 * relative:+.6g}%)'))
        if item['constantChange']:
            c = item['constantChange']
            lines.append(f'  constant: {c["before"]} -> {c["after"]}; delta {c["delta"]}')
        for key, change in item['fitStatusChanges'].items():
            lines.append(f'  {key}: {change["before"]} -> {change["after"]}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', help='GPU report directory, JSON, or JSON.gz')
    parser.add_argument('after', help='GPU report directory, JSON, or JSON.gz')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--atol', type=float, default=0, help='absolute coefficient tolerance')
    parser.add_argument('--rtol', type=float, default=0, help='relative coefficient tolerance')
    parser.add_argument('--site-map', help='one-to-one JSON object mapping old site IDs to new site IDs')
    parser.add_argument('--include-unchanged', action='store_true')
    parser.add_argument('--fail-on-change', action='store_true', help='return 1 when interfaces or metadata change')
    parser.add_argument('--region', default='*', help='case-sensitive region glob')
    parser.add_argument('--metric', default='*', help='case-sensitive metric glob')
    args = parser.parse_args(argv)
    try:
        site_map = None
        if args.site_map:
            with open(args.site_map, encoding='utf-8') as source:
                site_map = json.load(source)
        result = diff_reports(read_report(args.before), read_report(args.after), absolute=args.atol,
                              relative=args.rtol, site_map=site_map, include_unchanged=args.include_unchanged,
                              region_pattern=args.region, metric_pattern=args.metric)
        print(json.dumps(result, indent=2, allow_nan=False) if args.json else format_text(result), end='\n' if args.json else '')
        changed = result['changedMetrics'] or result['metadataChanges'] or result['beforeValidity'] != result['afterValidity']
        return 1 if args.fail_on_change and changed else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'diff_reports: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
