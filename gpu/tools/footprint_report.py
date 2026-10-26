#!/usr/bin/env python3
"""Show requested bytes beside exact footprints and derived reuse ratios."""
import argparse
import fnmatch
import json
import sys

from report_io import read_report


def summarize_counts(counts, available):
    accesses = []
    for space in ('global', 'generic'):
        for direction in ('read', 'write'):
            prefix = f'{space}_unique_{direction}'
            requested = counts.get(f'{space}_{direction}_bytes', 0)
            unique = counts.get(prefix + '_bytes', 0)
            if not requested and not unique:
                continue
            sectors = counts.get(prefix + '_sectors', 0)
            lines = counts.get(prefix + '_lines', 0)
            accesses.append({
                'space': space, 'direction': direction, 'requestedBytes': requested,
                'uniqueBytes': unique if available else None,
                'uniqueSectors32': sectors if available else None,
                'uniqueLines128': lines if available else None,
                'sectorBytes32': 32 * sectors if available else None,
                'lineBytes128': 128 * lines if available else None,
                'redundancyRatio': requested / unique if available and unique else None,
                'requestedPerSectorByte': requested / (32 * sectors) if available and sectors else None,
                'requestedPerLineByte': requested / (128 * lines) if available and lines else None,
                'sectorUtilization': unique / (32 * sectors) if available and sectors else None,
                'lineUtilization': unique / (128 * lines) if available and lines else None,
            })
    return accesses


def summarize_report(report, level='all', region_pattern='*'):
    records = []
    for region in report['regions']:
        if not fnmatch.fnmatchcase(region['name'], region_pattern):
            continue
        for observation in region['observations']:
            base = {key: observation.get(key) for key in ('pid', 'token', 'pcvs', 'scope', 'kernel')}
            base['region'] = region['name']

            def append(container, kind, context, site=None):
                if level not in ('all', kind):
                    return
                counts = container['metrics'] if site is None else container['sites'][site]
                available = (container.get('footprintVersion') == 1 and not any(
                    counts.get(key, 0) for key in ('footprint_unavailable_launches',
                                                  'footprint_unavailable_invocations')))
                accesses = summarize_counts(counts, available)
                if accesses or site is None:
                    records.append({**context, 'level': kind, 'site': site,
                                    'footprintAvailable': available, 'accesses': accesses})

            append(observation, 'region', base)
            for site in sorted(observation.get('sites', {})):
                append(observation, 'site', base, site)
            for index, launch in enumerate(observation.get('launches', [])):
                context = {**base, 'launchIndex': index, 'kernel': launch['kernel']}
                append(launch, 'launch', context)
                for site in sorted(launch.get('sites', {})):
                    append(launch, 'launch-site', context, site)
    return {'schema': 'drperf.gpu.footprint.v1', 'captureValidity': report.get('validity'),
            'semantics': 'Exact address unions; redundancyRatio = requestedBytes / uniqueBytes. '
                         '32-byte sectors and 128-byte lines are rounded address coverage, '
                         'not cache misses, HBM traffic, or latency. Ratios are derived, not fitted.',
            'records': records}


def format_text(summary):
    lines = ['GPU FOOTPRINT / REUSE', summary['semantics'],
             f'Capture validity: {summary["captureValidity"]}',
             'level region invocation launch site space direction requested unique sectors32 lines128 redundancy']
    for row in summary['records']:
        label = (f'{row["level"]} {json.dumps(row["region"])} '
                 f'{row["pid"]}:{row["token"]} {row.get("launchIndex", "-")} '
                 f'{json.dumps(row["site"])}')
        for access in row['accesses']:
            ratio = access['redundancyRatio']
            values = [access[k] for k in ('requestedBytes', 'uniqueBytes', 'uniqueSectors32', 'uniqueLines128')]
            lines.append(f'{label} {access["space"]} {access["direction"]} ' +
                         ' '.join('unavailable' if v is None else str(v) for v in values) +
                         (' n/a' if ratio is None else f' {ratio:.6g}x'))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', help='GPU report directory, JSON, or JSON.gz')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--level', choices=('all', 'region', 'launch', 'site', 'launch-site'), default='all')
    parser.add_argument('--region', default='*', help='case-sensitive region glob')
    args = parser.parse_args(argv)
    try:
        result = summarize_report(read_report(args.report), args.level, args.region)
        print(json.dumps(result, indent=2, allow_nan=False) if args.json else format_text(result), end='\n' if args.json else '')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'footprint_report: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
