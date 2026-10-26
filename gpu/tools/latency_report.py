#!/usr/bin/env python3
"""Rank modeled latency paths. These are model units, never measured GPU time."""
import argparse
import fnmatch
import json
import math
import sys

from report_io import read_report


def residency(m, sm, smem_per_sm=233472, threads_per_sm=2048, ctas_per_sm_cap=32, smem_reserved=1024):
    """Shared-memory/thread-limited CTAs per SM and waves for one launch.

    Registers are unknown at PTX level, so this is an upper bound on residency."""
    blocks = m.get('executed_blocks', m.get('blocks', 0))
    if not blocks:
        return None, None
    threads = max(1, m.get('launched_lanes', 0) // max(1, m.get('blocks', 1)))
    smem = m.get('dynamic_shared_bytes_per_launch', 0) + m.get('static_shared_bytes_per_launch', 0)
    by_smem = smem_per_sm // (smem + smem_reserved) if smem else ctas_per_sm_cap
    per_sm = max(1, min(ctas_per_sm_cap, threads_per_sm // threads, by_smem))
    return per_sm, math.ceil(blocks / (sm * per_sm))


def rank_regions(report, sms=None, lanes_per_sm=None, threshold=4, region_pattern='*'):
    for name, value in [('sms', sms), ('lanes_per_sm', lanes_per_sm), ('threshold', threshold)]:
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be positive and finite')
    records, unavailable = [], []
    for region in report['regions']:
        if not fnmatch.fnmatchcase(region['name'], region_pattern):
            continue
        launches = [launch for row in region['observations'] for launch in row.get('launches', [])]
        modeled = [launch for launch in launches if launch['metrics'].get('latency_model_version') == 1
                   and 'latency_critical_path' in launch['metrics']]
        if not modeled:
            unavailable.append(region['name'])
            continue
        critical = exposed = work = blocks = underfilled = 0
        per_sm_min, waves_max = None, 0
        throughput = 0.0
        configurations = set()
        for launch in modeled:
            m = launch['metrics']
            sm = sms if sms is not None else m['latency_model_sm_count']
            lanes = lanes_per_sm if lanes_per_sm is not None else m['latency_model_lanes_per_sm']
            if sm <= 0 or lanes <= 0:
                raise ValueError('invalid model SM or lane count')
            critical += m['latency_critical_path']
            exposed += m.get('dependent_global_loads_on_critical_path', 0)
            work += m.get('ptx_lane_instructions', 0)
            throughput += m.get('ptx_lane_instructions', 0) / (sm * lanes)
            b = m.get('executed_blocks', m['blocks'])
            blocks += b
            underfilled += b < sm
            per_sm, waves = residency(m, sm)
            if per_sm is not None:
                per_sm_min = per_sm if per_sm_min is None else min(per_sm_min, per_sm)
                waves_max = max(waves_max, waves)
            configurations.add(tuple(m.get('latency_model_' + k) for k in
                                     ('alu', 'global', 'shared', 'mma', 'sfu')) + (sm, lanes))
        ratio = critical / throughput if throughput else None
        records.append({'region': region['name'], 'launches': len(modeled),
                        'unmodeledLaunches': len(launches)-len(modeled),
                        'criticalPathTimesLaunches': critical,
                        'meanCriticalPath': critical / len(modeled),
                        'exposedGlobalLoads': exposed, 'laneInstructions': work,
                        'blocks': blocks, 'underfilledLaunches': underfilled,
                        'ctasPerSmUpperBound': per_sm_min, 'maxWaves': waves_max,
                        'throughputProxy': throughput, 'latencyToThroughputRatio': ratio,
                        'latencyBound': ratio is not None and ratio >= threshold,
                        'mixedModelConstants': len(configurations) > 1})
    records.sort(key=lambda r: (-r['criticalPathTimesLaunches'], r['region']))
    return {'schema': 'drperf.gpu.latency.v1', 'captureValidity': report.get('validity'),
            'semantics': 'Uncalibrated in-order register-ready model; no measured time or speedup. '
                         'Score sums one maximum lane path per launch (mean path x launches). '
                         'Latency-bound is a heuristic: path >= threshold x '
                         'lane instructions / (SMs x lane proxy). Inclusive launch envelopes '
                         'are ranked once; device-only scopes without launches are unavailable.',
            'threshold': threshold, 'records': records, 'unavailableRegions': unavailable}


def format_text(summary):
    lines = ['GPU LATENCY MODEL (not measured time)', summary['semantics'],
             f'Capture validity: {summary["captureValidity"]}',
             'Residency: CTAs/SM from shared memory and threads only (registers unknown); waves = '
             'ceil(blocks / (SMs x CTAs/SM)), maximum over launches.',
             'region launches path_x_launches mean_path exposed_loads blocks underfilled ctas_per_sm max_waves latency/throughput flag']
    for r in summary['records']:
        ratio = r['latencyToThroughputRatio']
        lines.append(f'{json.dumps(r["region"])} {r["launches"]} {r["criticalPathTimesLaunches"]} '
                     f'{r["meanCriticalPath"]:.6g} {r["exposedGlobalLoads"]} {r["blocks"]} '
                     f'{r["underfilledLaunches"]} {r["ctasPerSmUpperBound"]} {r["maxWaves"]} ' +
                     ('n/a' if ratio is None else f'{ratio:.4g}') +
                     (' latency-bound' if r['latencyBound'] else ' throughput-bound-proxy') +
                     (' MIXED-CONSTANTS' if r['mixedModelConstants'] else ''))
    if summary['unavailableRegions']:
        lines.append('Model unavailable: ' + ', '.join(summary['unavailableRegions']))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', help='GPU report directory, JSON, or JSON.gz')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--region', default='*')
    parser.add_argument('--sms', type=int, help='override captured SM count')
    parser.add_argument('--lanes-per-sm', type=int, help='override captured lane proxy')
    parser.add_argument('--threshold', type=float, default=4)
    args = parser.parse_args(argv)
    try:
        summary = rank_regions(read_report(args.report), args.sms, args.lanes_per_sm,
                               args.threshold, args.region)
        print(json.dumps(summary, indent=2, allow_nan=False) if args.json else format_text(summary))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'latency_report: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
