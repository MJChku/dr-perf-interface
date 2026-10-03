"""Check passive declared event order, without inferring application predicates.

Event IDs/generations are supplied by the annotator. Publication is a checkpoint
immediately before the real release. Waited follows the existing readiness check.
Neither checkpoint implements a wait; explicit probes may delay publication.
"""
from collections import defaultdict, Counter
import math

import composition
import waits
import wait_coverage
from report_storage import require_inline_waits

# Legacy end checkpoints remain readable; begin is no longer required.
DECLARED = {'declared_publish', 'declared_waited', 'declared_wait_end', 'declared_waited_null'}
WAITED = {'declared_waited', 'declared_wait_end', 'declared_waited_null'}


def check(model):
    report = _check_events(model)
    if report is not None:
        import event_interfaces
        report['interfaceChecks'] = event_interfaces.check(model, report)
    return report


def _check_events(model):
    require_inline_waits(model)
    capture = model.get('waits') or {}
    captured = capture.get('events', [])
    events = [e for e in captured if e['kind'] in DECLARED]
    if not capture:
        return None
    original_names = {r['id']: r.get('originalName', r['id']) for r in model.get('regions', [])}
    probe = waits.validate_delay_probes(captured, model.get('provenance', {}).get('runs', []),
                                        original_names)
    out = {'status': 'unverified', 'probe': probe['actual'] or bool(capture.get('probe')),
           'probeRequested': probe['requested'] or bool(capture.get('probeRequested')), 'edges': [],
           'violations': [], 'unverified': [], 'unexplained': [], 'interfaces': [], 'nullWaits': [],
           'coverage': {'status': 'unverified'},
           'contract': 'Observed CPU checkpoint order, not proof of necessary dependency. '
                       'No condition argument: markers follow original control flow. '
                       'Publish precedes the real release; waited follows observed readiness. '
                       'IDs/generations are process-scoped. GPU submission alone is not GPU completion. '
                       'waited(null) has no publisher; its indicator and coverage are checked, '
                       'and its refinement reason is for manual review.'}
    if probe['warnings']:
        out['unverified'].extend({'reason': reason} for reason in probe['warnings'])
        return out
    if (capture.get('status') != 'observed' or capture.get('warnings')
            or not model.get('trace', {}).get('complete')
            or model.get('validity', {}).get('errors') or model.get('validity', {}).get('traceErrors')):
        specific = probe['warnings'] or [w for w in capture.get('warnings', [])
                                         if 'delay' in w.lower() or 'probe' in w.lower()]
        out['unverified'].extend({'reason': reason} for reason in specific)
        out['unverified'].append({'reason': 'Incomplete capture; ordering and coverage cannot be checked.'})
        return out
    counts = Counter(e['group'] for e in captured)
    for index, run in enumerate(model.get('provenance', {}).get('runs', [])):
        measurement = run.get('measurement', {})
        if 'wait_records' not in measurement:  # Portable reports predating record metadata.
            continue
        group = f'{index}:{measurement.get("pid", 0)}'
        if measurement.get('wait_dropped') or counts[group] != measurement['wait_records']:
            out['unverified'].append({'reason':
                'Synchronization record count differs from capture metadata.'})
            return out
    regions = {r['id']: r for r in model['regions']}
    traces = model['trace']['events']
    composition._forest(regions, traces)
    by_call = {(t['group'], str(t['thread']), composition._integer(t['seq'])): t for t in traces}
    logical_calls, aliases, logical_parents = wait_coverage.logical_context(traces, captured)
    canonical = []
    for original in events:
        e = dict(original)
        e['regionSeq'] = composition._integer(e['regionSeq'])
        invocation = (e['group'], str(e['tid']), e['regionSeq'])
        parent = by_call.get(invocation)
        valid = True
        if parent is None:
            valid = False
            out['unverified'].append({'checkpoint': e['id'], 'reason':
                'Event checkpoint is outside a captured region invocation.'})
            e['region'] = ''
        else:
            e['region'] = parent['region']
        if not e.get('returned') or e.get('result') != 0 or e['end'] <= e['start']:
            valid = False
            out['unverified'].append({'checkpoint': e['id'], 'reason': 'Incomplete event checkpoint'})
        e['_valid'] = valid
        canonical.append(e)
    events = canonical
    publications = defaultdict(list)
    def key(e): return e['group'], str(e['object']), str(e['aux'])
    def thread(e): return e['group'], str(e['tid'])
    ids = [e['id'] for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate event checkpoint ID')
    for e in events:
        if e['kind'] == 'declared_publish' and e['_valid']:
            publications[key(e)].append(e)
    for identity, pubs in publications.items():
        if len(pubs) != 1:
            out['violations'].append({'event': list(identity), 'reason': 'Event generation published more than once',
                                      'publications': [p['id'] for p in pubs]})
    for e in sorted(events, key=lambda e: (e['group'], e['start'])):
        if e['kind'] not in WAITED:
            continue
        if e['kind'] == 'declared_waited_null':
            out['nullWaits'].append(dict(event=None, generation=None, group=e['group'],
                consumer=e['region'], consumerInvocation=e['regionSeq'], waited=e['id'],
                status='recorded' if e['_valid'] else 'unverified', producer=None,
                publication=None, refinement='null'))
            continue
        pubs = publications[key(e)]
        edge = {'event': str(e['object']), 'generation': str(e['aux']), 'group': e['group'],
                'consumer': e['region'], 'consumerInvocation': e['regionSeq'],
                'waited': e['id'], 'status': 'unverified',
                'producer': None, 'publication': None}
        if not e['_valid']:
            out['edges'].append(edge)
            continue
        if len(pubs) == 1:
            pub = pubs[0]
            edge.update(producer=pub['region'], publication=pub['id'], producerInvocation=pub['regionSeq'])
            publisher = by_call[(pub['group'], str(pub['tid']), pub['regionSeq'])]
            consumer = by_call[(e['group'], str(e['tid']), e['regionSeq'])]
            # Use dynamic invocation ancestry, never just region names. Another
            # thread can execute the same region independently and release us.
            nested = (thread(pub) == thread(e) and
                      int(consumer['seq']) <= int(publisher['seq']) and
                      int(publisher['end']) <= int(consumer['end']))
            consumer_key = aliases[wait_coverage.invocation(e)]
            current = aliases[wait_coverage.invocation(pub)]
            while current is not None:
                nested = nested or current == consumer_key
                current = logical_parents[current]
            if nested:
                edge['status'] = 'violation'
                out['violations'].append(dict(event=list(key(e)), consumer=e['region'],
                    producer=pub['region'], waited=e['id'], publication=pub['id'],
                    reason='Publisher is the waiter itself or its synchronous descendant; '
                           'call/return order is not a waited dependency. Use waited(null) '
                           'with a reason for an event-free refinement.'))
            elif pub['end'] < e['start']:
                edge['status'] = 'ordered'
                if 'beginUs' in e and 'endUs' in pub:
                    edge['distanceUs'] = max(0, int(e['beginUs'])-int(pub['endUs']))
            else:
                edge['status'] = 'violation'
                out['violations'].append({'event': list(key(e)), 'consumer': e['region'],
                    'producer': pub['region'], 'waited': e['id'], 'publication': pub['id'],
                    'reason': 'Consumer passed waited before declared publication completed'})
        else:
            out['unverified'].append({'event': list(key(e)), 'waited': e['id'],
                                      'reason': 'Missing or ambiguous publication generation'})
        out['edges'].append(edge)
    # A channel is process-scoped and belongs to one ordered region pair,
    # across all its observed generations. Same-pair repeated calls are valid.
    channels = defaultdict(list)
    for edge in out['edges']:
        if edge['producer'] is not None and edge['consumer']:
            channels[edge['group'], edge['event']].append(edge)
    out['eventPairs'] = []
    for (group, event), claims in sorted(channels.items()):
        pairs = sorted({(e['producer'], e['consumer']) for e in claims})
        out['eventPairs'].append(dict(group=group, event=event,
            pairs=[dict(producer=p, consumer=c) for p, c in pairs]))
        if len(pairs) > 1:
            out['violations'].append(dict(group=group, event=event,
                reason='Event channel is shared across region pairs; use a separate event ID per pair',
                pairs=[dict(producer=p, consumer=c) for p, c in pairs]))
            for edge in claims:
                edge['status'] = 'violation'
    # Declaration ownership follows the captured invocation containing waited,
    # irrespective of which descendant implements the native synchronization.
    # Record that context without claiming a native object/publication match.
    declarations = defaultdict(list)
    checkpoints = {e['id']: e for e in events}
    for edge in out['edges'] + out['nullWaits']:
        e = checkpoints[edge['waited']]
        if e['_valid']:
            declarations[aliases[wait_coverage.invocation(e)]].append(edge)
        edge['nativeContext'] = []
    contexts = defaultdict(Counter)
    known_count = 0
    # Reapply the recorded capture boundary when checking older reports too.
    # CUDA synchronization is retained; only proven direct native-GX CPU
    # implementation calls are outside the application coverage budget.
    operations, _ = waits.application_operations(capture.get('operations', []),
                                                 model.get('provenance', {}).get('runs', []))
    _, out['implementationSynchronization'] = waits.application_operations(captured,
                                                 model.get('provenance', {}).get('runs', []))
    for op in operations:
        invocation = (op['group'], str(op['tid']), int(op['regionSeq']))
        call = by_call.get(invocation)
        if call is None:
            continue
        # Canonicalize the region from its actual invocation, as for checkpoints.
        region = call['region']
        current = aliases[invocation]
        while current is not None:
            for edge in declarations[current]:
                marker = checkpoints[edge['waited']]
                if op.get('returned') and 0 < int(op.get('end', 0)) < int(marker['start']):
                    contexts[edge['waited']][region, op['api']] += 1
            current = logical_parents[current]
        if op.get('producers'):
            known_count += 1
    for edge in out['edges'] + out['nullWaits']:
        edge['nativeContext'] = [dict(region=region, api=api, count=count)
                                for (region, api), count in sorted(contexts[edge['waited']].items())]
    out['nativeResolved'] = known_count
    out['coverage'] = wait_coverage.check(operations, events, out['edges'] + out['nullWaits'],
                                          (logical_calls, aliases, logical_parents))
    for row in out['coverage']['regions']:
        if row['uncovered']:
            out['unexplained'].append(dict(region=row['region'], count=row['uncovered'],
                unit='wait obligations', term='unexplained(Wait[?])', examples=row['examples']))
    # Infer only occurrence counts from existing entry PCVs. No new condition is
    # supplied or evaluated by the marker, and identical states are not averaged.
    buckets = defaultdict(Counter)
    for e in events:
        if e['kind'] in WAITED and e['_valid'] and e['region'] in regions:
            identity = None if e['kind'] == 'declared_waited_null' else str(e['object'])
            buckets[e['region'], identity][aliases[wait_coverage.invocation(e)]] += 1
    for (region, event), counts in sorted(buckets.items(), key=lambda item:
                                          (item[0][0], '' if item[0][1] is None else item[0][1])):
        calls = [t for t in logical_calls.values() if t['region'] == region]
        names = regions[region]['states']
        values = [counts[t['group'], str(t['thread']), t['seq']] for t in calls]
        relation = composition.affine([[composition._integer(t['values'][n]) for n in names] for t in calls], values, names)
        out['interfaces'].append({'region': region, 'event': event, 'states': names,
            'countFormula': relation, 'present': sum(v > 0 for v in values), 'absent': sum(v == 0 for v in values)})
    out['status'] = 'violation' if out['violations'] else 'unverified' if out['unverified'] else 'ordered'
    return out


def probe_plan(report, margin_ms=100, maximum_ms=5000):
    """Plan a second execution; distance is diagnostic wall time, never cost."""
    if margin_ms < 1 or maximum_ms < margin_ms:
        raise ValueError('Probe bounds must be positive and maximum >= margin')
    if not report or report['probe']:
        raise ValueError('Plan from an unperturbed event capture')
    gaps = defaultdict(list)
    for edge in report['edges']:
        if edge['status'] == 'ordered' and edge['producer'] and 'distanceUs' in edge:
            gaps[edge['producer']].append(edge['distanceUs'])
    return [{'region': region, 'delayMs': min(maximum_ms, math.ceil(max(values)/1000)+margin_ms),
             'baselineMaxDistanceUs': max(values),
             'capped': math.ceil(max(values)/1000)+margin_ms > maximum_ms,
             'scope': 'all declared publications in this region; rerun one region per probe'}
            for region, values in sorted(gaps.items())]


def lines(report):
    """Keep ordering failures and missing annotation coverage visible in exports."""
    if not report:
        return []
    out = [f"declared event order: {report['status']}"]
    out.extend('  ERROR: '+v['reason'] for v in report['violations'])
    out.extend('  UNVERIFIED: '+v['reason'] for v in report['unverified'])
    coverage = report['coverage']
    if 'obligations' in coverage:
        out.append(f"annotation coverage: {coverage['covered']}/{coverage['obligations']} "
                   'wait obligations')
    else:
        out.append('annotation coverage: unverified')
    interfaces = report.get('interfaceChecks', {})
    for row in interfaces.get('claims', []):
        out.append(f"  {row['region']}: {row['term']} [{row['status']}]")
        if row.get('event') is None:
            out.append('    refinement reason (manual review): ' + row['reason'])
    for row in interfaces.get('unexplained', report['unexplained']):
        out.append(f"  {row['region']}: {row['term']} ({row['count']} observations)")
    return out
