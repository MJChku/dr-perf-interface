"""Explicit wait-interface claims checked against passive event checkpoints.

The annotator supplies the indicator; observed counts never replace that claim.
Checkpoint/native coverage remains a separate budget, not an object association.
"""
from collections import Counter, defaultdict

import composition
import wait_coverage
from wait_contracts import Indicator


def check(model, events):
    import inline_waits
    declaration = inline_waits.resolve(model, events)
    if not isinstance(declaration, dict) or declaration.get('version') != 1 or not isinstance(declaration.get('claims'), list):
        raise ValueError('waitDeclarations requires version 1 and a claims array')
    regions = {r['id']: r for r in model['regions']}
    claims, selectors, identities = [], set(), set()
    for claim in declaration['claims']:
        is_null = 'event' in claim and claim['event'] is None
        fields = {'id', 'region', 'event', 'indicator', 'producer'} | ({'reason'} if is_null else set())
        if set(claim) != fields:
            raise ValueError('Wait interface requires id, region, event, indicator and producer' +
                             ('; waited(null) additionally requires a reason' if is_null else ''))
        identity, region = claim['id'], claim['region']
        event = None if is_null else str(claim['event'])
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError('Wait interface IDs must be distinct nonempty strings')
        if region not in regions:
            raise ValueError('Unknown wait-interface region: ' + str(region))
        if not is_null and (not event.isascii() or not event.isdecimal() or not 0 <= int(event) < 2**64):
            raise ValueError('Wait interface event must be a uint64 identity')
        event = None if is_null else str(int(event))
        if (region, event) in selectors:
            raise ValueError('Use one declared indicator per region/event')
        if is_null and (claim['producer'] is not None or not isinstance(claim['reason'], str)
                        or not claim['reason'].strip()):
            raise ValueError('waited(null) requires producer null and a nonempty manual-review reason')
        if not is_null and (not isinstance(claim['producer'], str) or not claim['producer']):
            raise ValueError('Wait interface producer must name a region')
        identities.add(identity); selectors.add((region, event))
        claims.append((dict(claim, event=event), Indicator(claim['indicator'], regions[region]['states'])))

    result = dict(status='checked', claims=[], unexplained=[], declarations=declaration,
        contract='Explicit indicators predict completed-wait occurrence per invocation, '
                 'including zero occurrences; not blocked duration. Inline indicators are assertions, not control flow. '
                 'Multiplicity is fitted separately on active invocations. Native coverage uses '
                 'wait-operation budgets, not inferred object pairing. Null refinement reasons '
                 'are for manual review, not automatically verified. Only captured executions are checked.')
    # No absent marker can be checked when the capture itself is incomplete.
    complete = (model.get('trace', {}).get('complete')
                and not model.get('validity', {}).get('errors')
                and not model.get('validity', {}).get('traceErrors')
                and events.get('coverage', {}).get('status') != 'unverified')
    calls = defaultdict(list)
    logical_calls, aliases, _ = wait_coverage.logical_context(
        model.get('trace', {}).get('events', []), model.get('waits', {}).get('events', []))
    for call in logical_calls.values():
        calls[call['region']].append(call)
    checkpoints = {e['id']: e for e in model.get('waits', {}).get('events', [])}
    by_call = defaultdict(list)
    for edge in events['edges'] + events.get('nullWaits', []):
        marker = checkpoints[edge['waited']]
        key = aliases.get(wait_coverage.invocation(marker), wait_coverage.invocation(marker))
        by_call[(edge['consumer'], edge['event']) + key].append(edge)

    for claim, indicator in claims:
        region, event = claim['region'], claim['event']
        names = regions[region]['states']
        row = dict(claim, status='checked', present=0, absent=0, unexpectedPresence=0,
                   unexpectedAbsence=0, producerMismatches=0, invalidDependencies=0,
                   unverifiedDependencies=0, counterexamples=[], multiplicity=None,
                   multiplicityStates=names)
        active_states, active_counts = [], []
        if complete:
            for call in calls[region]:
                matching = by_call[region, event, call['group'], str(call['thread']), call['seq']]
                predicted, observed = indicator(call['values']), bool(matching)
                row['present' if observed else 'absent'] += 1
                reason = ('unexpectedPresence' if observed and not predicted else
                          'unexpectedAbsence' if predicted and not observed else None)
                if reason:
                    row[reason] += 1
                mismatch = any(e.get('producer') is not None and e['producer'] != claim['producer']
                               for e in matching)
                row['producerMismatches'] += int(mismatch)
                row['invalidDependencies'] += sum(e['status'] == 'violation' for e in matching)
                row['unverifiedDependencies'] += sum(e['status'] == 'unverified' for e in matching)
                if (reason or mismatch) and len(row['counterexamples']) < 3:
                    row['counterexamples'].append(dict(kind=reason or 'producerMismatch',
                        group=call['group'], thread=call['thread'], invocation=call['seq'],
                        state=call['values'], predicted=int(predicted), observed=int(observed)))
                if predicted and observed:
                    active_states.append([composition._integer(call['values'][n]) for n in names])
                    active_counts.append(len(matching))
            if active_counts:
                row['multiplicity'] = composition.affine(active_states, active_counts, names)
            if any(row[n] for n in ('unexpectedPresence', 'unexpectedAbsence',
                                   'producerMismatches', 'invalidDependencies')):
                row['status'] = 'invalid'
            elif row['unverifiedDependencies'] or not calls[region]:
                row['status'] = 'unverified'
            elif not row['present']:
                row['status'] = 'not-exercised'
            elif row['multiplicity'] is None:
                row['status'] = 'unexplained-multiplicity'
        else:
            row['status'] = 'unverified'
        multiple = composition.affine_text(row['multiplicity'], names) if row['multiplicity'] else None
        row['prefix'] = f"I[{claim['indicator']}] * "
        if multiple and multiple != '1':
            row['prefix'] += f'({multiple}) * '
        elif row['status'] == 'unexplained-multiplicity':
            row['prefix'] += 'unexplained(multiplicity) * '
        row['reference'] = 'waited(null)' if event is None else f"Wait[{claim['producer']}]"
        if event is None:
            row['refinement'] = 'null'
            row['reasonReview'] = 'manual'
        row['term'] = row['prefix'] + row['reference']
        result['claims'].append(row)

    # No implicit True indicator: undeclared semantic checkpoints remain terms
    # with a known target, even when their event order is valid.
    undeclared = Counter((e['consumer'], e.get('producer') or '?') for e in events['edges']
                         if (e['consumer'], e['event']) not in selectors)
    for (region, producer), count in sorted(undeclared.items()):
        result['unexplained'].append(dict(region=region, producer=producer, count=count,
            reason='No declared indicator for the observed event', kind='undeclared-event',
            term=f'unexplained(Wait[{producer}])'))
    nulls = Counter(e['consumer'] for e in events.get('nullWaits', [])
                    if (e['consumer'], None) not in selectors)
    for region, count in sorted(nulls.items()):
        result['unexplained'].append(dict(region=region, producer=None, count=count,
            reason='waited(null) needs a declared indicator and manual-review reason',
            kind='undeclared-null', refinement='null', term='unexplained(waited(null))'))
    for residual in events['unexplained']:
        result['unexplained'].append(dict(residual, producer='?', kind='native-coverage',
                                         reason='Observed synchronization has no covering checkpoint'))
    statuses = {r['status'] for r in result['claims']}
    result['status'] = ('invalid' if 'invalid' in statuses else
                        'unverified' if not complete or statuses & {'unverified', 'not-exercised'} else
                        'unexplained' if result['unexplained'] or 'unexplained-multiplicity' in statuses else
                        'checked')
    warnings = [w for w in model.get('inlineWaitSourceWarnings', []) if w['region'] in regions]
    if warnings:
        result['declarationWarnings'] = warnings
        if result['status'] != 'invalid': result['status'] = 'unverified'
    return result


def terms(report, region):
    checks = report.get('interfaceChecks', {})
    return [r['term'] for r in checks.get('claims', []) + checks.get('unexplained', [])
            if r['region'] == region]
