"""Remove marker blocks and estimate wrapper overhead before cost fitting.

Inside calibration belongs to each region once. Outside calibration belongs
only to its direct parent, once per child call at that parent's state. Block
profiles avoid subtracting child work or hiding application irregularity in an
aggregate constant. Calibration is an estimate, with any unmatched part exposed.
"""
from collections import Counter, defaultdict
from pathlib import Path

import derive

CALIB = "_perfmark_calibration"


def is_marker(symbol):
    module, name = symbol[:2]
    module = Path(module).name
    return (module == 'libperfmark.so' or module.startswith('libperfmark.so.') or
            module == '_perfmark.so' or module.startswith('_perfmark.') and module.endswith('.so') or
            name in ('perfmark_begin', 'perfmark_begin_v', 'perfmark_end', 'perfmark_state'))


def _run(key):
    return key[0] if isinstance(key, tuple) else 0


def adjust(keys, slots, records, trace_errors=()):
    """Return corrected keys and per-region/state accounting; keep raw intact."""
    profiles = defaultdict(lambda: defaultdict(lambda: [Counter(), 0]))
    schemas = {}
    for key, row in keys.items():
        if not row.get('overflow') and row['count']:
            schemas.setdefault(row['region'], tuple(n for n, _ in row['states']))
        if row['region'] in (CALIB, CALIB+'_outer', CALIB+'_loop'):
            vector, _ = profiles[_run(key)][row['region']]
            vector.update({b: c for b, c in row['vec'].items()
                           if not is_marker(slots.get(b, ('?', '?'))) and
                           not derive.is_runtime(slots.get(b, ('?', '?')))})
            profiles[_run(key)][row['region']][1] += row['count']
    calibration = {}
    for run, entries in profiles.items():
        if not all(entries[name][1] for name in (CALIB, CALIB+'_outer', CALIB+'_loop')):
            continue
        inner, ni = entries[CALIB]
        outer, no = entries[CALIB+'_outer']
        loop, nl = entries[CALIB+'_loop']
        # Outer and loop vectors are already EXCLUSIVE; subtracting the child
        # inside profile here would charge its overhead twice.
        outside = {b: (outer[b]/no - loop[b]/nl)/(ni/no) for b in set(outer) | set(loop)}
        if sum(outside.values()) < -1e-9:
            continue  # A negative total wrapper cost is an unusable calibration.
        calibration[run] = ({b: c/ni for b, c in inner.items()}, outside)
    expected, seen, children = Counter(), Counter(), Counter()
    for key, row in keys.items():
        if not row.get('overflow'):
            expected[(_run(key), row['region'], tuple(v for _, v in row['states']))] += row['count']
    active = defaultdict(list)
    valid = not trace_errors
    for record in sorted(records, key=lambda r: (r.get('run', 0), r.get('pid', 0), r['seq'])):
        run, name = record.get('run', 0), record['region']
        stack = active[(run, record.get('pid', 0), record['tid'])]
        while stack and stack[-1][1] < record['seq']:
            stack.pop()
        if stack and record['seq_end'] >= stack[-1][1]:
            valid = False
        names = schemas.get(name)
        try:
            state = tuple(int(record.get('state', {})[n]) for n in names) if names is not None else None
        except (KeyError, TypeError, ValueError):
            state = None
        identity = (run, name, state)
        seen[identity] += 1
        if stack:
            children[stack[-1][0]] += 1
        stack.append((identity, record['seq_end']))
    # The fit aggregates by state across callers. Apply calibration at that
    # same scope, rather than allocating an average child count to each root
    # bucket and clipping it against the wrong caller's available blocks.
    grouped = {}
    for key, row in keys.items():
        identity = (_run(key), row['region'], tuple(row['states']), bool(row.get('overflow')))
        if identity not in grouped:
            grouped[identity] = (key, dict(row, vec=Counter(), count=0))
        merged = grouped[identity][1]
        merged['vec'].update(row['vec'])
        merged['count'] += row['count']
    corrected, totals = {}, defaultdict(lambda: defaultdict(float))
    for key, row in grouped.values():
        run = _run(key)
        state = tuple(v for _, v in row['states'])
        identity = (run, row['region'], state)
        count = row['count']
        result = dict(row)
        vector = {b: c for b, c in row['vec'].items() if not is_marker(slots.get(b, ('?', '?')))}
        removed = sum(row['vec'].values()) - sum(vector.values())
        summary = totals[(row['region'], state)]
        summary['calls'] += count
        summary['exactBlocks'] += removed
        summary['directChildCalls'] += (children[identity] * count / expected[identity]
                                         if valid and expected[identity] else 0)
        if run in calibration and not row['region'].startswith(CALIB):
            if valid and seen[identity] == expected[identity] and count and not row.get('overflow'):
                inside, outside = calibration[run]
                child_count = children[identity] / expected[identity]
                summary['calibratedCalls'] += count
                for b in set(inside) | set(outside):
                    correction = count * (inside.get(b, 0) + child_count * outside.get(b, 0))
                    # A signed differential profile can add back loop-control
                    # work removed by the baseline. Keep it signed, not abs().
                    available = vector.get(b, 0)
                    applied = min(available, correction)
                    vector[b] = available - applied
                    summary['wrapperEstimate'] += applied
                    summary['unmatchedEstimate'] += correction - applied
            else:
                summary['uncalibratedCalls'] += count
        elif not row['region'].startswith(CALIB):
            summary['uncalibratedCalls'] += count
        result['vec'] = {b: c for b, c in vector.items() if c}
        corrected[key] = result
    return corrected, {key: dict(value) for key, value in totals.items()}


def metadata(region, states, accounting):
    points = []
    for (name, state), row in sorted(accounting.items()):
        if name != region or not row['calls']:
            continue
        n = row['calls']
        points.append({'state': list(state), 'exactBlocks': row['exactBlocks']/n,
                       'wrapperEstimate': row.get('wrapperEstimate', 0)/n,
                       'unmatchedEstimate': row.get('unmatchedEstimate', 0)/n,
                       'directChildCalls': row.get('directChildCalls', 0)/n,
                       'calibratedCalls': row.get('calibratedCalls', 0),
                       'calls': n})
    return {'method': 'block profiles; own inside once + direct-child outside per state',
            'scope': 'marker modules excluded exactly; wrapper calibration is an estimate; caller-side PCV computation may remain',
            'points': points}
