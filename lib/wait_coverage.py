"""Count-based annotation coverage, not native-object/event association.

Each completed wait contributes an obligation. Consecutive failed attempts at
the same native API/object/site may form one retry chain, ending at success.
Successful operations are never merged just because their region is the same.
Allocation is local-first (children before parents), then oldest-ready-first;
it is an accounting witness, not an inferred causal connection.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import heapq


def logical_operations(ops, checkpoint_orders=()):
    """Conservative retry grouping: never infer that two successes are retries."""
    result = []
    orders = sorted(checkpoint_orders)
    previous = None
    for op in sorted(ops, key=lambda o: int(o['start'])):
        signature = tuple(str(op.get(k, '')) for k in
                          ('api', 'object', 'aux', 'callerModule', 'callerOffset'))
        # Runtime adapters already delimit the entire await, including retries.
        retry = (previous is not None and op.get('kind') != 'runtime_wait'
                 and signature == previous[0]
                 and previous[1].get('returned')
                 and failed(previous[1])
                 and bisect_right(orders, int(previous[1]['end'])) ==
                     bisect_right(orders, int(op['start'])-1))
        if retry:
            result[-1].append(op)
        else:
            result.append([op])
        previous = (signature, op)
    return result


def failed(op):
    result = op.get('result')
    if result is None:
        return False
    return result < 0 if op.get('kind') == 'receive' else result != 0


def logical_context(traces, scope_events=()):
    """Join active steps of explicitly delimited async invocations.

    Ordinary synchronous invocations keep their identity. Task scopes are
    process-unique; sharing an OS thread does not establish ancestry.
    """
    physical = {invocation(t, True): t for t in traces}
    parents, stacks = {}, defaultdict(list)
    for key, call in sorted(physical.items(), key=lambda pair: (pair[0][0], pair[0][2])):
        stack = stacks[key[:2]]
        while stack and int(physical[stack[-1]]['end']) < key[2]:
            stack.pop()
        parents[key] = stack[-1] if stack else None
        stack.append(key)
    scopes, owner = {}, {}
    for e in scope_events:
        if e['kind'] != 'runtime_scope' or int(e['object']) == 0:
            continue
        key = invocation(e)
        if key not in physical:
            continue  # restoration outside a counted step
        scope = (e['group'], str(e['object']))
        # A restoration in an outer region repeats that region's scope.
        if key in owner and owner[key] != scope:
            raise ValueError('Conflicting async scope for a region invocation')
        owner[key] = scope
        row = scopes.setdefault(scope, dict(keys=set(), parent=str(e['aux'])))
        if row['parent'] != str(e['aux']):
            raise ValueError('Async scope changed parent')
        row['keys'].add(key)
    aliases = {key: key for key in physical}
    for scope, row in scopes.items():
        keys = sorted(row['keys'], key=lambda key: key[2])
        first = keys[0]
        for key in keys:
            if (physical[key]['region'], physical[key]['values'], key[:2]) != (
                    physical[first]['region'], physical[first]['values'], first[:2]):
                raise ValueError('Async region steps changed region, thread, or entry PCVs')
            aliases[key] = first
        row['first'] = first
    calls = {key: dict(call) for key, call in physical.items() if aliases[key] == key}
    logical_parents = {}
    for key, parent in parents.items():
        target = aliases[key]
        actual_parent = aliases.get(parent)
        if key in owner:
            row = scopes[owner[key]]
            if row['parent'] != '0':
                parent_scope = scopes.get((key[0], row['parent']))
                if parent_scope is None:
                    raise ValueError('Async scope references an uncaptured parent')
                actual_parent = parent_scope['first']
        if target in logical_parents and logical_parents[target] != actual_parent:
            raise ValueError('Async invocation changed parent across steps')
        logical_parents[target] = actual_parent
    # Validate cycles iteratively (deeply nested synchronous code is supported).
    postorder(calls, logical_parents)
    return calls, aliases, logical_parents


def postorder(calls, parents):
    children = defaultdict(list)
    for key in calls:
        children[parents.get(key)].append(key)
    result, stack = [], [(key, False) for key in children[None]]
    while stack:
        key, done = stack.pop()
        if done:
            result.append(key)
        else:
            stack.append((key, True))
            stack.extend((child, False) for child in children[key])
    if len(result) != len(calls):
        raise ValueError('Cyclic or missing invocation parent')
    return result


def invocation(record, trace=False):
    return (record['group'], str(record['thread' if trace else 'tid']),
            int(record['seq' if trace else 'regionSeq']))


def eligible(checkpoint):
    return checkpoint['status'] == 'ordered' or (
        checkpoint.get('refinement') == 'null' and checkpoint['status'] == 'recorded')


def check(operations, checkpoints, edges, context):
    calls, aliases, parents = context
    direct = defaultdict(list)
    runtime_ids = {op['id'] for op in operations if op.get('kind') == 'runtime_wait'
                   and op.get('returned') and invocation(op) in aliases}
    nested_native = 0
    for op in operations:
        if op.get('runtimeContainer') in runtime_ids:
            nested_native += 1
            continue
        key = aliases.get(invocation(op), invocation(op))
        if key in calls:
            direct[key].append(op)
        elif int(op['regionSeq']) != 0:
            # The exporter deliberately omits annotation/PCV-only regions.
            name = op.get('region', '')
            if not name.startswith(('_perfmark', 'perf.')) and ':' not in name:
                raise ValueError('Synchronization references an uncaptured region invocation')
    declarations = defaultdict(list)
    by_id = {e['id']: e for e in checkpoints}
    for edge in edges:
        edge['coverageCredit'] = None
        if eligible(edge):
            marker = by_id[edge['waited']]
            declarations[aliases.get(invocation(marker), invocation(marker))].append((int(marker['start']), edge))

    stats = {}
    obligations = {}
    owned = defaultdict(list)
    # Multiple obligations can belong to one invocation. An unfinished call
    # cannot be discharged by a completed-wait annotation.
    for key, ops in direct.items():
        call = calls[key]
        row = stats.setdefault(call['region'], dict(region=call['region'],
            obligations=0, covered=0, uncovered=0, nativeCalls=0, runtimeCalls=0, examples=[]))
        row['nativeCalls'] += sum(o.get('kind') != 'runtime_wait' for o in ops)
        row['runtimeCalls'] += sum(o.get('kind') == 'runtime_wait' for o in ops)
        for number, attempts in enumerate(logical_operations(ops, [s for s, _ in declarations[key]])):
            origin = key + (number,)
            owned[key].append(origin)
            row['obligations'] += 1
            obligations[origin] = obligation(call, key, attempts)

    pending = defaultdict(list)
    uncovered = []
    used = 0
    for key in postorder(calls, parents):
        heap = pending.pop(key, [])
        for origin in owned[key]:
            item = obligations[origin]
            heapq.heappush(heap, (item['ready'] if item['complete'] else float('inf'), origin))
        for start, edge in sorted(declarations[key], key=lambda p: p[0]):
            if heap and heap[0][0] < start:
                _, origin = heapq.heappop(heap)
                item = obligations[origin]
                stats[item['region']]['covered'] += 1
                used += 1
                edge['coverageCredit'] = dict(region=item['region'], group=origin[0],
                    thread=origin[1], invocation=origin[2], obligation=origin[3],
                    operations=item['operations'])
        parent = parents[key]
        if parent is not None:
            other = pending[parent]
            if len(heap) > len(other):
                heap, other = other, heap
                pending[parent] = other
            for entry in heap:
                heapq.heappush(other, entry)
        else:
            uncovered.extend(origin for _, origin in heap)

    for key in sorted(uncovered):
        item = obligations[key]
        row = stats[item['region']]
        row['uncovered'] += 1
        if len(row['examples']) < 3:
            row['examples'].append(item)
    total = len(obligations)
    assert total == used + len(uncovered)
    rows = [stats[name] for name in sorted(stats)]
    return dict(status='uncovered' if uncovered else 'covered', version=2,
        unit='observed wait operations (failed native retries grouped)',
        obligations=total, covered=used, uncovered=len(uncovered),
        eligibleCredits=sum(eligible(e) for e in edges),
        unusedCredits=sum(eligible(e) for e in edges)-used,
        nativeCalls=sum(row['nativeCalls'] for row in rows),
        runtimeCalls=sum(row['runtimeCalls'] for row in rows),
        nestedNativeCalls=nested_native, regions=rows,
        contract='Distinct completed waits retain distinct obligations, including within '
                 'one region. Consecutive failed attempts at one native API/object/site '
                 'are grouped until success. Native layers inside a scoped runtime primitive '
                 'are retained as evidence without a second obligation. Credits are never reused or banked. '
                 'Coverage allocations are accounting witnesses, not producer matches.')


def obligation(call, key, ops):
    complete = all(o.get('returned') and int(o.get('end', 0)) > int(o['start']) for o in ops)
    item = dict(region=call['region'], group=key[0], thread=key[1],
        invocation=key[2], nativeCalls=sum(o.get('kind') != 'runtime_wait' for o in ops),
        runtimeCalls=sum(o.get('kind') == 'runtime_wait' for o in ops),
        apis=dict(Counter(o['api'] for o in ops)),
        ready=max(int(o['end']) for o in ops) if complete else None,
        complete=complete, operations=[o['id'] for o in ops])
    callers = Counter((o['api'], o['callerModule'], str(o['callerOffset']))
                      for o in ops if 'callerModule' in o and 'callerOffset' in o)
    if callers:
        item['callers'] = [dict(api=api, module=module, offset=offset, calls=count)
            for (api, module, offset), count in callers.most_common(3)]
        item['otherCallerCalls'] = sum(callers.values()) - sum(c['calls'] for c in item['callers'])
    return item
