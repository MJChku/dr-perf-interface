"""Read wait-interface assertions from passive markers and literal Python sites.

Runtime metadata is authoritative evidence of the declaration that executed.
Literal source sites additionally preserve never-executed declarations so a true
indicator with no checkpoint can fail. No source code or predicate is executed.
"""
import ast
from collections import defaultdict
from pathlib import Path


def source_claims(root, paths):
    import explorer
    claims, warnings = [], []
    if not root:
        return claims, warnings
    for path in explorer.source_files(root, paths):
        if path.suffix != '.py':
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeError, OSError):
            continue
        functions, modules = {}, set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'perfmark':
                functions.update((a.asname or a.name, a.name) for a in node.names)
            elif isinstance(node, ast.Import):
                modules.update(a.asname or a.name for a in node.names if a.name == 'perfmark')
        def name(call):
            f = call.func
            if isinstance(f, ast.Name):
                return functions.get(f.id)
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in modules:
                return f.attr
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try: constants[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
        def literal(node):
            if isinstance(node, ast.Name) and node.id in constants:
                return constants[node.id]
            return ast.literal_eval(node)
        def walk(node, region=None):
            if isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    c = item.context_expr
                    if isinstance(c, ast.Call) and name(c) in ('region', 'async_region') and c.args:
                        try: region = literal(c.args[0])
                        except (ValueError, TypeError): region = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # A function nested lexically in a region need not run in that region.
                region = None
                for c in node.decorator_list:
                    if isinstance(c, ast.Call) and name(c) in ('region', 'async_region') and c.args:
                        try: region = literal(c.args[0])
                        except (ValueError, TypeError): pass
            if isinstance(node, ast.Call) and name(node) == 'wait' and isinstance(region, str):
                args = dict(zip(('event', 'generation', 'indicator', 'producer', 'reason'), node.args))
                args.update((k.arg, k.value) for k in node.keywords if k.arg)
                try:
                    event = literal(args['event'])
                    indicator = literal(args['indicator'])
                    producer = literal(args['producer']) if 'producer' in args else None
                    reason = literal(args['reason']) if 'reason' in args else None
                    if event is not None and (type(event) is not int or not 0 <= event < 2**64):
                        raise ValueError('event must be uint64')
                    if not isinstance(indicator, str) or not indicator.strip():
                        raise ValueError('indicator must be a nonempty expression string')
                    c = dict(region=region, event=None if event is None else str(event),
                             indicator=indicator, producer=producer)
                    if event is None: c['reason'] = reason
                    claims.append(c)
                except (KeyError, ValueError, TypeError):
                    warnings.append(dict(region=region, source=f'{path}:{node.lineno}',
                        reason='Wait declaration is not statically resolvable; an entirely unexecuted site cannot be checked.'))
            for child in ast.iter_child_nodes(node):
                walk(child, region)
        walk(tree)
    return claims, warnings


def resolve(model, events):
    legacy = model.get('waitDeclarations', {'version': 1, 'claims': []})
    if not isinstance(legacy, dict) or legacy.get('version') != 1 or not isinstance(legacy.get('claims'), list):
        raise ValueError('waitDeclarations requires version 1 and a claims array')
    rows = {}
    originals = defaultdict(list)
    for r in model['regions']:
        originals[r.get('originalName', r['id'])].append(r['id'])
    producers = defaultdict(set)
    for e in events['edges']:
        if e.get('producer'):
            producers[e['consumer'], e['event']].add(e['producer'])
    def add(c, inline=False):
        c = dict(c)
        if c.get('event') is not None:
            c['event'] = str(c['event'])
            if c['event'].isascii() and c['event'].isdecimal():
                c['event'] = str(int(c['event']))
        key = c['region'], c.get('event')
        if inline:
            if c.get('producer') is None and c.get('event') is not None:
                possible = producers[key]
                c['producer'] = next(iter(possible)) if len(possible) == 1 else '?'
            c['id'] = f'inline:{key[0]}:{key[1]}'
        previous = rows.get(key)
        if previous:
            if {k:v for k,v in previous.items() if k != 'id'} != {k:v for k,v in c.items() if k != 'id'}:
                raise ValueError(f'Conflicting wait declarations for region/event {key}')
        else:
            rows[key] = c
    seen = set()
    for c in legacy['claims']:
        event = c.get('event')
        if event is not None:
            event = str(event)
            if event.isascii() and event.isdecimal(): event = str(int(event))
        key = c.get('region'), event
        if key in seen:
            raise ValueError('Use one declared indicator per region/event')
        seen.add(key)
        add(c)
    for c in model.get('inlineWaitSources', []):
        for region in originals.get(c['region'], []):
            add(dict(c, region=region), inline=True)
    for e in model.get('waits', {}).get('events', []):
        if e.get('declarationError'):
            raise ValueError('Invalid or truncated inline wait declaration')
        if 'indicator' not in e:
            continue
        if e['kind'] not in ('declared_waited', 'declared_waited_null'):
            continue
        c = dict(region=e['region'], event=None if e['kind'] == 'declared_waited_null' else str(e['object']),
                 indicator=e['indicator'], producer=e.get('producer'))
        if c['event'] is None: c['reason'] = e.get('reason')
        if c['region'] not in {r['id'] for r in model['regions']}:
            continue  # event checker retains out-of-region checkpoint as unverified
        add(c, inline=True)
    return dict(version=1, claims=list(rows.values()))
