"""One run produces the agent report, checked JSON, and standalone human graph."""
from collections import Counter, defaultdict
import base64
import gzip
import html
import json
from pathlib import Path
import secrets
import shlex
import tempfile

import event_model
import execution
import explorer
import source_coverage

ASSETS = Path(__file__).resolve().parents[1] / 'extensions/vscode/media'


def graph_lines(model):
    """A complete adjacency list: each region once, without unfolding shared DAGs."""
    graph = model['executionGraph']
    lines = ['ARCHITECTURE GRAPH',
             'contains = direct subregion; next = observed sibling order, not a dependency;',
             'waited -> = declared consumer-to-publisher claim. Check status is shown.',
             'Wait[?] = uncovered synchronization; waited(null) = event-free refinement with a supplied reason, not a dependency.',
             'Graph aggregates observed invocations; it is not a latency or critical-path model.']
    if graph['status'] != 'observed':
        return lines + ['Graph unavailable: ' + '; '.join(graph.get('warnings', []))]
    outgoing, edges = defaultdict(list), defaultdict(list)
    for edge in graph['edges']:
        if edge['kind'] != 'wait': outgoing[edge['source']].append(edge)
    report = model.get('eventModel', {})
    for edge in report.get('edges', []):
        edges[edge['consumer']].append(edge)
    checks = defaultdict(list)
    for row in report.get('interfaceChecks', {}).get('claims', []):
        checks[row['region']].append(row)
    unknown = Counter()
    for row in report.get('unexplained', []): unknown[row['region']] += row['count']
    q = lambda value: json.dumps(value, ensure_ascii=False)
    for region in sorted(model['regions'], key=lambda row: row['id']):
        name = region['id']
        lines += ['', f'REGION {q(name)} (calls={region["calls"]}; PCVs={", ".join(region["states"]) or "none"})']
        for source in region.get('sources', []):
            lines.append(f'  source: {source["path"]}:{source["line"]}')
        for edge in sorted(outgoing[name], key=lambda e: (e['kind'], e['target'], str(e.get('context')))):
            label = 'contains' if edge['kind'] == 'contains' else 'next'
            context = f'; within={q(edge["context"])}' if edge.get('context') else ''
            lines.append(f'  {label} -> {q(edge["target"])} [observations={edge["observations"]}{context}]')
        # Preserve separate event channels and rejected claims; do not turn
        # native object matching into an inferred semantic dependency arrow.
        grouped = Counter((e.get('producer'), e['event'], e['status']) for e in edges[name])
        for (producer, event, status), count in sorted(grouped.items(), key=lambda row: repr(row[0])):
            claims = [c for c in checks[name] if c.get('event') == event]
            annotation = '; '.join(c['term'] + ' [interface=' + c['status'] + ']' for c in claims)
            if not annotation: annotation = 'unexplained: indicator not declared'
            lines.append(f'  waited -> {q(producer or "?")} [event={event}; order={status}; observations={count}]')
            lines.append('    ' + annotation)
        for claim in checks[name]:
            if claim.get('event') is None or not any(e['event'] == claim['event'] for e in edges[name]):
                lines.append('  interface: ' + claim['term'] + ' [' + claim['status'] + ']')
                if claim.get('reason'): lines.append('    reason (manual review): ' + claim['reason'])
        if unknown[name]: lines.append(f'  unexplained(Wait[?]): {unknown[name]} obligations')
        for claim in checks[name]:
            for counterexample in claim.get('counterexamples', []):
                lines.append('  counterexample: ' + json.dumps(counterexample, ensure_ascii=False))
    return lines


def text_report(model):
    report = model.get('eventModel', {})
    graph = model['executionGraph']
    coverage = report.get('coverage', {})
    lines = ['DRPERF PERFORMANCE REPORT',
             'Command: ' + (shlex.join(model.get('provenance', {}).get('command', [])) or 'not recorded'),
             f'Regions: {len(model["regions"])}; graph: {graph["status"]}',
             'Declared event ordering: ' + report.get('status', 'disabled / not captured')]
    claims = report.get('interfaceChecks', {}).get('claims', [])
    pairs = {(e['consumer'], e.get('producer'), e['event']) for e in report.get('edges', [])}
    nulls = sum(c.get('event') is None for c in claims)
    lines += [f'Declared dependency channels observed: {len(pairs)}; waited(null) refinements: {nulls}.',
              'Null refinements do not establish publishers; checking their markers/indicators does not validate their reasons.']
    for consumer, producer, event in sorted(pairs, key=repr):
        lines.append(f'  {consumer} -> {producer or "?"} [event={event}]')
    if 'obligations' in coverage:
        lines.append(f'Wait obligations: {coverage["covered"]}/{coverage["obligations"]} covered; '
                     f'{coverage["uncovered"]} unexplained')
        if coverage['uncovered']:
            lines.append('WAIT COVERAGE INCOMPLETE: declared event ordering does not mean all synchronization is explained.')
    lines += ['Read the graph below for structure; COST INTERFACES AND BREAKDOWN for formulas and functions.',
              'Queries: drperf --report REPORT.json --stats | --region NAME | --top costly | --top unexplained | --full',
              'graph.html: interactive region graph and selected-region performance details.',
              'profile.drperf.json: machine-readable formulas, checks, state tables and graph.',
              'Numeric displays are rounded to integers; the JSON retains full precision.',
              'Evidence sidecars support rechecking; keep them with the JSON.']
    from report_query import metrics
    costs = metrics(model)
    lines += [f'Cost evidence: {sum(r["singleState"] for r in costs)}/{len(costs)} regions have one observed PCV state; '
              f'{sum(r["ownUnexplainedInstructions"] > 0 for r in costs)} have nonzero own residual; '
              f'{sum((r["interfaceUnexplainedInstructionsEstimate"] or 0) > 0 for r in costs)} have nonzero interface residual including unresolved child terms.',
              'A one-state constant fit explains that state mean only; it does not establish growth or per-call variation.']
    if model.get('provenance', {}).get('reportNote'):
        lines.append('Capture note: ' + model['provenance']['reportNote'])
    for error in model.get('validity', {}).get('errors', []) + model.get('validity', {}).get('traceErrors', []):
        lines.append('MEASUREMENT ERROR: ' + error)
    for failure in report.get('violations', []) + report.get('unverified', []):
        lines.append('WAIT CHECK: ' + json.dumps(failure, ensure_ascii=False))
    for warning in report.get('interfaceChecks', {}).get('declarationWarnings', []):
        lines.append('WAIT DECLARATION: ' + json.dumps(warning, ensure_ascii=False))
    for probe in model.get('waitProbePlan', []):
        lines.append(f'Optional publication probe: region={probe["region"]}; delayMs={probe["delayMs"]}; '
                     f'capped={probe["capped"]} (not executed by this run)')
    lines += ['', *source_coverage.lines(model.get('codeCoverage')), '', *graph_lines(model), '', 'COST INTERFACES AND BREAKDOWN',
              'CPU instructions on observed states; symbolic child costs use F[child].',
              *explorer.cost_lines(model), '', 'OBSERVED PCV RELATIONSHIPS',
              'These are observed equalities, not causal propagation or latency predictions.']
    lines += [explorer.relation_text(row) for row in model.get('relations', [])] or ['None fitted.']
    return '\n'.join(lines) + '\n'


def standalone_html(model):
    """Use the existing interactive viewer and ELK layout, without Node or a server."""
    nonce = secrets.token_urlsafe(24)
    def read(name): return (ASSETS / name).read_text()
    def script(source, attrs=''):
        source = source.replace('</script', '<\\/script').replace('</SCRIPT', '<\\/SCRIPT')
        return f'<script nonce="{nonce}" {attrs}>{source}</script>'
    def data_url(name):
        return 'data:text/javascript;base64,' + base64.b64encode(read(name).encode()).decode()
    data = json.dumps(dict(report=model, view=dict(mode='top')), ensure_ascii=False, separators=(',', ':'))
    payload = base64.b64encode(gzip.compress(data.encode(), mtime=0)).decode()
    scripts = [script('window.DrperfStandalone = true;'),
               script(read('expressions.js')), script(read('model.js')),
               script(read('analysis-client.js'), ' '.join(
                   f'data-{kind}="{data_url(name)}"' for kind, name in
                   [('expressions', 'expressions.js'), ('model', 'model.js'), ('worker', 'analysis-worker.js')])),
               *[script(read(name)) for name in ['execution-graph.js', 'vendor/elk.bundled.js',
                                                 'graph-layout.js', 'explorer.js']],
               script('''(async () => {
  try {
    const bytes = Uint8Array.from(atob('PAYLOAD'), c => c.charCodeAt(0));
    const json = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text();
    const {report, view} = JSON.parse(json);
    window.DrperfStandalone = view;
    window.postMessage({type:'model', model:report, selected:view.selected}, '*');
  } catch (error) { document.getElementById('app').textContent = 'Cannot open graph: '+error.message; }
})();'''.replace('PAYLOAD', payload))]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; worker-src blob:; connect-src data:; img-src data:;">
<title>drperf region graph</title><style>{read('explorer.css')}</style></head>
<body><template id="elk-license">ELK.js 0.12.0 — https://github.com/kieler/elkjs
{html.escape(read('vendor/ELK-LICENSE.md'))}</template><div id="app">Loading embedded profile…</div>
{''.join(scripts)}
</body></html>'''


def atomic_text(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary: temporary.unlink(missing_ok=True)


def write(model, destination, refresh_checks=False):
    """Destination is the JSON path; report.txt and graph.html are adjacent."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_coverage.ensure(model)
    if refresh_checks and model.get('waits'):
        model['eventModel'] = event_model.check(model)
    report = model.get('eventModel')
    if report and not report.get('probe') and not report.get('unverified'):
        model['waitProbePlan'] = event_model.probe_plan(report)
    graph = execution.build(model)
    graph['contract'] += ' This report draws declared wait claims only; native API observations remain coverage evidence.'
    model['executionGraph'] = graph
    prior = None
    if destination.exists():
        try:
            prior = json.loads(destination.read_text()).get('waits', {}).get('evidence', {}).get('path')
        except (OSError, ValueError): pass
    explorer.write_model(model, destination)
    compact = explorer.load_model(destination)
    text = text_report(compact).replace('profile.drperf.json: machine-readable', destination.name+': machine-readable')
    atomic_text(destination.parent/'report.txt', text)
    atomic_text(destination.parent/'graph.html', standalone_html(compact))
    # Only retire the evidence referenced by the previous report at this exact
    # destination. Do not remove other experiments or arbitrary files.
    current = compact.get('waits', {}).get('evidence', {}).get('path')
    if prior and prior != current and Path(prior).name == prior and prior.startswith(destination.name+'.waits.'):
        (destination.parent/prior).unlink(missing_ok=True)
    return dict(text=destination.parent/'report.txt', graph=destination.parent/'graph.html', profile=destination)
