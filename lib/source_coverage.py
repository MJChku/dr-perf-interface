"""Source annotation coverage; do not equate instruction fit with path coverage."""
from collections import Counter
import ast
import hashlib
import io
from pathlib import Path
import re
import tokenize


def marked_lines(source_root, locations, source_paths=None):
    """Static lexical coverage, independent of the workload and fitted states."""
    if not source_root or not Path(source_root).is_dir():
        return dict(status='unavailable', reason='No accessible source root.')
    import explorer
    root = Path(source_root).resolve()
    spans = {}
    for sources in locations.values():
        for source in sources:
            if 'path' in source:
                spans.setdefault(source['path'], []).append((source['line'], source['endLine']))
    files, skipped = [], []
    paths = list(explorer.source_files(root, source_paths, limit=5001))
    truncated = len(paths) > 5000
    for path in paths[:5000]:
        name = str(path.relative_to(root))
        try:
            if path.stat().st_size > 2_000_000:
                raise ValueError('file exceeds 2 MB scan limit')
            raw = path.read_bytes()
            text = raw.decode('utf-8')
            file_spans = list(spans.get(name, []))
            if path.suffix == '.py':
                # Lexical coverage does not require a literal region name.
                # Dynamic names still delimit Python context/decorator bodies.
                for node in ast.walk(ast.parse(text)):
                    calls = ([item.context_expr for item in node.items]
                             if isinstance(node, (ast.With, ast.AsyncWith)) else
                             node.decorator_list if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [])
                    for call in calls:
                        if not isinstance(call, ast.Call):
                            continue
                        label = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, 'id', '')
                        if label in ('region', 'async_region', 'marked', 'active_marked'):
                            file_spans.append((call.lineno, node.end_lineno))
                # Lines carrying tokens, excluding comments and whitespace.
                code = set()
                ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                           tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING}
                for token in tokenize.generate_tokens(io.StringIO(text).readline):
                    if token.type not in ignored:
                        code.update(range(token.start[0], token.end[0]+1))
                code &= {i for i,line in enumerate(text.splitlines(),1) if line.strip()}
            else:
                clean = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/',
                    lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0])
                    if m[0].startswith('/') else m[0], text)
                code = {i for i,line in enumerate(clean.splitlines(),1) if line.strip()}
            covered = {i for start,end in file_spans for i in range(start,end+1)} & code
            uncovered = sorted(code-covered)
            ranges = []
            for line in uncovered:
                if ranges and ranges[-1][1]+1 == line:
                    ranges[-1][1] = line
                else:
                    ranges.append([line,line])
            files.append(dict(path=name, total=len(code), marked=len(covered),
                              unmarkedRanges=ranges, sha256=hashlib.sha256(raw).hexdigest()))
        except (OSError, UnicodeError, ValueError, tokenize.TokenError, SyntaxError) as error:
            skipped.append(dict(path=name, reason=str(error)))
    total, marked = sum(f['total'] for f in files), sum(f['marked'] for f in files)
    return dict(kind='marked-source-lines', status='partial' if skipped or truncated else 'measured',
        sourceRoot=str(root), total=total, marked=marked, unmarked=total-marked,
        share=marked/total if total else None, files=files, skipped=skipped, truncated=truncated,
        contract='Nonblank, noncomment source lines lexically inside discovered region spans / all such lines '
            'in the selected source tree. Nested spans count once; unexecuted regions still count. '
            'Includes marker/decorator lines and string literals. Callees outside a marked lexical span '
            'are not counted as marked. Supports Python, C/C++, headers and Rust; discovery exclusions '
            'and scan limits apply. Python region contexts/decorators support dynamic names; '
            'unrecognized wrappers and unmatched manual begin/end spans need explicit source support.')


def build(model, locations, source_root, source_paths=None):
    calls = Counter()
    for region in model['regions']:
        calls[region.get('originalName', region['id'])] += region['calls']
    complete = bool(model.get('trace', {}).get('complete')) and not (
        model.get('validity', {}).get('errors') or model.get('validity', {}).get('traceErrors'))
    declared = {name: sources for name, sources in locations.items()
                if not name.startswith(('_perfmark', 'perf.')) and ':' not in name}
    rows = [dict(region=name, calls=calls[name], sources=sources,
                 status='observed' if calls[name] else 'not-observed' if complete else 'unknown',
                 siteAttribution='ambiguous' if len(sources) > 1 else 'unique-name')
            for name, sources in sorted(declared.items())]
    return dict(kind='annotated-region names', status='observed' if complete else 'incomplete',
        sourceRoot=str(source_root) if source_root else None,
        declared=len(rows), observed=sum(row['calls'] > 0 for row in rows),
        notObserved=sum(row['status'] == 'not-observed' for row in rows), regions=rows,
        withoutSource=[dict(region=name,calls=count) for name,count in sorted(calls.items()) if name not in declared],
        lines=marked_lines(source_root, locations, source_paths),
        contract='Coverage of literal region names discovered in the selected source tree, not whole-program '
                 'line or branch coverage. Multiple sites sharing one name cannot be distinguished. '
                 'Dynamic marker names and excluded/unscanned source files are outside the denominator. '
                 'Source discovery is bounded to 5000 files and skips files over 2 MB. '
                 'A not-observed region has unknown cost; a complete fit says nothing about unexecuted code.')


def lines(coverage):
    if not coverage:
        return ['CODE COVERAGE', 'UNAVAILABLE: annotation coverage was not checked.']
    if coverage.get('status') == 'unavailable':
        return ['CODE COVERAGE', 'UNAVAILABLE: ' + coverage['reason']]
    source = coverage.get('lines', {})
    rows = ['CODE COVERAGE']
    if source.get('kind') == 'marked-source-lines':
        percent = f'{100*source["share"]:.0f}%' if source['share'] is not None else 'n/a'
        rows += [f'Source lines inside marked regions: {source["marked"]}/{source["total"]} ({percent}); '
                 f'unmarked: {source["unmarked"]}; status: {source["status"]}', source['contract']]
        rows.extend(f'  {f["path"]}: {f["marked"]}/{f["total"]} lines marked' for f in source['files'])
    else:
        rows.append('Marked source-line coverage unavailable: ' + source.get('reason', 'not present in this saved report'))
    rows += [
            f'Annotated region names observed: {coverage["observed"]}/{coverage["declared"]}; '
            f'not observed: {coverage["notObserved"]}; status: {coverage["status"]}',
            coverage['contract']]
    if coverage.get('basis'):
        rows.append('Evidence: ' + coverage['basis'])
    for row in coverage['regions']:
        if row['status'] != 'observed' or row['siteAttribution'] == 'ambiguous':
            sources = ', '.join(f'{s["path"]}:{s["line"]}' for s in row['sources'])
            rows.append(f'  {row["region"]}: {row["status"]}; {row["siteAttribution"]}; {sources}')
    for row in coverage['withoutSource']:
        rows.append(f'  {row["region"]}: observed, annotation source not located')
    return rows


def ensure(model):
    """Recover a source comparison for old reports, without inventing capture coverage."""
    if model.get('codeCoverage', {}).get('lines', {}).get('kind') == 'marked-source-lines':
        return
    root = model.get('provenance', {}).get('sourceRoot')
    def unavailable(reason):
        model['codeCoverage'] = dict(status='unavailable', reason=reason,
            lines=dict(status='not-collected'), branches=dict(status='not-collected'))
    if not root or not Path(root).is_dir():
        unavailable('Saved capture has no accessible source tree for annotation discovery.')
        return
    root = Path(root).resolve()
    recorded = {(s['path'], s['sha256']) for r in model['regions']
                for s in r.get('sources', []) if s.get('sha256')}
    if not recorded:
        unavailable('Saved capture has no source hashes; current annotations cannot be verified.')
        return
    mismatches = []
    for name, digest in sorted(recorded):
        path = (root / name).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            mismatches.append(name)
    if mismatches:
        unavailable('Saved source hashes differ or files are missing: ' + ', '.join(mismatches))
        return
    import explorer
    coverage = build(model, explorer.source_locations(root, None), root)
    coverage['basis'] = (f'Saved invocation counts compared with the current source annotation inventory; '
        f'{len(recorded)} recorded source hashes match. Files without recorded hashes are not '
        'verified as capture-time sources. The application was not rerun.')
    coverage['sourceComparison'] = 'post-capture'
    coverage['verifiedSourceFiles'] = len(recorded)
    model['codeCoverage'] = coverage
