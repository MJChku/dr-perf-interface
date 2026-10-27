"""Independent GPU collector, translation command and resource-formula reports.

Only the numerical affine checker is shared with CPU drperf. This module never
loads DynamoRIO or calls CPU perfmark, runner, composition, or wait analysis.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import re
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import native_fit

ROOT = Path(__file__).resolve().parents[1]
GPU = ROOT / 'gpu'
RUNTIME = GPU / 'build/libdrperf_gpu.so'
TRANSLATOR = GPU / 'vendor/gxptx/.lake/build/bin/gxptx'
SCHEMA = 'drperf.gpu.v1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_build():
    if not RUNTIME.is_file() or not TRANSLATOR.is_file():
        raise ValueError('GPU backend missing; run `drperf --gpu build`')
    native_fit.library()


PTX_FUNC = re.compile(r'\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([A-Za-z_$][\w$]*)')
CPP_FUNC = re.compile(r'^extern "C".*?\bvoid\s+([A-Za-z_$][\w$]*)\(void\*\* args\)')
CPP_COUNTER = r'drperf_gpu_instruction\((?:__func__|"[^"]+"), "'
CPP_SITE = re.compile(CPP_COUNTER + r'([^"]+)", __LINE__')


def ptx_instructions(ptx_text):
    """{function: [(ptx_line, opcode, 'file:line' or None)]} in PTX order."""
    # nvcc emits the .file table after the functions that reference it.
    files = dict(re.findall(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', ptx_text, re.M))
    out, func, loc, depth, inlined_at = {}, None, None, 0, {}
    for number, raw in enumerate(ptx_text.splitlines(), 1):
        line = raw.split('//')[0].strip()
        if not line or line.startswith('.file'):
            continue
        m = PTX_FUNC.search(line)
        if m and depth == 0:
            func, loc = m.group(1), None
            out.setdefault(func, [])
        m = re.match(r'\.loc\s+(\d+)\s+(\d+)\s+(\d+)(?:.*inlined_at\s+(\d+)\s+(\d+)\s+(\d+))?', line)
        if m:
            # Inlined helper code: follow inlined_at to the nearest caller in another
            # file and print "helper<caller" (e.g. barrier.h:424<sm100_mqa_logits.cuh:236).
            frame = m.group(1, 2, 3)
            if m.group(4):
                inlined_at[frame] = m.group(4, 5, 6)
            chain, seen = [frame], {frame}
            while chain[-1] in inlined_at and inlined_at[chain[-1]] not in seen and len(chain) < 32:
                chain.append(inlined_at[chain[-1]]); seen.add(chain[-1])
            name = lambda f: f'{Path(files.get(f[0], "?")).name}:{f[1]}'
            caller = next((f for f in chain[1:] if f[0] != chain[0][0]), None)
            loc = name(chain[0]) + (f'<{name(caller)}' if caller else '')
            continue
        if func is None:
            continue
        for part in line.replace('{', ';').replace('}', ';').split(';'):
            part = part.strip()
            if depth == 0 and not part:
                continue
            if not part or part.startswith('.') or part.endswith(':') or part.startswith('('):
                continue
            part = re.sub(r'^[A-Za-z_$][\w$]*:\s*(?!:)', '', part)  # inline label (not 'shared::')
            part = re.sub(r'^@!?\S+\s+', '', part)  # predicate guard
            opcode = part.split()[0] if part.split() else ''
            if opcode and re.match(r'^[a-z]', opcode):
                out[func].append((number, opcode, loc))
        depth += raw.count('{') - raw.count('}')
        if depth <= 0:
            depth = 0
    return out


IFCONV_MAX = 16
IFCONV_STOP = re.compile(CPP_COUNTER + r'(brx|bar|barrier|mbarrier|ret|exit|call|'
                         r'griddepcontrol|fence|membar|nanosleep|trap)')
CPP_LABEL = re.compile(r'\s*(L_\w+): ;')
CPP_BRANCH = re.compile(CPP_COUNTER + r'(bra(?:\.uni)?)", __LINE__')


def mark_if_conversion(cpp_text):
    """Tag branches of short forward regions as "<op>.ifconv".

    ptxas predicates short forward hammocks (`@p bra L; ld; mul; L:`, small
    if/else or switch diamonds) and schedules across them; the basic-block
    latency schedule otherwise resets its issue floor at every branch and
    serializes, e.g., guarded loads of an unrolled loop. A region starts at a
    conditional forward branch and extends to the furthest label any branch
    inside it targets; it qualifies with at most IFCONV_MAX instrumented
    instructions, no backward branch, no barrier/fence/call/return, and no jump
    into it from outside."""
    lines = cpp_text.split('\n')
    labels, gotos, branches = {}, defaultdict(list), {}
    for i, line in enumerate(lines):
        m = CPP_LABEL.match(line)
        if m:
            labels[m.group(1)] = i
        for target in re.findall(r'goto (L_\w+);', line):
            gotos[target].append(i)
        if CPP_BRANCH.search(line):
            m = re.search(r'goto (L_\w+);', ' '.join(lines[i:i+3]))
            if m:
                branches[i] = (m.group(1), line.lstrip().startswith('if ('))
    count = [0]
    for line in lines:
        count.append(count[-1] + bool(CPP_SITE.search(line)))
    marked = set()
    for start, (target, conditional) in sorted(branches.items()):
        if not conditional or start in marked or labels.get(target, -1) <= start:
            continue
        end, ok = labels[target], True
        while ok:
            inner = [k for k in branches if start < k < end]
            grown = max([labels.get(branches[k][0], -1) for k in inner] + [end])
            ok = all(labels.get(branches[k][0], -1) > start for k in inner)
            if grown == end:
                break
            end = grown
        ok = ok and count[end] - count[start + 1] <= IFCONV_MAX
        ok = ok and not any(IFCONV_STOP.search(lines[k]) for k in range(start, end))
        # No jump from outside into a label strictly inside the region.
        ok = ok and all(start <= g < end for name, at in labels.items() if start < at < end for g in gotos[name])
        if ok:
            marked.update(k for k in branches if start <= k < end)
    for k in marked:
        for j in range(k, min(k+3, len(lines))):
            lines[j] = re.sub(r'"(bra(?:\.uni)?)", __LINE__', r'"\1.ifconv", __LINE__', lines[j])
    return '\n'.join(lines), len(marked)


def build_site_map(ptx_text, cpp_text):
    """Align translated instruction sites (C++ lines) with PTX lines, in order.

    Returns {cpp_function: {cpp_line: [ptx_line, 'file:line' or None]}}; sites the
    alignment cannot place are omitted. The translator keeps PTX order within a
    function, so a greedy opcode match with bounded lookahead suffices."""
    ptx = ptx_instructions(ptx_text)
    result, func, j, stream = {}, None, 0, []
    for number, line in enumerate(cpp_text.splitlines(), 1):
        m = CPP_FUNC.match(line)
        if m and '{' in line or (m and line.rstrip().endswith(')')):
            func = m.group(1)
            stream, j = ptx.get(func.removesuffix('__gxptx'), []), 0
            result.setdefault(func, {})
        m = CPP_SITE.search(line)
        if not m or func is None:
            continue
        op = m.group(1).removesuffix('.ifconv')
        for k in range(j, min(len(stream), j + 64)):
            if stream[k][1] == op:
                result[func][number] = [stream[k][0], stream[k][2]]
                j = k + 1
                break
    return {f: sites for f, sites in result.items() if sites}


def translate(argv):
    parser = argparse.ArgumentParser(prog='drperf --gpu translate')
    parser.add_argument('ptx', type=Path)
    parser.add_argument('-o', '--output', required=True, type=Path)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    args = parser.parse_args(argv)
    require_build()
    source = args.ptx.resolve()
    output = args.output.resolve()
    cpp_output = Path(str(output)+'.cpp')
    manifest_output = Path(str(output)+'.gpu.json')
    if source in (output, cpp_output, manifest_output):
        raise ValueError('GPU outputs must differ from their PTX input')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.gpu-translate-', dir=output.parent) as scratch:
        scratch = Path(scratch)
        cpp = scratch/'module.cpp'
        env = {k: v for k, v in os.environ.items()
               if k not in ('LD_PRELOAD', 'GXPTX_SPLIT')}
        env['GXPTX_PRELUDE'] = str(GPU/'vendor/gxptx/Gxptx/prelude.h')
        subprocess.run([str(TRANSLATOR), str(source), str(cpp)], env=env, check=True)
        if cpp.read_text().count('drperf_gpu_instruction(') <= 1:
            raise ValueError('translator emitted no GPU counters; rebuild the instrumented GPU translator')
        if not re.search(r'drperf_gpu_instruction\("[^"]+:bb[0-9]+",', cpp.read_text()):
            raise ValueError('translator has no PTX basic-block IDs; run `drperf --gpu build`')
        text, _ = mark_if_conversion(cpp.read_text())
        cpp.write_text(text + '\nextern "C" const unsigned drperf_gpu_instrumented_abi = 3;\n')
        subprocess.run([args.cxx, '-std=c++17', '-O1', '-fPIC', '-shared',
                        '-fno-strict-aliasing', '-fwrapv', '-Wl,-z,defs',
                        str(cpp), '-L'+str(RUNTIME.parent), '-ldrperf_gpu',
                        '-Wl,-rpath,'+str(RUNTIME.parent), '-o', str(scratch/'module.so')],
                       env=env, check=True)
        site_map = build_site_map(source.read_text(errors='replace'), cpp.read_text())
        os.replace(cpp, cpp_output)
        os.replace(scratch/'module.so', output)
    sites_output = Path(str(output)+'.sites.json')
    sites_output.write_text(json.dumps({'schema': 'drperf.gpu.sites.v1', 'input': str(source),
                                        'functions': site_map}, separators=(',', ':')))
    manifest = {
        'schema': SCHEMA, 'backend': 'gpu', 'instrumentationAbi': 3, 'input': str(source),
        'inputSha256': sha256(source), 'translatorSha256': sha256(TRANSLATOR),
        'preludeSha256': sha256(GPU/'vendor/gxptx/Gxptx/prelude.h'),
        'runtimeSha256': sha256(RUNTIME), 'moduleSha256': sha256(output),
        'generatedCpp': str(cpp_output), 'siteMap': str(sites_output),
        'fittingUnit': 'PTX basic block; IDs emitted by Lean translation',
        'counting': 'executed PTX operations; predication respected; FMA = 2; '
                    'tensor tile work counted once per collective; requested bytes',
    }
    manifest_output.write_text(json.dumps(manifest, indent=2)+'\n')
    print('drperf GPU translated:', output)
    return 0


def read_capture(paths):
    rows = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not records or records[0].get('event') != 'start' or records[-1].get('event') != 'finish':
            raise ValueError(f'incomplete GPU capture: {path}')
        if sum(r.get('event') == 'start' for r in records) != 1 or sum(r.get('event') == 'finish' for r in records) != 1:
            raise ValueError(f'ambiguous GPU process lifecycle: {path}')
        if records[-1].get('open_regions') or records[-1].get('launch_failures'):
            raise ValueError(f'GPU capture has open markers or failed launches: {path}')
        completed = records[-1].get('regions_completed')
        if completed != len(records)-2 or records[-1].get('regions_started') != completed:
            raise ValueError(f'GPU region record count mismatch: {path}')
        if {r.get('token') for r in records[1:-1]} != set(range(1, completed+1)):
            raise ValueError(f'duplicate or missing GPU invocation tokens: {path}')
        pid = records[0]['pid']
        for r in records[1:-1]:
            if r.get('backend') != 'gpu' or r.get('schema') != 1 or r.get('pid') != pid:
                raise ValueError(f'invalid GPU record: {path}')
            if not isinstance(r.get('region'), str) or not r['region'] or not isinstance(r.get('pcvs'), dict):
                raise ValueError(f'invalid GPU region or PCVs: {path}')
            if len(r['pcvs']) > 256:
                raise ValueError('too many GPU PCVs')
            values = list(r['pcvs'].values()) + list(r['metrics'].values())
            values += [v for m in r['sites'].values() for v in m.values()]
            if any(type(v) is not int or abs(v) > 2**53 for v in values):
                raise ValueError('GPU counts/PCVs exceed exact double range or are not integers')
            if any(v < 0 for v in r['metrics'].values()):
                raise ValueError('negative GPU work count')
            rows.append(r)
            # Tokens/PIDs are process-local and can recur in independent captures.
            r['captureId'] = str(path.resolve())
    if not rows:
        raise ValueError('no GPU marker invocations were recorded')
    return rows


def capture_modules(paths):
    """Translated modules the captured processes loaded, with their site maps."""
    modules = set()
    for path in paths:
        lines = [line for line in path.read_text().splitlines() if line]
        modules.update(json.loads(lines[-1]).get('modules', []) if lines else [])
    return [{'module': m, 'siteMap': m+'.sites.json'} for m in sorted(modules)]


def load_site_maps(provenance):
    """{cpp_function: {cpp_line: [ptx_line, 'file:line' or None]}} from report provenance."""
    merged = {}
    for module in (provenance or {}).get('modules', []):
        try:
            data = json.loads(Path(module['siteMap']).read_text())
        except (OSError, ValueError, KeyError):
            continue
        for func, sites in data.get('functions', {}).items():
            merged.setdefault(func, {}).update(sites)
    return merged


def clean(value):
    if not math.isfinite(value):
        raise ValueError('nonfinite GPU fit')
    nearest = round(value)
    return nearest if abs(value-nearest) < 1e-8 else value


def diagnostic_metric(metric):
    # The old issue model is an assumption, not an observed resource count.
    return metric.startswith(('latency_', 'emulator_')) or metric == 'dependent_global_loads_on_critical_path'


def additive_metric(metric):
    return (not diagnostic_metric(metric) and '_unique_' not in metric
            and not metric.startswith(('sum_block_', 'sum_launch_')))


def state_groups(rows, pcvs):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row['pcvs'][p] for p in pcvs)].append(row)
    return sorted(groups.items())


def fit_metrics(rows, pcvs, get_counts, absolute, relative, *, means=True):
    if means:
        original = get_counts
        rows = [{'pcvs': dict(zip(pcvs, state)), 'counts': {
            metric: sum(original(r).get(metric, 0) for r in group)/len(group)
            for metric in set().union(*(original(r) for r in group))}}
            for state, group in state_groups(rows, pcvs)]
        get_counts = lambda r: r['counts']
    values = [[r['pcvs'][p] for p in pcvs] for r in rows]
    metrics = sorted(m for m in set().union(*(get_counts(r) for r in rows)) if not diagnostic_metric(m))
    if not metrics:
        return {}
    plan = native_fit.Plan(values)
    try:
        observations = [[get_counts(r).get(metric, 0) for r in rows] for metric in metrics]
        result = {}
        for metric, ys, fit in zip(metrics, observations, plan.fit(observations, absolute, relative)):
            k = len(pcvs)
            def errors(a, c):
                return [abs(y-(sum(v*x for v, x in zip(a, state))+c))
                        for state, y in zip(values, ys)]
            def exact(a, c):
                return all(e <= max(absolute, relative*y) for e, y in zip(errors(a, c), ys))
            # Least squares on large integer counts leaves float noise (e.g.
            # 2e-07 * pcv beside an 8.9e+06 term). Prefer the all-integer
            # rounding, then scale-aware snapping; each candidate must still
            # reconstruct every state mean, otherwise the raw fit is kept.
            scale = max([1.0]+[abs(v) for v in fit[:k+1]])
            def snap(v):
                nearest = round(v)
                return nearest if abs(v-nearest) <= max(1e-8, 1e-9*scale) else clean(v)
            for coefficients, constant in (([round(v) for v in fit[:k]], round(fit[k])),
                                           ([snap(v) for v in fit[:k]], snap(fit[k])),
                                           (fit[:k], fit[k])):
                if exact(coefficients, constant):
                    break
            deviations = errors(coefficients, constant)
            accepted = bool(fit[k+4]) and all(e <= max(absolute, relative*y)
                                             for e, y in zip(deviations, ys))
            result[metric] = {
                'coefficients': dict(zip(pcvs, coefficients)),
                'constant': constant, 'maxAbsoluteError': max(deviations),
                'accepted': accepted, 'observedMin': min(ys), 'observedMax': max(ys),
                'unexplainedMean': 0 if accepted else sum(ys)/len(ys),
                'dependentPCVs': [pcvs[j] for j in sorted(plan.dependent)],
            }
        return result
    finally:
        plan.close()


def own_interfaces(rows, pcvs, site_fits, absolute, relative):
    """Sum accepted block fits; never fit a second grouping of rejected blocks."""
    # Unions and maximum-lane metrics are not additive. Keep their separately
    # measured region-aggregate interfaces, never add instruction footprints.
    totals = fit_metrics(rows, pcvs, lambda r: {
        m: v for m, v in r['metrics'].items() if not additive_metric(m)}, absolute, relative)
    grouped = state_groups(rows, pcvs)
    for metric in sorted({m for r in rows for m in r['metrics'] if additive_metric(m)}):
        means = [sum(r['metrics'].get(metric, 0) for r in group)/len(group) for _, group in grouped]
        totals[metric] = {'observedMin': min(means), 'observedMax': max(means)}
    for metric, total in totals.items():
        if not additive_metric(metric):
            total['aggregation'] = 'observed-region-aggregate; not additive across sites or children'
            total['unexplainedByState'] = [
                {'pcvs': dict(zip(pcvs, state)), 'mean': 0 if total['accepted'] else
                 sum(r['metrics'].get(metric, 0) for r in group)/len(group), 'sites': {}}
                for state, group in grouped]
            if not total['accepted']:
                total['rejectedFit'] = {'coefficients': total['coefficients'], 'constant': total['constant']}
                total['coefficients'] = dict.fromkeys(pcvs, 0)
                total['constant'] = 0
            continue
        parts = {s: fs[metric] for s, fs in site_fits.items() if metric in fs}
        # Geometry and runtime overhead have no PTX site. Retain any measured
        # remainder as its own contribution, never as a compensating fit.
        remainder = []
        for row in rows:
            value = row['metrics'].get(metric, 0) - sum(fs.get(metric, 0) for fs in row['sites'].values())
            if value < 0:
                raise ValueError(f'GPU site costs exceed region total for {metric}')
            remainder.append(dict(row, remainder={metric: value}))
        if not parts or any(r['remainder'][metric] for r in remainder):
            parts['<region overhead>'] = fit_metrics(
                remainder, pcvs, lambda r: r['remainder'], absolute, relative)[metric]
        accepted = {s: f for s, f in parts.items() if f['accepted']}
        rejected = {s: f for s, f in parts.items() if not f['accepted']}
        total['dependentPCVs'] = sorted({p for f in parts.values() for p in f['dependentPCVs']})
        # Individual fits were already snapped and checked. Do not round the
        # sum again: tiny slopes can matter at large PCV values.
        coefficients = {p: math.fsum(f['coefficients'][p] for f in accepted.values()) for p in pcvs}
        constant = math.fsum(f['constant'] for f in accepted.values())
        breakdown = []
        errors = []
        for state, group in grouped:
            contributions = {s: sum(
                (r['metrics'].get(metric, 0) - sum(fs.get(metric, 0) for fs in r['sites'].values()))
                if s == '<region overhead>' else r['sites'].get(s, {}).get(metric, 0)
                for r in group)/len(group) for s in rejected}
            measured = sum(r['metrics'].get(metric, 0) for r in group)/len(group)
            unexplained = sum(contributions.values())
            errors.append(abs(measured - unexplained - constant - sum(coefficients[p]*v for p, v in zip(pcvs, state))))
            breakdown.append({'pcvs': dict(zip(pcvs, state)), 'mean': unexplained, 'sites': contributions})
        total.update(coefficients=coefficients, constant=constant, accepted=not rejected,
                     maxAbsoluteError=max(errors), aggregation='sum of accepted basic-block fits and region overhead',
                     unexplainedMean=sum(r['mean'] for r in breakdown)/len(breakdown),
                     unexplainedByState=breakdown, rejectedSites=sorted(rejected))
    return totals


def basic_block_rows(rows):
    """Use IDs supplied by Lean, not inferred paths or data-dependent groups.

    Preserve raw instruction observations for attribution. Old modules without
    IDs retain conservative singleton units and are explicitly marked legacy.
    """
    result, members, legacy = [], defaultdict(set), set()
    for row in rows:
        blocks = defaultdict(lambda: defaultdict(int))
        for site, metrics in row['sites'].items():
            additive = {m: v for m, v in metrics.items() if additive_metric(m)}
            if not additive:
                continue
            prefix, sep, _ = site.partition(':cpp')
            if sep and re.fullmatch(r'.+:bb[0-9]+', prefix):
                block = prefix
            else:
                block = site
                legacy.add(site)
            members[block].add(site)
            for metric, count in additive.items():
                blocks[block][metric] += count
        result.append(dict(row, sites={b: dict(ms) for b, ms in blocks.items()}))
    return result, {b: sorted(s) for b, s in sorted(members.items())}, sorted(legacy)


def compose_regions(rows, regions):
    """Freeze child interfaces; fit only exact per-invocation multiplicities."""
    if not all('token' in r and 'pid' in r and 'parent' in r for r in rows):
        for region in regions:
            region['composition'] = {'status': 'unavailable', 'reason': 'invocation identities missing', 'children': []}
        return
    def key(r, token=None):
        return (r.get('captureId', ''), r['pid'], r['token'] if token is None else token)
    by_key = {key(r): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('duplicate GPU invocation identity')
    children = defaultdict(list)
    for row in rows:
        if row['parent']:
            parent = key(row, row['parent'])
            if parent not in by_key:
                raise ValueError('missing GPU parent invocation')
            children[parent].append(row)
    # Calculate observed inclusive counts only for an unexplained child term.
    # An accepted F is symbolic, not a child's global mean multiplied by N.
    inclusive, visiting = {}, set()
    def total(k):
        if k in inclusive:
            return inclusive[k]
        if k in visiting:
            raise ValueError('cyclic GPU invocation parentage')
        visiting.add(k)
        counts = {m: v for m, v in by_key[k]['metrics'].items() if additive_metric(m)}
        for child in children[k]:
            for m, v in total(key(child)).items():
                counts[m] = counts.get(m, 0) + v
        visiting.remove(k)
        inclusive[k] = counts
        return counts
    for k in by_key:
        total(k)
    for region in regions:
        parents, pcvs = region['observations'], region['pcvs']
        names = sorted({c['region'] for p in parents for c in children[key(p)]})
        terms = []
        for name in names:
            samples = []
            for p in parents:
                calls = [c for c in children[key(p)] if c['region'] == name]
                histogram = defaultdict(int)
                for c in calls:
                    histogram[tuple(sorted(c['pcvs'].items()))] += 1
                counts = defaultdict(int)
                for c in calls:
                    for m, v in inclusive[key(c)].items():
                        counts[m] += v
                samples.append(dict(p, calls={'count': len(calls)}, childCounts=dict(counts),
                    childStates=[{'pcvs': dict(s), 'count': n} for s, n in sorted(histogram.items())]))
            fit = fit_metrics(samples, pcvs, lambda r: r['calls'], 1e-8, 1e-8, means=False)['count']
            # Rational verification disallows averaging away differing child
            # call counts, and disallows tolerance hiding a missing call.
            a = [Fraction(fit['coefficients'][p]).limit_denominator(10**9) for p in pcvs]
            d = Fraction(fit['constant']).limit_denominator(10**9)
            exact = all(d + sum(v*s['pcvs'][p] for p, v in zip(pcvs, a)) == s['calls']['count'] for s in samples)
            multiplier = dict(fit, coefficients={p: float(v) for p, v in zip(pcvs, a)},
                              constant=float(d), accepted=exact,
                              exactRational={'coefficients': {p: str(v) for p, v in zip(pcvs, a)}, 'constant': str(d)})
            terms.append({'child': name, 'accepted': exact, 'multiplier': multiplier,
                          'metrics': sorted(set().union(*(s['childCounts'] for s in samples))),
                          'term': f'({formula(multiplier)}) * F[{name}]' if exact else f'unexplained(F[{name}])',
                          'invocations': [{'captureId': p.get('captureId', ''), 'pid': p['pid'],
                              'token': p['token'], 'count': p['calls']['count'], 'childStates': p['childStates']}
                              for p in samples],
                          'unexplainedByState': [] if exact else [
                              {'pcvs': dict(zip(pcvs, state)), 'metrics': {
                                  m: sum(s['childCounts'].get(m, 0) for s in group)/len(group)
                                  for m in set().union(*(s['childCounts'] for s in group))}}
                              for state, group in state_groups(samples, pcvs)]})
        region['composition'] = {'status': 'checked', 'children': terms,
            'semantics': 'exact call multiplicities; F includes child unexplained costs and descendants; additive resources only'}
        region['interfaces'] = {}
        metrics = set(region['fits']) | {m for term in terms for m in term['metrics']}
        for metric in sorted(metrics):
            own = region['fits'].get(metric)
            parts = [formula(own)] if own else []
            if own and not own['accepted']:
                parts.append('unexplained')
            if additive_metric(metric):
                parts += [t['term'] for t in terms if metric in t['metrics']]
            states = []
            for state, group in state_groups(parents, pcvs):
                values = dict(zip(pcvs, state))
                own_u = next((s['mean'] for s in (own or {}).get('unexplainedByState', []) if s['pcvs'] == values), 0)
                child_u = sum(s['metrics'].get(metric, 0) for t in terms if not t['accepted']
                              for s in t['unexplainedByState'] if s['pcvs'] == values)
                states.append({'pcvs': values, 'own': own_u, 'unexplainedChild': child_u, 'total': own_u+child_u})
            region['interfaces'][metric] = {'formula': ' + '.join(parts) or '0',
                'unexplainedByState': states,
                'unexplainedMean': sum(s['total'] for s in states)/len(states),
                'aggregation': 'additive composition' if additive_metric(metric) else 'region aggregate only'}


def build_model(rows, absolute=1e-6, relative=1e-6):
    if any(not math.isfinite(v) or v < 0 for v in (absolute, relative)):
        raise ValueError('tolerances must be finite and nonnegative')
    groups = defaultdict(list)
    schemas = {}
    for r in rows:
        names = tuple(sorted(r['pcvs']))
        if r['region'] in schemas and schemas[r['region']] != names:
            raise ValueError('GPU PCV schema changed for region '+r['region'])
        schemas[r['region']] = names
        groups[r['region']].append(r)
    regions = []
    for name, invocations in sorted(groups.items()):
        scopes={r.get('scope','host') for r in invocations}
        if len(scopes)!=1:
            raise ValueError('GPU observation scope changed for region '+name)
        pcvs = list(schemas[name])
        block_rows, block_members, legacy_sites = basic_block_rows(invocations)
        sites = sorted(block_members)
        site_fits = {site: fit_metrics(block_rows, pcvs,
                    lambda r, s=site: r['sites'].get(s, {}), absolute, relative) for site in sites}
        fits = own_interfaces(block_rows, pcvs, site_fits, absolute, relative)
        states = sorted({tuple(r['pcvs'][p] for p in pcvs) for r in invocations})
        unmodeled = sum(r['metrics'].get('unmodeled_work_events', 0) for r in invocations)
        regions.append({'name': name, 'scope': next(iter(scopes)),
                        'fitSemantics': ('state-means/legacy-or-mixed-units-v3' if legacy_sites
                                         else 'state-means/basic-block-sum-v3'),
                        'basicBlockMembers': block_members, 'legacyInstructionSites': legacy_sites,
                        'metricScope': 'exclusive', 'pcvs': pcvs, 'invocations': len(invocations),
                        'stateCount': len(states), 'singleState': len(states) == 1,
                        'fits': fits, 'sites': site_fits, 'observations': invocations,
                        'stateStatistics': [
                            {'pcvs': dict(zip(pcvs, state)), 'invocations': len(group), 'metrics': {
                                m: {'mean': sum(r['metrics'].get(m, 0) for r in group)/len(group),
                                    'min': min(r['metrics'].get(m, 0) for r in group),
                                    'max': max(r['metrics'].get(m, 0) for r in group)}
                                for m in fits}}
                            for state, group in state_groups(invocations, pcvs)],
                        'diagnosticMetrics': sorted({m for r in invocations for m in r['metrics'] if diagnostic_metric(m)}),
                        'unmodeledWorkEvents': unmodeled,
                        'validity': 'incomplete' if unmodeled else 'observed-work-counts'})
    compose_regions(rows, regions)
    return {
        'schema': SCHEMA, 'backend': 'gpu', 'regions': regions,
        'validity': 'incomplete' if any(r['unmodeledWorkEvents'] for r in regions) else 'complete',
        'fit': {'absoluteTolerance': absolute, 'relativeTolerance': relative,
                'unitOfObservation': 'mean at each distinct PCV state',
                'unitOfFitting': 'PTX basic block; no grouping across blocks',
                'scope': 'observed states only; same-state invocation variation is retained'},
        'semantics': {
            'work': 'dynamic PTX operations, separate from CPU instructions',
            'memory': 'requested PTX bytes; unresolved generic addresses remain generic; '
                      'not cache transactions or HBM traffic',
            'parallelism': 'logical launch geometry, per-block maximum lane work, and '
                           'lane-sequence span with block-barrier joins; no GPU cycle estimate',
            'spanLimitations': 'lane instruction order serializes independent instructions; '
                               'warp-exchange joins and cross-block dependency edges are not modeled',
            'diagnostics': 'emulator_* counters describe execution machinery, not GPU barriers; '
                           'block_barrier_lane_arrivals and collective_lane_arrivals come from PTX',
            'composition': 'exclusive own costs plus exact affine child-call multiplicities times F[child]; '
                           'unfitted child multiplicities remain unexplained(F[child]); additive resources only',
            'deviceRegions': 'lane scopes retain individual invocations and PCVs; warp/block '
                             'scopes require complete physical membership and identical PCVs; '
                             'no implicit synchronization; scoped span is not inferred',
            'annotationOverhead': 'marker calls, bodies and device constructor registration are excluded; '
                                  'PCV expressions evaluated before entry and caller parameter staging may be counted in a parent; '
                                  'annotations may affect compiler optimization',
            'time': 'assumed latency and model-dependent load-depth metrics are raw diagnostics only, never fitted; '
                    'actual load instructions and requested bytes are fitted resource counts',
        },
    }


def number(v):
    # %g would print the exact integer 8912896 as 8.9129e+06.
    return str(int(v)) if float(v).is_integer() and abs(v) < 2**53 else f'{v:.10g}'


def formula(fit):
    terms = [f'{number(v)} * {p}' for p, v in fit['coefficients'].items() if v]
    if fit['constant'] or not terms:
        terms.append(number(fit['constant']))
    return ' + '.join(terms).replace('+ -', '- ')


def short_sites(names, provenance=None):
    # Site keys include :bbID:cppLINE:opcode; older ones omit :bbID.
    # Demangle once (c++filt) and
    # drop template arguments so report lines name the kernel readably.
    funcs = sorted({n.split('_gxptx:')[0] for n in names if '_gxptx:' in n})
    # The translator appends '_' to the symbol before the _gxptx: suffix.
    plain = [f[:-1] if f.endswith('_') else f for f in funcs]
    try:
        out = subprocess.run(['c++filt'], input='\n'.join(plain), capture_output=True,
                             text=True, timeout=30, check=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        out = plain
    def strip(d):
        r, depth = [], 0
        for c in d:
            depth += c == '<'
            if depth == 0:
                r.append(c)
            depth -= c == '>' and depth > 0
        r = ''.join(r).split('(')[0]
        return r[5:] if r.startswith('void ') else r
    table = dict(zip(funcs, map(strip, out))) if len(out) == len(funcs) else {}
    maps = load_site_maps(provenance)
    result = {}
    for n in names:
        f, _, rest = n.partition('_gxptx:')
        # With a site map: CUDA source line (when compiled with -lineinfo) and PTX line.
        block = ''
        location = rest
        if re.match(r'^bb[0-9]+:', rest):
            block, _, location = rest.partition(':')
        line, _, op = location.partition(':')
        where = maps.get(f+'_gxptx', {}).get(line[3:]) if line.startswith('cpp') else None
        if where:
            rest = (block+' ' if block else '') + (f'{where[1]} ' if where[1] else '') + f'ptx{where[0]}:{op}'
        result[n] = f'{table.get(f, f)} @ {rest}' if rest else n
    return result


def text_report(model):
    lines = ['DRPERF GPU RESOURCE INTERFACES',
             'Collector: standalone translated PTX; CPU instruction collector disabled.',
             'Bytes are requested accesses. Parallelism is logical. No latency prediction.',
             'Fitting unit: '+model.get('fit', {}).get('unitOfFitting', 'legacy instruction sites')+'.',
             'Fits check means at each observed PCV state; invocation variation is retained.', '']
    for region in model['regions']:
        lines += [f'REGION "{region["name"]}"',
                  f'  {region["invocations"]} invocations, {region["stateCount"]} states']
        if region['singleState']:
            lines.append('  SINGLE STATE: a constant fit does not establish scaling.')
        if region.get('legacyInstructionSites'):
            lines.append('  LEGACY: instruction sites without block IDs remain separate; retranslate for basic-block fitting.')
        if region['unmodeledWorkEvents']:
            lines.append(f'  INCOMPLETE: {region["unmodeledWorkEvents"]} unmodeled work events')
        for metric in sorted(set(region['fits']) | set(region.get('interfaces', {}))):
            fit = region['fits'].get(metric)
            interface = region.get('interfaces', {}).get(metric)
            expression = interface['formula'] if interface else formula(fit) + (' + unexplained' if not fit['accepted'] else '')
            lines.append(f'  {metric} = {expression}')
            if fit and not fit['accepted']:
                lines.append(f'    unexplained own mean: {number(fit["unexplainedMean"])}')
            if interface and interface['unexplainedMean']:
                lines.append(f'    unexplained including unfitted children: {number(interface["unexplainedMean"])}')
        varying = sorted({m for state in region.get('stateStatistics', []) for m, s in state['metrics'].items() if s['min'] != s['max']})
        if varying:
            lines.append('  Same-state invocation variation: '+', '.join(varying))
        dependent = sorted({p for fit in region['fits'].values() for p in fit['dependentPCVs']})
        if dependent:
            lines.append('  PCVs not independently identifiable: '+', '.join(dependent))
        names = short_sites(list(region['sites']), model.get('provenance'))
        for site, fits in region['sites'].items():
            rejected = [m for m, f in fits.items() if not f['accepted']]
            if rejected:
                label = 'basic block' if ':bb' in site else 'site'
                lines.append('  Unexplained '+label+' '+names[site]+': '+', '.join(rejected))
        for metric, fit in region['fits'].items():
            for state in fit.get('unexplainedByState', []):
                if state['mean']:
                    lines.append(f'    {metric} at {state["pcvs"]}: '+', '.join(
                        f'{names.get(s, s)}={number(v)}' for s, v in state['sites'].items() if v))
        lines.append('')
    return '\n'.join(lines)+'\n'


def capture_paths(directory, provenance):
    """Resolve the named capture after relocation; never pool older runs."""
    names = provenance.get('rawCaptures', [])
    if not names:
        raise ValueError('report has no capture provenance; cannot safely select a run to refit')
    paths = []
    for name in names:
        old = Path(name)
        candidates = [directory/old] if not old.is_absolute() else [old, directory/old.parent.name/old.name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise ValueError('missing GPU capture named in report: '+str(old))
        paths.append(path)
    return paths


def write_status(directory, status, **metadata):
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as f:
        json.dump(dict(status=status, **metadata), f)
    os.replace(f.name, directory/'capture-status.json')


def gx_environment(root, library=None, inherited=None):
    """Reusable by GX's future `explain gpu` launcher; no standalone emulator.

    GX owns the CUDA path. The optional observer library only collects markers
    and executed resource counts. Existing GX selection/configuration is kept.
    """
    root = Path(root).resolve()
    library = Path(library).resolve() if library else root/'src/sims/gpu/gx_cuda.so'
    collector = GPU/'build/libdrperf_gpu_gx.so'
    for path in (library, collector, ROOT/'build/libdrperf_fit.so'):
        if not path.is_file():
            raise ValueError(f'missing GX capture component: {path}; build GX and run gpu/gx/build.sh')
    env = dict(os.environ if inherited is None else inherited)
    # Refuse a second GPU emulator: both export execution/context symbols.
    if 'libdrperf_gpu.so' in env.get('LD_PRELOAD', ''):
        raise ValueError('GX capture cannot preload the standalone DrPerf GPU runtime')
    env['GX_HOME'] = str(root)
    env['GX_EXPLAIN_GPU_LIBRARY'] = str(collector)
    preload = env.get('LD_PRELOAD', '').split(':')
    env['LD_PRELOAD'] = ':'.join(dict.fromkeys([str(library)] + [p for p in preload if p]))
    for key, value in dict(GX_KERNEL_CHECK='1', GX_ALLOC_ALL_REAL='1', GX_COMM_ONLY='0',
                           GX_PTX_LIBS='1', GX_CHROME_TRACE='0').items():
        env.setdefault(key, value)
    return env


def report_capture(paths, output, *, absolute=1e-6, relative=1e-6, provenance=None):
    """Fit all captured invocations and write the same report from either runtime."""
    paths = [Path(p) for p in paths]
    model = build_model(read_capture(paths), absolute, relative)
    model['provenance'] = dict(provenance or {}, rawCaptures=[str(p) for p in paths],
                               modules=capture_modules(paths))
    if model['provenance'].get('execution') == 'gx':
        model['semantics'].update(
            parallelism='not collected by the GX observer; these are additive resource counts, not latency',
            spanLimitations='no lane span or critical-path estimate in this capture',
            deviceRegions='host perfmark/GPU regions own launches at submission; device markers are not connected',
            captureCoverage='prediction-only launches, omitted lanes/payloads, and unconnected dynamic resource operations are explicitly incomplete')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in [('profile.gpu.drperf.json', json.dumps(model, indent=2)+'\n'),
                          ('report.txt', text_report(model))]:
        with tempfile.NamedTemporaryFile(mode='w', dir=output, delete=False) as f:
            f.write(content)
        os.replace(f.name, output/name)
    return model


def main(argv):
    try:
        if argv[:1] == ['build']:
            return subprocess.run(['sh', str(GPU/'build.sh')], cwd=ROOT).returncode
        if argv[:1] == ['translate']:
            return translate(argv[1:])
        if argv[:1] == ['--report']:
            parser = argparse.ArgumentParser(prog='drperf --gpu --report')
            parser.add_argument('path', nargs='?', type=Path, default=Path('drperf-gpu-report/profile.gpu.drperf.json'))
            parser.add_argument('--json', action='store_true')
            args = parser.parse_args(argv[1:])
            path = args.path/'profile.gpu.drperf.json' if args.path.is_dir() else args.path
            status = path.parent/'capture-status.json'
            if status.is_file() and json.loads(status.read_text())['status'] != 'complete':
                raise ValueError('latest GPU capture did not complete; previous report is not the latest run')
            model = json.loads(path.read_text())
            if model.get('schema') != SCHEMA:
                raise ValueError('this is not a GPU report; CPU and GPU schemas are separate')
            print(json.dumps(model, indent=2) if args.json else text_report(model), end='\n' if args.json else '')
            return 0
        if argv[:1] == ['--refit']:
            # Re-fit retained raw captures (new tolerances/fitter) without re-emulating.
            parser = argparse.ArgumentParser(prog='drperf --gpu --refit')
            parser.add_argument('path', type=Path)
            parser.add_argument('--absolute-tolerance', type=float, default=1e-6)
            parser.add_argument('--relative-tolerance', type=float, default=1e-6)
            args = parser.parse_args(argv[1:])
            old = json.loads((args.path/'profile.gpu.drperf.json').read_text())
            if old.get('schema') != SCHEMA:
                raise ValueError('this is not a GPU report')
            paths = capture_paths(args.path, old.get('provenance', {}))
            model = build_model(read_capture(paths), args.absolute_tolerance, args.relative_tolerance)
            model['provenance'] = dict(old.get('provenance', {}), rawCaptures=[str(p) for p in paths],
                                       modules=capture_modules(paths), refitFrom=str(args.path))
            for name, content in [('profile.gpu.drperf.json', json.dumps(model, indent=2)+'\n'),
                                  ('report.txt', text_report(model))]:
                with tempfile.NamedTemporaryFile(mode='w', dir=args.path, delete=False) as f:
                    f.write(content)
                os.replace(f.name, args.path/name)
            print(f'drperf GPU: refit {len(model["regions"])} regions; work capture {model["validity"]}')
            return 0 if model['validity'] == 'complete' else 2
        parser = argparse.ArgumentParser(prog='drperf --gpu',
                    description='Run GPU markers, execute translated PTX, and fit GPU resource interfaces. No DynamoRIO.')
        parser.add_argument('--gx', type=Path, help='GX checkout: run the application through GX, with resource capture')
        parser.add_argument('--gx-library', type=Path, help='override the GX runtime shared library')
        parser.add_argument('--output', type=Path, default=Path('drperf-gpu-report'))
        parser.add_argument('--absolute-tolerance', type=float, default=1e-6)
        parser.add_argument('--relative-tolerance', type=float, default=1e-6)
        parser.add_argument('--timeout', type=float, default=120)
        parser.add_argument('command', nargs=argparse.REMAINDER)
        args = parser.parse_args(argv)
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not command:
            parser.error('supply a command, or use `build`, `translate`, or `--report`')
        if any(not math.isfinite(v) or v < 0 for v in
               (args.absolute_tolerance, args.relative_tolerance)) or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error('tolerances must be finite and nonnegative; timeout must be positive')
        if args.gx:
            env = gx_environment(args.gx, args.gx_library)
        else:
            if args.gx_library:
                parser.error('--gx-library requires --gx')
            require_build()
            env = {k: v for k, v in os.environ.items()
                   if k != 'LD_PRELOAD' and (k.startswith('DRPERF_GPU_') or
                       not k.startswith(('DRPERF_', 'DYNAMORIO_', 'GX_', 'NEX_')))}
        args.output.mkdir(parents=True, exist_ok=True)
        capture = Path(tempfile.mkdtemp(prefix='capture-', dir=args.output.resolve()))
        write_status(args.output, 'running', capture=str(capture))
        env['DRPERF_GPU_CAPTURE'] = str(capture/'gpu')
        result = subprocess.run(command, env=env, timeout=args.timeout)
        if result.returncode:
            write_status(args.output, 'failed', capture=str(capture), exitCode=result.returncode)
            print(f'drperf GPU: program failed ({result.returncode}); capture retained at {capture}', file=sys.stderr)
            return result.returncode if result.returncode > 0 else 128-result.returncode
        paths = sorted(capture.glob('gpu.*.jsonl'))
        provenance = {'command': command, 'execution': 'gx' if args.gx else 'standalone'}
        if args.gx:
            library = args.gx_library or args.gx/'src/sims/gpu/gx_cuda.so'
            provenance.update(gxRoot=str(args.gx.resolve()), gxRuntimeSha256=sha256(library),
                              collectorSha256=sha256(GPU/'build/libdrperf_gpu_gx.so'))
        else:
            provenance.update(runtimeSha256=sha256(RUNTIME),
                              vendoredGX=json.loads((GPU/'vendor/provenance.json').read_text())['revision'])
        model = report_capture(paths, args.output, absolute=args.absolute_tolerance,
                               relative=args.relative_tolerance, provenance=provenance)
        write_status(args.output, 'complete', capture=str(capture))
        print(f'drperf GPU: {len(model["regions"])} regions; work capture {model["validity"]}')
        print('  '+str(args.output/'report.txt'))
        print('  '+str(args.output/'profile.gpu.drperf.json'))
        return 0 if model['validity'] == 'complete' else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('drperf GPU: '+str(error), file=sys.stderr)
        return 2
