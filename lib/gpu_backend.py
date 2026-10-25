"""Independent GPU collector, translation command and resource-formula reports.

Only the numerical affine checker is shared with CPU drperf. This module never
loads DynamoRIO or calls CPU perfmark, runner, composition, or wait analysis.
"""
import argparse
from collections import defaultdict
import hashlib
import json
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
        with cpp.open('a') as f:
            f.write('\nextern "C" const unsigned drperf_gpu_instrumented_abi = 2;\n')
        subprocess.run([args.cxx, '-std=c++17', '-O1', '-fPIC', '-shared',
                        '-fno-strict-aliasing', '-fwrapv', '-Wl,-z,defs',
                        str(cpp), '-L'+str(RUNTIME.parent), '-ldrperf_gpu',
                        '-Wl,-rpath,'+str(RUNTIME.parent), '-o', str(scratch/'module.so')],
                       env=env, check=True)
        os.replace(cpp, cpp_output)
        os.replace(scratch/'module.so', output)
    manifest = {
        'schema': SCHEMA, 'backend': 'gpu', 'instrumentationAbi': 2, 'input': str(source),
        'inputSha256': sha256(source), 'translatorSha256': sha256(TRANSLATOR),
        'preludeSha256': sha256(GPU/'vendor/gxptx/Gxptx/prelude.h'),
        'runtimeSha256': sha256(RUNTIME), 'moduleSha256': sha256(output),
        'generatedCpp': str(cpp_output),
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
    if not rows:
        raise ValueError('no GPU marker invocations were recorded')
    return rows


def clean(value):
    if not math.isfinite(value):
        raise ValueError('nonfinite GPU fit')
    nearest = round(value)
    return nearest if abs(value-nearest) < 1e-8 else value


def fit_metrics(rows, pcvs, get_counts, absolute, relative):
    values = [[r['pcvs'][p] for p in pcvs] for r in rows]
    metrics = sorted(set().union(*(get_counts(r) for r in rows)))
    if not metrics:
        return {}
    plan = native_fit.Plan(values)
    try:
        observations = [[get_counts(r).get(metric, 0) for r in rows] for metric in metrics]
        result = {}
        for metric, ys, fit in zip(metrics, observations, plan.fit(observations, absolute, relative)):
            k = len(pcvs)
            coefficients, constant = [clean(v) for v in fit[:k]], clean(fit[k])
            def errors(a, c):
                return [abs(y-(sum(v*x for v, x in zip(a, state))+c))
                        for state, y in zip(values, ys)]
            deviations = errors(coefficients, constant)
            # Pretty integer coefficients must still reconstruct every call.
            # A tiny coefficient can contribute substantial work at large PCVs.
            if any(e > max(absolute, relative*y) for e, y in zip(deviations, ys)):
                coefficients, constant = fit[:k], fit[k]
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


def build_model(rows, absolute=1e-6, relative=1e-6):
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
        pcvs = list(schemas[name])
        sites = sorted(set().union(*(r['sites'] for r in invocations)))
        fits = fit_metrics(invocations, pcvs, lambda r: r['metrics'], absolute, relative)
        site_fits = {site: fit_metrics(invocations, pcvs,
                    lambda r, s=site: r['sites'].get(s, {}), absolute, relative) for site in sites}
        states = sorted({tuple(r['pcvs'][p] for p in pcvs) for r in invocations})
        unmodeled = sum(r['metrics'].get('unmodeled_work_events', 0) for r in invocations)
        regions.append({'name': name, 'pcvs': pcvs, 'invocations': len(invocations),
                        'stateCount': len(states), 'singleState': len(states) == 1,
                        'fits': fits, 'sites': site_fits, 'observations': invocations,
                        'unmodeledWorkEvents': unmodeled,
                        'validity': 'incomplete' if unmodeled else 'observed-work-counts'})
    return {
        'schema': SCHEMA, 'backend': 'gpu', 'regions': regions,
        'validity': 'incomplete' if any(r['unmodeledWorkEvents'] for r in regions) else 'complete',
        'fit': {'absoluteTolerance': absolute, 'relativeTolerance': relative,
                'unitOfObservation': 'individual GPU region invocation',
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
            'composition': 'nested regions own their direct launches; parent and child work is exclusive',
            'time': 'no latency, occupancy, throughput or hardware calibration is inferred',
        },
    }


def formula(fit):
    terms = [f'{v:g} * {p}' for p, v in fit['coefficients'].items() if v]
    if fit['constant'] or not terms:
        terms.append(f'{fit["constant"]:g}')
    return ' + '.join(terms).replace('+ -', '- ')


def text_report(model):
    lines = ['DRPERF GPU RESOURCE INTERFACES',
             'Collector: standalone translated PTX; CPU instruction collector disabled.',
             'Bytes are requested accesses. Parallelism is logical. No latency prediction.',
             'Fits check every individual invocation at observed states only.', '']
    for region in model['regions']:
        lines += [f'REGION "{region["name"]}"',
                  f'  {region["invocations"]} invocations, {region["stateCount"]} states']
        if region['singleState']:
            lines.append('  SINGLE STATE: a constant fit does not establish scaling.')
        if region['unmodeledWorkEvents']:
            lines.append(f'  INCOMPLETE: {region["unmodeledWorkEvents"]} unmodeled work events')
        for metric, fit in region['fits'].items():
            status = 'accepted' if fit['accepted'] else 'UNEXPLAINED'
            lines.append(f'  {metric} = {formula(fit)} [{status}; max error {fit["maxAbsoluteError"]:g}]')
        dependent = sorted({p for fit in region['fits'].values() for p in fit['dependentPCVs']})
        if dependent:
            lines.append('  PCVs not independently identifiable: '+', '.join(dependent))
        for site, fits in region['sites'].items():
            rejected = [m for m, f in fits.items() if not f['accepted']]
            if rejected:
                lines.append('  Unexplained site '+site+': '+', '.join(rejected))
        lines.append('')
    return '\n'.join(lines)+'\n'


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
            model = json.loads(path.read_text())
            if model.get('schema') != SCHEMA:
                raise ValueError('this is not a GPU report; CPU and GPU schemas are separate')
            print(json.dumps(model, indent=2) if args.json else text_report(model), end='\n' if args.json else '')
            return 0
        parser = argparse.ArgumentParser(prog='drperf --gpu',
                    description='Run GPU markers, execute translated PTX, and fit GPU resource interfaces. No DynamoRIO.')
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
        require_build()
        args.output.mkdir(parents=True, exist_ok=True)
        capture = Path(tempfile.mkdtemp(prefix='capture-', dir=args.output.resolve()))
        env = {k: v for k, v in os.environ.items()
               if k != 'LD_PRELOAD' and not k.startswith(('DRPERF_', 'DYNAMORIO_', 'GX_', 'NEX_'))}
        env['DRPERF_GPU_CAPTURE'] = str(capture/'gpu')
        result = subprocess.run(command, env=env, timeout=args.timeout)
        if result.returncode:
            print(f'drperf GPU: program failed ({result.returncode}); capture retained at {capture}', file=sys.stderr)
            return result.returncode if result.returncode > 0 else 128-result.returncode
        paths = sorted(capture.glob('gpu.*.jsonl'))
        model = build_model(read_capture(paths), args.absolute_tolerance, args.relative_tolerance)
        model['provenance'] = {'command': command, 'rawCaptures': [str(p) for p in paths],
                               'runtimeSha256': sha256(RUNTIME),
                               'vendoredGX': json.loads((GPU/'vendor/provenance.json').read_text())['revision']}
        for name, content in [('profile.gpu.drperf.json', json.dumps(model, indent=2)+'\n'),
                              ('report.txt', text_report(model))]:
            target = args.output/name
            with tempfile.NamedTemporaryFile(mode='w', dir=args.output, delete=False) as f:
                f.write(content)
            os.replace(f.name, target)
        print(f'drperf GPU: {len(model["regions"])} regions; work capture {model["validity"]}')
        print('  '+str(args.output/'report.txt'))
        print('  '+str(args.output/'profile.gpu.drperf.json'))
        return 0 if model['validity'] == 'complete' else 2
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('drperf GPU: '+str(error), file=sys.stderr)
        return 2
