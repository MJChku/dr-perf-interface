#!/usr/bin/env python3
"""Validate paired measurement infrastructure, not agent discovery accuracy.

Sources and profiles live in a temporary directory and are removed on exit.
Only the compact smoke record is retained. Do not use marked smoke workspaces
as starting points for blind agents.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench


def run(archive, output):
    summary = dict(kind='harness-smoke-not-agent-trials',
                   model_for_future_trials='gpt-5.6-sol', cases=[])
    with tempfile.TemporaryDirectory(prefix='drperf-paired-smoke-') as temporary:
        for case in ('sqlglot', 'libcst', 'comfyui'):
            pair = Path(temporary)/case
            bench.prepare_pair(case, pair, archive, 'gpt-5.6-sol', 0)
            row = dict(case=case, pair=bench.read(pair/'pair.json'), measurements={})
            for arm in ('timing', 'drperf'):
                workspace = pair/arm
                if arm == 'drperf':
                    driver = workspace/'workload.py'
                    code = driver.read_text()
                    if case == 'sqlglot':
                        target = '    result = optimize(expression, schema=schema)'
                        states = 'n_joins=n'
                    elif case == 'libcst':
                        target = '    module = libcst.parse_module(src)'
                        states = 'n_terms=n, shape=shape'
                    else:
                        target = '    await keys.add_keys(list(prompt))'
                        states = 'n=n, shape=shape'
                    assert code.count(target) == 1
                    code = 'import perfmark\n' + code.replace(target,
                        f'    with perfmark.region("target", {states}):\n    ' + target)
                    driver.write_text(code)
                    row['smoke_driver_sha256'] = hashlib.sha256(code.encode()).hexdigest()
                print('SMOKE', case, arm, flush=True)
                rc = bench.measure(workspace, arm, workspace/'plan.json', workspace/'hypothesis.json')
                measured = bench.read(workspace/'measurements/round-01/result.json')
                row['measurements'][arm] = measured
                assert rc == 0, (case, arm, measured)
                if arm == 'drperf':
                    report = bench.read(workspace/'measurements/round-01/profile.drperf.json')
                    row['drperf_regions'] = len(report['regions'])
                    row['drperf_trace_records'] = report['trace']['recordCount']
                    target = next(r for r in report['regions'] if r['id'] == 'target')
                    assert any(s['path'] == 'workload.py' for s in target['sources'])
                    row['smoke_annotations'] = 'Evaluator-inserted raw inputs; not agent discovery.'
                measured['elapsed_seconds'] = sum(p['wall_seconds'] for p in measured['points'])
                # Point logs retain process diagnostics only until this scope exits.
                row.setdefault('workload_outputs', {})[arm] = [
                    [line for line in (workspace/f'measurements/round-01/point-{i:02d}/output.txt').read_text().splitlines()
                     if line.startswith(('n_joins=', 'n_terms=', 'n='))]
                    for i in range(len(measured['points']))]
            assert row['workload_outputs']['timing'] == row['workload_outputs']['drperf']
            assert all(row['workload_outputs']['timing'])
            summary['cases'].append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            bench.write(output, summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.archive.resolve(), args.output.resolve())
