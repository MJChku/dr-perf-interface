#!/usr/bin/env python3
"""Resource interfaces: state means, unexplained sites and child composition."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'lib'))
import gpu_backend as gpu


def row(n, sites, **extra):
    metrics = {}
    for counts in sites.values():
        for m, v in counts.items():
            metrics[m] = metrics.get(m, 0) + v
    return dict(region='R', pcvs={'n': n}, sites=sites, metrics=metrics, **extra)


def model(rows):
    return {r['name']: r for r in gpu.build_model(rows)['regions']}


class Interfaces(unittest.TestCase):
    def test_cancellation_cannot_hide_unexplained_statements(self):
        rows = [row(n, {'A': {'fp32_flops': n*n}, 'B': {'fp32_flops': 100-n*n}}) for n in range(1, 5)]
        r = model(rows)['R']; f = r['fits']['fp32_flops']
        self.assertFalse(f['accepted'])
        self.assertEqual(f['constant'], 0)
        self.assertEqual(f['unexplainedMean'], 100)
        self.assertEqual(f['rejectedSites'], ['A', 'B'])
        self.assertTrue(all(s['mean'] == 100 for s in f['unexplainedByState']))

    def test_keep_explained_portion_and_function_site_breakdown(self):
        r = model([row(n, {'copy:1': {'global_read_bytes': 8*n}, 'hidden:2': {'global_read_bytes': n*n}})
                   for n in range(1, 5)])['R']
        f = r['fits']['global_read_bytes']
        self.assertEqual(f['coefficients'], {'n': 8})
        self.assertEqual(f['unexplainedMean'], 7.5)
        self.assertEqual(f['unexplainedByState'][3]['sites'], {'hidden:2': 16})

    def test_same_state_means_and_variation(self):
        rows = [row(n, {'A': {'global_load_instructions': v}})
                for n, values in [(1, [0, 4]*5), (2, [2, 6]), (3, [4, 8])]
                for v in values]
        r = model(rows)['R']; f = r['fits']['global_load_instructions']
        self.assertTrue(f['accepted'])
        self.assertEqual(f['coefficients'], {'n': 2})
        self.assertEqual(r['stateStatistics'][0]['metrics']['global_load_instructions'], {'mean': 2, 'min': 0, 'max': 4})
        self.assertEqual(r['stateStatistics'][0]['invocations'], 10)

    def test_nonaffine_state_means_remain_unexplained(self):
        r = model([row(n, {'A': {'fp32_flops': n*n+d}}) for n in range(1, 5) for d in (0, 2)])['R']
        self.assertFalse(r['fits']['fp32_flops']['accepted'])

    def test_latency_and_model_dependent_load_depth_never_fitted(self):
        rows = [row(n, {'A': {'global_load_instructions': n, 'latency_critical_path': n*400,
                             'dependent_global_loads_on_critical_path': n}}) for n in range(4)]
        r = model(rows)['R']
        self.assertEqual(set(r['fits']), {'global_load_instructions'})
        self.assertEqual(set(r['sites']['A']), {'global_load_instructions'})
        self.assertIn('latency_critical_path', r['observations'][1]['metrics'])

    def test_unique_footprint_is_not_summed_over_sites(self):
        rows = [row(n, {'A': {'global_unique_read_bytes': 4*n}, 'B': {'global_unique_read_bytes': 4*n}})
                for n in range(1, 4)]
        for r in rows: r['metrics']['global_unique_read_bytes'] //= 2
        f = model(rows)['R']['fits']['global_unique_read_bytes']
        self.assertEqual(f['coefficients'], {'n': 4})
        self.assertIn('not additive', f['aggregation'])

    def tree(self, multiplicity=lambda n: n, captures=('one',)):
        rows = []
        for capture in captures:
            token = 0
            for n in range(4):
                token += 1; parent = token
                p = row(n, {'host': {'global_load_instructions': 2}}, pid=7, token=token, parent=0, captureId=capture)
                rows.append(p)
                for j in range(multiplicity(n)):
                    token += 1
                    c = row(j, {'child': {'global_load_instructions': j*j+1}}, pid=7, token=token, parent=parent, captureId=capture)
                    c['region'] = 'child'; rows.append(c)
        return rows

    def test_child_composition_including_zero_calls_and_irregular_child(self):
        rs = model(self.tree(lambda n: n+1))
        parent = rs['R']; term = parent['composition']['children'][0]
        self.assertTrue(term['accepted'])
        self.assertEqual(term['multiplier']['coefficients'], {'n': 1})
        self.assertEqual(term['multiplier']['constant'], 1)
        self.assertIn('(1 * n + 1) * F[child]', parent['interfaces']['global_load_instructions']['formula'])
        self.assertFalse(rs['child']['fits']['global_load_instructions']['accepted'])
        self.assertEqual(parent['interfaces']['global_load_instructions']['unexplainedMean'], 0)
        zero = model(self.tree())['R']['composition']['children'][0]['invocations'][0]
        self.assertEqual(zero['count'], 0)

    def test_unexplained_child_includes_full_observed_descendants(self):
        rows = self.tree(lambda n: n*n)
        r = model(rows)['R']; term = r['composition']['children'][0]
        self.assertFalse(term['accepted'])
        self.assertEqual(term['term'], 'unexplained(F[child])')
        self.assertEqual(term['unexplainedByState'][-1]['metrics']['global_load_instructions'], sum(j*j+1 for j in range(9)))
        self.assertGreater(r['interfaces']['global_load_instructions']['unexplainedMean'], 0)

    def test_same_state_call_count_is_not_averaged_for_composition(self):
        rows = self.tree()
        for r in rows:
            if r['region'] == 'R': r['pcvs']['n'] = 0
        self.assertFalse(model(rows)['R']['composition']['children'][0]['accepted'])

    def test_unexplained_child_cost_includes_grandchildren(self):
        rows = self.tree(lambda n: n*n)
        token = max(r['token'] for r in rows)
        for child in list(rows):
            if child['region'] != 'child': continue
            token += 1
            leaf = row(0, {'leaf': {'global_load_instructions': 7}},
                       pid=7, token=token, parent=child['token'], captureId='one')
            leaf['region'] = 'leaf'; rows.append(leaf)
        rs = model(rows)
        self.assertIn('F[leaf]', rs['child']['interfaces']['global_load_instructions']['formula'])
        u = rs['R']['composition']['children'][0]['unexplainedByState'][-1]
        self.assertEqual(u['metrics']['global_load_instructions'], sum(j*j+8 for j in range(9)))

    def test_fractional_child_multiplier(self):
        rows = self.tree()
        for r in rows:
            if r['region'] == 'R': r['pcvs']['n'] *= 8
        term = model(rows)['R']['composition']['children'][0]
        self.assertTrue(term['accepted'])
        self.assertEqual(term['multiplier']['exactRational']['coefficients']['n'], '1/8')

    def test_capture_identity_prevents_cross_run_child_pairing(self):
        r = model(self.tree(captures=('one', 'two')))['R']
        self.assertTrue(r['composition']['children'][0]['accepted'])
        self.assertEqual(len(r['composition']['children'][0]['invocations']), 8)

    def test_recursive_region_names_stay_symbolic(self):
        rows = self.tree()
        for r in rows: r['region'] = 'R'
        r = model(rows)['R']
        self.assertEqual(r['composition']['children'][0]['child'], 'R')

    def test_broken_parentage_rejected(self):
        rows = self.tree()
        rows[-1]['parent'] = 10000
        with self.assertRaisesRegex(ValueError, 'missing GPU parent'): model(rows)
        rows = self.tree(); rows[0]['parent'] = rows[0]['token']
        with self.assertRaisesRegex(ValueError, 'cyclic GPU'): model(rows)

    def test_refit_relocation_selects_only_named_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ('capture-old', 'capture-current'):
                (root/d).mkdir(); (root/d/'gpu.1.jsonl').write_text('')
            result = gpu.capture_paths(root, {'rawCaptures': ['/removed/capture-current/gpu.1.jsonl']})
            self.assertEqual(result, [root/'capture-current/gpu.1.jsonl'])
            with self.assertRaisesRegex(ValueError, 'missing GPU capture'):
                gpu.capture_paths(root, {'rawCaptures': ['/removed/capture-failed/gpu.1.jsonl']})

    def test_tolerances_validated_for_refit_too(self):
        for v in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): gpu.build_model([row(1, {})], v, 0)

    def test_failed_rerun_is_not_shown_as_previous_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'profile.gpu.drperf.json').write_text(json.dumps(gpu.build_model([row(1, {})])))
            p = subprocess.run([ROOT/'bin/drperf', '--gpu', '--output', root, '--', '/bin/false'], capture_output=True)
            self.assertEqual(p.returncode, 1)
            p = subprocess.run([ROOT/'bin/drperf', '--gpu', '--report', root], capture_output=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn(b'latest GPU capture did not complete', p.stderr)


if __name__ == '__main__': unittest.main()
