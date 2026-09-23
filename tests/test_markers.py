"""Marker removal uses exclusive profiles and direct-child counts per state."""
from collections import Counter
import copy
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
import derive
import explorer
import markers
import runner


class Capture:
    slots = {0: ('libperfmark.so', 'perfmark_begin', 0),
             1: ('python', 'inside_wrapper', 1),
             2: ('python', 'outside_wrapper', 2),
             3: ('app', 'work', 3), 4: ('python', 'loop', 4)}
    def __init__(self):
        self.keys, self.records, self.seq = {}, [], 0
        self.identities = {}

    def call(self, name, value, vector, body=None, run=0):
        state = [('n', value)]
        identity = run, name, value
        if identity not in self.identities:
            key = run, len(self.keys)
            self.identities[identity] = key
            self.keys[key] = {'region': name, 'states': state, 'count': 0, 'vec': {}, 'root': ''}
        row = self.keys[self.identities[identity]]
        row['count'] += 1
        vec = Counter(row['vec']); vec.update(vector); row['vec'] = dict(vec)
        self.seq += 1
        record = {'run': run, 'pid': 1, 'tid': 1, 'region': name, 'state': dict(state), 'nk': 1, 'seq': self.seq}
        self.records.append(record)
        if body:
            body()
        self.seq += 1
        record['seq_end'] = self.seq

    def calibrate(self, inside=20, outside=3, run=0):
        self.call(markers.CALIB+'_loop', 0, {0: 10, 1: inside, 4: 30}, run=run)
        self.call(markers.CALIB+'_outer', 0, {0: 50, 1: inside, 2: 4*outside, 4: 30},
                  lambda: [self.call(markers.CALIB, 0, {0: 10, 1: inside}, run=run) for _ in range(4)], run=run)


class MarkerAccounting(unittest.TestCase):
    def test_nested_slope_is_corrected_before_fitting(self):
        cap = Capture(); cap.calibrate()
        for n in (0, 1, 3, 7, 12):
            cap.call('parent', n, {0: 10+10*n, 1: 20, 2: 3*n, 3: 100*n+7},
                     lambda: [cap.call('child', n, {0: 10, 1: 20, 3: 50*n+2}) for _ in range(n)])
        adjusted, accounting = markers.adjust(cap.keys, cap.slots, cap.records)
        vecs, calls = derive.per_trigger(derive.per_state(adjusted, 'parent')[0])
        fit = derive.derive(vecs, cap.slots, split=False)[0]
        self.assertAlmostEqual(fit.a[0], 100)
        self.assertAlmostEqual(fit.c, 7)
        for state, vec in vecs.items():
            self.assertEqual(vec, {3: 100*state[0]+7})
            row = accounting[('parent', state)]
            self.assertEqual(row['wrapperEstimate'], 20 + 3*state[0])
            self.assertEqual(row['unmatchedEstimate'], 0)

    def test_grandchildren_do_not_charge_outer_and_recursion_counts(self):
        cap = Capture(); cap.calibrate()
        def middle(n):
            cap.call('middle', n, {0: 10, 1: 20, 2: 3*n, 3: 17},
                     lambda: [cap.call('leaf', 0, {0: 10, 1: 20, 3: 2}) for _ in range(n)])
        cap.call('outer', 10, {0: 10, 1: 20, 2: 3, 3: 19}, lambda: middle(10))
        def recursive(n):
            cap.call('recursive', n, {0: 10, 1: 20, 2: 3*(n>0), 3: 23},
                     lambda: recursive(n-1) if n else None)
        recursive(4)
        adjusted, accounting = markers.adjust(cap.keys, cap.slots, cap.records)
        self.assertEqual(accounting[('outer', (10,))]['wrapperEstimate'], 23)
        self.assertEqual(accounting[('middle', (10,))]['wrapperEstimate'], 50)
        for row in adjusted.values():
            if row['region'] in ('outer', 'middle', 'recursive', 'leaf'):
                self.assertEqual(set(row['vec']), {3})

    def test_different_run_calibrations_are_not_mixed(self):
        cap = Capture()
        for run, inside, outside in ((0, 20, 3), (1, 100, 11)):
            cap.calibrate(inside, outside, run)
            cap.call('parent', 1, {0: 10, 1: inside, 2: outside, 3: 7},
                     lambda: cap.call('child', 1, {0: 10, 1: inside, 3: 5}, run=run), run=run)
        adjusted, _ = markers.adjust(cap.keys, cap.slots, cap.records)
        for row in adjusted.values():
            if row['region'] == 'parent':
                self.assertEqual(row['vec'], {3: 7})

    def test_nonlinear_child_counts_do_not_leave_marker_irregularity(self):
        cap = Capture(); cap.calibrate()
        for n in (0, 1, 3, 7, 12):
            count = n*n
            cap.call('parent', n, {0: 10, 1: 20, 2: 3*count, 3: 100*n+7},
                     lambda: [cap.call('child', n, {0: 10, 1: 20, 3: 7}) for _ in range(count)])
        adjusted, _ = markers.adjust(cap.keys, cap.slots, cap.records)
        vecs, _ = derive.per_trigger(derive.per_state(adjusted, 'parent')[0])
        fit = derive.derive(vecs, cap.slots, split=False)[0]
        self.assertEqual(fit.n_irr, 0)
        self.assertAlmostEqual(fit.a[0], 100)

    def test_bad_or_missing_calibration_is_not_silently_over_subtracted(self):
        cap = Capture(); cap.calibrate()
        cap.call('tiny', 1, {0: 10, 1: 2, 3: 7})
        adjusted, accounting = markers.adjust(cap.keys, cap.slots, cap.records)
        row = next(r for r in adjusted.values() if r['region'] == 'tiny')
        self.assertEqual(row['vec'], {3: 7})
        self.assertEqual(accounting[('tiny', (1,))]['unmatchedEstimate'], 18)
        adjusted, accounting = markers.adjust(cap.keys, cap.slots, cap.records, ['truncated'])
        row = next(r for r in adjusted.values() if r['region'] == 'tiny')
        self.assertEqual(row['vec'], {1: 2, 3: 7})
        self.assertEqual(accounting[('tiny', (1,))]['uncalibratedCalls'], 1)

    def test_module_only_removal_does_not_claim_caller_preparation(self):
        cap = Capture(); cap.call('native', 1, {0: 10, 3: 19})
        adjusted, accounting = markers.adjust(cap.keys, cap.slots, cap.records)
        self.assertEqual(next(iter(adjusted.values()))['vec'], {3: 19})
        self.assertEqual(accounting[('native', (1,))]['uncalibratedCalls'], 1)

    def test_shared_state_root_buckets_are_adjusted_at_the_fitting_scope(self):
        cap = Capture(); cap.calibrate()
        cap.call('parent', 1, {0: 10, 1: 20, 2: 12, 3: 7},
                 lambda: [cap.call('child', 1, {0: 10, 1: 20, 3: 5}) for _ in range(4)])
        cap.call('parent', 1, {0: 10, 1: 20, 3: 7})
        key = cap.identities[(0, 'parent', 1)]
        cap.keys[key].update(count=1, root='busy', vec={0: 10, 1: 20, 2: 12, 3: 7})
        cap.keys[(0, 1000)] = dict(cap.keys[key], root='idle', vec={0: 10, 1: 20, 3: 7})
        saved = copy.deepcopy(cap.keys)
        adjusted, accounting = markers.adjust(cap.keys, cap.slots, cap.records)
        own, _, _ = derive.per_state(adjusted, 'parent')
        self.assertEqual(own[(1,)], ({3: 14}, 2))
        self.assertEqual(accounting[('parent', (1,))]['unmatchedEstimate'], 0)
        self.assertEqual(cap.keys, saved)


class PythonMarkers(unittest.TestCase):
    def test_real_nested_python_wrapper_profiles(self):
        self.check_python(False)

    def test_real_nested_ctypes_wrapper_profiles(self):
        self.check_python(True)

    def check_python(self, ctypes):
        with tempfile.TemporaryDirectory(prefix='markers-', dir=ROOT/'out') as temporary:
            folder = Path(temporary)
            app = folder/'app.py'
            app.write_text('''import perfmark
def leaf(n):
    with perfmark.region("leaf", n=n):
        total = 0
        for i in range(100*n): total += i
    return total
def middle(n):
    with perfmark.region("middle", n=n):
        for i in range(n): leaf(n)
for n in (0,1,2,4,8):
    with perfmark.region("outer", n=n): middle(n)
''')
            with patch.dict(os.environ, {'DRPERF_FOLLOW_THREADS':'0', 'PERFMARK_NO_EXT': '1' if ctypes else ''}):
                rc, log, files = runner.run([sys.executable, str(app)], str(folder/'raw'), timeout=90)
            self.assertEqual(rc, 0, log); self.assertTrue(files, log)
            model = explorer.build_model(folder/'raw', discover=False)
            self.assertEqual(model['composition']['status'], 'observed')
            regions = {r['id']: r for r in model['regions']}
            for name, region in regions.items():
                self.assertTrue(all(p['observed'] >= 0 for p in region['points']))
                self.assertTrue(any(p['recorded'] > p['observed'] for p in region['points']))
                self.assertTrue(all(p['calibratedCalls'] == p['calls'] for p in region['markerAdjustment']['points']))
                for fit in region['regimes']:
                    for group in fit['attribution']['coefficients'] + [fit['attribution']['constant'], fit['attribution']['unexplained']]:
                        self.assertFalse(any(markers.is_marker((r['module'], r['function'])) for r in group))
            self.assertTrue(all(p['directChildCalls'] == 1 for p in regions['outer']['markerAdjustment']['points']))
            self.assertTrue(all(p['directChildCalls'] == p['state'][0] for p in regions['middle']['markerAdjustment']['points']))
            self.assertTrue(all(p['directChildCalls'] == 0 for p in regions['leaf']['markerAdjustment']['points']))


if __name__ == '__main__':
    unittest.main()
