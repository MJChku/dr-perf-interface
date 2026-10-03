"""Declared indicators are assertions, never fitted replacements."""
import copy
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_wait_coverage import Capture
import event_model
import explorer

ROOT = Path(__file__).resolve().parents[1]


def fixture(needs=(0, 1), counts=(0, 1), indicator='need == 1'):
    c = Capture()
    for generation, (need, count) in enumerate(zip(needs, counts), 1):
        c.publish(generation=generation)
        with c.region('A'):
            c.stack[-1]['values'] = {'need': need}
            for _ in range(count):
                c.sync()
                c.waited(generation=generation)
    m = c.model()
    region = next(r for r in m['regions'] if r['id'] == 'A')
    region['states'] = ['need']
    region['points'] = [dict(state=[n], calls=k) for n, k in Counter(needs).items()]
    m['waitDeclarations'] = dict(version=1, claims=[dict(id='ready', region='A',
        event='1', indicator=indicator, producer='publisher')])
    return m


class DeclaredWaitInterfaces(unittest.TestCase):
    def result(self, model):
        return event_model.check(model)['interfaceChecks']

    def test_conditional_indicator_keeps_zero_invocations(self):
        r = self.result(fixture())
        self.assertEqual(r['status'], 'checked')
        row = r['claims'][0]
        self.assertEqual((row['present'], row['absent']), (1, 1))
        self.assertEqual(row['term'], 'I[need == 1] * Wait[publisher]')

    def test_absent_and_unexpected_waits_are_failed_claims(self):
        r = self.result(fixture(indicator='need == 0'))
        self.assertEqual(r['status'], 'invalid')
        self.assertEqual((r['claims'][0]['unexpectedPresence'],
                          r['claims'][0]['unexpectedAbsence']), (1, 1))
        self.assertEqual(len(r['claims'][0]['counterexamples']), 2)

    def test_same_pcv_state_is_not_averaged(self):
        row = self.result(fixture((1, 1), (0, 1)))['claims'][0]
        self.assertEqual(row['status'], 'invalid')
        self.assertEqual(row['unexpectedAbsence'], 1)

    def test_indicator_must_be_explicit_even_when_always_present(self):
        m = fixture((1,), (1,)); del m['waitDeclarations']
        r = self.result(m)
        self.assertEqual(r['claims'], [])
        self.assertEqual(r['unexplained'][0]['term'], 'unexplained(Wait[publisher])')

    def test_declared_true_is_not_simplified_away(self):
        self.assertEqual(self.result(fixture((1,), (1,), 'True'))['claims'][0]['term'],
                         'I[True] * Wait[publisher]')

    def test_wrong_publisher_is_invalid_despite_correct_presence(self):
        m = fixture(); m['waitDeclarations']['claims'][0]['producer'] = 'other'
        r = self.result(m)
        self.assertEqual(r['status'], 'invalid')
        self.assertEqual(r['claims'][0]['producerMismatches'], 1)

    def test_failed_event_order_cannot_pass_interface(self):
        m = fixture((1,), (1,))
        pub = m['waits']['events'][0]; pub['start'] = 500; pub['end'] = 501
        self.assertEqual(self.result(m)['claims'][0]['status'], 'invalid')

    def test_native_wait_without_checkpoint_remains_unexplained(self):
        m = fixture((0,), (0,))
        c = Capture(); c.publish()
        with c.region('A'): c.sync()
        m = c.model()
        r = self.result(m)
        self.assertEqual(r['unexplained'][0]['term'], 'unexplained(Wait[?])')
        self.assertEqual(r['unexplained'][0]['count'], 1)

    def test_presence_and_multiplicity_are_separate(self):
        row = self.result(fixture((0, 1), (0, 3)))['claims'][0]
        self.assertEqual(row['status'], 'checked')
        self.assertEqual(row['term'], 'I[need == 1] * (3) * Wait[publisher]')

    def test_unfitted_multiplicity_is_not_hidden(self):
        r = self.result(fixture((1, 1), (1, 2)))
        self.assertEqual(r['status'], 'unexplained')
        self.assertIn('unexplained(multiplicity)', r['claims'][0]['term'])

    def test_never_taken_branch_does_not_verify_publisher(self):
        row = self.result(fixture((0, 0), (0, 0)))['claims'][0]
        self.assertEqual(row['status'], 'not-exercised')
        self.assertEqual(row['absent'], 2)

    def test_incomplete_capture_cannot_establish_absence(self):
        m = fixture(); m['trace']['complete'] = False
        self.assertEqual(self.result(m)['claims'][0]['status'], 'unverified')

    def test_no_state_or_marker_mutation(self):
        m = fixture(); before = copy.deepcopy(m)
        self.result(m)
        self.assertEqual(m, before)

    def test_indicator_is_validated_and_does_not_execute_code(self):
        for expression in ('missing == 1', '__import__("os")', 'need + 2', '1 // 0'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                self.result(fixture(indicator=expression))

    def test_duplicate_selector_is_rejected(self):
        m = fixture(); m['waitDeclarations']['claims'].append(dict(m['waitDeclarations']['claims'][0], id='again'))
        with self.assertRaises(ValueError): self.result(m)

    def test_sidecar_roundtrip_keeps_declarations_and_checks(self):
        m = fixture(); m['eventModel'] = event_model.check(m)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.drperf.json'
            explorer.write_model(m, path)
            full = explorer.load_model(path, wait_evidence=True)
            self.assertEqual(full['waitDeclarations'], m['waitDeclarations'])
            self.assertEqual(event_model.check(full), m['eventModel'])

    def test_cli_exit_and_unexplained_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.drperf.json'
            for m, expected in [(fixture(), 0), (fixture(indicator='False'), 1), (fixture(), 2)]:
                if expected == 2: del m['waitDeclarations']
                explorer.write_model(m, path)
                run = subprocess.run([sys.executable, str(ROOT/'tools/drperf-check-events'), str(path)],
                                     text=True, capture_output=True)
                self.assertEqual(run.returncode, expected, run.stderr + run.stdout)
                if expected == 2:
                    self.assertIn('unexplained(Wait[publisher])', run.stdout)
                    self.assertNotIn('WARNING', run.stdout)


if __name__ == '__main__': unittest.main()
