"""Null refinements retain the ordinary coverage and indicator contracts."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_wait_coverage import Capture
import event_model
import explorer
import runner

ROOT = Path(__file__).resolve().parents[1]


def declaration(region='B', indicator='True', reason='Protect a bookkeeping counter.'):
    return dict(version=1, claims=[dict(id='internal', region=region, event=None,
        producer=None, indicator=indicator, reason=reason)])


class NullRefinement(unittest.TestCase):
    def model(self):
        c = Capture()
        for parent in ('A', 'C'):
            with c.region(parent):
                with c.region('B'):
                    c.sync(3)
                    c.record('declared_waited_null')
        m = c.model(); m['waitDeclarations'] = declaration()
        return m

    def test_shared_helper_has_no_event_pair_or_dependency_edge(self):
        r = event_model.check(self.model())
        self.assertEqual(r['status'], 'ordered')
        self.assertEqual(r['edges'], [])
        self.assertEqual(r['eventPairs'], [])
        self.assertEqual(len(r['nullWaits']), 2)
        self.assertEqual(r['coverage']['covered'], 2)
        self.assertEqual(r['unexplained'], [])
        claim = r['interfaceChecks']['claims'][0]
        self.assertEqual(claim['term'], 'I[True] * waited(null)')
        self.assertEqual(claim['status'], 'checked')
        self.assertEqual(claim['reasonReview'], 'manual')
        self.assertIn('manual review', '\n'.join(event_model.lines(r)))

    def test_wrong_indicator_and_missing_reason_not_silently_accepted(self):
        m = self.model(); m['waitDeclarations'] = declaration(indicator='False')
        row = event_model.check(m)['interfaceChecks']['claims'][0]
        self.assertEqual(row['status'], 'invalid')
        self.assertEqual(row['unexpectedPresence'], 2)
        for reason in ('', ' ', None):
            m['waitDeclarations'] = declaration(reason=reason)
            with self.assertRaises(ValueError): event_model.check(m)
        m['waitDeclarations'] = declaration()
        del m['waitDeclarations']['claims'][0]['reason']
        with self.assertRaises(ValueError): event_model.check(m)

    def test_absent_marker_is_checked(self):
        m = self.model(); m['waits']['events'] = [e for e in m['waits']['events']
                                               if e['kind'] != 'declared_waited_null']
        r = event_model.check(m)
        self.assertEqual(r['interfaceChecks']['claims'][0]['unexpectedAbsence'], 2)
        self.assertEqual(r['coverage']['uncovered'], 2)

    def test_undeclared_null_stays_visible(self):
        m = self.model(); del m['waitDeclarations']
        r = event_model.check(m)
        self.assertEqual(r['interfaceChecks']['status'], 'unexplained')
        self.assertEqual(r['interfaceChecks']['unexplained'][0]['term'], 'unexplained(waited(null))')

    def test_ancestor_null_has_one_credit_and_cannot_cover_future_sync(self):
        for early in (False, True):
            c = Capture()
            with c.region('A'):
                if early: c.record('declared_waited_null')
                with c.region('middle'):
                    with c.region('B'): c.sync()
                    with c.region('B'): c.sync()
                if not early: c.record('declared_waited_null')
            m = c.model(); m['waitDeclarations'] = declaration('A')
            r = event_model.check(m)
            self.assertEqual(r['coverage']['covered'], 0 if early else 1)
            self.assertEqual(r['coverage']['uncovered'], 2 if early else 1)

    def test_null_cannot_name_a_producer(self):
        m = self.model(); m['waitDeclarations']['claims'][0]['producer'] = 'A'
        with self.assertRaises(ValueError): event_model.check(m)

    def test_null_and_event_zero_coexist_without_aliasing(self):
        c = Capture(); c.publish(event=0, generation=0)
        with c.region('A'):
            with c.region('B'): c.sync()
            c.waited(event=0, generation=0)
            with c.region('C'): c.sync()
            c.record('declared_waited_null', event=0, generation=0)
        m = c.model(); m['waitDeclarations'] = declaration('A')
        m['waitDeclarations']['claims'].append(dict(id='real-zero', region='A',
            event='0', producer='publisher', indicator='True'))
        r = event_model.check(m)
        self.assertEqual(r['interfaceChecks']['status'], 'checked')
        self.assertEqual(r['coverage']['covered'], 2)
        self.assertEqual(len(r['edges']), 1)
        self.assertEqual(len(r['nullWaits']), 1)
        self.assertEqual(len(r['eventPairs']), 1)

    def test_incomplete_null_is_not_credited(self):
        m = self.model()
        for e in m['waits']['events']:
            if e['kind'] == 'declared_waited_null': e['returned'] = False
        r = event_model.check(m)
        self.assertEqual(r['status'], 'unverified')
        self.assertEqual(r['coverage']['covered'], 0)


class SynchronousDescendant(unittest.TestCase):
    def test_direct_and_transitive_child_publications_are_rejected(self):
        for depth in (0, 3):
            c = Capture()
            def child(n):
                with c.region('child'):
                    if n: child(n-1)
                    else:
                        c.record('declared_publish')
                        c.sync()
            with c.region('A'):
                child(depth)
                c.waited()
            r = c.check()
            self.assertEqual(r['status'], 'violation')
            self.assertIn('synchronous descendant', r['violations'][0]['reason'])
            self.assertEqual(r['coverage']['covered'], 0)

    def test_same_invocation_cannot_publish_to_itself(self):
        c = Capture()
        with c.region('A'):
            c.record('declared_publish'); c.sync(); c.waited()
        self.assertEqual(c.check()['status'], 'violation')

    def test_null_refinement_does_not_erase_an_invalid_event_claim(self):
        c = Capture()
        with c.region('A'):
            with c.region('B'):
                c.record('declared_publish'); c.sync()
            c.waited(); c.record('declared_waited_null')
        r = c.check()
        self.assertEqual(r['coverage']['covered'], 1)
        self.assertEqual(r['status'], 'violation')
        self.assertEqual(r['edges'][0]['status'], 'violation')

    def test_other_thread_is_not_a_synchronous_child(self):
        c = Capture()
        with c.region('A'):
            # Build a second-thread invocation whose sequence interval lies
            # within A. Numeric containment alone must not reject it.
            c.thread = '2'
            with c.region('B'): c.record('declared_publish')
            c.thread = '1'; c.sync(); c.waited()
        self.assertEqual(c.check()['status'], 'ordered')

    def test_completed_sibling_publication_remains_valid(self):
        c = Capture(); c.publish()
        with c.region('A'): c.sync(); c.waited()
        self.assertEqual(c.check()['status'], 'ordered')


class PythonNullCheckpoint(unittest.TestCase):
    def test_ctypes_scope_refinement_matches_context_and_decorator_semantics(self):
        sys.path.insert(0, str(ROOT/'perfmark/python'))
        try:
            import perfmark
        finally:
            sys.path.pop(0)
        log = []
        with patch.object(perfmark, '_open', side_effect=lambda *a: log.append('begin')), \
             patch.object(perfmark, '_end', side_effect=lambda *a: log.append('end')), \
             patch.object(perfmark, 'event_waited', side_effect=lambda e: log.append(('waited', e))):
            with perfmark._region_ctypes('ordinary'):
                pass
            scope = perfmark._region_ctypes('reviewed').waited_null_on_exit()
            scope.waited_null_on_exit()
            @scope
            def failure():
                raise ValueError('preserved')
            with self.assertRaisesRegex(ValueError, 'preserved'):
                failure()
            with scope:
                pass
        self.assertEqual(log, ['begin', 'end', 'begin', ('waited', None), 'end',
                               'begin', ('waited', None), 'end'])

    def test_explicit_scope_exit_refinement_preserves_exceptions_and_default(self):
        with tempfile.TemporaryDirectory(prefix='wait-null-exit-', dir=ROOT/'out') as tmp:
            folder = Path(tmp); script = folder/'app.py'
            script.write_text('''import perfmark
for fail in (0, 1):
    try:
        with perfmark.region('outer'):
            scope = perfmark.region('reviewed', fail=fail)
            assert scope.waited_null_on_exit() is scope
            scope.waited_null_on_exit()  # idempotent, not an extra credit
            with scope:
                if fail:
                    raise ValueError('preserved')
    except ValueError as error:
        assert fail and str(error) == 'preserved'
    else:
        assert not fail
with perfmark.region('ordinary'):
    pass
''')
            with patch.dict(os.environ, {'DRPERF_WAITS':'1', 'DRPERF_FOLLOW_THREADS':'0',
                                         'DRPERF_WAIT_DELAY_MS':'0'}):
                rc, log, _ = runner.run([sys.executable, str(script)], str(folder/'raw'), timeout=30)
            self.assertEqual(rc, 0, log)
            m = explorer.build_model(folder/'raw', discover=False,
                wait_declarations=declaration('reviewed'))
            markers = [e for e in m['waits']['events'] if e['kind']=='declared_waited_null']
            self.assertEqual(len(markers), 2)
            self.assertTrue(all(e['region']=='reviewed' and e['instructionExcluded'] for e in markers))
            self.assertEqual(m['eventModel']['interfaceChecks']['claims'][0]['status'], 'checked')
            self.assertFalse(m['validity']['traceErrors'])

    def test_native_python_marker_and_argument_validation(self):
        with tempfile.TemporaryDirectory(prefix='wait-null-', dir=ROOT/'out') as tmp:
            folder = Path(tmp); script = folder/'app.py'
            script.write_text('''import perfmark
with perfmark.region('B', need=1):
    perfmark.event_waited(None)
    for function, args in [(perfmark.event_waited, (None, 1)),
                           (perfmark.event_publish, (None,)),
                           (perfmark.event_waited, (1,))]:
        try:
            function(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid checkpoint accepted')
''')
            with patch.dict(os.environ, {'DRPERF_WAITS':'1', 'DRPERF_FOLLOW_THREADS':'0',
                                         'DRPERF_WAIT_DELAY_MS':'0'}):
                rc, log, _ = runner.run([sys.executable, str(script)], str(folder/'raw'), timeout=30)
            self.assertEqual(rc, 0, log)
            m = explorer.build_model(folder/'raw', discover=False,
                wait_declarations=declaration(indicator='need == 1'))
            r = m['eventModel']
            self.assertEqual(r['status'], 'ordered', r)
            self.assertEqual(len(r['nullWaits']), 1)
            self.assertEqual(r['interfaceChecks']['claims'][0]['status'], 'checked')
            self.assertEqual(r['edges'], [])
            markers = [e for e in m['waits']['events'] if e['kind']=='declared_waited_null']
            self.assertEqual(len(markers), 1)
            self.assertTrue(markers[0]['instructionExcluded'])
            path = folder/'null.drperf.json'; explorer.write_model(m, path)
            full = explorer.load_model(path, True)
            self.assertEqual(event_model.check(full), r)

if __name__ == '__main__': unittest.main()
