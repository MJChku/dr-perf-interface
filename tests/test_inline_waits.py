"""Inline declarations travel with the marker, including negative observations."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import event_model
import explorer
import inline_waits
from test_event_interfaces import fixture

ROOT = Path(__file__).resolve().parents[1]

class InlineDeclarations(unittest.TestCase):
    def inline(self, **kw):
        m = fixture(**kw)
        del m['waitDeclarations']
        for e in m['waits']['events']:
            if e['kind'] == 'declared_waited':
                e.update(indicator='need == 1')
        return m

    def test_runtime_declaration_checks_all_invocations_without_mutation(self):
        m = self.inline(); before = copy.deepcopy(m)
        r = event_model.check(m)['interfaceChecks']
        self.assertEqual(r['status'], 'checked')
        self.assertEqual((r['claims'][0]['present'], r['claims'][0]['absent']), (1, 1))
        self.assertEqual(r['claims'][0]['producer'], 'publisher')
        self.assertEqual(m, before)

    def test_inline_false_presence_and_missing_occurrence(self):
        for needs, counts, field in [((0,), (1,), 'unexpectedPresence'), ((1, 1), (0, 1), 'unexpectedAbsence')]:
            r = event_model.check(self.inline(needs=needs, counts=counts))['interfaceChecks']
            self.assertEqual(r['status'], 'invalid')
            self.assertEqual(r['claims'][0][field], 1)

    def test_conflicting_or_truncated_runtime_declarations_fail(self):
        m = self.inline(needs=(1, 1), counts=(1, 1))
        waits = [e for e in m['waits']['events'] if e['kind'] == 'declared_waited']
        waits[-1]['indicator'] = 'False'
        with self.assertRaisesRegex(ValueError, 'Conflicting'): event_model.check(m)
        waits[-1]['indicator'] = 'need == 1'; waits[-1]['declarationError'] = True
        with self.assertRaisesRegex(ValueError, 'truncated'): event_model.check(m)

    def test_literal_sites_and_dynamic_sites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'app.py').write_text('''from perfmark import region as scope, wait as checkpoint
import perfmark as pm
EVENT = 42
with scope("A", need=1):
    if False: checkpoint(EVENT, 1, indicator="need == 1", producer="P")
    pm.wait(None, indicator="True", reason="counter")
    checkpoint(dynamic_event(), 1, indicator="need == 1")
    def later(): checkpoint(EVENT, 1, indicator="True")
''')
            claims, warnings = inline_waits.source_claims(root, None)
            self.assertEqual(len(claims), 2)
            self.assertEqual(claims[0]['event'], '42')
            self.assertEqual(len(warnings), 1)
            m = self.inline(needs=(1,), counts=(0,))
            m['inlineWaitSources'] = [dict(region='A', event='1', producer='publisher', indicator='need == 1')]
            r = event_model.check(m)['interfaceChecks']
            self.assertEqual(r['claims'][0]['unexpectedAbsence'], 1)

class InlineCapture(unittest.TestCase):
    def run_app(self, source, expected=0):
        with tempfile.TemporaryDirectory(prefix='drperf-inline-') as temp:
            root = Path(temp); (root/'app.py').write_text(source)
            env = {k:v for k,v in os.environ.items() if not k.startswith('DRPERF')}
            p = subprocess.run([str(ROOT/'bin/drperf'), sys.executable, 'app.py'],
                cwd=root, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(p.returncode, expected, p.stdout+p.stderr)
            self.assertFalse((root/'drperf.waits.json').exists())
            return explorer.load_model(root/'drperf-report/profile.drperf.json', wait_evidence=True)

    def test_real_marker_and_inline_metadata(self):
        m = self.run_app('''from perfmark import region, release, wait
for generation, need in enumerate((0, 1, 0, 1)):
    with region("producer"): release(42, generation)
    with region("consumer", need=need):
        if need: wait(42, generation, indicator="need == 1", producer="producer")
''')
        c = m['eventModel']['interfaceChecks']['claims'][0]
        self.assertEqual((c['present'], c['absent']), (2, 2))
        self.assertEqual(c['status'], 'checked')
        self.assertEqual(m['waitDeclarations']['claims'][0]['indicator'], 'need == 1')
        self.assertTrue(any(e['api']=='perfmark_wait' and e['indicator']=='need == 1' for e in m['waits']['events']))

    def test_false_condition_does_not_skip_marker(self):
        m = self.run_app('''from perfmark import region, release, wait
with region("producer"): release(42, 1)
with region("consumer", need=0): wait(42, 1, "need == 1")
''', expected=1)
        self.assertEqual(m['eventModel']['interfaceChecks']['claims'][0]['unexpectedPresence'], 1)

    def test_entirely_unexecuted_site_is_not_silently_lost(self):
        m = self.run_app('''from perfmark import region, wait
with region("consumer", need=1):
    if False: wait(42, 1, indicator="need == 1", producer="producer")
''', expected=1)
        self.assertEqual(m['eventModel']['interfaceChecks']['claims'][0]['unexpectedAbsence'], 1)

    def test_null_reason_is_inline(self):
        m = self.run_app('''from perfmark import region, wait
with region("counter", need=1):
    wait(None, indicator="need == 1", reason="Protect a bookkeeping counter.")
''')
        c = m['eventModel']['interfaceChecks']['claims'][0]
        self.assertEqual(c['status'], 'checked')
        self.assertEqual(c['reasonReview'], 'manual')
        self.assertEqual(m['eventModel']['edges'], [])

    def test_native_c_api_copies_inline_strings(self):
        with tempfile.TemporaryDirectory(prefix='drperf-inline-c-') as temp:
            root = Path(temp)
            (root/'app.c').write_text('''#include "perfmark.h"
int main(void) {
  perfmark_begin("P", "", 0);
  perfmark_release(7, 1);
  perfmark_end("P");
  perfmark_begin("C", "need", 1);
  char indicator[] = "need == 1";
  perfmark_wait(7, 1, indicator, "P");
  indicator[0] = 'x'; /* metadata must outlive the argument storage */
  perfmark_end("C");
  return 0;
}
''')
            subprocess.run(['gcc', '-O2', '-I'+str(ROOT/'perfmark'), str(root/'app.c'),
                '-L'+str(ROOT/'build'), '-lperfmark', '-Wl,-rpath,'+str(ROOT/'build'),
                '-o', str(root/'app')], check=True, capture_output=True)
            env = {k:v for k,v in os.environ.items() if not k.startswith('DRPERF')}
            p = subprocess.run([str(ROOT/'bin/drperf'), str(root/'app')], cwd=root,
                env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
            m = explorer.load_model(root/'drperf-report/profile.drperf.json')
            c = m['eventModel']['interfaceChecks']['claims'][0]
            self.assertEqual(c['indicator'], 'need == 1')
            self.assertEqual(c['status'], 'checked')

    def test_native_binding_rejects_bad_arguments(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT/'perfmark/python'))
        p = subprocess.run([sys.executable, '-c', '''from perfmark import wait, release
for args, kwargs in [((1,1),{}), ((True,1),dict(indicator="True")),
    ((1,-1),dict(indicator="True")), ((1,2**64),dict(indicator="True")),
    ((None,),dict(indicator="True")), ((None,1),dict(indicator="True",reason="x")),
    ((1,1),dict(indicator="True",reason="x")), ((1,1),dict(indicator="x"*2049))]:
    try: wait(*args, **kwargs)
    except (ValueError, OverflowError): pass
    else: raise AssertionError((args, kwargs))
release(1,1)
wait(1,1,indicator="False")  # passive even outside capture
'''], env=env, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)

if __name__ == '__main__': unittest.main()
