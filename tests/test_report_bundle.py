"""The sole public command emits checked, readable reports without extra steps."""
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import report_bundle
import source_coverage
import explorer


class ReportBundle(unittest.TestCase):
    def test_historical_coverage_scans_absent_annotations_and_checks_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'app.py'
            source.write_text('@marked("executed")\ndef f(): pass\n'
                              '@active_marked("absent")\nasync def g(): pass\n')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            model=dict(regions=[dict(id='executed',calls=1,sources=[dict(path='app.py',sha256=digest)])],
                       provenance=dict(sourceRoot=str(root)),trace=dict(complete=True))
            source_coverage.ensure(model)
            self.assertEqual((model['codeCoverage']['observed'],model['codeCoverage']['declared']),(1,2))
            self.assertEqual(model['codeCoverage']['sourceComparison'],'post-capture')
            self.assertIn('application was not rerun', '\n'.join(source_coverage.lines(model['codeCoverage'])))
            del model['codeCoverage']
            source.write_text('# changed\n')
            source_coverage.ensure(model)
            self.assertEqual(model['codeCoverage']['status'],'unavailable')
            self.assertIn('app.py',model['codeCoverage']['reason'])
            del model['codeCoverage']
            model['provenance'].clear()
            source_coverage.ensure(model)
            self.assertEqual(model['codeCoverage']['status'],'unavailable')

    def test_null_refinements_are_not_reported_as_dependency_channels(self):
        model=dict(regions=[],executionGraph=dict(status='observed',edges=[]),
            eventModel=dict(status='ordered',edges=[dict(consumer='A',producer='B',event='1')],
                interfaceChecks=dict(claims=[dict(region='C',event=None)]),
                coverage=dict(covered=1,obligations=3,uncovered=2)))
        from unittest.mock import patch
        with patch.object(explorer,'cost_lines',return_value=[]):
            text=report_bundle.text_report(model)
        self.assertIn('dependency channels observed: 1; waited(null) refinements: 1',text)
        self.assertIn('WAIT COVERAGE INCOMPLETE',text)
        self.assertIn('A -> B',text)

    def test_one_command_default_outputs_and_missing_region_coverage(self):
        with tempfile.TemporaryDirectory(prefix='drperf-cli-') as temporary:
            root=Path(temporary)
            (root/'app.py').write_text('''from perfmark import region
with region("executed", n=2):
    result=sum(range(2))
if False:
    with region("unreached", n=3):
        result=sum(range(3))
''')
            env=dict(os.environ)
            for name in ['DRPERF_REPORT','DRPERF_REPORT_DIR','DRPERF_WAIT_INTERFACES','DRPERF_SOURCE_ROOT','DRPERF_WAITS']:
                env.pop(name,None)
            result=subprocess.run([str(ROOT/'bin/drperf'),sys.executable,'app.py'],cwd=root,
                                  env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            output=root/'drperf-report'
            self.assertTrue((output/'graph.html').exists())
            text=(output/'report.txt').read_text()
            model=explorer.load_model(output/'profile.drperf.json',wait_evidence=True)
            self.assertIn('eventModel',model)
            self.assertIn('REGION "executed"',text)
            self.assertIn('COST INTERFACES AND BREAKDOWN',text)
            self.assertEqual((model['codeCoverage']['observed'],model['codeCoverage']['declared']),(1,2))
            self.assertIn('unreached: not-observed',text)
            page=(output/'graph.html').read_text()
            payload=re.search(r"atob\('([A-Za-z0-9+/=]+)'\)",page)[1]
            embedded=json.loads(gzip.decompress(base64.b64decode(payload)))['report']
            self.assertEqual(embedded['id'],model['id'])
            self.assertEqual(embedded['executionGraph'],model['executionGraph'])
            self.assertNotIn('<script src=',page)
            self.assertIn('ELK.js',page)

    def test_default_indicator_file_is_checked_without_second_command(self):
        with tempfile.TemporaryDirectory(prefix='drperf-cli-') as temporary:
            root=Path(temporary)
            (root/'app.py').write_text('''from perfmark import region, event_waited, event_publish
with region("producer"): event_publish(42,1)
with region("consumer", need=0): event_waited(42,1)
''')
            (root/'drperf.waits.json').write_text(json.dumps(dict(version=1,claims=[dict(
                id='ready',region='consumer',event='42',producer='producer',indicator='need == 1')])))
            env=dict(os.environ)
            for name in ['DRPERF_REPORT','DRPERF_REPORT_DIR','DRPERF_WAIT_INTERFACES','DRPERF_SOURCE_ROOT','DRPERF_WAITS']:
                env.pop(name,None)
            result=subprocess.run([str(ROOT/'bin/drperf'),sys.executable,'app.py'],cwd=root,
                                  env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,1,result.stdout+result.stderr)
            text=(root/'drperf-report/report.txt').read_text()
            self.assertIn('waited -> "producer"',text)
            self.assertIn('unexpectedPresence',text)
            self.assertIn('interface=invalid',text)

    def test_html_report_names_are_not_executable_markup(self):
        report=dict(id='</script><script>EVIL()</script>',regions=[])
        page=report_bundle.standalone_html(report)
        self.assertNotIn('EVIL()',page)
        payload=re.search(r"atob\('([A-Za-z0-9+/=]+)'\)",page)[1]
        self.assertEqual(json.loads(gzip.decompress(base64.b64decode(payload)))['report'],report)

    def test_coverage_does_not_claim_all_same_name_sites_or_uncaptured_paths(self):
        model=dict(regions=[dict(id='shared',calls=2)],trace=dict(complete=True),validity={})
        coverage=source_coverage.build(model,{'shared':[{},{}],'absent':[{}]},None)
        self.assertEqual(coverage['regions'][1]['siteAttribution'],'ambiguous')
        self.assertEqual(coverage['lines']['status'],'unavailable')
        model['trace']['complete']=False
        coverage=source_coverage.build(model,{'absent':[{}]},None)
        self.assertEqual(coverage['notObserved'],0)
        self.assertEqual(coverage['regions'][0]['status'],'unknown')

if __name__ == '__main__': unittest.main()
