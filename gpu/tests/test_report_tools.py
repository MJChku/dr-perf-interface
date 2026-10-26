#!/usr/bin/env python3
"""Report diff and reuse-viewer tests; no build, CUDA, or GPU required."""
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

GPU = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GPU / 'tools'))
from diff_reports import diff_reports, format_text
from footprint_report import summarize_report
from report_io import read_report


def fit(coefficient=4, constant=0, accepted=True):
    return {'coefficients': {'elements': coefficient}, 'constant': constant,
            'accepted': accepted, 'dependentPCVs': [], 'maxAbsoluteError': 0}


def report():
    return {'schema': 'drperf.gpu.v1', 'backend': 'gpu', 'validity': 'complete', 'regions': [
        {'name': 'kernel', 'scope': 'host', 'metricScope': 'exclusive', 'pcvs': ['elements'],
         'validity': 'observed-work-counts', 'fits': {'global_read_bytes': fit()},
         'sites': {'kernel:cpp10:ld.global.u32': {'global_read_bytes': fit()}}, 'observations': []}]}


class ReportTools(unittest.TestCase):
    def test_child_interfaces_are_compared_without_run_identifiers(self):
        old, new = report(), report()
        child = {'child': 'C', 'accepted': True, 'term': '(1) * F[C]',
                 'metrics': ['global_read_bytes'], 'invocations': [{'token': 1}]}
        old['regions'][0]['composition'] = {'children': [copy.deepcopy(child)]}
        new['regions'][0]['composition'] = {'children': [copy.deepcopy(child)]}
        new['regions'][0]['composition']['children'][0]['invocations'] = [{'token': 200}]
        self.assertFalse(diff_reports(old, new)['metadataChanges'])
        new['regions'][0]['composition']['children'][0]['term'] = '(elements) * F[C]'
        self.assertIn('childInterfaces', diff_reports(old, new)['metadataChanges'][0]['changes'])

    def test_region_and_site_coefficient_deltas(self):
        old, new = report(), report()
        new['regions'][0]['fits']['global_read_bytes'] = fit(1)
        new['regions'][0]['sites']['kernel:cpp10:ld.global.u32']['global_read_bytes'] = fit(1)
        result = diff_reports(old, new)
        self.assertEqual(result['changedMetrics'], 2)
        for change in result['changes']:
            self.assertEqual(change['coefficientChanges']['elements'],
                             {'before': 4, 'after': 1, 'delta': -3, 'relativeChange': -0.75})
            self.assertTrue(change['bothAccepted'])
            self.assertTrue(change['comparable'])
        self.assertIn('-75%', format_text(result))

    def test_added_removed_metrics_sites_and_regions(self):
        old, new = report(), report()
        new['regions'][0]['fits'] = {'global_write_bytes': fit(2)}
        new['regions'][0]['sites'] = {'kernel:cpp20:st.global.u32': {'global_write_bytes': fit(2)}}
        added = copy.deepcopy(new['regions'][0]); added['name'] = 'new'
        new['regions'].append(added)
        result = diff_reports(old, new)
        statuses = {(r['region'], r['site'], r['afterSite'], r['metric']): r['status'] for r in result['changes']}
        self.assertEqual(statuses[('kernel', None, None, 'global_read_bytes')], 'removed')
        self.assertEqual(statuses[('kernel', None, None, 'global_write_bytes')], 'added')
        self.assertEqual(statuses[('kernel', 'kernel:cpp10:ld.global.u32', None, 'global_read_bytes')], 'removed')
        self.assertEqual(statuses[('kernel', None, 'kernel:cpp20:st.global.u32', 'global_write_bytes')], 'added')
        self.assertEqual(result['metadataChanges'][0]['region'], 'new')
        reverse = diff_reports(new, old)
        self.assertEqual(reverse['metadataChanges'][0]['status'], 'removed')

    def test_constant_zero_coefficients_and_rejected_fits(self):
        old, new = report(), report()
        old['regions'][0]['fits']['global_read_bytes'] = fit(0, 10, False)
        new['regions'][0]['fits']['global_read_bytes'] = fit(2, 3, True)
        change = diff_reports(old, new)['changes'][0]
        self.assertIsNone(change['coefficientChanges']['elements']['relativeChange'])
        self.assertEqual(change['constantChange']['delta'], -7)
        self.assertEqual(change['fitStatusChanges']['accepted'], {'before': False, 'after': True})
        self.assertFalse(change['bothAccepted'])

    def test_schema_scope_and_dependent_pcvs_are_disclosed(self):
        old, new = report(), report()
        region = new['regions'][0]
        region['scope'] = 'block'; region['pcvs'] = ['elements', 'tiles']
        region['fits']['global_read_bytes']['dependentPCVs'] = ['tiles']
        result = diff_reports(old, new)
        self.assertIn('scope', result['metadataChanges'][0]['changes'])
        self.assertIn('pcvs', result['metadataChanges'][0]['changes'])
        self.assertFalse(result['changes'][0]['comparable'])
        self.assertIn('dependentPCVs', result['changes'][0]['fitStatusChanges'])
        self.assertIn('[dependent PCVs: tiles]', format_text(result))

    def test_tolerances_unchanged_and_filters(self):
        old, new = report(), report()
        self.assertFalse(diff_reports(old, new)['changes'])
        self.assertEqual(len(diff_reports(old, new, include_unchanged=True)['changes']), 2)
        new['regions'][0]['fits']['global_read_bytes']['coefficients']['elements'] += 1e-8
        self.assertEqual(diff_reports(old, new)['changedMetrics'], 1)
        self.assertEqual(diff_reports(old, new, absolute=1e-7)['changedMetrics'], 0)
        self.assertEqual(diff_reports(old, new, relative=1e-7)['changedMetrics'], 0)
        self.assertFalse(diff_reports(old, new, region_pattern='other*')['changes'])
        self.assertFalse(diff_reports(old, new, metric_pattern='*write*')['changes'])
        with self.assertRaises(ValueError):
            diff_reports(old, new, absolute=-1)

    def test_site_mapping_and_ambiguities(self):
        old, new = report(), report()
        old_site, new_site = 'kernel:cpp10:ld.global.u32', 'kernel:cpp20:ld.global.u32'
        new['regions'][0]['sites'][new_site] = new['regions'][0]['sites'].pop(old_site)
        self.assertEqual(diff_reports(old, new)['changedMetrics'], 2)
        self.assertEqual(diff_reports(old, new, site_map={old_site: new_site})['changedMetrics'], 0)
        new['regions'][0]['sites'][new_site]['global_read_bytes'] = fit(1)
        change = diff_reports(old, new, site_map={old_site: new_site})['changes'][0]
        self.assertEqual(change['site'], old_site)
        self.assertEqual(change['afterSite'], new_site)
        with self.assertRaises(ValueError):
            diff_reports(old, new, site_map={'absent': new_site})
        with self.assertRaises(ValueError):
            diff_reports(old, new, site_map={old_site: new_site, 'absent': new_site})
        new['regions'][0]['sites'][old_site] = {'global_read_bytes': fit()}
        with self.assertRaises(ValueError):
            diff_reports(new, new, site_map={old_site: new_site})

    def test_duplicate_regions_and_nonfinite_coefficients_fail(self):
        old, new = report(), report()
        new['regions'].append(copy.deepcopy(new['regions'][0]))
        with self.assertRaises(ValueError):
            diff_reports(old, new)
        for field, value in [('sites', []), ('fits', []), ('pcvs', {})]:
            malformed = report(); malformed['regions'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                diff_reports(old, malformed)
        new = report(); new['regions'][0]['fits']['global_read_bytes']['constant'] = float('nan')
        with self.assertRaises(ValueError):
            diff_reports(old, new)

    def test_legacy_footprints_are_unavailable_and_zero_is_not_a_ratio(self):
        model = report()
        observation = {'pid': 1, 'token': 1, 'pcvs': {}, 'metrics': {'global_read_bytes': 128}, 'sites': {}}
        model['regions'][0]['observations'] = [observation]
        record = summarize_report(model)['records'][0]
        self.assertFalse(record['footprintAvailable'])
        self.assertIsNone(record['accesses'][0]['redundancyRatio'])
        self.assertIsNone(record['accesses'][0]['uniqueSectors32'])
        observation.update(footprintVersion=1, metrics={})
        record = summarize_report(model)['records'][0]
        self.assertTrue(record['footprintAvailable'])
        self.assertEqual(record['accesses'], [])

    def test_cli_json_text_gzip_directories_and_exit_codes(self):
        scratch = GPU / 'build/ext-work'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            work = Path(directory)
            before, after = work / 'before', work / 'after.json.gz'
            before.mkdir()
            (before / 'profile.gpu.drperf.json').write_text(json.dumps(report()))
            new = report(); new['regions'][0]['fits']['global_read_bytes'] = fit(1)
            with gzip.open(after, 'wt') as source:
                json.dump(new, source)
            self.assertEqual(read_report(after), new)

            def cli(*args):
                return subprocess.run([sys.executable, str(GPU / 'tools/diff_reports.py'), *map(str, args)],
                                      capture_output=True, text=True, timeout=10)
            result = cli(before, after, '--json', '--fail-on-change')
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(result.stdout)['changedMetrics'], 1)
            result = cli(before, before, '--fail-on-change')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('0 changed', result.stdout)
            result = cli(before, after, '--atol', 'nan')
            self.assertEqual(result.returncode, 2)
            wrong = work / 'cpu.json'; wrong.write_text('{"backend":"cpu"}')
            self.assertEqual(cli(before, wrong).returncode, 2)
            with self.assertRaises(ValueError):
                read_report(wrong)


if __name__ == '__main__':
    unittest.main()
