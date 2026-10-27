#!/usr/bin/env python3
"""Lean-assigned block identity, predication, and one-stage resource fitting."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BasicBlocks(unittest.TestCase):
    def test_translated_blocks_fit_once_and_survive_refit_without_sources(self):
        with tempfile.TemporaryDirectory(prefix='drperf-basic-blocks-') as directory:
            work = Path(directory)

            def run(*args):
                p = subprocess.run(list(map(str, args)), cwd=ROOT, capture_output=True,
                                   text=True, timeout=120)
                self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
                return p

            run(ROOT/'bin/drperf', '--gpu', 'translate', ROOT/'gpu/tests/basic_blocks.ptx',
                '-o', work/'blocks.so')
            run(os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-Igpu/include',
                'gpu/tests/basic_blocks_driver.cc', '-Lgpu/build', '-ldrperf_gpu',
                '-Wl,-rpath,'+str(ROOT/'gpu/build'), '-o', work/'driver')
            run(ROOT/'bin/drperf', '--gpu', '--output', work/'report', '--',
                work/'driver', work/'blocks.so')
            path = work/'report/profile.gpu.drperf.json'
            model = json.loads(path.read_text())
            regions = {r['name']: r for r in model['regions']}
            same, different = regions['predicated_pair'], regions['branched_pair']
            self.assertEqual(len(same['basicBlockMembers']), 1)
            self.assertGreater(len(different['basicBlockMembers']), 1)
            fit = same['fits']['global_read_bytes']
            self.assertTrue(fit['accepted'])
            self.assertEqual(fit['coefficients'], {'n': 4})
            self.assertFalse(different['fits']['global_read_bytes']['accepted'])
            for r in regions.values():
                self.assertEqual(r['legacyInstructionSites'], [])
                for row in r['observations']:
                    self.assertEqual(row['metrics']['global_read_bytes'], 4*row['pcvs']['n'])
                    loads = [site for site, counts in row['sites'].items()
                             if counts.get('global_read_bytes', 0)]
                    blocks = {site.partition(':cpp')[0] for site in loads}
                    if row['pcvs']['n'] > 32:
                        self.assertEqual(len(blocks), 1 if r is same else 2)
            text = (work/'report/report.txt').read_text()
            self.assertIn('Unexplained basic block', text)
            self.assertNotIn('LEGACY:', text)
            # No C++ or PTX parser is needed to refit retained raw observations.
            for p in work.glob('blocks.so*'):
                p.unlink()
            run(ROOT/'bin/drperf', '--gpu', '--refit', work/'report')
            refit = json.loads(path.read_text())
            self.assertEqual(model['regions'], refit['regions'])


if __name__ == '__main__':
    unittest.main()
