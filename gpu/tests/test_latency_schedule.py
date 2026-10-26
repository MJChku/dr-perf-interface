#!/usr/bin/env python3
"""Latency model schedules: PTX-order in-order issue vs basic-block (ptxas-like)."""
import json, os, pathlib, subprocess, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT/'gpu/tests/latency_schedule'

def launch_metrics(report):
    model = json.loads((report/'profile.gpu.drperf.json').read_text())
    raw = model['provenance']['rawCaptures']
    for path in raw:
        for line in open(path):
            e = json.loads(line)
            for launch in e.get('launches', []):
                if 'latency_critical_path' in launch.get('metrics', {}):
                    return launch['metrics']
    raise AssertionError('no latency metrics')

class ScheduleTest(unittest.TestCase):
    def test_interleaved_loads(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            subprocess.run([ROOT/'bin/drperf', '--gpu', 'translate', HERE/'interleaved.ptx', '-o', tmp/'k.so'], check=True, capture_output=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT/'gpu/include'), HERE/'driver.cc', '-L'+str(ROOT/'gpu/build'),
                            '-ldrperf_gpu', '-Wl,-rpath,'+str(ROOT/'gpu/build'), '-o', tmp/'driver'], check=True)
            got = {}
            for schedule in ('inorder', 'block'):
                env = dict(os.environ, DRPERF_GPU_LATENCY_SCHEDULE=schedule)
                run = subprocess.run([ROOT/'bin/drperf', '--gpu', '--output', tmp/schedule, '--', tmp/'driver', tmp/'k.so'],
                                     env=env, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                got[schedule] = launch_metrics(tmp/schedule)
            self.assertEqual(got['inorder']['dependent_global_loads_on_critical_path'], 4)
            self.assertEqual(got['block']['dependent_global_loads_on_critical_path'], 1)
            self.assertEqual(got['block']['latency_model_schedule'], 1)
            self.assertLess(got['block']['latency_critical_path'], got['inorder']['latency_critical_path'])

if __name__ == '__main__':
    unittest.main()
