#!/usr/bin/env python3
"""Read-before-write detection (initcheck) on buffers marked uninitialized."""
import json, pathlib, subprocess, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[2]
NVCC = '/home/ubuntu/GX/toolchains/cuda13/bin/nvcc'

class InitcheckTest(unittest.TestCase):
    def test_partial_write_then_read(self):
        here = ROOT/'gpu/tests/initcheck'
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            ptx = tmp/'initcheck.ptx'
            if not pathlib.Path(NVCC).exists():
                self.skipTest('CUDA compiler unavailable for regenerating PTX')
            subprocess.run([NVCC, '-arch=compute_100f', '-ptx', here/'initcheck.cu', '-o', ptx], check=True)
            subprocess.run([ROOT/'bin/drperf', '--gpu', 'translate', ptx, '-o', tmp/'k.so'], check=True, capture_output=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT/'gpu/include'), here/'driver.cc',
                            '-L'+str(ROOT/'gpu/build'), '-ldrperf_gpu', '-Wl,-rpath,'+str(ROOT/'gpu/build'),
                            '-o', tmp/'driver'], check=True)
            run = subprocess.run([ROOT/'bin/drperf', '--gpu', '--output', tmp/'report', '--', tmp/'driver', tmp/'k.so'],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('uninitialized byte(s) read at consume', run.stderr)
            model = json.loads((tmp/'report/profile.gpu.drperf.json').read_text())
            regions = {r['name']: r for r in model['regions']}
            fit = regions['consume']['fits']['uninitialized_read_bytes']
            self.assertTrue(fit['accepted'])
            self.assertEqual(fit['coefficients'], {'n': 4, 'written': -4})
            self.assertNotIn('uninitialized_read_bytes', regions['partial_write']['fits'])

if __name__ == '__main__':
    unittest.main()
