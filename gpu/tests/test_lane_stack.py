#!/usr/bin/env python3
"""Lane fiber stacks: large translated frames need more than boost's 128 KiB default."""
import os, pathlib, subprocess, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = ROOT/'gpu/tests/lane_stack'

class LaneStackTest(unittest.TestCase):
    def test_big_local_frame(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            subprocess.run([ROOT/'bin/drperf', '--gpu', 'translate', HERE/'big_local.ptx', '-o', tmp/'k.so'],
                           check=True, capture_output=True)
            subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT/'gpu/include'), HERE/'driver.cc',
                            '-L'+str(ROOT/'gpu/build'), '-ldrperf_gpu', '-Wl,-rpath,'+str(ROOT/'gpu/build'),
                            '-o', tmp/'driver'], check=True)
            def run(value, profiled=False):
                env = {k: v for k, v in os.environ.items() if k not in ('DRPERF_GPU_LANE_STACK_BYTES', 'GX_FIBER_STACK_BYTES')}
                if value is not None:
                    env['DRPERF_GPU_LANE_STACK_BYTES'] = value
                cmd = [tmp/'driver', tmp/'k.so']
                if profiled:
                    cmd = [ROOT/'bin/drperf', '--gpu', '--output', tmp/'report', '--'] + cmd
                return subprocess.run(cmd, env=env, capture_output=True, text=True)
            for value in (None, '', '1048576'):
                result = run(value)
                self.assertEqual(result.returncode, 0, (value, result.stderr))
            result = run(None, profiled=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(run('131072').returncode, 0)   # the old default overflows
            bad = run('12k')
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('lane stack bytes', bad.stderr)

if __name__ == '__main__':
    unittest.main()
