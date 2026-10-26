#!/usr/bin/env python3
"""Latency model across lanes: critical-lane breakdown, mbarrier spin rollback, tensor pipe."""
import json, os, pathlib, subprocess, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS = ROOT/'gpu/tests'

def build(tmp, ptx, driver, name):
    subprocess.run([ROOT/'bin/drperf', '--gpu', 'translate', ptx, '-o', tmp/(name+'.so')], check=True, capture_output=True)
    subprocess.run(['g++', '-std=c++17', '-O2', '-pthread', '-I'+str(ROOT/'gpu/include'), driver, '-L'+str(ROOT/'gpu/build'),
                    '-ldrperf_gpu', '-Wl,-rpath,'+str(ROOT/'gpu/build'), '-o', tmp/(name+'-driver')], check=True)

def capture(tmp, out, args, **env):
    run = subprocess.run([ROOT/'bin/drperf', '--gpu', '--output', tmp/out, '--', *args],
                         env=dict(os.environ, **env), capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return json.loads((tmp/out/'profile.gpu.drperf.json').read_text())

def launches(model, region):
    return [(row['pcvs'], launch) for r in model['regions'] if r['name'] == region
            for row in r['observations'] for launch in row['launches']]

class LatencyAsyncTest(unittest.TestCase):
    def check_identity(self, lane):
        self.assertEqual(lane['issue'] + lane['stalled'], lane['path'])
        for cause in lane.get('chain', []) + lane.get('warps', []):
            self.assertEqual(cause['issue'] + cause['stalled'], cause['path'])

    def test_critical_lane_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            build(tmp, TESTS/'latency_schedule/interleaved.ptx', TESTS/'latency_schedule/driver.cc', 'k')
            for schedule, top in (('inorder', 399), ('block', 399)):
                model = capture(tmp, schedule, [tmp/'k-driver', tmp/'k.so'], DRPERF_GPU_LATENCY_SCHEDULE=schedule)
                (_, launch), = launches(model, 'interleaved')
                lane = launch['criticalLane']
                self.check_identity(lane)
                self.assertEqual(lane['path'], launch['metrics']['latency_critical_path'])
                self.assertEqual(lane['stalls'][0][1], top)  # the first load's consumer

    def test_spin_rollback_and_cause(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            build(tmp, TESTS/'latency_async/spin.ptx', TESTS/'latency_async/driver.cc', 'spin')
            model = capture(tmp, 'spin', [tmp/'spin-driver', tmp/'spin.so', 'spin'])
            (_, launch), = launches(model, 'spin')
            self.assertGreater(launch['metrics'].get('mbarrier_spin_lane_instructions', 0), 0)
            lane = launch['criticalLane']
            self.check_identity(lane)
            self.assertEqual(lane['thread'], 0)
            self.assertLess(lane['issue'], 30)  # spun probes were rolled back
            site, cycles = lane['stalls'][0]
            self.assertIn('try_wait', site.split(' <- ')[0])
            self.assertIn('mbarrier.arrive', site.split(' <- ')[1])
            self.assertTrue(site.endswith('[thread 1]'))
            arriver, = [c for c in lane['chain'] if c['thread'] == 1]
            # Resumes at the arrival (+ its own 400-unit load/store tail), not after the spin.
            self.assertLessEqual(lane['path'], arriver['path'] + 420)

    def test_tensor_pipe_completion(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            build(tmp, TESTS/'tcgen05.ptx', TESTS/'tcgen05_driver.cc', 'tcgen')
            def paths(**env):
                model = capture(tmp, 'r'+str(len(env)), [tmp/'tcgen-driver', tmp/'tcgen.so'], **env)
                out = {}
                for pcvs, launch in launches(model, 'tcgen-bf16'):
                    if pcvs['blocks'] == 1:
                        out[pcvs['mma_tiles']] = launch
                return out
            default, fast = paths(), paths(DRPERF_GPU_TENSOR_FLOPS_PER_UNIT='1000000000')
            # 128x64x16 BF16 = 262144 FLOPs = 32 units each at 8192 FLOPs/unit.
            slope = lambda p: p[4]['metrics']['latency_critical_path'] - p[1]['metrics']['latency_critical_path']
            self.assertGreaterEqual(slope(default), 3*32)
            self.assertLess(slope(fast), 3*32)
            self.assertEqual(default[1]['metrics']['latency_model_tensor_rate'], 8192)
            lane = default[4]['criticalLane']
            self.check_identity(lane)
            self.assertTrue(any('tcgen05.mma' in s and s.endswith('[tensor pipe, thread 0]') for s, _ in lane['stalls']),
                            lane['stalls'][:4])

    def test_site_map_names_ptx_and_source_lines(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'gpu/build') as tmp:
            tmp = pathlib.Path(tmp)
            build(tmp, TESTS/'latency_async/lineinfo.ptx', TESTS/'latency_schedule/driver.cc', 'li')
            sites = json.loads((tmp/'li.so.sites.json').read_text())['functions']['interleaved__gxptx']
            ptx_lines = sorted(v[0] for v in sites.values())
            self.assertEqual(len(ptx_lines), 12)  # every translated instruction placed
            self.assertIn([20, 'helper.cuh:5<kernel.cu:13'], sites.values())  # inlined helper add.f32
            self.assertIn([29, 'kernel.cu:20'], sites.values())
            capture(tmp, 'r', [tmp/'li-driver', tmp/'li.so'], DRPERF_GPU_LATENCY_SCHEDULE='block')
            run = subprocess.run(['python3', ROOT/'gpu/tools/latency_report.py', tmp/'r', '--stalls', '3'],
                                 capture_output=True, text=True, check=True)
            # The first load's consumer is the inlined helper's add.f32.
            self.assertIn('interleaved @ helper.cuh:5<kernel.cu:13 ptx20:add.f32', run.stdout)
            model = json.loads((tmp/'r'/'profile.gpu.drperf.json').read_text())
            self.assertEqual([m['module'] for m in model['provenance']['modules']], [str((tmp/'li.so').resolve())])

if __name__ == '__main__':
    unittest.main()
