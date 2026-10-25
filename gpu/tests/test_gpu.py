#!/usr/bin/env python3
"""End-to-end PTX execution and independent work-count/fit oracles. No GPU/CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'lib'))
import gpu_backend


def run(args, expected=0):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, text=True, timeout=120)
    if p.returncode != expected:
        raise AssertionError(f'{args}: status {p.returncode}\n{p.stdout}\n{p.stderr}')
    return p


class GPUIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gpu_backend.require_build()
        cls.temp = tempfile.TemporaryDirectory(prefix='drperf-gpu-test-')
        cls.work = Path(cls.temp.name)
        cls.module = cls.work/'kernels.so'
        cls.saxpy = cls.work/'saxpy.so'
        cls.driver = cls.work/'fixture'
        cls.saxpy_driver = cls.work/'saxpy'
        cxx = os.environ.get('CXX', 'g++')
        for source, target in [('gpu/tests/fixture.cpp', cls.driver),
                               ('gpu/examples/saxpy.cpp', cls.saxpy_driver)]:
            run([cxx, '-std=c++17', '-O2', '-pthread', '-Igpu/include', source,
                 '-Lgpu/build', '-ldrperf_gpu', '-Wl,-rpath,'+str(ROOT/'gpu/build'), '-o', target])
        for source, target in [('gpu/tests/kernels.ptx', cls.module),
                               ('gpu/examples/saxpy.ptx', cls.saxpy)]:
            run([ROOT/'bin/drperf', '--gpu', 'translate', source, '-o', target])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def capture(self, name, command, expected=0):
        dest = self.work/name
        result = run([ROOT/'bin/drperf', '--gpu', '--output', dest, '--', *command], expected)
        if expected:
            self.assertFalse((dest/'profile.gpu.drperf.json').exists())
            return result
        return json.loads((dest/'profile.gpu.drperf.json').read_text())

    def test_saxpy_counts_and_affine_interfaces(self):
        model = self.capture('saxpy-report', [self.saxpy_driver, self.saxpy])
        self.assertEqual(model['backend'], 'gpu')
        self.assertEqual(model['validity'], 'complete')
        r = model['regions'][0]
        for row in r['observations']:
            n, b = row['pcvs']['elements'], row['pcvs']['blocks']
            metrics = row['metrics']
            self.assertEqual(metrics.get('fp32_flops', 0), 2*n)
            self.assertEqual(metrics.get('global_read_bytes', 0), 8*n)
            self.assertEqual(metrics.get('global_write_bytes', 0), 4*n)
            self.assertEqual(metrics['launched_lanes'], 128*b)
            self.assertEqual(metrics['warps'], 4*b)
        for metric, coefficient in [('fp32_flops', 2), ('global_read_bytes', 8),
                                    ('global_write_bytes', 4)]:
            f = r['fits'][metric]
            self.assertTrue(f['accepted'], f)
            self.assertEqual(f['coefficients']['elements'], coefficient)
            self.assertEqual(f['constant'], 0)
            # Evaluate the learned formula at an unobserved shape against
            # the independent algorithmic count, including a different grid.
            self.assertEqual(f['constant']+f['coefficients']['elements']*511+
                             f['coefficients']['blocks']*4, coefficient*511)
        self.assertTrue(all(f['accepted'] for site in r['sites'].values() for f in site.values()))
        parallel = r['fits']['sum_block_max_lane_flops']
        self.assertTrue(parallel['accepted'])
        self.assertEqual(parallel['coefficients']['active_blocks'], 2)

    def test_collectives_parallelism_residuals_and_same_state_variation(self):
        model = self.capture('collectives', [self.driver, self.module])
        self.assertEqual(model['validity'], 'complete')
        rs = {r['name']: r for r in model['regions']}
        for row in rs['reduce']['observations']:
            n, b = row['pcvs']['elements'], row['pcvs']['blocks']
            counts = row['metrics']
            self.assertEqual(counts['fp32_flops'], 63*b)
            self.assertEqual(counts['global_read_bytes'], 4*n)
            self.assertEqual(counts['global_write_bytes'], 4*b)
            self.assertEqual(counts['shared_read_bytes'], (63*2+1)*4*b)
            self.assertEqual(counts['shared_write_bytes'], (64+63)*4*b)
            self.assertEqual(counts['emulator_barrier_generations'], 7*b)
            self.assertEqual(counts['block_barrier_lane_arrivals'], 64*7*b)
            self.assertGreater(counts['sum_block_lane_sequence_span'], 0)
        for row in rs['shuffle']['observations']:
            self.assertEqual(row['metrics']['fp32_flops'], 32*5*row['pcvs']['warps'])
        for row in rs['bf16-matrix']['observations']:
            tiles = row['pcvs']['tiles']
            counts = row['metrics']
            self.assertEqual(counts['tensor_float_flops'], 2*16*16*16*tiles)
            self.assertEqual(counts['global_read_bytes'], (512+512+1024)*tiles)
            self.assertEqual(counts['global_write_bytes'], 1024*tiles)
        self.assertFalse(rs['missing-pcv']['fits']['fp32_flops']['accepted'])
        self.assertTrue(rs['triangular']['fits']['fp32_flops']['accepted'])
        self.assertEqual(rs['triangular']['fits']['fp32_flops']['coefficients']['pairs'], 1)
        variation = rs['same-state-variation']
        self.assertEqual(variation['stateCount'], 1)
        self.assertFalse(variation['fits']['fp32_flops']['accepted'])
        self.assertEqual(sorted(r['metrics'].get('fp32_flops', 0) for r in variation['observations']), [0,0,32,32])
        self.assertEqual(rs['parent']['observations'][0]['metrics'], {})
        self.assertEqual(rs['child']['observations'][0]['metrics']['fp32_flops'], 160)
        self.assertNotEqual(rs['child']['observations'][0]['parent'], 0)
        self.assertEqual(rs['bf16-matrix']['fits']['tensor_float_flops']['coefficients']['tiles'], 8192)

    def test_zero_pcv_constant_region(self):
        model = self.capture('constant', [self.driver, self.module, 'constant'])
        r = model['regions'][0]
        self.assertTrue(r['singleState'])
        self.assertEqual(r['fits']['fp32_flops']['constant'], 160)
        self.assertTrue(r['fits']['fp32_flops']['accepted'])

    def test_host_threads_keep_lane_state_and_regions_separate(self):
        model = self.capture('threads', [self.driver, self.module, 'threads'])
        r = model['regions'][0]
        self.assertEqual(r['invocations'], 12)
        self.assertEqual(len({x['token'] for x in r['observations']}), 12)
        self.assertTrue(all(x['metrics']['fp32_flops'] == 320 for x in r['observations']))

    def test_unmodeled_work_is_explicitly_incomplete(self):
        dest = self.work/'unmodeled'
        run([ROOT/'bin/drperf', '--gpu', '--output', dest, '--',
             self.driver, self.module, 'unmodeled'], 2)
        model = json.loads((dest/'profile.gpu.drperf.json').read_text())
        self.assertEqual(model['validity'], 'incomplete')
        self.assertGreater(model['regions'][0]['unmodeledWorkEvents'], 0)

    def test_uninstrumented_and_wrong_abi_modules_fail(self):
        cxx = os.environ.get('CXX', 'g++')
        for abi in [1, 2]:
            src, so = self.work/f'fake-{abi}.cpp', self.work/f'fake-{abi}.so'
            src.write_text('extern "C" void shuffle__gxptx(void**) {}\n'
                           'extern "C" const unsigned shuffle__gxptx_nparams=1;\n'
                           f'extern "C" const unsigned drperf_gpu_instrumented_abi={abi};\n')
            run([cxx, '-shared', '-fPIC', src, '-o', so])
            p = subprocess.run([str(ROOT/'bin/drperf'), '--gpu', '--output', str(self.work/f'fake-{abi}'),
                                '--', str(self.driver), str(so), 'constant'],
                               cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('ABI mismatch' if abi == 1 else 'without instrumentation', p.stderr)

    def test_lifecycle_failures_are_not_successful_captures(self):
        self.capture('open', [self.driver, self.module, 'unclosed'], 2)
        self.capture('truncated', [self.driver, self.module, 'truncated'], 2)
        p = subprocess.run([str(ROOT/'bin/drperf'), '--gpu', '--output', str(self.work/'unmarked'),
                            '--', str(self.driver), str(self.module), 'unmarked'],
                           cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('no enclosing GPU region', p.stderr)
        self.assertFalse((self.work/'unmarked/profile.gpu.drperf.json').exists())

    def test_gpu_path_does_not_import_cpu_collector(self):
        code = """import runpy,sys
m=runpy.run_path('bin/drperf')
try:
    m['main'](['--gpu','--help'])
except SystemExit as error:
    assert error.code == 0
assert 'runner' not in sys.modules
assert 'derive' not in sys.modules
"""
        run([sys.executable, '-c', code])

    def test_gpu_report_query(self):
        model = self.capture('query', [self.driver, self.module, 'constant'])
        p = run([ROOT/'bin/drperf', '--gpu', '--report', self.work/'query', '--json'])
        self.assertEqual(json.loads(p.stdout), model)

    def test_capture_record_loss_is_detected(self):
        model = self.capture('record-loss', [self.driver, self.module, 'constant'])
        path = Path(model['provenance']['rawCaptures'][0])
        records = path.read_text().splitlines()
        damaged = self.work/'missing-record.jsonl'
        damaged.write_text('\n'.join([records[0], *records[2:]])+'\n')
        with self.assertRaisesRegex(ValueError, 'record count mismatch'):
            gpu_backend.read_capture([damaged])

    def test_generic_bytes_are_not_claimed_as_global_transactions(self):
        source = self.work/'generic.ptx'
        source.write_text((ROOT/'gpu/examples/saxpy.ptx').read_text()
                          .replace('ld.global.', 'ld.').replace('st.global.', 'st.'))
        module = self.work/'generic.so'
        run([ROOT/'bin/drperf', '--gpu', 'translate', source, '-o', module])
        model = self.capture('generic', [self.saxpy_driver, module])
        for row in model['regions'][0]['observations']:
            n, c = row['pcvs']['elements'], row['metrics']
            self.assertEqual(c.get('generic_read_bytes', 0), 8*n)
            self.assertEqual(c.get('generic_write_bytes', 0), 4*n)
            self.assertEqual(c.get('global_read_bytes', 0), 0)

    def test_translation_does_not_overwrite_input_or_sibling_host_cpp(self):
        module = self.work/'protected.so'
        sibling = self.work/'protected.cpp'
        sibling.write_text('HOST SOURCE MUST REMAIN\n')
        run([ROOT/'bin/drperf', '--gpu', 'translate', ROOT/'gpu/examples/saxpy.ptx', '-o', module])
        self.assertEqual(sibling.read_text(), 'HOST SOURCE MUST REMAIN\n')
        source = self.work/'protected-input.ptx'
        original = (ROOT/'gpu/examples/saxpy.ptx').read_text()
        source.write_text(original)
        run([ROOT/'bin/drperf', '--gpu', 'translate', source, '-o', source], 2)
        self.assertEqual(source.read_text(), original)

    def test_small_coefficients_survive_report_formatting(self):
        rows = [{'region': 'small-coefficient', 'pcvs': {'n': i*10**9},
                 'metrics': {'global_read_bytes': i}, 'sites': {}, 'launches': []}
                for i in range(4)]
        f = gpu_backend.build_model(rows)['regions'][0]['fits']['global_read_bytes']
        self.assertTrue(f['accepted'])
        self.assertAlmostEqual(f['coefficients']['n']*10**9, 1)


if __name__ == '__main__':
    unittest.main()
