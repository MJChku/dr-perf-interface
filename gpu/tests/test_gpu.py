#!/usr/bin/env python3
"""End-to-end PTX execution and independent work-count/fit oracles. No GPU/CUDA."""
import json
import os
import shutil
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
                               ('gpu/examples/saxpy.cpp', cls.saxpy_driver),
                               ('gpu/tests/tcgen05_driver.cc', cls.work/'tcgen-driver'),
                               ('gpu/tests/device_regions_driver.cc', cls.work/'device-driver')]:
            run([cxx, '-std=c++17', '-O2', '-pthread', '-Igpu/include', source,
                 '-Lgpu/build', '-ldrperf_gpu', '-Wl,-rpath,'+str(ROOT/'gpu/build'), '-o', target])
        for source, target in [('gpu/tests/kernels.ptx', cls.module),
                               ('gpu/examples/saxpy.ptx', cls.saxpy),
                               ('gpu/tests/tcgen05.ptx', cls.work/'tcgen.so'),
                               ('gpu/tests/device_regions.ptx', cls.work/'device.so'),
                               ('gpu/tests/device_tensor_regions.ptx', cls.work/'device-tensor.so')]:
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
            self.assertEqual(metrics.get('global_load_instructions', 0), 2*n)
            self.assertEqual(metrics['launched_lanes'], 128*b)
            self.assertEqual(metrics['warps'], 4*b)
        for metric, coefficient in [('fp32_flops', 2), ('global_read_bytes', 8),
                                    ('global_write_bytes', 4), ('global_load_instructions', 2)]:
            f = r['fits'][metric]
            self.assertTrue(f['accepted'], f)
            self.assertEqual(f['coefficients']['elements'], coefficient)
            self.assertEqual(f['constant'], 0)
            # Evaluate the learned formula at an unobserved shape against
            # the independent algorithmic count, including a different grid.
            self.assertEqual(f['constant']+f['coefficients']['elements']*511+
                             f['coefficients']['blocks']*4, coefficient*511)
        # Sector/line footprints need rounded-size and allocation-alignment
        # PCVs; element counts alone still explain every original work metric
        # and exact unique-byte footprint.
        rounded = ('_sectors', '_lines', '_sector_bytes', '_line_bytes')
        # Max-plus timing paths can change at zero/partial launch boundaries;
        # their fits must retain rejection instead of weakening work oracles.
        modeled = {'latency_critical_path', 'dependent_global_loads_on_critical_path'}
        self.assertTrue(all(f['accepted'] for site in r['sites'].values()
                            for metric, f in site.items()
                            if metric not in modeled and not metric.endswith(rounded)))
        parallel = r['fits']['sum_block_max_lane_flops']
        self.assertTrue(parallel['accepted'])
        self.assertEqual(parallel['coefficients']['active_blocks'], 2)

    def test_device_lane_pcvs_nested_exclusive_work_and_output_oracle(self):
        model=self.capture('device-lanes',[self.work/'device-driver',self.work/'device.so','lane'])
        rs={r['name']:r for r in model['regions']}
        loops=rs['device.lane.loop'];outer=rs['device.lane.outer']
        self.assertEqual(loops['fits']['fp32_flops']['coefficients']['iterations'],1)
        self.assertTrue(loops['fits']['fp32_flops']['accepted'])
        self.assertEqual(outer['fits']['fp32_flops']['coefficients']['active'],1)
        self.assertTrue(outer['fits']['fp32_flops']['accepted'])
        parents={r['token']:r for r in outer['observations']}
        for r in loops['observations']:
            self.assertEqual(r['scope'],'lane')
            self.assertEqual(r['metrics']['fp32_flops'],r['pcvs']['iterations'])
            self.assertIn(r['parent'],parents)
            self.assertEqual(r['kernel'],'device_sections')
            self.assertTrue(any(site.startswith('annotated_adds:') for site in r['sites']))
            self.assertEqual(r['metrics']['region_lane_participants'],1)
        for r in outer['observations']:
            self.assertEqual(r['metrics'].get('fp32_flops',0),r['pcvs']['active'])
        for r in rs['device-envelope']['observations']:
            self.assertEqual(r['metrics'].get('fp32_flops',0),0)
            self.assertEqual(r['launches'][0]['metricScope'],'inclusive')
        total=sum(r['metrics'].get('fp32_flops',0) for v in rs.values() for r in v['observations'])
        expected=sum(1+((i*7+11)%17)%3+1 for n in [0,1,7,35,53] for i in range(n))
        self.assertEqual(total,expected)

    def test_device_annotations_need_no_host_envelope(self):
        model=self.capture('device-unwrapped',[self.work/'device-driver',self.work/'device.so','unwrapped'])
        rs={r['name']:r for r in model['regions']}
        root=rs['kernel.device_sections']
        self.assertEqual(root['invocations'],5)
        tokens={r['token'] for r in root['observations']}
        self.assertTrue(all(r['parent'] in tokens for r in rs['device.lane.outer']['observations']))
        self.assertTrue(all(r['scope']=='kernel' and r['parent']==0 for r in root['observations']))

    def test_device_cooperative_regions_partial_warps_and_barrier_fibers(self):
        model=self.capture('device-cooperative',[self.work/'device-driver',self.work/'device.so','cooperative'])
        rs={r['name']:r for r in model['regions']}
        blocks=rs['device.block.cooperative']['observations']
        warps=rs['device.warp.cooperative']['observations']
        self.assertEqual(len(blocks),24)
        parents={r['token'] for r in blocks}
        self.assertEqual({r['pcvs']['lanes'] for r in warps},{16,32})
        for r in warps:
            lanes=r['pcvs']['lanes'];m=r['metrics']
            self.assertEqual(m.get('fp32_flops',0),lanes*r['pcvs']['iterations'])
            self.assertEqual(m['region_lane_participants'],lanes)
            self.assertIn(r['parent'],parents)
        for r in blocks:
            lanes=r['pcvs']['lanes'];m=r['metrics']
            self.assertEqual(m['fp32_flops'],lanes)
            self.assertEqual(m['shared_write_bytes'],4*lanes)
            self.assertEqual(m['shared_read_bytes'],4*lanes)
            self.assertEqual(m['block_barrier_lane_arrivals'],lanes)
            self.assertEqual(m['global_write_bytes'],4*lanes)
            self.assertEqual(m['region_lane_participants'],lanes)

    def test_device_repeated_named_region_occurrences(self):
        model=self.capture('device-repeated',[self.work/'device-driver',self.work/'device.so','repeated'])
        region=next(r for r in model['regions'] if r['name']=='device.block.repeated')
        self.assertEqual(region['invocations'],8)
        for r in region['observations']:
            self.assertEqual(r['metrics']['fp32_flops'],32)
            self.assertEqual(r['metrics']['region_lane_participants'],32)

    def test_device_invalid_markers_never_produce_successful_profiles(self):
        errors=['unbalanced GPU device','invalid GPU device region','unclosed GPU device',
                'incomplete cooperative','inconsistent cooperative device PCV value',
                'inconsistent cooperative device PCVs','duplicate or excessive',
                'PCV registered after region entry','missing entry PCV completion']
        for mode,error in enumerate(errors):
            with self.subTest(mode=mode):
                result=self.capture('device-invalid-'+str(mode),
                    [self.work/'device-driver',self.work/'device.so',str(mode)],expected=1)
                self.assertIn(error,result.stderr)

    def test_device_inner_region_detects_hidden_history_scaling(self):
        bad=self.capture('device-sparse',[self.work/'device-driver',self.work/'device.so','sparse'])
        good=self.capture('device-sparse-refined',[self.work/'device-driver',self.work/'device.so','sparse-refined'])
        before={r['name']:r for r in bad['regions']};after={r['name']:r for r in good['regions']}
        clear='device.block.dense_clear';math='device.block.selected_math'
        self.assertFalse(before[clear]['fits']['global_write_bytes']['accepted'])
        fit=after[clear]['fits']['global_write_bytes']
        self.assertTrue(fit['accepted'])
        self.assertEqual(fit['coefficients']['history'],4)
        self.assertEqual(fit['coefficients']['selected'],0)
        self.assertEqual(fit['constant'],0)
        for r in after[clear]['observations']:
            self.assertEqual(r['metrics']['global_write_bytes'],4*r['pcvs']['history'])
        fit=before[math]['fits']['fp32_flops']
        self.assertTrue(fit['accepted'])
        self.assertEqual(fit['coefficients']['selected'],1)

    def test_device_tcgen05_semantic_resources_are_scoped_once(self):
        model=self.capture('device-tensor',[self.work/'tcgen-driver',self.work/'device-tensor.so'])
        rs={r['name']:r for r in model['regions']}
        for name,tile_work in [('bf16_tile',2*128*64*16),('nvfp4_tile',2*128*64*64)]:
            r=rs['device.block.'+name]
            for row in r['observations']:
                self.assertEqual(row['metrics']['tensor_float_flops'],tile_work*row['pcvs']['repeats'])
                self.assertEqual(row['metrics']['tensor_mma_instructions'],row['pcvs']['repeats'])
                self.assertEqual(row['metrics']['region_lane_participants'],128)
            fit=r['fits']['tensor_float_flops']
            self.assertTrue(fit['accepted'])
            self.assertEqual(fit['coefficients']['repeats'],tile_work)
        for name in ['tcgen-bf16','tcgen-nvfp4']:
            for row in rs[name]['observations']:
                self.assertEqual(row['metrics'].get('tensor_float_flops',0),0)
                self.assertGreater(row['launches'][0]['metrics']['tensor_float_flops'],0)

    def test_nvcc_device_marker_abi_and_native_builds(self):
        nvcc=os.environ.get('NVCC') or shutil.which('nvcc')
        if not nvcc:self.skipTest('NVCC optional; checked-in device PTX tested above')
        common=[nvcc,'-std=c++17','-Igpu/include','-arch='+os.environ.get('DRPERF_GPU_TEST_ARCH','sm_80')]
        late=self.work/'late-pcv.cu'
        late.write_text('#include "drperf_gpu_device.cuh"\n'
                        '__global__ void k(){ drperf::gpu::DeviceRegion r("r"); r.pcv("late",1); }\n')
        error=run([*common,'-ptx',late,'-o',self.work/'late-pcv.ptx'],expected=1)
        self.assertIn('pcv',error.stderr)
        for opt in ['-O0','-O2','-O3']:
            ptx=self.work/('device'+opt+'.ptx');module=self.work/('device'+opt+'.so')
            run([*common,opt,'-ptx','gpu/tests/device_regions.cu','-o',ptx])
            run([ROOT/'bin/drperf','--gpu','translate',ptx,'-o',module])
            self.capture('nvcc-device'+opt,[self.work/'device-driver',module,'lane'])
        obj=self.work/'device-native.o';stubs=self.work/'device-stubs.o'
        run([*common,'-O2','-dc','gpu/tests/device_regions.cu','-o',obj])
        run([*common,'-O2','-dc','gpu/runtime/device_marker_stubs.cu','-o',stubs])
        run([*common,'-dlink',obj,stubs,'-o',self.work/'device-linked.o'])
        disabled=self.work/'device-disabled.ptx'
        run([*common,'-O2','-DDRPERF_GPU_DISABLE_DEVICE_REGIONS','-ptx','gpu/tests/device_regions.cu','-o',disabled])
        self.assertNotIn('__drperf_gpu_device_',disabled.read_text())
        module=self.work/'device-disabled.so'
        run([ROOT/'bin/drperf','--gpu','translate',disabled,'-o',module])
        self.capture('nvcc-device-disabled',[self.work/'device-driver',module,'lane'])
        run([*common,'-O2','-DDRPERF_GPU_DISABLE_DEVICE_REGIONS','-cubin','gpu/tests/device_regions.cu',
             '-o',self.work/'device-disabled.cubin'])

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
        self.assertTrue(variation['fits']['fp32_flops']['accepted'])
        self.assertEqual(variation['fits']['fp32_flops']['constant'], 16)
        self.assertEqual(variation['stateStatistics'][0]['metrics']['fp32_flops'],
                         {'mean': 16, 'min': 0, 'max': 32})
        self.assertEqual(sorted(r['metrics'].get('fp32_flops', 0) for r in variation['observations']), [0,0,32,32])
        self.assertEqual(rs['parent']['observations'][0]['metrics'], {})
        self.assertEqual(rs['child']['observations'][0]['metrics']['fp32_flops'], 160)
        self.assertNotEqual(rs['child']['observations'][0]['parent'], 0)
        self.assertIn('F[child]', rs['parent']['interfaces']['fp32_flops']['formula'])
        self.assertEqual(rs['bf16-matrix']['fits']['tensor_float_flops']['coefficients']['tiles'], 8192)

    def test_tcgen05_numerics_and_descriptor_resource_counts(self):
        model = self.capture('tcgen-report', [self.work/'tcgen-driver', self.work/'tcgen.so'])
        self.assertEqual(model['validity'], 'complete')
        rs = {r['name']: r for r in model['regions']}
        for name, work in [('tcgen-bf16', 2*128*64*16), ('tcgen-nvfp4', 2*128*64*64)]:
            r=rs[name]
            for row in r['observations']:
                m=row['metrics'];tiles=row['pcvs']['mma_tiles'];blocks=row['pcvs']['blocks']
                self.assertEqual(m['tensor_float_flops'], work*tiles)
                self.assertEqual(m['tensor_mma_instructions'], tiles)
                self.assertEqual(m['global_write_bytes'], 4*128*64*blocks)
                self.assertEqual(m.get('unmodeled_work_events',0), 0)
                # Tensor work belongs to original MMA statement sites, not C++ loops.
                sites=[v for k,v in row['sites'].items() if ':tcgen05.mma.' in k]
                self.assertEqual(sum(x.get('tensor_float_flops',0) for x in sites), work*tiles)
            self.assertTrue(r['fits']['tensor_float_flops']['accepted'])
            self.assertEqual(r['fits']['tensor_float_flops']['coefficients']['mma_tiles'],work)
        for row in rs['tcgen-bf16']['observations']:
            tiles=row['pcvs']['mma_tiles'];m=row['metrics']
            self.assertEqual(m['tmem_read_bytes'],32768*tiles)
            self.assertEqual(m['tmem_write_bytes'],32768*tiles)
        for row in rs['tcgen-nvfp4']['observations']:
            tiles=row['pcvs']['mma_tiles'];blocks=row['pcvs']['blocks'];m=row['metrics']
            self.assertEqual(m['tmem_read_bytes'],(32768+4*(128+64))*tiles)
            self.assertEqual(m['tmem_write_bytes'],32768*tiles+4096*blocks)

    def test_tcgen05_masked_accounting_is_honestly_incomplete(self):
        dest=self.work/'tcgen-masked'
        run([ROOT/'bin/drperf','--gpu','--output',dest,'--',self.work/'tcgen-driver',self.work/'tcgen.so','mask'],2)
        model=json.loads((dest/'profile.gpu.drperf.json').read_text())
        self.assertEqual(model['validity'],'incomplete')
        region=next(r for r in model['regions'] if r['name']=='tcgen-bf16')
        self.assertGreater(region['unmodeledWorkEvents'],0)

    def test_tcgen05_two_cta_group_requires_cluster_geometry(self):
        result=self.capture('tcgen-group2',[self.work/'tcgen-driver',self.work/'tcgen.so','group2'],expected=1)
        self.assertIn('cluster rank outside launch geometry',result.stderr)

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

    def test_async_copy_work_is_accounted(self):
        dest = self.work/'unmodeled'
        run([ROOT/'bin/drperf', '--gpu', '--output', dest, '--',
             self.driver, self.module, 'unmodeled'])
        model = json.loads((dest/'profile.gpu.drperf.json').read_text())
        self.assertEqual(model['validity'], 'complete')
        row=model['regions'][0]['observations'][0]
        self.assertEqual(row['metrics']['global_read_bytes'],16)
        self.assertEqual(row['metrics']['shared_write_bytes'],16)
        self.assertEqual(row['metrics']['shared_read_bytes'],4)
        self.assertEqual(row['metrics']['global_write_bytes'],4)

    def test_uninstrumented_and_wrong_abi_modules_fail(self):
        cxx = os.environ.get('CXX', 'g++')
        for abi in [1, 2, 3]:
            src, so = self.work/f'fake-{abi}.cpp', self.work/f'fake-{abi}.so'
            src.write_text('extern "C" void shuffle__gxptx(void**) {}\n'
                           'extern "C" const unsigned shuffle__gxptx_nparams=1;\n'
                           'extern "C" const unsigned shuffle__gxptx_syncfree=0;\n'
                           'extern "C" const unsigned drperf_gpu_latency_abi=1;\n'
                           'extern "C" const unsigned shuffle__gxptx_latency_abi=1;\n'
                           f'extern "C" const unsigned drperf_gpu_instrumented_abi={abi};\n')
            run([cxx, '-shared', '-fPIC', src, '-o', so])
            p = subprocess.run([str(ROOT/'bin/drperf'), '--gpu', '--output', str(self.work/f'fake-{abi}'),
                                '--', str(self.driver), str(so), 'constant'],
                               cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn('ABI mismatch' if abi != 3 else 'without instrumentation', p.stderr)

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
