#!/usr/bin/env python3
"""Exact memory footprint, fitting, and derived reuse oracles. CPU only."""
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
GPU = ROOT / 'gpu'
sys.path.insert(0, str(GPU / 'tools'))
from footprint_report import summarize_report
from diff_reports import diff_reports


def run(*args, expected=0):
    result = subprocess.run([str(a) for a in args], cwd=ROOT, text=True,
                            capture_output=True, timeout=180)
    if result.returncode != expected:
        raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result


class FootprintIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = GPU / 'build/ext-work'
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix='footprint-test-', dir=scratch)
        cls.work = Path(cls.temp.name)
        cxx = os.environ.get('CXX', 'g++')
        run(cxx, '-std=c++17', '-O2', GPU / 'tests/footprint_unit.cc', '-o', cls.work / 'unit')
        run(cls.work / 'unit')
        run(cxx, '-std=c++17', '-O2', '-Igpu/include', GPU / 'tests/footprint_driver.cc',
            '-Lgpu/build', '-ldrperf_gpu', '-Wl,-rpath,' + str(GPU / 'build'), '-o', cls.work / 'driver')
        run(ROOT / 'bin/drperf', '--gpu', 'translate', GPU / 'tests/footprint.ptx', '-o', cls.work / 'kernel.so')
        cls.dest = cls.work / 'report'
        run(ROOT / 'bin/drperf', '--gpu', '--output', cls.dest, '--', cls.work / 'driver', cls.work / 'kernel.so')
        cls.report = json.loads((cls.dest / 'profile.gpu.drperf.json').read_text())
        cls.regions = {r['name']: r for r in cls.report['regions']}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_broadcast_redundancy_and_site_fits(self):
        region = self.regions['broadcast']
        for row in region['observations']:
            n = row['pcvs']['lanes']
            self.assertEqual(row['footprintVersion'], 1)
            for counts in [row['metrics'], row['launches'][0]['metrics']]:
                for direction in ('read', 'write'):
                    self.assertEqual(counts[f'global_{direction}_bytes'], 4 * n)
                    self.assertEqual(counts[f'global_unique_{direction}_bytes'], 4)
                    self.assertEqual(counts[f'global_unique_{direction}_sectors'], 1)
                    self.assertEqual(counts[f'global_unique_{direction}_lines'], 1)
            sites = [counts for counts in row['sites'].values() if counts.get('global_read_bytes')]
            self.assertEqual(len(sites), 1)
            self.assertEqual(sites[0]['global_unique_read_bytes'], 4)
        for direction in ('read', 'write'):
            for suffix in ('bytes', 'sectors', 'lines', 'sector_bytes', 'line_bytes'):
                fit = region['fits'][f'global_unique_{direction}_{suffix}']
                self.assertTrue(fit['accepted'], fit)
                self.assertEqual(fit['coefficients']['lanes'], 0)
        # Address unions cannot be added across the instructions of a block.
        # Keep their exact observations and region fit, not a fabricated sum.
        self.assertTrue(all('global_unique_read_bytes' not in b for b in region['sites'].values()))

    def test_streaming_unique_equals_requested_and_fits(self):
        region = self.regions['streaming']
        for row in region['observations']:
            n = row['pcvs']['lanes']
            for direction in ('read', 'write'):
                counts = row['metrics']
                self.assertEqual(counts[f'global_unique_{direction}_bytes'], 4 * n)
                self.assertEqual(counts[f'global_unique_{direction}_sector_bytes'], 4 * n)
                self.assertEqual(counts[f'global_unique_{direction}_line_bytes'], 4 * n)
                fit = region['fits'][f'global_unique_{direction}_bytes']
                self.assertTrue(fit['accepted'])
                self.assertEqual(fit['coefficients']['lanes'], 4)

    def test_strided_partial_sectors_and_lines(self):
        for name, lines_per_lane in [('sector_partial', 1 / 4), ('line_stride', 1)]:
            for row in self.regions[name]['observations']:
                counts, n = row['metrics'], row['pcvs']['lanes']
                self.assertEqual(counts['global_read_bytes'], 4 * n)
                self.assertEqual(counts['global_unique_read_bytes'], 4 * n)
                self.assertEqual(counts['global_unique_read_sectors'], n)
                self.assertEqual(counts['global_unique_read_lines'], n * lines_per_lane)

    def test_region_union_across_launches_not_sum(self):
        row = self.regions['two_launches']['observations'][0]
        self.assertEqual(row['metrics']['global_read_bytes'], 256)
        self.assertEqual(row['metrics']['global_unique_read_bytes'], 4)
        self.assertEqual(row['metrics']['global_unique_write_bytes'], 128)
        self.assertEqual(len(row['launches']), 2)
        self.assertEqual(sum(l['metrics']['global_unique_read_bytes'] for l in row['launches']), 8)
        site = next(c for c in row['sites'].values() if c.get('global_read_bytes'))
        self.assertEqual(site['global_read_bytes'], 256)
        self.assertEqual(site['global_unique_read_bytes'], 4)
        self.assertTrue(all('sites' in l for l in row['launches']))

    def test_exclusive_nested_lane_warp_block_ownership(self):
        for scope, participants, observations in [('lane', 1, 64), ('warp', 32, 2), ('block', 64, 1)]:
            host = self.regions['scoped_' + scope]['observations'][0]
            self.assertEqual(host['metrics']['global_read_bytes'], 256)
            self.assertEqual(host['metrics']['global_unique_read_bytes'], 4)
            launch = host['launches'][0]
            self.assertEqual(launch['metrics']['global_read_bytes'], 768)
            self.assertEqual(launch['metrics']['global_unique_read_bytes'], 8)
            self.assertEqual(launch['metrics']['global_unique_read_sectors'], 2)
            for name in ('outer', 'inner'):
                rows = self.regions[f'device.{scope}.{name}']['observations']
                self.assertEqual(len(rows), observations)
                for row in rows:
                    self.assertEqual(row['metrics']['global_read_bytes'], 4 * participants)
                    self.assertEqual(row['metrics']['global_unique_read_bytes'], 4)
                    self.assertEqual(row['metrics']['global_unique_read_sectors'], 1)
                    self.assertEqual(row['metrics']['region_lane_participants'], participants)
                    self.assertEqual(sum(c.get('global_unique_read_bytes', 0) for c in row['sites'].values()), 4)

    def test_generic_atomic_vector_accesses(self):
        counts = self.regions['generic']['observations'][0]['metrics']
        self.assertEqual(counts['generic_unique_read_bytes'], 4)
        self.assertEqual(counts['generic_unique_write_bytes'], 4)
        self.assertNotIn('global_unique_read_bytes', counts)
        counts = self.regions['atomic']['observations'][0]['metrics']
        for direction in ('read', 'write'):
            self.assertEqual(counts[f'global_{direction}_bytes'], 128)
            self.assertEqual(counts[f'global_unique_{direction}_bytes'], 4)
        counts = self.regions['vector']['observations'][0]['metrics']
        for direction in ('read', 'write'):
            self.assertEqual(counts[f'global_{direction}_bytes'], 512)
            self.assertEqual(counts[f'global_unique_{direction}_bytes'], 512)
            self.assertEqual(counts[f'global_unique_{direction}_sectors'], 16)
            self.assertEqual(counts[f'global_unique_{direction}_lines'], 4)

    def test_ratio_tool_text_json_and_partial_utilization(self):
        summary = summarize_report(self.report, level='region', region_pattern='broadcast')
        for record in summary['records']:
            n = record['pcvs']['lanes']
            for access in record['accesses']:
                self.assertEqual(access['redundancyRatio'], n)
                self.assertEqual(access['sectorUtilization'], 1 / 8)
        summary = summarize_report(self.report, level='region', region_pattern='sector_partial')
        for record in summary['records']:
            access = next(a for a in record['accesses'] if a['direction'] == 'read')
            self.assertEqual(access['redundancyRatio'], 1)
            self.assertEqual(access['requestedPerSectorByte'], 1 / 8)
        tool = GPU / 'tools/footprint_report.py'
        result = run(sys.executable, tool, self.dest, '--json', '--level', 'launch', '--region', 'broadcast')
        self.assertEqual(len(json.loads(result.stdout)['records']), 3)
        result = run(sys.executable, tool, self.dest, '--level', 'site', '--region', 'broadcast')
        self.assertIn('32x', result.stdout)

    def test_wmma_strided_tile_counts_exclude_holes(self):
        row = self.regions['wmma_stride']['observations'][0]
        for counts in [row['metrics'], row['launches'][0]['metrics']]:
            for direction in ('read', 'write'):
                self.assertEqual(counts[f'global_{direction}_bytes'], 1024)
                self.assertEqual(counts[f'global_unique_{direction}_bytes'], 1024)
                self.assertEqual(counts[f'global_unique_{direction}_sectors'], 32)
                self.assertEqual(counts[f'global_unique_{direction}_lines'], 16)
        for direction in ('read', 'write'):
            site = next(c for c in row['sites'].values() if c.get(f'global_{direction}_bytes'))
            self.assertEqual(site[f'global_unique_{direction}_bytes'], 1024)

    def semantic_report(self, name):
        cache = self.__class__.__dict__.get('semantic_reports')
        if cache is None:
            cache = self.__class__.semantic_reports = {}
        if name not in cache:
            driver, module, dest = self.work / name, self.work / (name + '.so'), self.work / (name + '-report')
            run(os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-Igpu/include',
                GPU / f'tests/{name}_driver.cc', '-Lgpu/build', '-ldrperf_gpu',
                '-Wl,-rpath,' + str(GPU / 'build'), '-o', driver)
            run(ROOT / 'bin/drperf', '--gpu', 'translate', GPU / f'tests/{name}.ptx', '-o', module)
            run(ROOT / 'bin/drperf', '--gpu', '--output', dest, '--', driver, module)
            cache[name] = {r['name']: r for r in json.loads((dest / 'profile.gpu.drperf.json').read_text())['regions']}
        return cache[name]

    def test_async_tail_bulk_mask_and_tma_reduction_footprints(self):
        regions = self.semantic_report('tma_ext')
        for row in regions['async.tail']['observations']:
            unique = 4 if row['pcvs']['source_bytes'] == 7 else 16
            self.assertEqual(row['metrics']['global_unique_read_bytes'], unique)
            self.assertEqual(row['metrics']['global_unique_write_bytes'], 48)
            self.assertEqual(row['metrics']['global_read_bytes'], row['pcvs']['source_bytes'])
        bulk = regions['tma.bulk']['observations'][0]
        self.assertEqual(bulk['metrics']['global_unique_read_bytes'], 768)
        self.assertEqual(bulk['metrics']['global_unique_write_bytes'], 640)
        mask = next(c for site, c in bulk['sites'].items() if '.cp_mask' in site)
        self.assertEqual(mask['global_unique_write_bytes'], 128)
        self.assertEqual(mask['global_unique_write_sectors'], 8)
        self.assertEqual(mask['global_unique_write_lines'], 2)
        for name, size in [('k_reduce_f32', 256), ('k_reduce_s32', 1024)]:
            row = regions[name]['observations'][0]
            counts = next(c for site, c in row['sites'].items() if ':cp.reduce.async.bulk.tensor.' in site)
            self.assertEqual(counts['global_unique_read_bytes'], size)
            self.assertEqual(counts['global_unique_write_bytes'], size)
        row = regions['tma.gather_scatter']['observations'][0]
        gather = next(c for site, c in row['sites'].items() if 'tile::gather4' in site)
        scatter = next(c for site, c in row['sites'].items() if 'tile::scatter4' in site)
        self.assertEqual(gather['global_unique_read_bytes'], 384)
        self.assertEqual(gather['global_unique_read_sectors'], 12)
        self.assertEqual(gather['global_unique_read_lines'], 6)
        self.assertEqual(scatter['global_unique_write_bytes'], 512)
        self.assertEqual(scatter['global_unique_write_sectors'], 16)
        self.assertEqual(scatter['global_unique_write_lines'], 8)

    def test_tma_oob_subbyte_strided_source_and_multicast_unions(self):
        regions = self.semantic_report('tma')
        # Independent geometry/byte-set oracle for the retained parity fixtures.
        cases = [
            (16, (104,37,3), (64,16,2), (1,1,1), (8,5,0)),
            (16, (104,37,3), (64,16,2), (1,1,1), (72,30,2)),
            (16, (104,37,3), (32,8,1), (1,1,1), (-8,-3,1)),
            (8, (208,37,3), (32,24,3), (1,1,1), (16,2,0)),
            (32, (52,37,3), (12,15,3), (1,3,2), (4,1,0)),
            (4, (416,37,3), (256,8,1), (1,1,1), (64,7,2)),
            (4, (512,37,3), (128,8,1), (1,1,1), (128,4,1)),
        ]
        self.assertEqual(len(regions['tma.load']['observations']), len(cases))
        for row, (bits, dims, box, step, coords) in zip(regions['tma.load']['observations'], cases):
            addresses = set()
            for z in range(0, box[2], step[2]):
                for y in range(0, box[1], step[1]):
                    for x in range(box[0]):
                        gx, gy, gz = x + coords[0], y + coords[1], z + coords[2]
                        if not (0 <= gx < dims[0] and 0 <= gy < dims[1] and 0 <= gz < dims[2]):
                            continue
                        first_bit = gx * bits + (gy * 256 + gz * 256 * 37) * 8
                        addresses.update(range(first_bit // 8, (first_bit + bits + 7) // 8))
            counts = next(c for site, c in row['sites'].items() if ':cp.async.bulk.tensor.' in site)
            self.assertEqual(counts['global_unique_read_bytes'], len(addresses))
            self.assertEqual(counts['global_unique_read_sectors'], len({a // 32 for a in addresses}))
            self.assertEqual(counts['global_unique_read_lines'], len({a // 128 for a in addresses}))
        for row in regions['tma.multicast']['observations']:
            self.assertEqual(row['metrics']['global_unique_read_bytes'], row['pcvs']['source_bytes'])
            self.assertEqual(row['metrics']['global_unique_read_sectors'], 160)
            self.assertEqual(row['metrics']['global_unique_read_lines'], 64)

    def test_elected_lane_fix_changes_requests_but_keeps_footprint(self):
        # A real PTX candidate: broadcast-mode loads elect one lane per warp,
        # then shuffle the loaded word. All other copy modes keep their loads.
        source = (GPU / 'tests/footprint.ptx').read_text()
        source = source.replace('.reg .b32 %r<8>;', '.reg .b32 %r<8>;\n .reg .pred %p<2>;', 1)
        original_load = ' ld.global.u32 %r5,[%d3];'
        candidate_load = ''' setp.ne.u32 %p0,%r0,0;
 @%p0 bra regular_load;
 and.b32 %r7,%r3,31;
 setp.eq.u32 %p1,%r7,0;
 @%p1 ld.global.u32 %r5,[%d3];
 shfl.sync.idx.b32 %r5,%r5,0,31,0xffffffff;
 bra loaded;
regular_load:
 ld.global.u32 %r5,[%d3];
loaded:'''
        self.assertEqual(source.count(original_load), 1)
        ptx, module, dest = self.work / 'elected.ptx', self.work / 'elected.so', self.work / 'elected-report'
        ptx.write_text(source.replace(original_load, candidate_load, 1))
        run(ROOT / 'bin/drperf', '--gpu', 'translate', ptx, '-o', module)
        run(ROOT / 'bin/drperf', '--gpu', '--output', dest, '--', self.work / 'driver', module)
        candidate = json.loads((dest / 'profile.gpu.drperf.json').read_text())
        before = self.regions['broadcast']
        after = next(r for r in candidate['regions'] if r['name'] == 'broadcast')
        # The edit splits the old load/store block. Do not fabricate a
        # one-to-one instruction mapping between differently structured blocks.
        changes = diff_reports(self.report, candidate, region_pattern='broadcast', metric_pattern='global_*',
                               include_unchanged=False)
        region_changes = [c for c in changes['changes'] if c['site'] is None and c['afterSite'] is None]
        self.assertEqual(len(region_changes), 2)
        self.assertTrue(any(c['site'] or c['afterSite'] for c in changes['changes']))
        for change in region_changes:
            self.assertIn(change['metric'], ('global_read_bytes', 'global_load_instructions'))
            scale = 4 if change['metric'] == 'global_read_bytes' else 1
            self.assertEqual(change['coefficientChanges']['lanes']['before'], scale)
            self.assertEqual(change['coefficientChanges']['lanes']['after'], scale / 32)
            self.assertTrue(change['bothAccepted'])
        for row in after['observations']:
            self.assertEqual(row['metrics']['global_read_bytes'], row['pcvs']['lanes'] / 8)
            self.assertEqual(row['metrics']['global_unique_read_bytes'], 4)
            self.assertEqual(row['metrics']['global_unique_read_sectors'], 1)
            self.assertEqual(row['metrics']['global_unique_read_lines'], 1)
        tool = run(sys.executable, GPU / 'tools/diff_reports.py', self.dest, dest, '--json',
                   '--region', 'broadcast', '--metric', 'global_*')
        result = json.loads(tool.stdout)
        self.assertTrue(any(c['site'] is None and c['metric'] == 'global_read_bytes' for c in result['changes']))

    def test_split_translation_capability_marker_links_once(self):
        prefix = self.work / 'split'
        environment = {**os.environ, 'GXPTX_PRELUDE': str(GPU / 'vendor/gxptx/Gxptx/prelude.h'), 'GXPTX_SPLIT': '2'}
        result = subprocess.run([str(GPU / 'vendor/gxptx/.lake/build/bin/gxptx'),
                                 str(GPU / 'tests/footprint.ptx'), str(prefix)],
                                cwd=ROOT, env=environment, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        first = Path(str(prefix) + '.0.cpp')
        source = first.read_text()
        self.assertEqual(first.read_text(), source)
        first.write_text(source + '\nextern "C" const unsigned drperf_gpu_instrumented_abi = 3;\n')
        module = self.work / 'split.so'
        run(os.environ.get('CXX', 'g++'), '-std=c++17', '-O1', '-shared', '-fPIC', '-fno-strict-aliasing',
            '-fwrapv', first, Path(str(prefix) + '.1.cpp'), '-Lgpu/build', '-ldrperf_gpu',
            '-Wl,-z,defs', '-Wl,-rpath,' + str(GPU / 'build'), '-o', module)
        dest = self.work / 'split-report'
        run(ROOT / 'bin/drperf', '--gpu', '--output', dest, '--', self.work / 'driver', module)
        report = json.loads((dest / 'profile.gpu.drperf.json').read_text())
        for region in report['regions']:
            for row in region['observations']:
                self.assertEqual(row['footprintVersion'], 1)
                if region['name'] == 'broadcast':
                    self.assertEqual(row['metrics']['global_unique_read_bytes'], 4)

    def test_older_modules_keep_work_counts_but_mark_footprints_unavailable(self):
        # Either half of the address-coverage capability can be stale, while
        # the pre-existing work-counter ABI is still compatible.
        source = Path(str(self.work / 'kernel.so') + '.cpp').read_text()
        patterns = {
            'prelude': r'extern "C" __attribute__\(\(weak\)\) const unsigned drperf_gpu_footprint_abi = 1;',
            'translator': r'extern "C" const unsigned \w+__gxptx_footprint_abi = 1;',
        }
        for label, pattern in patterns.items():
            with self.subTest(capability=label):
                self.assertRegex(source, pattern)
                cpp, module = self.work / (label + '.cpp'), self.work / (label + '.so')
                cpp.write_text(re.sub(pattern, '', source))
                run(os.environ.get('CXX', 'g++'), '-std=c++17', '-O1', '-shared', '-fPIC', '-fno-strict-aliasing',
                    '-fwrapv', cpp, '-Lgpu/build', '-ldrperf_gpu', '-Wl,-rpath,' + str(GPU / 'build'), '-o', module)
                dest = self.work / (label + '-report')
                run(ROOT / 'bin/drperf', '--gpu', '--output', dest, '--', self.work / 'driver', module)
                report = json.loads((dest / 'profile.gpu.drperf.json').read_text())
                summary = summarize_report(report)
                self.assertTrue(summary['records'])
                for record in summary['records']:
                    self.assertFalse(record['footprintAvailable'])
                    for access in record['accesses']:
                        self.assertIsNone(access['uniqueBytes'])
                        self.assertIsNone(access['redundancyRatio'])


if __name__ == '__main__':
    unittest.main()
