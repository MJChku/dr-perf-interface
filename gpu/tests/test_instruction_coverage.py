#!/usr/bin/env python3
"""CPU-only packed-dot, matrix spelling, uniform-load and preflight regressions."""
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


def run(argv, expected=0):
    result = subprocess.run([str(x) for x in argv], cwd=ROOT, capture_output=True, text=True, timeout=180)
    if result.returncode != expected:
        raise AssertionError(f'{argv}: {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result


def entry(name, body, extra=''):
    return f'''.visible .entry {name}(.param .u64 input, .param .u64 output, .param .u64 stored) {{
 .reg .b64 %d<8>;
 .reg .b32 %r<32>;
 .reg .pred %p;
 {extra}
 ld.param.u64 %d0,[input];
 ld.param.u64 %d1,[output];
 mov.u32 %r20,%tid.x;
 {body}
 ret;
}}
'''


def fixtures():
    ptx = '.version 8.0\n.target sm_90\n.address_size 64\n'
    for width in (16, 8):
        for half in range(2 if width == 16 else 1):
            for sa in range(2):
                for sb in range(2):
                    name = f'dot_{width}_{half}_{sa}_{sb}'
                    op = ('dp2a.'+('hi' if half else 'lo') if width == 16 else 'dp4a')
                    op += '.'+('s32' if sa else 'u32')+'.'+('s32' if sb else 'u32')
                    ptx += entry(name, f'''
 mul.wide.u32 %d2,%r20,12;
 add.u64 %d2,%d0,%d2;
 ld.global.u32 %r0,[%d2]; ld.global.u32 %r1,[%d2+4]; ld.global.u32 %r2,[%d2+8];
 {op} %r3,%r0,%r1,%r2;
 mul.wide.u32 %d3,%r20,4; add.u64 %d3,%d1,%d3;
 st.global.u32 [%d3],%r3;''')
    for count in (1, 2, 4):
        for trans in (0, 1):
            for spelling in range(3):
                for cta in (0, 1):
                    name = f'matrix_{count}_{trans}_{spelling}_{cta}'
                    t = '.trans' if trans else ''
                    mods = [f'm8n8.x{count}{t}', f'x{count}.m8n8{t}', f'x{count}{t}.m8n8'][spelling]
                    mods += '.shared'+('::cta' if cta else '')+'.b16'
                    regs = ','.join(f'%r{i+4}' for i in range(count))
                    stores = '\n'.join(f'st.global.u32 [%d3+{i*4}],%r{i+4};' for i in range(count))
                    ptx += entry(name, f'''
 ld.param.u64 %d7,[stored];
 mul.wide.u32 %d2,%r20,16; add.u64 %d2,%d0,%d2;
 ld.global.v4.u32 {{%r0,%r1,%r2,%r3}},[%d2];
 // Permuted, non-contiguous rows: address suppliers cannot be ignored.
 mad.lo.u32 %r21,%r20,5,3; and.b32 %r21,%r21,31;
 mul.wide.u32 %d4,%r21,32; mov.u64 %d5,smem;
 add.u64 %d4,%d4,%d5;
 st.shared.v4.u32 [%d4],{{%r0,%r1,%r2,%r3}};
 add.u64 %d6,%d4,2048;
 st.shared.v4.u32 [%d6],{{0,0,0,0}};
 bar.sync 0;
 ldmatrix.sync.aligned.{mods} {{{regs}}},[%d4];
 mul.wide.u32 %d3,%r20,16; add.u64 %d3,%d1,%d3;
 {stores}
 stmatrix.sync.aligned.{mods} [%d6],{{{regs}}};
 bar.sync 0;
 ld.shared.v4.u32 {{%r0,%r1,%r2,%r3}},[%d6];
 mul.wide.u32 %d2,%r20,16; add.u64 %d2,%d7,%d2;
 st.global.v4.u32 [%d2],{{%r0,%r1,%r2,%r3}};''', '.shared .align 16 .b8 smem[4096];')
    for width in (1, 2, 4):
        regs = ','.join(f'%r{i}' for i in range(width))
        operand = regs if width == 1 else '{'+regs+'}'
        vec = '' if width == 1 else f'.v{width}'
        stores = '\n'.join(f'st.global.u32 [%d3+{i*4}],%r{i};' for i in range(width))
        ptx += entry(f'uniform_{width}', f'''
 ldu.global{vec}.u32 {operand},[%d0];
 mul.wide.u32 %d3,%r20,16; add.u64 %d3,%d1,%d3;
 {stores}''')
    for number, mode in enumerate(('rn', 'rz', 'rm', 'rp')):
        ptx += entry(f'bf16_{number}', f'''
 mul.wide.u32 %d2,%r20,4; add.u64 %d2,%d0,%d2;
 ld.global.f32 %f,[%d2];
 cvt.{mode}.bf16.f32 %h,%f;
 mul.wide.u32 %d3,%r20,2; add.u64 %d3,%d1,%d3;
 st.global.b16 [%d3],%h;''', '.reg .f32 %f; .reg .b16 %h;')
    return ptx


class Instructions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gpu_backend.require_build()
        cls.temp = tempfile.TemporaryDirectory(prefix='drperf-instructions-')
        cls.work = Path(cls.temp.name)
        cls.ptx = cls.work/'kernels.ptx'
        cls.ptx.write_text(fixtures())
        run([ROOT/'bin/drperf', '--gpu', 'translate', cls.ptx, '-o', cls.work/'kernels.so'])
        run([os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-Igpu/include',
             'gpu/tests/packed_matrix_driver.cc', '-Lgpu/build', '-ldrperf_gpu',
             '-Wl,-rpath,'+str(ROOT/'gpu/build'), '-o', cls.work/'driver'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def capture(self, mode):
        output = self.work/mode
        run([ROOT/'bin/drperf', '--gpu', '--output', output, '--', self.work/'driver', self.work/'kernels.so', mode])
        report = json.loads((output/'profile.gpu.drperf.json').read_text())
        self.assertEqual(report['validity'], 'complete')
        return report['regions']

    def test_packed_dot_all_signedness_halves_overflow_and_counts(self):
        regions = self.capture('dot')
        self.assertEqual(len(regions), 12)
        for region in regions:
            ops = 4 if region['name'].startswith('dot_16') else 8
            for row in region['observations']:
                sites = [v for k,v in row['sites'].items() if ':dp2a.' in k or ':dp4a.' in k]
                self.assertEqual(len(sites), 1)
                self.assertEqual(sites[0]['integer_ops'], ops*row['pcvs']['lanes'])
                self.assertEqual(sites[0]['ptx_lane_instructions'], row['pcvs']['lanes'])
                self.assertFalse(any('flops' in k for k in sites[0]))

    def test_matrix_spellings_fragment_oracle_and_byte_counts(self):
        regions = self.capture('matrix')
        self.assertEqual(len(regions), 36)
        for region in regions:
            row = region['observations'][0]
            for op, metric in [('ldmatrix.', 'shared_read_bytes'), ('stmatrix.', 'shared_write_bytes')]:
                sites = [v for k,v in row['sites'].items() if ':'+op in k]
                self.assertEqual(len(sites), 1)
                self.assertEqual(sites[0][metric], 128*row['pcvs']['matrices'])

    def test_uniform_loads_count_requested_bytes_and_footprints(self):
        for region in self.capture('ldu'):
            width = int(region['name'].split('_')[1])
            for row in region['observations']:
                n = row['pcvs']['lanes']; metrics = row['metrics']
                self.assertEqual(metrics['global_read_bytes'], 4*width*n)
                self.assertEqual(metrics['global_load_instructions'], n)
                self.assertEqual(metrics['global_unique_read_bytes'], 4*width)

    def test_supported_manifest(self):
        manifest = json.loads((self.work/'kernels.so.gpu.json').read_text())
        self.assertEqual(manifest['instructionCoverage']['unsupportedSites'], [])

    def test_bf16_directed_rounding_special_values_and_counts(self):
        regions = self.capture('bf16')
        self.assertEqual(len(regions), 4)
        for region in regions:
            for row in region['observations']:
                n = row['pcvs']['lanes']
                self.assertEqual(row['metrics']['conversion_ops'], n)
                self.assertEqual(row['metrics']['global_read_bytes'], 4*n)
                self.assertEqual(row['metrics']['global_write_bytes'], 2*n)

    def test_runtime_assertions_are_not_unsupported_instructions(self):
        source = self.work/'copy.ptx'
        source.write_text('.version 8.0\n.target sm_90\n.address_size 64\n'+entry('copy', '''
 cp.async.ca.shared.global [%d1],[%d0],16,%r20;
 cp.async.wait_all;'''))
        result = run([ROOT/'bin/drperf', '--gpu', 'translate', source, '-o', self.work/'copy.so'])
        coverage = json.loads((self.work/'copy.so.gpu.json').read_text())['instructionCoverage']
        self.assertEqual(coverage['unsupportedSites'], [])
        self.assertEqual(len(coverage['runtimeValidationSites']), 1)
        self.assertEqual(coverage['runtimeValidationSites'][0]['reason'], 'cp.async.src_size_exceeds_cp_size')
        self.assertNotIn('unsupported GPU instruction', result.stderr)

    def test_unsupported_forms_warn_with_origin_before_execution(self):
        text = '.version 8.0\n.target sm_90\n.address_size 64\n'
        text += entry('bad', '''
 .loc 1 42 0
 @%p dp2a.lo.u16.u32 %r0,%r1,%r2,%r3;
 dp4a.sat.u32.u32 %r0,%r1,%r2,%r3;
 ldmatrix.sync.aligned.x3.trans.m8n8.shared.b16 {%r0,%r1,%r2},[%d0];
 ldmatrix.sync.aligned.x4.trans.m8n8.shared.b8 {%r0,%r1,%r2,%r3},[%d0];''')
        text += '\n.file 1 "fixture.cu"\n'
        source = self.work/'bad.ptx'; source.write_text(text)
        result = run([ROOT/'bin/drperf', '--gpu', 'translate', source, '-o', self.work/'bad.so'])
        self.assertIn('4 unsupported GPU instruction site(s)', result.stderr)
        manifest = json.loads((self.work/'bad.so.gpu.json').read_text())
        sites = manifest['instructionCoverage']['unsupportedSites']
        self.assertEqual(len(sites), 4)
        for site in sites:
            self.assertEqual(site['function'], 'bad')
            self.assertEqual(site['sourceLocation'], 'fixture.cu:42')
            self.assertIn(site['opcode'], text.splitlines()[site['ptxLine']-1])


if __name__ == '__main__': unittest.main()
