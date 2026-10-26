#!/usr/bin/env python3
"""CPU-only dependency, predication, synchronization and report model oracles."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

GPU = Path(__file__).resolve().parents[1]
ROOT = GPU.parent
sys.path.insert(0, str(GPU / 'tools'))
from latency_report import rank_regions
from report_io import read_report


def run(*args, env=None):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise AssertionError(f'{args}: {result.returncode}\n{result.stdout}\n{result.stderr}')
    return result


def entry(name, body):
    return f"""
.visible .entry {name}(.param .u64 input, .param .u64 output) {{
 .reg .b64 %d<3>;
 .reg .b32 %r<32>;
 .reg .pred %p;
 ld.param.u64 %d0,[input];
 ld.param.u64 %d1,[output];
 mov.u32 %r20,0;
 {body}
 st.global.u32 [%d1],%r20;
 ret;
}}
"""


class LatencyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = GPU / 'build/ext-work3'
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=scratch)
        cls.work = Path(cls.temp.name)
        bodies = {}
        for n in (1, 2, 5, 11):
            bodies[f'chain{n}'] = '\n'.join('ld.global.u32 %r0,[%d0]; add.u32 %r20,%r20,%r0;' for _ in range(n))
            bodies[f'upfront{n}'] = '\n'.join(f'ld.global.u32 %r{i},[%d0];' for i in range(n)) + '\n' + '\n'.join(f'add.u32 %r20,%r20,%r{i};' for i in range(n))
        bodies['prefetch11'] = """
 mov.u32 %r21,11;
 ld.global.u32 %r0,[%d0];
L_loop:
 add.u32 %r20,%r20,%r0;
 sub.u32 %r21,%r21,1;
 setp.eq.u32 %p,%r21,0;
 @%p bra L_done;
 ld.global.u32 %r0,[%d0];
 bra L_loop;
L_done:
 """
        bodies['unused'] = 'ld.global.u32 %r0,[%d0];'
        bodies['baseline'] = 'mov.u32 %r0,1;'
        bodies['disabled'] = """
 ld.global.u32 %r0,[%d0];
 setp.eq.u32 %p,0,1;
 @%p add.u32 %r20,%r20,%r0;
 """
        bodies['disabled_write'] = """
 mov.u32 %r0,1;
 setp.eq.u32 %p,0,1;
 @%p ld.global.u32 %r0,[%d0];
 add.u32 %r20,%r20,%r0;
 """
        bodies['predicate_load'] = """
 ld.global.u32 %r0,[%d0];
 setp.eq.u32 %p,%r0,1;
 @%p mov.u32 %r20,1;
 """
        bodies['mixed_predication'] = """
 mov.u32 %r22,%tid.x;
 setp.eq.u32 %p,%r22,0;
 @%p add.u32 %r20,%r20,1;
 """
        bodies['vector'] = """
 ld.global.v4.u32 {%r0,%r1,%r2,%r3},[%d0];
 add.u32 %r20,%r2,%r3;
 """
        # The long pre-barrier lane and long post-barrier lane differ. A
        # maximum of independent lane times cannot satisfy this oracle.
        before = 'ld.global.u32 %r0,[%d0]; add.u32 %r20,%r20,%r0;'
        after = '\n'.join(before for _ in range(3))
        divergent = f"""
 mov.u32 %r22,%tid.x;
 setp.eq.u32 %p,%r22,0;
 @!%p bra L_join;
 {before}
L_join:
 {{join}}
 @%p bra L_finish;
 {after}
L_finish:
 """
        bodies['barrier'] = divergent.replace('{join}', 'bar.sync 0;')
        bodies['named'] = divergent.replace('{join}', 'bar.sync 3,2;')
        bodies['cluster'] = divergent.replace('{join}', 'barrier.cluster.arrive; barrier.cluster.wait;')
        bodies['grid'] = divergent.replace('{join}', 'call.uni gx_ptx_grid_sync,();')
        bodies['no_barrier'] = divergent.replace('{join}', 'mov.u32 %r23,0;')
        # Explicit empty-loop wait annotation parks the CPU fiber; it joins
        # the completed phase's producer clock without counting scheduler
        # polling iterations as an invented hardware delay.
        bodies['mbarrier'] = f"""
 mbarrier.init.shared.b64 [latency_bar],1;
 bar.sync 0;
 mov.u32 %r22,%tid.x;
 setp.eq.u32 %p,%r22,0;
 @!%p bra L_wait;
 {before}
 mbarrier.arrive.shared.b64 _,[latency_bar];
 bra L_finish;
L_wait:
 cvta.shared.u64 %d2,latency_bar;
 {{
 .param .b64 address;
 .param .b64 token;
 .param .b32 parity;
 st.param.b64 [address],%d2;
 st.param.b64 [token],0;
 st.param.b32 [parity],1;
 call.uni gx_ptx_mbarrier_wait,(address,token,parity);
 }}
 {after}
L_finish:
 """
        ptx = cls.work / 'latency.ptx'
        ptx.write_text('.version 8.0\n.target sm_90\n.address_size 64\n'
                       '.shared .align 8 .b64 latency_bar;\n'
                       '.extern .func gx_ptx_grid_sync();\n'
                       '.extern .func gx_ptx_mbarrier_wait(.param .b64 a, .param .b64 t, .param .b32 p);\n' +
                       ''.join(entry(k, v) for k, v in bodies.items()))
        cls.module = cls.work / 'latency.so'
        run(ROOT / 'bin/drperf', '--gpu', 'translate', ptx, '-o', cls.module)
        driver = cls.work / 'driver.cc'
        names = ','.join(json.dumps(k) for k in bodies)
        driver.write_text("""
#include "drperf_gpu.h"
#include <cstdint>
#include <cstring>
int main(int argc,char** argv) {
 if(argc!=2)return 2;
 drperf::gpu::Module m(argv[1]);
 uint32_t input[16];for(auto& x:input)x=1;
 uint32_t output=0;auto* in=input;auto* out=&output;
 void* args[]={&in,&out};
 for(const char* name:{""" + names + """}) {
  unsigned lanes=(!strcmp(name,"barrier")||!strcmp(name,"named")||!strcmp(name,"cluster")||
                  !strcmp(name,"grid")||!strcmp(name,"mbarrier")||!strcmp(name,"no_barrier")||
                  !strcmp(name,"mixed_predication"))?2:1;
  drperf::gpu::Region r(name);m.launch(name,1,lanes,args);
  if(!strncmp(name,"chain",5)&&output!=unsigned(atoi(name+5)))return 3;
  if(!strncmp(name,"upfront",7)&&output!=unsigned(atoi(name+7)))return 4;
  if(!strcmp(name,"prefetch11")&&output!=11)return 5;
  if(!strcmp(name,"disabled")&&output!=0)return 6;
  if(!strcmp(name,"disabled_write")&&output!=1)return 7;
  if(!strcmp(name,"vector")&&output!=2)return 8;
 }
 {
  drperf::gpu::Region r("repeated");
  m.launch("chain2",1,1,args);m.launch("upfront2",1,1,args);
 }
}
""")
        cls.driver = cls.work / 'driver'
        run(os.environ.get('CXX', 'g++'), '-std=c++17', '-O2', '-Igpu/include', driver,
            '-Lgpu/build', '-ldrperf_gpu', '-Wl,-rpath,' + str(GPU / 'build'), '-o', cls.driver)
        cls.report = cls.capture('default')
        cls.rows = {r['name']: r['observations'][0] for r in cls.report['regions']}

    @classmethod
    def capture(cls, name, settings=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DRPERF_GPU_LATENCY_')
               and k not in ('DRPERF_GPU_SM_COUNT', 'DRPERF_GPU_LANES_PER_SM')}
        dest = cls.work / name
        # The launcher deliberately strips DRPERF_* from its own environment.
        command = ['env', *(f'{k}={v}' for k, v in (settings or {}).items()), cls.driver, cls.module]
        run(ROOT / 'bin/drperf', '--gpu', '--output', dest, '--', *command, env=env)
        return read_report(dest)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def metric(self, name, metric='latency_critical_path'):
        return self.rows[name]['metrics'][metric]

    def test_dependent_chains_and_all_loads_upfront(self):
        key = 'dependent_global_loads_on_critical_path'
        for n in (1, 2, 5, 11):
            self.assertEqual(self.metric(f'chain{n}', key), n)
            self.assertEqual(self.metric(f'upfront{n}', key), 1)
        self.assertGreater(self.metric('chain11'), 7 * self.metric('upfront11'))
        self.assertEqual(self.metric('prefetch11', key), 11)
        self.assertGreater(self.metric('prefetch11'), 7 * self.metric('upfront11'))

    def test_unused_loads_and_predicated_off_data_do_not_stall(self):
        key = 'dependent_global_loads_on_critical_path'
        self.assertEqual(self.metric('unused'), self.metric('baseline'))
        for name in ('unused', 'disabled', 'disabled_write'):
            self.assertEqual(self.metric(name, key), 0)
            self.assertLess(self.metric(name), 100)
        self.assertEqual(self.metric('predicate_load', key), 1)
        self.assertEqual(self.metric('vector', key), 1)
        disabled = [m for site, m in self.rows['disabled']['sites'].items() if ':add.u32' in site]
        self.assertEqual(len(disabled), 1)
        self.assertGreater(disabled[0]['latency_critical_path'], 0)
        self.assertNotIn('ptx_lane_instructions', disabled[0])
        mixed = [m for site, m in self.rows['mixed_predication']['sites'].items() if ':add.u32' in site]
        self.assertEqual(len(mixed), 1)  # Enabled and disabled arms share a PTX site.
        self.assertEqual(mixed[0]['ptx_lane_instructions'], 1)

    def test_block_and_named_barriers_join_before_following_work(self):
        for name in ('barrier', 'named', 'cluster', 'grid', 'mbarrier'):
            self.assertGreater(self.metric(name), self.metric('no_barrier') + 350)
            self.assertEqual(self.metric(name, 'dependent_global_loads_on_critical_path'), 4)

    def test_configured_constants_are_retained_and_change_only_model(self):
        other = self.capture('configured', {'DRPERF_GPU_LATENCY_GLOBAL': '80', 'DRPERF_GPU_SM_COUNT': '7'})
        rows = {r['name']: r['observations'][0] for r in other['regions']}
        a, b = self.rows['chain11']['metrics'], rows['chain11']['metrics']
        self.assertEqual(a['latency_critical_path'] - b['latency_critical_path'], 11 * 320)
        self.assertEqual(b['latency_model_sm_count'], 7)
        for k in ('global_read_bytes', 'global_write_bytes', 'ptx_lane_instructions'):
            self.assertEqual(a[k], b[k])

    def test_site_paths_and_ordinary_metric_fits(self):
        for r in self.report['regions']:
            self.assertNotIn('latency_critical_path', r['fits'])
            self.assertNotIn('dependent_global_loads_on_critical_path', r['fits'])
            self.assertTrue(r['fits']['global_load_instructions']['accepted'] if 'global_load_instructions' in r['fits'] else True)
            row = r['observations'][0]
            if r['name'] == 'repeated':
                continue  # Region sums per-launch maxima; sites sum separately.
            self.assertEqual(max(m['latency_critical_path'] for m in row['sites'].values()),
                             row['metrics']['latency_critical_path'])

    def test_launch_paths_sum_once_with_paired_exposure_counts(self):
        for metric in ('latency_critical_path', 'dependent_global_loads_on_critical_path'):
            self.assertEqual(self.metric('repeated', metric),
                             self.metric('chain2', metric) + self.metric('upfront2', metric))
        record = next(r for r in rank_regions(self.report)['records'] if r['region'] == 'repeated')
        self.assertEqual(record['launches'], 2)
        self.assertEqual(record['criticalPathTimesLaunches'], self.metric('repeated'))

    def test_loader_rejects_missing_or_wrong_latency_capabilities(self):
        original = Path(str(self.module) + '.cpp').read_text()
        for old, new in [('drperf_gpu_latency_abi = 1', 'drperf_gpu_latency_abi = 0'),
                         ('drperf_gpu_latency_abi = 1', 'old_latency_abi = 1'),
                         ('chain1__gxptx_latency_abi = 1', 'chain1__gxptx_latency_abi = 0')]:
            with self.subTest(capability=new):
                self.assertIn(old, original)
                cpp, so = self.work / 'stale.cpp', self.work / 'stale.so'
                cpp.write_text(original.replace(old, new))
                run(os.environ.get('CXX', 'g++'), '-std=c++17', '-shared', '-fPIC', cpp,
                    '-Lgpu/build', '-ldrperf_gpu', '-Wl,-rpath,' + str(GPU / 'build'), '-o', so)
                result = subprocess.run([str(self.driver), str(so)], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('latency instrumentation ABI mismatch', result.stderr)


class LatencyReport(unittest.TestCase):
    def test_ranking_sums_launch_paths_once_and_flags_work_pressure(self):
        def region(name, work, paths):
            launches = [{'metrics': {'latency_model_version': 1, 'latency_critical_path': p,
                        'ptx_lane_instructions': work, 'blocks': 8, 'latency_model_sm_count': 148,
                        'latency_model_lanes_per_sm': 128}} for p in paths]
            return {'name': name, 'observations': [{'launches': launches}]}
        model = {'regions': [region('overlap', 100000000, [500]), region('serial', 30000, [5000, 5000]),
                             {'name': 'old', 'observations': []}]}
        result = rank_regions(model)
        first, second = result['records']
        self.assertEqual(first['region'], 'serial')
        self.assertEqual(first['criticalPathTimesLaunches'], 10000)
        self.assertTrue(first['latencyBound'])
        self.assertFalse(second['latencyBound'])
        self.assertEqual(result['unavailableRegions'], ['old'])
        for setting in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                rank_regions(model, threshold=setting)

    def test_cli_directories_gzip_and_invalid_settings(self):
        with tempfile.TemporaryDirectory(dir=GPU / 'build/ext-work3') as directory:
            work = Path(directory)
            model = {'schema': 'drperf.gpu.v1', 'backend': 'gpu', 'validity': 'complete',
                     'regions': [{'name': 'old', 'observations': []}]}
            (work / 'profile.gpu.drperf.json').write_text(json.dumps(model))
            compressed = work / 'report.json.gz'
            with gzip.open(compressed, 'wt') as f:
                json.dump(model, f)
            tool = GPU / 'tools/latency_report.py'
            for source in (work, compressed):
                output = json.loads(run(sys.executable, tool, source, '--json').stdout)
                self.assertEqual(output['unavailableRegions'], ['old'])
                self.assertIn('not measured', run(sys.executable, tool, source).stdout)
            for option in (('--sms', '0'), ('--lanes-per-sm', '-1'), ('--threshold', 'nan')):
                result = subprocess.run([sys.executable, str(tool), str(work), *option],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
