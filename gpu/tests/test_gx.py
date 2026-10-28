"""GX observer lifecycle and optional real GX/CUDA application regression."""
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'lib'))
import gpu_backend as gpu


class GXCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(['sh', str(ROOT/'gpu/gx/build.sh')], check=True)

    def collector(self, code, *, failed=False):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, DRPERF_GPU_CAPTURE=tmp+'/gpu')
            env.pop('GX_EXPLAIN_GPU_LIBRARY', None)
            proc = subprocess.run([sys.executable, '-c', '''
import ctypes as C
import threading
lib=C.CDLL('''+repr(str(ROOT/'gpu/build/libdrperf_gpu_gx.so'))+''')
V=C.c_void_p; U=C.c_uint64
Submit=C.CFUNCTYPE(V,C.c_char_p,C.c_int)
Count=C.CFUNCTYPE(None,V,C.c_char_p,C.c_char_p,U)
Finish=C.CFUNCTYPE(None,V,C.c_int)
class API(C.Structure):
 _fields_=[('abi',C.c_uint32),('submit',Submit),('count',Count),('finish',Finish)]
lib.drperf_gpu_gx_observer_v1.restype=C.POINTER(API)
a=lib.drperf_gpu_gx_observer_v1().contents
assert a.abi==1
lib.drperf_gpu_region_begin.argtypes=[C.c_char_p,C.c_size_t,C.POINTER(C.c_char_p),C.POINTER(C.c_int64)]
lib.drperf_gpu_region_begin.restype=U
lib.drperf_gpu_region_end.argtypes=[U]
def begin(name,n):
 return lib.drperf_gpu_region_begin(name.encode(),1,(C.c_char_p*1)(b'n'),(C.c_int64*1)(n))
def count(handle,n):
 a.count(handle,b'k:bb0:cpp1:ld.global.f32',b'ptx_lane_instructions',n)
 a.count(handle,b'k:bb0:cpp1:ld.global.f32',b'global_read_bytes',4*n)
''' + code], env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            if failed:
                with self.assertRaises(ValueError):
                    gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl')))
                return
            return gpu.build_model(gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl'))))

    def test_delayed_completion_and_same_state_mean(self):
        m=self.collector('''
for n in (3,7,11):
 for factor in (1,3):
  parent=begin('outer',n);child=begin('inner',n)
  handle=a.submit(b'k',1)
  lib.drperf_gpu_region_end(child);lib.drperf_gpu_region_end(parent)
  t=threading.Thread(target=lambda: (count(handle,factor*n),a.finish(handle,1)))
  t.start();t.join()
''')
        r={r['name']:r for r in m['regions']}
        self.assertEqual(r['inner']['invocations'],6)
        self.assertEqual(r['inner']['stateCount'],3)
        self.assertEqual(r['inner']['fits']['global_read_bytes']['coefficients'],{'n':8})
        self.assertEqual(r['outer']['fits'],{})
        self.assertEqual(m['validity'],'complete')

    def test_skipped_and_uninstrumented_launches_incomplete(self):
        for translated in (0,1):
            m=self.collector(f"r=begin('R',4);h=a.submit(b'unknown',{translated});a.finish(h,1);lib.drperf_gpu_region_end(r)")
            self.assertEqual(m['validity'],'incomplete')
            self.assertEqual(m['regions'][0]['unmodeledWorkEvents'],1)

    def test_cancelled_or_pending_launch_rejects_capture(self):
        self.collector("r=begin('R',4);h=a.submit(b'k',1);count(h,4);a.finish(h,0);lib.drperf_gpu_region_end(r)",failed=True)
        self.collector("r=begin('R',4);h=a.submit(b'k',1);lib.drperf_gpu_region_end(r)",failed=True)

    def test_unannotated_kernel_is_visible(self):
        m=self.collector("h=a.submit(b'unmarked',1);count(h,5);a.finish(h,1)")
        self.assertEqual(m['regions'][0]['name'],'kernel:unmarked')
        self.assertEqual(m['regions'][0]['pcvs'],[])

    def test_descendant_incompleteness_is_visible_on_parent_interface(self):
        m=self.collector('''
p=begin('outer',4)
own=a.submit(b'known',1);count(own,4);a.finish(own,1)
c=begin('middle',4);leaf=begin('leaf',4)
h=a.submit(b'unsupported',0)
lib.drperf_gpu_region_end(leaf);lib.drperf_gpu_region_end(c);lib.drperf_gpu_region_end(p)
a.finish(h,1)
other=begin('unrelated',4);h=a.submit(b'other_unsupported',0)
a.finish(h,1);lib.drperf_gpu_region_end(other)
''')
        r={r['name']:r for r in m['regions']}
        self.assertEqual(m['validity'],'incomplete')
        for name in ('outer','middle'):
            self.assertEqual(r[name]['metricScope'],'exclusive')
            self.assertEqual(r[name]['unmodeledWorkEvents'],0)
            self.assertEqual(r[name]['validity'],'observed-work-counts')
            self.assertEqual(r[name]['inclusiveUnmodeledWorkEvents'],1)
            self.assertEqual(r[name]['inclusiveValidity'],'incomplete')
        self.assertEqual(r['leaf']['inclusiveUnmodeledWorkEvents'],1)
        self.assertEqual(r['outer']['fits']['global_read_bytes']['constant'],16)
        report=gpu.text_report(dict(m,regions=[r['outer']]))
        self.assertIn('1 descendant unmodeled work events',report)
        self.assertIn('composed resource capture is incomplete',report)

    def test_entry_pcvs_and_out_of_order_nested_completion(self):
        m=self.collector('''
for n in (2,5,9):
 names=(C.c_char_p*1)(b'n');values=(C.c_int64*1)(n)
 p=lib.drperf_gpu_region_begin(b'outer',1,names,values)
 c=begin('inner',n)
 handles=[a.submit(b'k',1) for _ in range(3)]
 names[0]=b'changed';values[0]=100
 lib.drperf_gpu_region_end(c);lib.drperf_gpu_region_end(p)
 for i in (2,0,1):
  count(handles[i],(i+1)*n);a.finish(handles[i],1)
''')
        r={r['name']:r for r in m['regions']}
        self.assertEqual(m['validity'],'complete')
        self.assertEqual(r['inner']['fits']['global_read_bytes']['coefficients'],{'n':24})
        self.assertEqual([row['pcvs'] for row in r['outer']['observations']],
                         [{'n':n} for n in (2,5,9)])
        self.assertEqual([row['gxLaunchCount'] for row in r['inner']['observations']],[3,3,3])
        self.assertEqual(r['outer']['inclusiveUnmodeledWorkEvents'],0)
        self.assertEqual(r['outer']['inclusiveValidity'],'observed-work-counts')
        legacy=dict(r['outer']['observations'][0])
        for field in ('pid','token','parent'):
            legacy.pop(field)
        unknown=gpu.build_model([legacy])['regions'][0]
        self.assertIsNone(unknown['inclusiveUnmodeledWorkEvents'])
        self.assertEqual(unknown['inclusiveValidity'],'unavailable')

    def test_environment_keeps_gx_and_avoids_second_emulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib=Path(tmp)/'gx.so';lib.touch()
            env=gpu.gx_environment(tmp,lib,{'GX_DEVICE_MODEL':'H100','GX_CHROME_TRACE':'0'})
            self.assertEqual(env['GX_DEVICE_MODEL'],'H100')
            self.assertEqual(env['LD_PRELOAD'],str(lib))
            self.assertEqual(env['GX_KERNEL_CHECK'],'1')
            with self.assertRaises(ValueError):
                gpu.gx_environment(tmp,lib,{'LD_PRELOAD':'/x/libdrperf_gpu.so'})


@unittest.skipUnless(os.environ.get('GX_TEST_LIBRARY') and os.environ.get('GX_HOME'),
                     'set GX_TEST_LIBRARY and GX_HOME for the real GX application test')
class GXApplication(unittest.TestCase):
    def test_scalar_division_ftz_and_float_shuffle_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/gpu', GX_PTX_CACHE_ROOT=tmp+'/cache',
                       GX_NUM_LOCAL_GPUS='1', GX_DEVICE_MEM_GB='1', GX_PTX_JOBS='2')
            proc=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_scalar_bits_application.py')],
                                env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            rows=gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl')))
            self.assertEqual(len(rows),5)
            self.assertEqual({r['region'] for r in rows},
                             {'div_approx_compatibility','shfl_float_compatibility'})
            self.assertEqual(gpu.build_model(rows)['validity'],'complete')
            for row in rows:
                n=row['pcvs']['n'];division=row['region']=='div_approx_compatibility'
                self.assertEqual(row['metrics']['global_read_bytes'],(8 if division else 4)*n)
                self.assertEqual(row['metrics']['global_write_bytes'],(8 if division else 48)*n)
                self.assertEqual(row['metrics']['fp32_flops'],(2 if division else 1)*n)
                if not division:
                    self.assertEqual(row['metrics']['collective_lane_arrivals'],9*n)

    def test_packed_integer_dots(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/gpu', GX_PTX_CACHE_ROOT=tmp+'/cache',
                       GX_NUM_LOCAL_GPUS='1', GX_DEVICE_MEM_GB='1', GX_PTX_JOBS='2')
            proc=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_packed_dot_application.py')],
                                env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            model=gpu.build_model(gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl'))))
            self.assertEqual(model['validity'],'complete')
            self.assertEqual(len(model['regions']),24)
            for region in model['regions']:
                self.assertEqual(region['invocations'],3)
                for row in region['observations']:
                    sites=[v for k,v in row['sites'].items() if ':dp2a.' in k or ':dp4a.' in k]
                    self.assertEqual(len(sites),1)
                    active=row['pcvs']['active']; n=row['pcvs']['lanes']
                    self.assertEqual(sites[0]['ptx_lane_instructions'],active)
                    self.assertEqual(sites[0]['integer_ops'],active*(4 if region['name'].startswith('dp2a') else 8))
                    self.assertFalse(any('flops' in metric for metric in sites[0]))
                    self.assertEqual(row['metrics']['global_read_bytes'],12*n)
                    self.assertEqual(row['metrics']['global_write_bytes'],4*n)

    def test_uninitialized_module_globals(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/gpu', GX_PTX_CACHE_ROOT=tmp+'/cache',
                       GX_NUM_LOCAL_GPUS='1', GX_DEVICE_MEM_GB='1', GX_PTX_JOBS='2')
            proc=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_globals_application.py')],
                                env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            rows=gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl')))
            self.assertEqual(len(rows),12)
            for row in rows:
                for metric in ('const_read_bytes','global_read_bytes','global_write_bytes'):
                    self.assertEqual(row['metrics'][metric],4*row['pcvs']['n'])
            model=gpu.build_model(rows)
            self.assertEqual(model['validity'],'complete')
            for metric in ('const_read_bytes','global_read_bytes','global_write_bytes'):
                fit=model['regions'][0]['fits'][metric]
                self.assertTrue(fit['accepted'])
                self.assertEqual(fit['coefficients'],{'n':4})

    def test_scalar_b128_shared_memory_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/gpu', GX_PTX_CACHE_ROOT=tmp+'/cache',
                       GX_NUM_LOCAL_GPUS='1',GX_DEVICE_MEM_GB='1',GX_PTX_JOBS='2')
            proc=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_b128_application.py')],
                                env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            rows=gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl')))
            model=gpu.build_model(rows)
            self.assertEqual(model['validity'],'complete')
            self.assertEqual(len(rows),3)
            for row in rows:
                n=row['pcvs']['n']
                for metric in ('shared_read_bytes','shared_write_bytes'):
                    self.assertEqual(row['metrics'][metric],32*n)
                # Each scalar access alone requests 16 bytes, independently
                # checked from the surrounding 16-byte vector accesses.
                for op,metric in [('ld.shared.b128','shared_read_bytes'),
                                  ('st.shared.b128','shared_write_bytes')]:
                    self.assertEqual(sum(counts.get(metric,0) for site,counts in row['sites'].items()
                                         if site.endswith(':'+op)),16*n)
            region=model['regions'][0]
            for metric in ('shared_read_bytes','shared_write_bytes'):
                self.assertTrue(region['fits'][metric]['accepted'])
                self.assertEqual(region['fits'][metric]['coefficients'],{'n':32})

    def test_cuda_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/gpu', GX_PTX_CACHE_ROOT=tmp+'/cache',
                       GX_NUM_LOCAL_GPUS='1',GX_DEVICE_MEM_GB='1',GX_PTX_JOBS='2')
            for skip in (False,True):
                case=Path(tmp)/str(skip);case.mkdir()
                runenv=dict(env,DRPERF_GPU_CAPTURE=str(case/'gpu'))
                if skip:runenv.update(GX_TEST_SKIP='1',GX_COMM_ONLY='1',GX_KERNEL_CHECK='0')
                proc=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_application.py')],
                                    env=runenv,capture_output=True,text=True,timeout=180)
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                model=gpu.report_capture(list(case.glob('gpu.*.jsonl')),case,
                                         provenance={'execution':'gx'})
                if skip:
                    self.assertEqual(model['validity'],'incomplete');continue
                self.assertEqual(model['validity'],'complete')
                r={r['name']:r for r in model['regions']}
                self.assertEqual(set(r),{'request','transform','replay','threaded','same_symbol'})
                for metric,coef in [('global_read_bytes',8),('global_write_bytes',8),('fp32_flops',4)]:
                    self.assertEqual(r['transform']['fits'][metric]['coefficients'],{'n':coef})
                    self.assertTrue(r['transform']['fits'][metric]['accepted'])
                self.assertEqual(r['transform']['invocations'],8)
                self.assertEqual(r['transform']['stateCount'],4)
                self.assertEqual(r['replay']['invocations'],4)
                self.assertEqual(r['replay']['fits']['global_read_bytes']['coefficients'],{'repeats':52})
                self.assertEqual(r['threaded']['fits']['global_read_bytes']['coefficients'],{'n':4})
                self.assertEqual(r['same_symbol']['fits']['global_read_bytes']['coefficients'],{'n':12})
                sites=r['same_symbol']['basicBlockMembers']
                self.assertEqual(len({site.split('/')[0] for site in sites}),2)

if __name__=='__main__':unittest.main()
