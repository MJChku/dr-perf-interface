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
