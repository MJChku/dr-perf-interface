#!/usr/bin/env python3
"""CPU-only numerical/resource checks of GX runtime parity; no CUDA needed."""
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import gpu_backend

def run(argv,expected=0):
    r=subprocess.run([str(x) for x in argv],cwd=ROOT,text=True,capture_output=True,timeout=120)
    if r.returncode!=expected:
        raise AssertionError(f'{argv}: {r.returncode}\n{r.stdout}\n{r.stderr}')
    return r

class RuntimeParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gpu_backend.require_build()
        cls.temp=tempfile.TemporaryDirectory(prefix='drperf-runtime-parity-')
        cls.work=Path(cls.temp.name)
        cxx=os.environ.get('CXX','g++')
        for name in ['wgmma','tma','tma_ext','cluster','tcgen05_pair','clc']:
            run([cxx,'-std=c++17','-O2','-Igpu/include',f'gpu/tests/{name}_driver.cc',
                 '-Lgpu/build','-ldrperf_gpu','-Wl,-rpath,'+str(ROOT/'gpu/build'),'-o',cls.work/name])
            if name!='wgmma':
                run([ROOT/'bin/drperf','--gpu','translate',f'gpu/tests/{name}.ptx','-o',cls.work/(name+'.so')])
        cls.wg=cls.work/'wg'
        run([cls.work/'wgmma','emit',cls.wg])
        cls.wg_cases=sorted(cls.wg.glob('*.ptx'))
        for path in cls.wg_cases:
            run([ROOT/'bin/drperf','--gpu','translate',path,'-o',path.with_suffix('.so')])

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def capture(self,name,argv,expected=0):
        dest=self.work/name
        result=run([ROOT/'bin/drperf','--gpu','--output',dest,'--',*argv],expected)
        if expected:
            self.assertFalse((dest/'profile.gpu.drperf.json').exists())
            return result
        model=json.loads((dest/'profile.gpu.drperf.json').read_text())
        self.assertEqual(model['validity'],'complete')
        return {r['name']:r for r in model['regions']}

    @staticmethod
    def sites(row,fragment,metric):
        return sum(m.get(metric,0) for site,m in row['sites'].items() if fragment in site)

    def test_every_prelude_runtime_symbol_resolves(self):
        prelude=(ROOT/'gpu/vendor/gxptx/Gxptx/prelude.h').read_text()
        names=set(re.findall(r'extern\s+"C"[^;{}]*?\b(\w+)\s*\([^;{}]*\);',prelude))
        runtime=ctypes.CDLL(str(ROOT/'gpu/build/libdrperf_gpu.so'))
        self.assertFalse([n for n in sorted(names) if not hasattr(runtime,n)])

    def test_wgmma_33_independent_numerical_oracles_and_tile_counts(self):
        self.assertEqual(len(self.wg_cases),33)
        rs=self.capture('wg-capture',[self.work/'wgmma','run',self.wg])
        self.assertEqual(len(rs),33)
        for r in rs.values():
            for row in r['observations']:
                p=row['pcvs'];m=row['metrics'];groups=p['groups']*p['operations']
                metric='tensor_integer_ops' if p['integer'] else 'tensor_float_flops'
                self.assertEqual(m[metric],2*64*p['N']*p['K']*groups,r['name'])
                self.assertEqual(m['tensor_mma_instructions'],groups,r['name'])
                self.assertEqual(m['shared_read_bytes'],((0 if p['register_a'] else 2048)+32*p['N'])*groups,r['name'])
                self.assertEqual(self.sites(row,':wgmma.mma_async.',metric),m[metric])

    def test_wgmma_partial_warpgroup_is_rejected(self):
        result=self.capture('wg-partial',[self.work/'wgmma','partial',self.wg],expected=1)
        self.assertIn('requires_full_warpgroup',result.stderr)

    def test_tma_swizzles_packing_oob_multicast_and_store(self):
        rs=self.capture('tma-capture',[self.work/'tma',self.work/'tma.so'])
        self.assertEqual(sum(r['invocations'] for r in rs.values()),14)
        for name in ['tma.load','tma.multicast']:
            for row in rs[name]['observations']:
                p=row['pcvs']
                self.assertEqual(self.sites(row,':cp.async.bulk.tensor.','global_read_bytes'),p['source_bytes'])
                self.assertEqual(self.sites(row,':cp.async.bulk.tensor.','shared_write_bytes'),p['destination_bytes'])
                self.assertEqual(row['metrics']['tma_transaction_bytes'],p['destination_bytes'])
                self.assertEqual(row['metrics']['tma_oob_fill_bytes'],p['destination_bytes']-p['source_bytes']*p.get('destinations',1))
        for row in rs['tma.store']['observations']:
            self.assertEqual(self.sites(row,':cp.async.bulk.tensor.','global_write_bytes'),row['pcvs']['written_bytes'])

    def test_tma_gather_scatter_edits_reductions_mask_and_async_tail(self):
        rs=self.capture('tma-ext-capture',[self.work/'tma_ext',self.work/'tma_ext.so'])
        self.assertEqual(sum(r['invocations'] for r in rs.values()),9)
        for name,bytes_,ops in [('k_reduce_f32',256,64),('k_reduce_s32',1024,256)]:
            row=rs[name]['observations'][0]
            self.assertEqual(self.sites(row,':cp.reduce.async.bulk.tensor.','global_write_bytes'),bytes_)
            self.assertEqual(self.sites(row,':cp.reduce.async.bulk.tensor.','global_read_bytes'),bytes_)
            self.assertEqual(row['metrics']['reduction_ops'],ops)
        row=rs['tma.gather_scatter']['observations'][0]
        self.assertEqual(self.sites(row,'tile::gather4','global_read_bytes'),384)
        self.assertEqual(self.sites(row,'tile::gather4','shared_write_bytes'),512)
        self.assertEqual(self.sites(row,'tile::scatter4','global_write_bytes'),512)
        row=rs['tma.bulk']['observations'][0]
        self.assertEqual(row['metrics']['bulk_reduce_instructions'],2)
        self.assertEqual(row['metrics']['reduction_ops'],128)
        self.assertEqual(self.sites(row,'.cp_mask','global_write_bytes'),128)
        for row in rs['async.tail']['observations']:
            self.assertEqual(self.sites(row,':cp.async.ca.','global_read_bytes'),row['pcvs']['source_bytes'])
            self.assertEqual(self.sites(row,':cp.async.ca.','shared_write_bytes'),48)

    def test_cluster_geometry_remote_shared_mbarriers_grid_and_votes(self):
        rs=self.capture('cluster-capture',[self.work/'cluster',self.work/'cluster.so'])
        self.assertEqual(rs['cluster']['invocations'],4)
        for row in rs['cluster']['observations']:
            self.assertEqual(row['metrics'].get('generic_read_bytes',0),0)
            self.assertEqual(row['metrics']['shared_read_bytes'],8*row['pcvs']['blocks']*row['pcvs']['threads'])
        self.assertEqual(rs['grid']['invocations'],1)
        self.assertEqual(rs['votes']['invocations'],1)

    def test_paired_cta_bf16_numerics_and_counts(self):
        rs=self.capture('pair-capture',[self.work/'tcgen05_pair',self.work/'tcgen05_pair.so'])
        region=rs['tcgen-pair']
        for row in region['observations']:
            repeats=row['pcvs']['repeats'];m=row['metrics']
            self.assertEqual(m['tensor_float_flops'],2*256*128*16*repeats)
            self.assertEqual(m['tensor_mma_instructions'],repeats)
            self.assertEqual(m['tmem_write_bytes'],4*256*128*repeats)
        self.assertTrue(region['fits']['tensor_float_flops']['accepted'])
        self.assertEqual(region['fits']['tensor_float_flops']['coefficients']['repeats'],2*256*128*16)

    def test_pending_cluster_cancellation_and_persistent_work_ownership(self):
        rs=self.capture('clc-capture',[self.work/'clc',self.work/'clc.so'])
        self.assertEqual(rs['clc']['invocations'],3)
        for row in rs['clc']['observations']:
            p=row['pcvs'];m=row['metrics']
            self.assertEqual(m['executed_blocks'],p['cluster'])
            self.assertEqual(m['cancelled_blocks'],p['blocks']-p['cluster'])
            self.assertEqual(m['cancelled_clusters'],p['blocks']//p['cluster']-1)

if __name__=='__main__':unittest.main()
