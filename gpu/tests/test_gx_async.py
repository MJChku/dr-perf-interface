"""Actual GX execution: dynamic copy-byte accounting and copy-control validity."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import gpu_backend as gpu


@unittest.skipUnless(os.environ.get('GX_HOME') and os.environ.get('GX_TEST_LIBRARY'),
                     'set GX_HOME and GX_TEST_LIBRARY for actual GX execution')
class AsyncCopy(unittest.TestCase):
    def capture(self, negative=False):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/gpu',GX_PTX_CACHE_ROOT=tmp+'/cache',
                GX_NUM_LOCAL_GPUS='1',GX_DEVICE_MEM_GB='1',GX_PTX_JOBS='2',
                PERFMARK_LIB=str(ROOT/'build/libperfmark.so'),
                PYTHONPATH=str(ROOT/'perfmark/python'))
            env.pop('DRPERF_TEST_UNMODELED_BULK',None)
            if negative:env['DRPERF_TEST_UNMODELED_BULK']='1'
            proc=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_async_copy_application.py')],
                env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            rows=gpu.read_capture(list(Path(tmp).glob('gpu.*.jsonl')))
            return rows

    def test_copy_bytes_and_ordering_operations(self):
        rows=self.capture()
        self.assertEqual(len(rows),8)
        for row in rows:
            size=row['pcvs']['bytes']
            for metric in ['global_read_bytes','global_write_bytes','shared_read_bytes','shared_write_bytes']:
                self.assertEqual(row['metrics'][metric],size,(row['region'],size,metric))
            self.assertEqual(row['metrics'].get('unmodeled_work_events',0),0)
            if row['region']=='simple_copy':
                for suffix in ['cp.async.commit_group','cp.async.wait_group','cp.async.wait_all']:
                    sites=[v for k,v in row['sites'].items() if k.endswith(':'+suffix)]
                    self.assertEqual(len(sites),1)
                    self.assertEqual(sites[0],{'ptx_lane_instructions':1})
        model=gpu.build_model(rows)
        self.assertEqual(model['validity'],'complete')
        for region in model['regions']:
            self.assertEqual(region['invocations'],4)
            self.assertEqual(region['stateCount'],3)
            for metric in ['global_read_bytes','global_write_bytes','shared_read_bytes','shared_write_bytes']:
                self.assertTrue(region['fits'][metric]['accepted'])
                self.assertEqual(region['fits'][metric]['coefficients'],{'bytes':1})

    def test_unconnected_bulk_direction_stays_incomplete(self):
        rows=self.capture(negative=True)
        model=gpu.build_model(rows)
        self.assertEqual(model['validity'],'incomplete')
        bad=next(r for r in model['regions'] if r['name']=='bulk_s2g')
        self.assertEqual(bad['unmodeledWorkEvents'],4)
        for row in bad['observations']:
            self.assertEqual(row['metrics']['unmodeled_work_events'],1)
            # Copy group controls are counted instructions, not extra missing
            # data transfers; only the unconnected S2G copy contributes one.
            missed=[site for site,counts in row['sites'].items() if counts.get('unmodeled_work_events')]
            self.assertEqual(len(missed),1)
            self.assertTrue(missed[0].endswith(':cp.async.bulk.global.shared::cta.bulk_group'))


if __name__=='__main__':unittest.main()
