"""Barrier metric aliases tested on both runtime paths, with real memory exchange."""
import ctypes
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
from gpu.tests.gx_barrier_application import FORMS,ptx


def check_counts(test,report):
    test.assertEqual(report['validity'],'complete')
    test.assertEqual({r['name'] for r in report['regions']},{x[0] for x in FORMS})
    for r in report['regions']:
        test.assertEqual(r['invocations'],2)
        for o in r['observations']:
            n=64*o['pcvs']['blocks'];m=o['metrics'];warp=r['name']=='warp_sync'
            test.assertEqual(m.get('block_barrier_lane_arrivals',0),0 if warp else n)
            test.assertEqual(m.get('collective_lane_arrivals',0),n if warp else 0)
            test.assertEqual(m['shared_read_bytes'],4*n)
            test.assertEqual(m['shared_write_bytes'],4*n)


class Barriers(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('GX_TEST_LIBRARY'),'requires GX runtime')
    def test_actual_gx_aliases_and_warp(self):
        with tempfile.TemporaryDirectory() as tmp:
            env=gpu.gx_environment(os.environ['GX_HOME'],os.environ['GX_TEST_LIBRARY'])
            env.update(DRPERF_GPU_CAPTURE=tmp+'/capture',GX_PTX_CACHE_ROOT=tmp+'/cache',GX_NUM_LOCAL_GPUS='1')
            result=subprocess.run([sys.executable,str(ROOT/'gpu/tests/gx_barrier_application.py')],env=env,
                                  capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            check_counts(self,gpu.build_model(gpu.read_capture(list(Path(tmp).glob('capture.*.jsonl')))))

    @unittest.skipUnless((ROOT/'gpu/build/libdrperf_gpu.so').exists(),'requires standalone GPU build')
    def test_standalone_aliases_and_warp(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'kernel.ptx').write_text(ptx())
            subprocess.run([str(ROOT/'bin/drperf'),'--gpu','translate',str(p/'kernel.ptx'),'-o',str(p/'kernel.so')],check=True,capture_output=True,text=True)
            names=','.join('"'+name+'"' for name,_,_ in FORMS)
            (p/'driver.cc').write_text('''#include <drperf_gpu.h>
#include <cassert>
#include <cstdint>
int main(int argc,char** argv){assert(argc==2);drperf::gpu::Module module(argv[1]);
const char* names[]={'''+names+'''};
for(unsigned mode=0;mode<7;mode++)for(unsigned blocks:{1u,3u}){
 unsigned n=64*blocks;drperf::gpu::Buffer<uint32_t> output(n+8,0xdeadbeef);
 uint32_t* start=output.data()+4;void* args[]={&start};
 {drperf::gpu::Region region(names[mode],{{"blocks",blocks}});module.launch(names[mode],dim3(blocks),dim3(64),args);}
 for(unsigned i=0;i<n;i++)assert(start[i]==(i^(mode==6?1u:32u)));
 for(unsigned i=0;i<4;i++)assert(output.data()[i]==0xdeadbeef&&output.data()[n+4+i]==0xdeadbeef);
}}
''')
            subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'gpu/include'),str(p/'driver.cc'),
                            '-L'+str(ROOT/'gpu/build'),'-ldrperf_gpu','-Wl,-rpath,'+str(ROOT/'gpu/build'),'-o',str(p/'driver')],check=True,capture_output=True,text=True)
            subprocess.run([str(ROOT/'bin/drperf'),'--gpu','--output',str(p/'report'),'--',str(p/'driver'),str(p/'kernel.so')],check=True,capture_output=True,text=True)
            check_counts(self,json.loads((p/'report/profile.gpu.drperf.json').read_text()))

    @unittest.skipUnless(os.environ.get('GX_HOME'),'requires GX source')
    def test_gx_classifier_preserves_scopes(self):
        # Exercise the actual counter implementation with an observer, without
        # claiming execution support for every synchronization spelling.
        gx=Path(os.environ['GX_HOME'])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'probe.cc').write_text('''#include <light/resource_capture.hh>
#include <map>
#include <string>
static std::map<std::string,uint64_t> counts;
static void* submit(const char*,int){counts.clear();return &counts;}
static void count(void*,const char*,const char* m,uint64_t n){counts[m]+=n;}
static void finish(void*,int){}
extern "C" const gx_resource_observer_v1* drperf_gpu_gx_observer_v1(){static gx_resource_observer_v1 api{1,submit,count,finish};return &api;}
extern "C" uint64_t probe(const char* op,const char* metric){GxResourceCapture c("probe",true);{GxResourceScope s(&c);gx_resource_instruction("probe:bb0",op,1,0,0,0,0,0,0);}return counts[metric];}
''')
            so=p/'probe.so'
            subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(gx/'src/sims/gpu/include'),
                            str(p/'probe.cc'),str(gx/'src/sims/gpu/common/resource_capture.cc'),'-ldl','-o',str(so)],check=True,capture_output=True,text=True)
            code='''import ctypes as C,os
lib=C.CDLL(os.environ['GX_EXPLAIN_GPU_LIBRARY']);lib.probe.argtypes=[C.c_char_p,C.c_char_p];lib.probe.restype=C.c_uint64
cta=['bar.sync','bar.cta.sync','bar.arrive','bar.cta.red.popc.u32','barrier.sync','barrier.cta.sync.aligned','barrier.arrive','barrier.cta.red.and.pred']
collective=['bar.warp.sync','barrier.cluster.arrive','barrier.cluster.wait','shfl.sync.idx.b32']
for ops,block in [(cta,True),(collective,False)]:
 for op in ops:
  assert lib.probe(op.encode(),b'block_barrier_lane_arrivals')==int(block),op
  assert lib.probe(op.encode(),b'collective_lane_arrivals')==int(not block),op
  assert lib.probe(op.encode(),b'ptx_lane_instructions')==1,op
'''
            subprocess.run([sys.executable,'-c',code],env=dict(os.environ,GX_EXPLAIN_GPU_LIBRARY=str(so)),check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
