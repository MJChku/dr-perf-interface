"""Real Ditto lease-flush methods, with the upstream CPU GC test harness.

The FTL and compaction control logic are real. Encoder completion is a controlled
native semaphore fixture, NOT real compression, GX execution, or a vLLM model run.
Only isolated in-memory methods are changed; the Ditto checkout is read-only.
"""
import ast
import ctypes
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import perfmark

root, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'build/shared-core'))
from src import perf
perf.set_enabled(False)

spec=importlib.util.spec_from_file_location('ditto_lease_test',root/'tests/test_compaction_lease.py')
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
from src.cache.runtime import CacheRuntime

# Load the actual worker method without importing/constructing vLLM. This is
# the same method-extraction technique used by Ditto's own control-path tests.
worker_path=output/'source/worker.py'
module=ast.parse(worker_path.read_text())
worker_class=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='DittoOffloadingWorker')
method=next(n for n in worker_class.body if isinstance(n,ast.FunctionDef) and n.name=='wait')
ns={'marked':perf.marked,'n':perf.n}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(worker_path),'exec'),ns)
worker_wait=ns['wait']

fixture=ctypes.CDLL(str(output/'completion.so'))
fixture.completion_start.argtypes=[ctypes.c_uint64]
fixture.completion_wait.argtypes=[]
fixture.completion_ready.restype=ctypes.c_int
fixture.completion_finish.argtypes=[]

class Fence:
    def synchronize(self):
        fixture.completion_wait()

forced=mode in ('forced','omitted')
records=[]
for generation in range(1,4):
    perf.set_enabled(False)
    h=tests._lease_harness()
    key,blocks,target=h.bind_run(owner='request'+str(generation))
    owner=tests._own(key);lease=-generation
    h.engine.add_lease_ids({owner:lease})
    h.engine.run_gc_trigger()
    assert len(h.engine.pending_compactions)==1
    batch=h.engine.pending_compactions[0]
    h.engine.compaction_stream=Fence()
    if forced:
        # Restore the unwanted synchronous fence in the real aborted branch.
        source=output/'source/compaction_forced.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompactionEngine')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='on_lease_flush')
        namespace=dict(h.engine.on_lease_flush.__wrapped__.__globals__)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
        h.engine.on_lease_flush=MethodType(namespace['on_lease_flush'],h.engine)
    cache=SimpleNamespace(compaction_engine=h.engine,pending_jobs={})
    cache.flush_source_lease=MethodType(CacheRuntime.flush_source_lease,cache)
    cache.wait=MethodType(CacheRuntime.wait,cache)
    worker=SimpleNamespace(cache=cache)
    if generation == 1:
        # Calibrate before the producer starts: first-use marker setup must
        # not accidentally let a false dependency appear ordered.
        perf.set_enabled(True)
        with perf.region('_perfmark_setup'): pass
        perf.set_enabled(False)
        with perfmark.region('capture', n=1): pass
    fixture.completion_start(generation)
    perf.set_enabled(True)
    with perfmark.region('vllm.reclaim_pages', leases=1, forced=int(forced)):
        worker_wait(worker,[lease])
        if mode in ('forced','false-claim'):
            perfmark.event_waited(7001,generation)
    perf.set_enabled(False)
    ready=bool(fixture.completion_ready())
    assert batch.members[0].aborted
    assert bool(h.engine.pending_compactions)
    if forced: assert ready
    # Join outside the reclaim region. A delayed producer tests whether reclaim
    # really returns before compression completion in the fixed code.
    fixture.completion_finish()
    # Restore the test harness's nonblocking stream for cleanup and drive actual
    # compaction completion: aborted output must be discarded, never adopted.
    h.engine.compaction_stream=tests._queue._Stream()
    h.engine.run_gc_trigger();h.finish()
    assert h.encoder.published==[]
    assert h.engine.take_lease_releases()==[(owner,lease)]
    records.append(dict(generation=generation,readyAtReclaimReturn=ready,aborted=True,
                        outputDiscarded=True))
print('RECLAIM_RESULT='+json.dumps(dict(mode=mode,cases=records)))
