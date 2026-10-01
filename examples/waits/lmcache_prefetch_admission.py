"""Exercise LMCache's real disk backend and existing prefetch serializers on CPU.

Baseline uses AsyncSingleSerializer, selected by StorageManager for async loading.
Candidate uses the installed AsyncMultiSerializer with its chunk-budget rule.
Only CPU buffer allocation and workload construction are supplied by this driver.
No artificial I/O delay or removed synchronization is used in normal runs.
"""
import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import threading
import time

import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
from lmcache.v1.storage_backend.storage_manager import AsyncSingleSerializer, AsyncMultiSerializer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['single', 'multi', 'both'], default='both')
parser.add_argument('--chunks', default='4,16')
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--capture', action='store_true')
parser.add_argument('--omit-peer', action='store_true')
parser.add_argument('--force-peer', action='store_true',
                    help='Negative control: claim a peer dependency even with free admission capacity.')
args = parser.parse_args()
if args.capture:
    import perfmark


def region(name, **state):
    return perfmark.region(name, **state) if args.capture else nullcontext()


class CPUBuffers:
    """Actual unpinned TensorMemoryObj allocation; explicit admission budget."""
    def __init__(self, budget):
        self.budget = budget

    def calculate_chunk_budget(self):
        return self.budget

    def allocate(self, shape, dtype, fmt, **kwargs):
        data = torch.empty(shape.numel() * dtype.itemsize, dtype=torch.uint8)
        meta = MemoryObjMetadata(shape, dtype, data.data_ptr(), data.numel(), 1, fmt=fmt)
        return TensorMemoryObj(data, meta, None)


def main():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    rows = []
    generation = 0
    with tempfile.TemporaryDirectory(prefix='lmc-admission-') as directory:
        cfg = LMCacheEngineConfig.from_defaults()
        cfg.local_disk = directory
        cfg.max_local_disk_size = 1.0
        cfg.local_cpu = False
        cfg.extra_config = {'disk_io_threads': 4}
        PinMonitor.GetOrCreate(cfg)
        pool = CPUBuffers(128)
        backend = LocalDiskBackend(cfg, loop, pool, dst_device='cpu')
        original_read = backend.batched_async_load_bytes_from_disk
        current = {}
        completion = {}

        def observed_read(paths, keys, memory_objs, **kw):
            which = 'small' if keys[0].chunk_hash == 10000 else 'large'
            with region('lmc.disk.read.' + which, chunks=len(keys)):
                result = original_read(paths, keys, memory_objs, **kw)
                completion[which] = time.perf_counter_ns()
                if args.capture:
                    perfmark.event_publish(9201 if which == 'large' else 9202, current['generation'])
                return result

        backend.batched_async_load_bytes_from_disk = observed_read
        max_chunks = max(map(int, args.chunks.split(',')))
        keys = [CacheEngineKey('admission', 1, 0, i, torch.uint8) for i in range(max_chunks)]
        small_key = CacheEngineKey('admission', 1, 0, 10000, torch.uint8)
        payload = bytes([37]) * (1024 * 1024)
        digest = hashlib.sha256(payload).hexdigest()
        shape = torch.Size([len(payload)])
        for key in keys + [small_key]:
            path = backend._key_to_path(key)
            Path(path).write_bytes(payload)
            backend.insert_key(key, len(payload), shape, torch.uint8, MemoryFormat.KV_2LTD)
        try:
            with region('capture', n=1):
                pass
            for chunks in map(int, args.chunks.split(',')):
                for roomy in (True, False):
                    cap = chunks + int(roomy)
                    pool.budget = 2 * cap
                    for trial in range(args.repeats + 1):
                        modes = ['single', 'multi'] if trial % 2 == 0 else ['multi', 'single']
                        for mode in modes:
                            if args.mode != 'both' and mode != args.mode:
                                continue
                            generation += 1
                            current['generation'] = generation
                            completion.clear()
                            current['large_request_done'] = False
                            serializer = (AsyncSingleSerializer(loop) if mode == 'single'
                                          else AsyncMultiSerializer(pool, loop))
                            entry = {}
                            entry_ready = threading.Event()

                            async def one(which, requested_keys):
                                if which == 'small':
                                    # Observe original admission state independently of the interface predicate.
                                    blocked = (serializer.lock is not None and serializer.lock.locked()
                                               if mode == 'single' else serializer._sem._current_chunks < 1)
                                    entry.update(blocked=bool(blocked), pending=not current['large_request_done'])
                                    entry_ready.set()
                                try:
                                    return await serializer.run(backend.batched_get_non_blocking(
                                        which, requested_keys), len(requested_keys))
                                finally:
                                    if which == 'large':
                                        current['large_request_done'] = True

                            large = asyncio.run_coroutine_threadsafe(one('large', keys[:chunks]), loop)
                            submitted = time.perf_counter_ns()
                            small = asyncio.run_coroutine_threadsafe(one('small', [small_key]), loop)
                            assert entry_ready.wait(30)
                            start = time.perf_counter_ns()
                            state = dict(serial=int(mode == 'single'), chunks=chunks,
                                         concurrent_budget=cap, peer_active=int(entry['pending']))
                            with region('lmc.prefetch.small', **state):
                                small_result = small.result(60)
                                if args.capture:
                                    perfmark.event_waited(9202, generation)
                                    if (entry['blocked'] or args.force_peer) and not args.omit_peer:
                                        perfmark.event_waited(9201, generation)
                            observed = time.perf_counter_ns()
                            large_result = large.result(60)
                            assert len(small_result) == 1 and len(large_result) == chunks
                            for obj in small_result + large_result:
                                assert hashlib.sha256(obj.byte_array).hexdigest() == digest
                                obj.unpin()
                                obj.ref_count_down()
                            if trial:
                                rows.append(dict(mode=mode, chunks=chunks, roomy=roomy, trial=trial,
                                    state=state, admissionWait=entry['blocked'],
                                    smallBeforeLarge=completion['small'] < completion['large'],
                                    callerWaitMs=(observed-start)/1e6,
                                    requestLatencyMs=(observed-submitted)/1e6,
                                    readCompletionGapMs=(completion['small']-completion['large'])/1e6,
                                    payloadsChecked=chunks+1))
        finally:
            backend.close()
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            assert not thread.is_alive()
            loop.close()
    sources = [Path(inspect.getsourcefile(cls)) for cls in
               (LocalDiskBackend, AsyncSingleSerializer, AsyncMultiSerializer)] + [Path(__file__)]
    args.output.write_text(json.dumps(dict(scope='LMCache disk-prefetch components, real files and bytes, CPU allocation adapter; not full serving.',
        captured=args.capture, omitPeer=args.omit_peer, forcePeer=args.force_peer, rows=rows,
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2)+'\n')
    print('RESULT=' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
