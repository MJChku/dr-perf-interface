"""Exercise LMCache CPU disk-prefetch operations with two independent requests."""
import argparse
import asyncio
from contextlib import nullcontext
import hashlib
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
from lmcache.v1.storage_backend.storage_manager import AsyncSingleSerializer

parser = argparse.ArgumentParser()
parser.add_argument('--capture', action='store_true')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
if args.capture:
    import perfmark

def region(name, **state):
    return perfmark.region(name, **state) if args.capture else nullcontext()

class CPUBuffers:
    def calculate_chunk_budget(self):
        return 128
    def allocate(self, shape, dtype, fmt, **kwargs):
        data = torch.empty(shape.numel() * dtype.itemsize, dtype=torch.uint8)
        return TensorMemoryObj(data, MemoryObjMetadata(shape, dtype, data.data_ptr(),
                               data.numel(), 1, fmt=fmt), None)

def main():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    rows = []
    with tempfile.TemporaryDirectory(prefix='lmc-workload-') as directory:
        cfg = LMCacheEngineConfig.from_defaults()
        cfg.local_disk = directory
        cfg.max_local_disk_size = 1.0
        cfg.local_cpu = False
        cfg.extra_config = {'disk_io_threads': 4}
        PinMonitor.GetOrCreate(cfg)
        backend = LocalDiskBackend(cfg, loop, CPUBuffers(), dst_device='cpu')
        read = backend.batched_async_load_bytes_from_disk
        def observed_read(paths, keys, memory_objs, **kwargs):
            label = 'lmc.disk.read_b' if keys[0].chunk_hash == 10000 else 'lmc.disk.read_a'
            with region(label, chunks=len(keys)):
                return read(paths, keys, memory_objs, **kwargs)
        backend.batched_async_load_bytes_from_disk = observed_read
        keys = [CacheEngineKey('prefetch-trial', 1, 0, i, torch.uint8) for i in range(16)]
        key_b = CacheEngineKey('prefetch-trial', 1, 0, 10000, torch.uint8)
        data = bytes([37]) * (1024 * 1024)
        digest = hashlib.sha256(data).hexdigest()
        for key in keys + [key_b]:
            Path(backend._key_to_path(key)).write_bytes(data)
            backend.insert_key(key, len(data), torch.Size([len(data)]), torch.uint8,
                               MemoryFormat.KV_2LTD)
        try:
            with region('capture', n=1):
                pass
            for count in (4, 16):
                for trial in range(args.repeats):
                    serializer = AsyncSingleSerializer(loop)
                    async def request(label, requested):
                        return await serializer.run(backend.batched_get_non_blocking(
                            label, requested), len(requested))
                    submitted = time.perf_counter_ns()
                    a = asyncio.run_coroutine_threadsafe(request('a', keys[:count]), loop)
                    b = asyncio.run_coroutine_threadsafe(request('b', [key_b]), loop)
                    with region('lmc.prefetch.consume_b', chunks=1, peer_chunks=count):
                        got_b = b.result(60)
                    observed_b = time.perf_counter_ns()
                    with region('lmc.prefetch.consume_a', chunks=count):
                        got_a = a.result(60)
                    for obj in got_a + got_b:
                        assert hashlib.sha256(obj.byte_array).hexdigest() == digest
                        obj.unpin()
                        obj.ref_count_down()
                    assert len(got_a) == count and len(got_b) == 1
                    rows.append(dict(chunks=count, trial=trial,
                                     bLatencyMs=(observed_b-submitted)/1e6,
                                     checkedPayloads=len(got_a)+len(got_b)))
        finally:
            backend.close()
            loop.call_soon_threadsafe(loop.stop)
            thread.join(5)
            assert not thread.is_alive()
            loop.close()
    args.output.write_text(json.dumps(dict(rows=rows), indent=2)+'\n')
    print('RESULT='+str(args.output), flush=True)

if __name__ == '__main__':
    main()
