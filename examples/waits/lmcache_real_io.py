"""Real LMCache TCP server/connector comparison; no GPU or injected delays.

Run in the LMCache environment. Only allocation is adapted to unpinned CPU
TensorMemoryObj buffers. The server, protocol and original connector are real.
--capture enables passive drperf markers; native runs provide latency evidence.
"""
import argparse
import asyncio
from concurrent.futures import Future
from contextlib import nullcontext
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--patched', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--capture', action='store_true')
parser.add_argument('--mode', choices=['before', 'after', 'both'], default='both')
parser.add_argument('--repeats', type=int, default=7)
parser.add_argument('--sizes', default='1,16,64')
args = parser.parse_args()
if args.capture:
    import perfmark


def region(name, **pcvs):
    return perfmark.region(name, **pcvs) if args.capture else nullcontext()


class CPUBuffers:
    """No pinning/device transfer needed to exercise the TCP connector."""
    def __init__(self):
        self.config = LMCacheEngineConfig.from_defaults()
        self.metadata = LMCacheMetadata('io-check', 1, 1, 0, 0, torch.uint8,
                                       (1, 2, 256, 1, 128))

    def allocate(self, shape, dtype, fmt):
        data = torch.empty(shape.numel() * dtype.itemsize, dtype=torch.uint8)
        meta = MemoryObjMetadata(shape, dtype, data.data_ptr(), data.numel(), 1, fmt=fmt)
        return TensorMemoryObj(data, meta, None)


class Steps:
    """Keep regions on active coroutine steps, never across suspension."""
    def __init__(self, coro, name, event, generation, size):
        self.iterator = coro.__await__()
        self.name, self.event, self.generation, self.size = name, event, generation, size

    def __await__(self): return self
    def __iter__(self): return self
    def __next__(self): return self.send(None)

    def step(self, method, *values):
        with region(self.name, mib=self.size):
            try:
                return method(*values)
            except StopIteration:
                if args.capture:
                    perfmark.event_publish(self.event, self.generation)
                raise

    def send(self, value): return self.step(self.iterator.send, value)
    def throw(self, *values): return self.step(self.iterator.throw, *values)
    def close(self): return self.step(self.iterator.close)


def main():
    spec = importlib.util.spec_from_file_location('lm_connector_candidate', args.patched)
    patched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patched)
    classes = {'before': LMCServerConnector, 'after': patched.LMCServerConnector}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Bind an ephemeral local port, then start the original standalone server.
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    server_log = args.output.with_suffix('.server.log').open('w')
    env = dict(os.environ)
    for key in ('LD_PRELOAD', 'DYNAMORIO_OPTIONS', 'DRPERF_LATE'):
        env.pop(key, None)
    server = subprocess.Popen([sys.executable, '-m', 'lmcache.v1.server',
                               '127.0.0.1', str(port), 'cpu'], env=env,
                              stdout=server_log, stderr=subprocess.STDOUT)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    connections = []
    rows = []
    protocol_checks = []
    try:
        deadline = time.monotonic() + 40
        while True:
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=1): pass
                break
            except OSError:
                if time.monotonic() > deadline or server.poll() is not None:
                    raise RuntimeError('LMCache server failed to start')
                time.sleep(.05)
        pool = CPUBuffers()
        for mode, cls in classes.items():
            connections.append((mode, cls('127.0.0.1', port, loop, pool),
                                 cls('127.0.0.1', port, loop, pool)))
        generation = 0
        # Late attach before measured sockets are used; server stays native.
        with region('capture', n=1): pass
        for mib in map(int, args.sizes.split(',')):
            key = CacheEngineKey('io-check', 1, 0, mib, torch.uint8)
            obj = pool.allocate(torch.Size([2, 1, 256, mib * 2048]), torch.uint8,
                                MemoryFormat.KV_2LTD)
            obj.raw_data.fill_(37)
            expected = hashlib.sha256(obj.byte_array).hexdigest()
            writer = connections[0][1]
            asyncio.run_coroutine_threadsafe(writer.put(key, obj), loop).result(30)
            assert asyncio.run_coroutine_threadsafe(writer.exists(key), loop).result(30)
            for trial in range(args.repeats + 1):
                # Alternate order to avoid always favoring one arm's warm caches.
                ordered = connections if trial % 2 == 0 else list(reversed(connections))
                for mode, bulk, lookup in ordered:
                    if args.mode != 'both' and mode != args.mode: continue
                    generation += 1
                    result = Future()
                    t = {}

                    async def op(coro, name, event):
                        value = await Steps(coro, name, event, generation, mib)
                        t[name] = time.perf_counter_ns()
                        return value

                    async def pair():
                        # Both independent operations are ready on the same loop.
                        get = asyncio.create_task(op(bulk.get(key), 'lmc.bulk.get', 8101))
                        exists = asyncio.create_task(op(lookup.exists(key), 'lmc.lookup.exists', 8102))
                        try:
                            found = await exists
                            result.set_result(found)
                            return await get
                        except BaseException as error:
                            if not result.done(): result.set_exception(error)
                            raise

                    start = time.perf_counter_ns()
                    future = asyncio.run_coroutine_threadsafe(pair(), loop)
                    with region('lmc.lookup.consume', mib=mib):
                        assert result.result(60) is True
                        if args.capture: perfmark.event_waited(8102, generation)
                    with region('lmc.bulk.consume', mib=mib):
                        received = future.result(60)
                        if args.capture: perfmark.event_waited(8101, generation)
                    digest = hashlib.sha256(received.byte_array).hexdigest()
                    assert digest == expected, (mode, mib, 'payload mismatch')
                    if trial:
                        rows.append(dict(mode=mode, mib=mib, trial=trial,
                            existsMs=(t['lmc.lookup.exists'] - start) / 1e6,
                            getMs=(t['lmc.bulk.get'] - start) / 1e6,
                            existsBeforeGet=t['lmc.lookup.exists'] < t['lmc.bulk.get'],
                            sha256=digest))
            if not args.capture:
                # Concurrent commands on ONE connection must stay framed even
                # when the new receive coroutine yields in the payload loop.
                missing = CacheEngineKey('missing-io-check', 1, 0, mib, torch.uint8)
                for mode, conn, _ in connections:
                    async def verify():
                        return await asyncio.gather(conn.get(key), conn.exists(key),
                            conn.get(key), conn.exists(missing), conn.get(missing))
                    a, found, b, absent, no_obj = asyncio.run_coroutine_threadsafe(verify(), loop).result(60)
                    assert found is True and absent is False and no_obj is None
                    assert hashlib.sha256(a.byte_array).hexdigest() == expected
                    assert hashlib.sha256(b.byte_array).hexdigest() == expected
                    protocol_checks.append(dict(mode=mode, mib=mib, concurrentCommands=5, passed=True))
        original = Path(inspect.getsourcefile(LMCServerConnector))
        sources = [original, args.patched, Path(__file__), original.parents[2]/'server/__main__.py']
        report = dict(scope='Real LMCache CPU TCP server and connector; two connections on one event loop; no injected delay; CPU-only allocation adapter.',
                      captured=args.capture, rows=rows, protocolChecks=protocol_checks,
                      sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('RESULT=' + str(args.output), flush=True)
    finally:
        for _, a, b in connections:
            for conn in (a, b):
                asyncio.run_coroutine_threadsafe(conn.close(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
        server.terminate()
        server.wait(10)
        server_log.close()


if __name__ == '__main__': main()
