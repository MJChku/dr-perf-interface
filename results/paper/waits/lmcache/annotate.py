"""Create a separate LMCache overlay with passive future checkpoints."""
from pathlib import Path
import shutil
import difflib

BASE=Path('/home/ubuntu/compression/ditto_kv/example/qwen2.5B/lmcache/lmcache_overlay')
OUT=Path(__file__).resolve().parent/'overlay'
shutil.copytree(BASE,OUT,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
p=OUT/'lmcache/_drperf.py'
p.write_text(p.read_text()+'''

# Passive checkpoints: preserve the real future and its original result wait.
# No region spans an await: concurrent coroutines share an execution thread.
import asyncio as _asyncio
import itertools as _itertools
import src.perf as _perf
_wait_generations = _itertools.count(1)


def submit_waitable(coro, loop, kind):
    if not (_perf.available and _perf._enabled):
        return _asyncio.run_coroutine_threadsafe(coro, loop)
    import perfmark
    identity = ({'exists': 2001, 'get': 2002, 'put': 2003}[kind], next(_wait_generations))
    async def publish_result():
        # Each active coroutine step is a real operation region. Close it
        # before suspension so unrelated tasks never become its children.
        return await _OperationSteps(coro, 'lmc.remote.' + kind + '_operation', identity)
    future = _asyncio.run_coroutine_threadsafe(publish_result(), loop)
    future._drperf_identity = identity
    future._drperf_kind = kind
    return future


class _OperationSteps:
    """Preserve await semantics while marking only active coroutine execution."""
    def __init__(self, coro, name, identity):
        self.iterator = coro.__await__()
        self.name, self.identity = name, identity

    def __await__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def _step(self, method, *args):
        import perfmark
        with region(self.name):
            try:
                return method(*args)
            except StopIteration:
                # Original coroutine returned successfully. Its Task/Future
                # is released only after this publication and region exit.
                perfmark.event_publish(*self.identity)
                raise

    def send(self, value):
        return self._step(self.iterator.send, value)

    def throw(self, *args):
        return self._step(self.iterator.throw, *args)

    def close(self):
        with region(self.name):
            return self.iterator.close()


def wait_result(future, timeout=None):
    identity = getattr(future, '_drperf_identity', None)
    if identity is None:
        return future.result(timeout)
    import perfmark
    # No primitive/helper region: this checkpoint belongs to the caller.
    result = future.result(timeout)
    perfmark.event_waited(*identity)
    return result
''')
p=OUT/'lmcache/v1/storage_backend/remote_backend.py';s=p.read_text()
s=s.replace('from lmcache._drperf import _nb, marked, n, pcv, region', 'from lmcache._drperf import _nb, marked, n, pcv, region, submit_waitable, wait_result')
s=s.replace('@marked("lmc.remote.contains")', '''@marked("lmc.remote.contains", lambda self, *a, **k: dict(
        connected=int(self.connection is not None),
        async_exists=int(not (self.config.extra_config or {}).get("use_exists_sync", False))))''')
s=s.replace('@marked("lmc.remote.get", lambda self, keys, *a, **k: dict(chunks=n(keys)))', '''@marked("lmc.remote.get", lambda self, keys, *a, **k: dict(
        chunks=n(keys),
        ready=int(self.local_cpu_backend is not None and self.connection is not None),
        individual=int(self.connection is not None and not self.connection.support_batched_get())))''')
s=s.replace('future = asyncio.run_coroutine_threadsafe(\n                    self.connection.exists(key), self.loop\n                )\n                res = future.result()', "future = submit_waitable(self.connection.exists(key), self.loop, 'exists')\n                res = wait_result(future)")
s=s.replace('asyncio.run_coroutine_threadsafe(self.connection.get(key), self.loop)', "submit_waitable(self.connection.get(key), self.loop, 'get')")
s=s.replace('memory_obj = future.result(self.config.blocking_timeout_secs)', 'memory_obj = wait_result(future, self.config.blocking_timeout_secs)')
s=s.replace('memory_obj = fut.result(self.config.blocking_timeout_secs)', 'memory_obj = wait_result(fut, self.config.blocking_timeout_secs)')
s=s.replace('    def put_callback(self, future: Future, key: CacheEngineKey):',
            '    @marked("lmc.remote.put_callback")\n    def put_callback(self, future: Future, key: CacheEngineKey):')
s=s.replace('            future.result()\n        except Exception as e:',
            '            wait_result(future)\n        except Exception as e:', 1)
s=s.replace('future = asyncio.run_coroutine_threadsafe(\n            self.connection.put(key, compressed_memory_obj), self.loop\n        )',
            "future = submit_waitable(\n            self.connection.put(key, compressed_memory_obj), self.loop, 'put'\n        )")
p.write_text(s)
patch=[]
for rel in ['lmcache/_drperf.py','lmcache/v1/storage_backend/remote_backend.py']:
 patch.extend(difflib.unified_diff((BASE/rel).read_text().splitlines(True),(OUT/rel).read_text().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(OUT.parent/'annotations.patch').write_text(''.join(patch))
