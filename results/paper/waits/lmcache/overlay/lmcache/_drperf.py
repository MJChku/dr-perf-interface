"""drperf support for the annotated LMCache overlay (not part of upstream).

Re-exports the tier's marker API (src.perf) and one size helper, so the
annotated modules add a single import. Region names are written literally at
each site, because drperf maps a region to source by finding its literal name
in a region(...) or marked(...) call.

Data-movement primitives are regions named by direction, each with a `bytes`
state: mv.h2d, mv.d2h, mv.d2d, mv.h2h (host copy), mv.net.send, mv.net.recv,
and mv.sync for a stream synchronisation. drperf composes a parent as
multiples of these, so redundant movement shows up as a multiplicity.
"""

from src.perf import marked, n, pcv, region  # noqa: F401


def _nb(x):
    """Bytes of a tensor or memory object; -1 when unsized."""
    try:
        return int(x.numel() * x.element_size())
    except Exception:
        try:
            return int(x.get_size())
        except Exception:
            try:
                return len(x)
            except Exception:
                return -1


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


# Broader architecture annotations. Generators and async methods are measured
# only during active steps, so yielded/suspended work cannot absorb siblings.
import functools as _functools
import inspect as _inspect
import hashlib as _hashlib

def lookup_checkpoint(kind, lookup_id):
    if not (_perf.available and _perf._enabled):
        return
    import perfmark
    generation = pcv(lambda: int.from_bytes(_hashlib.blake2b(
        lookup_id.encode('utf-8'), digest_size=8).digest(), 'big'))
    getattr(perfmark, 'event_' + kind)(2101, generation)

def copy_event(kind, memory_obj=None):
    if not (_perf.available and _perf._enabled):
        return None
    # Tensor properties may release/reacquire the GIL. The host/device test is
    # annotation computation too: evaluate it inside the excluded PCV scope.
    return pcv(lambda: True
        if memory_obj is None or not memory_obj.tensor.is_cuda else None)

def copy_checkpoint(kind, identity):
    if kind != 'waited':
        raise ValueError('Copy refinements have no publisher')
    if identity is not None and _perf.available and _perf._enabled:
        import perfmark
        perfmark.event_waited(None)

def slot_needs_copy(tensor, device):
    import torch
    # Tensor.to still returns a tensor for an empty mapping, but submits no
    # data transfer and performs no transfer-completion wait in this path.
    if tensor.numel() == 0:
        return False
    origin = tensor.device
    destination = torch.device(device)
    if origin.type != destination.type:
        return True
    if destination.type == 'cuda' and destination.index is None:
        destination = torch.device('cuda', torch.cuda.current_device())
    return origin != destination

def slot_event(kind, tensor, device):
    if not (_perf.available and _perf._enabled):
        return None
    return pcv(lambda: True
        if slot_needs_copy(tensor, device) else None)

class _ArchitectureSteps:
    def __init__(self, iterator, name):
        self.iterator, self.name = iterator, name
    def __iter__(self):
        return self
    def __await__(self):
        return self
    def __next__(self):
        return self.send(None)
    def send(self, value):
        with region(self.name):
            return self.iterator.send(value)
    def throw(self, *args):
        with region(self.name):
            return self.iterator.throw(*args)
    def close(self):
        with region(self.name):
            return self.iterator.close()

def active_marked(name):
    def decorate(function):
        if _inspect.iscoroutinefunction(function):
            @_functools.wraps(function)
            async def wrapped(*args, **kwargs):
                return await _ArchitectureSteps(function(*args, **kwargs).__await__(), name)
        elif _inspect.isgeneratorfunction(function):
            @_functools.wraps(function)
            def wrapped(*args, **kwargs):
                return (yield from _ArchitectureSteps(function(*args, **kwargs), name))
        else:
            raise TypeError('active_marked requires a coroutine or generator')
        return wrapped
    return decorate

_SCOPE_NULL_REASONS = {'lmc.adapter.LMCacheConnectorV1Impl.register_kv_caches': 'Request/cache metadata and slot-index construction may acquire interpreter and tensor-runtime locks. This checkpoint acknowledges the completed scope, without claiming a region publisher or that each invocation blocked.', 'lmc.adapter.ReqMeta.from_request_tracker': 'Request/cache metadata and slot-index construction may acquire interpreter and tensor-runtime locks. This checkpoint acknowledges the completed scope, without claiming a region publisher or that each invocation blocked.', 'lmc.vllm.build_connector_meta': 'Request/cache metadata and slot-index construction may acquire interpreter and tensor-runtime locks. This checkpoint acknowledges the completed scope, without claiming a region publisher or that each invocation blocked.', 'lmc.allocator.tensor_memory_allocator.TensorMemoryAllocator._get_buffer_slice': 'Local storage allocation, slicing, free-list updates and their runtime locks complete in this scope. No paired region publisher is claimed; allocation can block in the runtime.', 'lmc.memory.AddressManager.allocate': 'Local storage allocation, slicing, free-list updates and their runtime locks complete in this scope. No paired region publisher is claimed; allocation can block in the runtime.', 'lmc.memory.AddressManager.free': 'Local storage allocation, slicing, free-list updates and their runtime locks complete in this scope. No paired region publisher is claimed; allocation can block in the runtime.', 'lmc.memory.PinnedAllocFree.alloc': 'Local storage allocation, slicing, free-list updates and their runtime locks complete in this scope. No paired region publisher is claimed; allocation can block in the runtime.', 'lmc.local_cpu_backend.LocalCPUBackend.initialize_allocator': 'Local storage allocation, slicing, free-list updates and their runtime locks complete in this scope. No paired region publisher is claimed; allocation can block in the runtime.', 'lmc.memory.TensorMemoryObj.unpin': 'Cache lookup, pin/reference bookkeeping and local memory lifetime changes use shared locks. This acknowledges those completed operations, including possible contention, without asserting a producer dependency.', 'lmc.local_cpu_backend.LocalCPUBackend.remove': 'Cache lookup, pin/reference bookkeeping and local memory lifetime changes use shared locks. This acknowledges those completed operations, including possible contention, without asserting a producer dependency.', 'lmc.local.get1': 'Cache lookup, pin/reference bookkeeping and local memory lifetime changes use shared locks. This acknowledges those completed operations, including possible contention, without asserting a producer dependency.', 'lmc.cache_engine.LMCacheEngine.post_init': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.cache_engine.LMCacheEngineBuilder._Create_token_database': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.cache_engine.LMCacheEngineBuilder.get_or_create': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.cache_policy.__init__.get_cache_policy': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.lookup_client.factory.LookupClientFactory._create_zmq_client_transport': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.lookup_client.factory.LookupClientFactory._create_zmq_server_transport': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.lookup_client.factory.LookupClientFactory.create_lookup_client': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.lookup_client.factory.LookupClientFactory.create_lookup_server': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.manager.LMCacheManager._init_health_monitor': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.storage_manager.StorageManager.create_backends': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.token_database.TokenDatabase._get_vllm_hash_func': 'Startup imports, configuration and service construction can synchronize with the interpreter, allocator or newly started service threads. This scope-end refinement leaves these waits visible without inventing a publisher; it does not claim startup is nonblocking.', 'lmc.gpu.GPUConnectorInterface.initialize_kvcaches_ptr': 'GPU connector metadata and tensor-view setup use tensor/interpreter runtime locks. This is a scope-end refinement with no CPU-region publisher claim, not a GPU completion event.', 'lmc.gpu.VLLMPagedMemGPUConnectorV2.from_metadata': 'GPU connector metadata and tensor-view setup use tensor/interpreter runtime locks. This is a scope-end refinement with no CPU-region publisher claim, not a GPU completion event.', 'lmc.gpu.contiguous_view': 'GPU connector metadata and tensor-view setup use tensor/interpreter runtime locks. This is a scope-end refinement with no CPU-region publisher claim, not a GPU completion event.', 'lmc.gpu.from_gpu_chunk': 'This scope submits or performs tensor transfers and can synchronize in the tensor/CUDA runtime. A host-destination path can also fence the stream. No CPU child publication is claimed and asynchronous submission does not imply device completion.', 'lmc.gpu.to_gpu_chunk': 'This scope submits or performs tensor transfers and can synchronize in the tensor/CUDA runtime. A host-destination path can also fence the stream. No CPU child publication is claimed and asynchronous submission does not imply device completion.', 'mv.d2d': 'This scope submits or performs tensor transfers and can synchronize in the tensor/CUDA runtime. A host-destination path can also fence the stream. No CPU child publication is claimed and asynchronous submission does not imply device completion.', 'mv.d2h': 'This scope submits or performs tensor transfers and can synchronize in the tensor/CUDA runtime. A host-destination path can also fence the stream. No CPU child publication is claimed and asynchronous submission does not imply device completion.', 'mv.h2d': 'This scope submits or performs tensor transfers and can synchronize in the tensor/CUDA runtime. A host-destination path can also fence the stream. No CPU child publication is claimed and asynchronous submission does not imply device completion.', 'mv.sync': 'The explicit stream fence waits for previously enqueued stream work. That work has no declared region-event publisher in this capture; this null refinement preserves the fence without fabricating a parent-child dependency.', 'lmc.gpu.store_slot_mapping': 'Slot-mapping conversion can perform a blocking device upload for nonempty data, while empty data only exercises tensor metadata/runtime handling. This scope-end refinement covers the completed conversion; it does not label empty mappings as transfers.', 'lmc.net.exists': 'The client reads a response or payload from the external TCP cache server (or terminates on EOF/error); receive retries and buffer/runtime locks occur in this scope. The server publisher is outside this capture, so the wait is explicitly unresolved to a region, not dismissed as bookkeeping.', 'lmc.net.get_meta': 'The client reads a response or payload from the external TCP cache server (or terminates on EOF/error); receive retries and buffer/runtime locks occur in this scope. The server publisher is outside this capture, so the wait is explicitly unresolved to a region, not dismissed as bookkeeping.', 'mv.net.recv': 'The client reads a response or payload from the external TCP cache server (or terminates on EOF/error); receive retries and buffer/runtime locks occur in this scope. The server publisher is outside this capture, so the wait is explicitly unresolved to a region, not dismissed as bookkeeping.', 'lmc.rpc.zmq_transport.ZmqRouterServerTransport.recv_request': 'The server receives an incoming lookup request or finishes a timeout/error poll. Internal transport/interpreter locks can also occur. This is an external request-availability wait, not a result-completion dependency on the requesting caller.', 'lmc.rpc.zmq_transport.ZmqRouterServerTransport.send_response': 'ZMQ socket setup or response submission can encounter transport locking and backpressure. No captured region publisher is asserted for these transport-internal waits.', 'lmc.rpc.zmq_transport.ZmqReqRepClientTransport._create_socket': 'ZMQ socket setup or response submission can encounter transport locking and backpressure. No captured region publisher is asserted for these transport-internal waits.', 'lmc.remote.put1': 'Connection bookkeeping, socket submission and active coroutine steps may acquire locks or encounter socket backpressure. This scope-end refinement does not replace the separately declared future-completion dependency and does not span coroutine suspension.', 'lmc.remote.put_operation': 'Connection bookkeeping, socket submission and active coroutine steps may acquire locks or encounter socket backpressure. This scope-end refinement does not replace the separately declared future-completion dependency and does not span coroutine suspension.', 'lmc.remote_backend.RemoteBackend.init_connection': 'Connection bookkeeping, socket submission and active coroutine steps may acquire locks or encounter socket backpressure. This scope-end refinement does not replace the separately declared future-completion dependency and does not span coroutine suspension.', 'lmc.cache_engine.LMCacheEngine._process_tokens_internal': 'Token hashing, chunk/key construction, tensor masks and memory-reference bookkeeping use interpreter/tensor locks. Each generator resume is a separate active scope; no publisher is claimed and suspension is not charged as active execution.', 'lmc.token_database.ChunkedTokenDatabase.process_tokens': 'Token hashing, chunk/key construction, tensor masks and memory-reference bookkeeping use interpreter/tensor locks. Each generator resume is a separate active scope; no publisher is claimed and suspension is not charged as active execution.', 'lmc.lookup': 'Serving-path metadata, health checks, cache bookkeeping and tensor-runtime work can synchronize directly in this scope. Child operations and declared result dependencies keep their own coverage obligations. This null checkpoint supplies only one scope-end coverage credit, not blanket coverage of descendants.', 'lmc.retrieve': 'Serving-path metadata, health checks, cache bookkeeping and tensor-runtime work can synchronize directly in this scope. Child operations and declared result dependencies keep their own coverage obligations. This null checkpoint supplies only one scope-end coverage credit, not blanket coverage of descendants.', 'lmc.store': 'Serving-path metadata, health checks, cache bookkeeping and tensor-runtime work can synchronize directly in this scope. Child operations and declared result dependencies keep their own coverage obligations. This null checkpoint supplies only one scope-end coverage credit, not blanket coverage of descendants.', 'lmc.vllm.get_num_new_matched_tokens': 'Serving-path metadata, health checks, cache bookkeeping and tensor-runtime work can synchronize directly in this scope. Child operations and declared result dependencies keep their own coverage obligations. This null checkpoint supplies only one scope-end coverage credit, not blanket coverage of descendants.', 'lmc.vllm.start_load_kv': 'Nonempty slot uploads have explicit completion checkpoints; this scope also has one exit checkpoint for request/lifetime bookkeeping and interpreter/runtime synchronization. The interface counts both. No CPU-region publisher is claimed, and empty mappings are not classified as uploads.', 'lmc.vllm.wait_for_save': 'Nonempty slot uploads have explicit completion checkpoints; this scope also has one exit checkpoint for request/lifetime bookkeeping and interpreter/runtime synchronization. The interface counts both. No CPU-region publisher is claimed, and empty mappings are not classified as uploads.', 'lmc.lookup_client.lmcache_lookup_client.LMCacheLookupServer.close': 'Graceful request-server shutdown joins the stopped receive thread before closing its transport. This is an explicit lifecycle wait; no declared publication on the thread-exit event is available.', 'lmc.rpc.zmq_transport.ZmqRouterServerTransport.close': 'Transport teardown closes the ZMQ socket without lingering for outbound data. Internal transport/interpreter synchronization remains an explicit null refinement.', 'lmc.remote.get_operation': 'The active network-get step reads and decodes the server response and handles buffers and runtime locks. Its result publication remains a separate declared dependency; this refinement names no producer for transport/runtime synchronization and does not span coroutine suspension.'}

_base_region = region

def region(name, states=None, **static):
    context = _base_region(name, states, **static)
    if name in _SCOPE_NULL_REASONS and _perf.available and _perf._enabled:
        # Arming is annotation computation, excluded like PCV preparation.
        _perf._pcv_enter()
        try:
            context.waited_null_on_exit()
        finally:
            _perf._pcv_exit()
    return context

def marked(name, states=None):
    def decorate(function):
        if not _perf.available:
            return function
        @_functools.wraps(function)
        def wrapped(*args, **kwargs):
            if not _perf._enabled:
                _perf._seen.add(name)
                return function(*args, **kwargs)
            _perf._pcv_enter()
            try:
                declared = states(*args, **kwargs) if states else {}
            except Exception:
                declared = {}
            finally:
                _perf._pcv_exit()
            with region(name, **declared):
                return function(*args, **kwargs)
        return wrapped
    return decorate
