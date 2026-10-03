"""Add broad, source-visible region annotations to the isolated LMCache overlay.

Run annotate.py first. This preserves its manually named regions and waits.
No semantic wait is inferred or added by this inventory pass.
"""
import ast
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
OUT = BASE / 'overlay/lmcache'
for name in ('memory_allocators', 'lookup_client', 'rpc'):
    shutil.copytree(BASE / 'upstream' / name, OUT / 'v1' / name, dirs_exist_ok=True)
shutil.copytree(BASE / 'upstream/cache_policy', OUT / 'v1/storage_backend/cache_policy', dirs_exist_ok=True)
for name, target in [('memory_management.py', 'v1/memory_management.py'),
                     ('abstract_backend.py', 'v1/storage_backend/abstract_backend.py'),
                     ('utils.py', 'utils.py'), ('manager.py', 'v1/manager.py')]:
    shutil.copy2(BASE / 'upstream' / name, OUT / target)

support = '''

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
'''
helper = OUT / '_drperf.py'
helper.write_text(helper.read_text() + support)
# Declare the scheduler/worker result handoff at its semantic caller, not at
# the shared ZMQ primitive. This capture has one rank in one process.
lookup = OUT / 'v1/lookup_client/lmcache_lookup_client.py'
text = lookup.read_text()
text = text.replace('    def lookup(\n', '''    @marked("lmc.lookup_rpc.request", lambda self, token_ids, *a, **k: dict(
        tokens=len(token_ids), blending=int(self.enable_blending),
        chunks=(len(token_ids) + (self.config.chunk_size - 1 if self.config.save_unfull_chunk else 0)) // self.config.chunk_size))
    def lookup(
''')
text = text.replace('        results = [int.from_bytes(resp, "big") for resp in responses]',
    '''        lookup_checkpoint('waited', lookup_id)
        results = [int.from_bytes(resp, "big") for resp in responses]''')
start = text.index('                    if not self.enable_blending:', text.index('        def process_request():'))
end = text.index('                    self.transport.send_response(identity, response)', start)
block = text[start:end]
text = text[:start] + '                    with region("lmc.lookup_rpc.handle"):\n' + ''.join(
    '    ' + line if line.strip() else line for line in block.splitlines(True)) + (
    "                        lookup_checkpoint('publish', lookup_id)\n") + text[end:]
lookup.write_text(text)
inventory = []
for path in sorted(OUT.rglob('*.py')):
    rel = str(path.relative_to(OUT))
    if rel == '_drperf.py' or 'cachegen' in rel:
        continue  # Requires device-derived numerical values; separate experiment.
    source = path.read_text()
    tree = ast.parse(source)
    edits = []
    # Keep region names shorter than the native marker name capacity.
    module = rel[:-3].replace('/', '.')
    module = module.replace('v1.storage_backend.', '').replace('v1.', '')
    module = module.replace('integration.vllm.vllm_v1_adapter', 'adapter')
    module = module.replace('gpu_connector.gpu_connectors', 'gpu')
    module = module.replace('memory_allocators.', 'allocator.')
    module = module.replace('memory_management', 'memory')
    def visit(body, classes=()):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, classes + (node.name,))
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = [ast.unparse(d) for d in node.decorator_list]
            old = next((d for d in node.decorator_list if isinstance(d, ast.Call)
                        and isinstance(d.func, ast.Name) and d.func.id == 'marked'), None)
            record = dict(path=str(path.relative_to(OUT.parent)), function='.'.join(classes + (node.name,)))
            if old is not None:
                inventory.append(dict(record, region=old.args[0].value, status='existing'))
                continue
            reason = None
            if node.name.startswith('__'):
                reason = 'constructor/protocol method'
            elif any('abstractmethod' in d or d == 'property' or d.endswith('.setter') for d in decorators):
                reason = 'abstract declaration/property'
            elif rel == 'utils.py' and ('CacheEngineKey' not in classes):
                reason = 'generic utility outside cache-key architecture'
            elif rel == 'v1/storage_backend/connector/lm_connector.py' and node.name in ('get','put','exists'):
                reason = 'already covered by explicit operation-step regions'
            if reason:
                inventory.append(dict(record, status='excluded', reason=reason))
                continue
            name = 'lmc.' + module + '.' + '.'.join(classes + (node.name,))
            # Native region names support long strings; preserve unique lexical ownership.
            # Walk this body, excluding nested functions, to detect its own yields.
            def yields(n):
                if isinstance(n, (ast.Yield, ast.YieldFrom)):
                    return True
                return any(yields(x) for x in ast.iter_child_nodes(n)
                           if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))
            active = isinstance(node, ast.AsyncFunctionDef) or yields(node)
            decorator = 'active_marked' if active else 'marked'
            edits.append((node.lineno-1, ' ' * node.col_offset + '@' + decorator + '(' + repr(name) + ')\n'))
            inventory.append(dict(record, region=name, status='added',
                                  unit='active step' if active else 'call'))
    visit(tree.body)
    if not edits:
        continue
    # Preserve module docstrings and future imports.
    insertion = 0
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)) or (isinstance(node, ast.ImportFrom) and node.module == '__future__'):
            insertion = node.end_lineno
        else:
            break
    edits.append((insertion, 'from lmcache._drperf import marked, active_marked, region, lookup_checkpoint\n'))
    lines = source.splitlines(True)
    for index, text in sorted(edits, reverse=True):
        lines.insert(index, text)
    result = ''.join(lines)
    ast.parse(result)
    path.write_text(result)
(BASE / 'annotation-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
from collections import Counter
print(Counter(row['status'] for row in inventory))

# Explicit null refinements follow the existing GPU fences. A synchronous
# child is not a publisher. Markers add no CUDA events or synchronization;
# reasons in the interface remain for manual review.
p = OUT / 'v1/gpu_connector/gpu_connectors.py'
s = p.read_text()
s = 'from lmcache._drperf import copy_event, copy_checkpoint\n' + s
start = s.index('    @marked("lmc.gpu.to_gpu_chunk"')
end = s.index("    @marked('lmc.gpu.VLLMPagedMemGPUConnectorV2.get_shape')", start)
part = s[start:end]
old = '''        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        with region("mv.sync"):
            self.load_stream.synchronize()
'''
new = old + '        copy_checkpoint("waited", True)\n'
assert part.count(old) == 1
part = part.replace(old, new)
old = '''        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)
'''
new = '''        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            identity = copy_event("store", memory_obj)
            self.from_gpu(memory_obj, start, end, **kwargs)
            copy_checkpoint("waited", identity)
'''
assert part.count(old) == 1
part = part.replace(old, new)
part = part.replace('lambda: dict(chunks=n(memory_objs), tokens=sum(ends) - sum(starts))))',
    'lambda: dict(chunks=min(n(memory_objs), n(starts), n(ends)), tokens=sum(ends) - sum(starts))))', 1)
needle = 'lambda: dict(chunks=n(memory_objs), tokens=sum(ends) - sum(starts))))'
assert part.count(needle) == 1
part = part.replace(needle, '''lambda: dict(chunks=min(n(memory_objs), n(starts), n(ends)),
            host_chunks=sum(not obj.tensor.is_cuda for obj, _, _ in zip(memory_objs, starts, ends)),
            tokens=sum(ends) - sum(starts))))''')
s = s[:start] + part + s[end:]
ast.parse(s)
p.write_text(s)
for filename in ('wait-interfaces.json', 'wait-interfaces-cpu.json'):
    path = BASE / filename
    declarations = json.loads(path.read_text())
    declarations['claims'] = [c for c in declarations['claims'] if not c['id'].startswith('gpu-chunks-')]
    declarations['claims'] += [dict(id='gpu-chunks-load', region='lmc.gpu.to_gpu', event=None,
        indicator='True', producer=None, reason='The existing load-stream fence completes the submitted batch, including an empty batch; no CPU-region publisher is claimed.'),
        dict(id='gpu-chunks-store', region='lmc.gpu.from_gpu', event=None,
        indicator='host_chunks > 0', producer=None, reason='The called chunk routine fences the store stream for each host destination; this is not a dependency on the synchronous child.')]
    path.write_text(json.dumps(declarations, indent=2) + '\n')

# The adapter's slot-mapping .to(device) uses the default blocking copy. These
# are distinct from the asynchronous KV chunk submissions annotated above.
p = OUT / 'integration/vllm/vllm_v1_adapter.py'
s = p.read_text()
s = 'from lmcache._drperf import slot_event, slot_needs_copy, copy_checkpoint\n' + s
old = '''    @marked("lmc.vllm.start_load_kv", lambda self, forward_context, **kw: pcv(lambda: dict(reqs=n(self._parent._get_connector_metadata().requests))))'''
new = '''    @marked("lmc.vllm.start_load_kv", lambda self, forward_context, **kw: pcv(lambda: dict(
        reqs=n(self._parent._get_connector_metadata().requests),
        load_uploads=sum(r.load_spec is not None and r.load_spec.can_load
                         and slot_needs_copy(r.slot_mapping, self.device)
                         for r in self._parent._get_connector_metadata().requests)
            if forward_context.attn_metadata is not None and self.lmcache_engine is not None else 0)))'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''            with region("mv.h2d", states=lambda: dict(bytes=_nb(request.slot_mapping))):
                slot_mapping = request.slot_mapping.to(self.device)'''
new = '''            slot_identity = slot_event("load", request.slot_mapping, self.device)
            with region("lmc.gpu.load_slot_mapping", states=lambda: dict(bytes=_nb(request.slot_mapping))):
                slot_mapping = request.slot_mapping.to(self.device)
            copy_checkpoint("waited", slot_identity)'''
assert s.count(old) == 1
s = s.replace(old, new)
old = '''    @marked("lmc.vllm.wait_for_save", lambda self: pcv(lambda: dict(reqs=n(self._parent._get_connector_metadata().requests))))'''
new = '''    @marked("lmc.vllm.wait_for_save", lambda self: pcv(lambda: dict(
        reqs=n(self._parent._get_connector_metadata().requests),
        store_uploads=sum(((r.save_spec is not None and r.save_spec.can_save) or self.kv_role == "kv_producer")
                          and len(r.slot_mapping) == len(r.token_ids)
                          and slot_needs_copy(r.slot_mapping, self.device)
                          for r in self._parent._get_connector_metadata().requests)
            if self.lmcache_engine is not None and self.kv_role != "kv_consumer" and not self.use_layerwise else 0)))'''
assert s.count(old) == 1
s = s.replace(old, new)
start = s.index('    def wait_for_save(self):')
part = s[start:]
old = '''            with region("mv.h2d", states=lambda: dict(bytes=_nb(slot_mapping))):
                slot_mapping = slot_mapping.to(self.device)'''
new = '''            slot_identity = slot_event("store", slot_mapping, self.device)
            with region("lmc.gpu.store_slot_mapping", states=lambda: dict(bytes=_nb(slot_mapping))):
                slot_mapping = slot_mapping.to(self.device)
            copy_checkpoint("waited", slot_identity)'''
assert part.count(old) == 1
s = s[:start] + part.replace(old, new)
ast.parse(s)
p.write_text(s)
for filename in ('wait-interfaces.json', 'wait-interfaces-cpu.json'):
    path = BASE / filename
    declarations = json.loads(path.read_text())
    declarations['claims'] = [c for c in declarations['claims'] if not c['id'].startswith('slot-mapping-')]
    declarations['claims'] += [dict(id='slot-mapping-load', region='lmc.vllm.start_load_kv', event=None,
        indicator='load_uploads > 0', producer=None, reason='Blocking upload of nonempty slot mappings returns before use; no CPU-region publisher is claimed.'),
        dict(id='slot-mapping-store', region='lmc.vllm.wait_for_save', event=None,
        indicator='store_uploads > 0', producer=None, reason='Blocking upload of nonempty slot mappings returns before store; no CPU-region publisher is claimed.')]
    path.write_text(json.dumps(declarations, indent=2) + '\n')

# Explicitly reviewed null refinements, not inferred from native observations.
# One scope-end checkpoint supplies at most ONE coverage credit. It does not
# consume all descendant waits, and always remains visible in the interface.
refinements = json.loads((BASE / 'null-refinements.json').read_text())
declared_names = {row['region'] for row in inventory if 'region' in row}
for path in OUT.rglob('*.py'):
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'region' and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            declared_names.add(node.args[0].value)
declared_names.update('lmc.remote.' + kind + '_operation' for kind in ('exists', 'get', 'put'))
assert set(refinements) <= declared_names, set(refinements) - declared_names
assert all(isinstance(reason, str) and reason.strip() for reason in refinements.values())
helper = OUT / '_drperf.py'
helper.write_text(helper.read_text() + "\n_SCOPE_NULL_REASONS = " + repr(refinements) + "\n" + r'''
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
''')
