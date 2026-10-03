"""Exercise the annotated connector methods with a deterministic device double.

No GPU latency/completion claim: this checks placement relative to the existing
synchronize call and ensures markers do not add device operations or waits.
The serving capture separately executes the real connector under GX.
"""
import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import itertools

source = Path(__file__).parent / 'overlay/lmcache/v1/gpu_connector/gpu_connectors.py'
cls = next(n for n in ast.parse(source.read_text()).body
           if isinstance(n, ast.ClassDef) and n.name == 'VLLMPagedMemGPUConnectorV2')
names = {'to_gpu', 'from_gpu', 'batched_to_gpu', 'batched_from_gpu'}
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
for node in methods:
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    if node.args.kwarg:
        node.args.kwarg.annotation = None
code = compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), str(source), 'exec')
log, active = [], []
generations = itertools.count(1)
fail = None

@contextmanager
def region(name, **kwargs):
    active.append(name)
    try:
        yield
    finally:
        active.pop()

def event(kind, memory_obj=None):
    return ((kind, next(generations))
            if memory_obj is None or not memory_obj.tensor.is_cuda else None)

def checkpoint(kind, identity):
    assert kind == 'waited', 'null refinement must not publish'
    if identity is not None:
        log.append((kind, identity, active[-1]))

def transfer(*args, **kwargs):
    log.append(('transfer',))
    if fail == 'transfer':
        raise RuntimeError('transfer failed')

class Stream:
    def synchronize(self):
        log.append(('sync',))
        if fail == 'sync':
            raise RuntimeError('sync failed')

scope = dict(region=region, copy_event=event, copy_checkpoint=checkpoint,
    torch=NS(cuda=NS(stream=lambda s: nullcontext())),
    device_ops=NS(multi_layer_kv_transfer=transfer),
    lmcache_native=NS(TransferDirection=NS(H2D=1, D2H=2)),
    MemoryFormat=NS(KV_2LTD=1, KV_MLA_FMT=2), _nb=lambda t: 128)
exec(code, scope)
region_names = dict(to_gpu='lmc.gpu.to_gpu_chunk', from_gpu='lmc.gpu.from_gpu_chunk',
                   batched_to_gpu='lmc.gpu.to_gpu', batched_from_gpu='lmc.gpu.from_gpu')
class Connector:
    def initialize_kvcaches_ptr(self, **kwargs): pass
    def _initialize_pointers(self, kvcaches): return []

def bind(name):
    def wrapped(self, *args, **kwargs):
        with region(region_names[name]):
            return scope[name](self, *args, **kwargs)
    return wrapped
for name in names:
    setattr(Connector, name, bind(name))
c = Connector()
c.kvcaches = [NS(device='cuda')]
c.use_mla = False
c.gpu_buffer = None
c.store_stream = Stream()
c.load_stream = Stream()
c.device = 'cuda'
c.page_buffer_size = c.block_size = c.head_size = c.block_stride_elems = 1
c.engine_kv_format = 0

class Tensor:
    shape = (1, 1, 1, 1)
    device = 'cuda'
    def __init__(self, cuda): self.is_cuda = cuda
    def __getitem__(self, key): return self
    def copy_(self, other, non_blocking):
        assert non_blocking
        log.append(('copy',))
        if fail == 'copy':
            raise RuntimeError('copy failed')
        return self

def obj(cuda):
    return NS(tensor=Tensor(cuda), metadata=NS(fmt=1))

cases = 0
for buffer in (None, Tensor(True)):
    c.gpu_buffer = buffer
    for direction in ('load', 'store'):
        for cuda_flags in ([], [False], [True], [False, True, False]):
            for fail_mode in (None, 'transfer', 'copy', 'sync'):
                log.clear()
                fail = fail_mode
                method = c.batched_to_gpu if direction == 'load' else c.batched_from_gpu
                raised = False
                try:
                    method([obj(cuda) for cuda in cuda_flags], [0]*len(cuda_flags),
                           [1]*len(cuda_flags), slot_mapping=[0])
                except RuntimeError:
                    raised = True
                publications = {e[1]: i for i,e in enumerate(log) if e[0] == 'publish'}
                for i, e in enumerate(log):
                    if e[0] != 'waited': continue
                    assert any(x[0] == 'sync' for x in log[:i]), log
                    assert fail != 'sync', log
                    assert e[2] == 'lmc.gpu.' + ('to_gpu' if direction == 'load' else 'from_gpu'), log
                if fail is None:
                    expected = 1 if direction == 'load' else sum(not x for x in cuda_flags)
                    assert sum(e[0] == 'transfer' for e in log) == len(cuda_flags), log
                    assert sum(e[0] == 'sync' for e in log) == (1 if direction == 'load' else expected), log
                    assert sum(e[0] == 'waited' for e in log) == expected, log
                    assert not publications, log
                cases += 1
# An exception after submission must never acquire a waited checkpoint.
assert not active
print(f'PASS: {cases} connector cases; empty/host/device/mixed batches and failures; no added device operations')

# Exercise the actual helper, rather than the device double's event allocator.
# Accessing tensor properties can perform native GIL synchronization. It must
# happen under perf.pcv, and must not happen at all with measurement disabled.
helper = ast.parse((source.parents[2] / '_drperf.py').read_text())
function = next(n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name == 'copy_event')
def excluded_pcv(fn):
    with region('perf.pcv'):
        return fn()
flags = NS(available=True, _enabled=True)
env = dict(pcv=excluded_pcv, _perf=flags, _wait_generations=itertools.count(1))
exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
             '<copy_event>', 'exec'), env)
class GuardedTensor:
    def __init__(self, cuda): self.cuda = cuda
    @property
    def is_cuda(self):
        assert active and active[-1] == 'perf.pcv', 'annotation tensor access escaped exclusion'
        return self.cuda
for cuda in (False, True):
    result = env['copy_event']('store', NS(tensor=GuardedTensor(cuda)))
    assert (result is None) == cuda
flags._enabled = False
assert env['copy_event']('store', NS(tensor=GuardedTensor(False))) is None
print('PASS: actual annotation helper evaluates tensor properties only in excluded PCV scope')

# Slot-mapping uploads: compare full bodies to the original overlay after
# removing passive annotations, then execute actual branch prefixes against
# device doubles (the remainder consumes KV data and is covered by GX runs).
import copy
import sys
from unittest.mock import patch
adapter = source.parents[2] / 'integration/vllm/vllm_v1_adapter.py'
original = Path('/home/ubuntu/compression/ditto_kv/example/qwen2.5B/lmcache/lmcache_overlay/lmcache/integration/vllm/vllm_v1_adapter.py')
def adapter_methods(path):
    c = next(n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == 'LMCacheConnectorV1Impl')
    return {n.name:n for n in c.body if isinstance(n, ast.FunctionDef)
            and n.name in ('start_load_kv', 'wait_for_save')}
class StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, n):
        n.decorator_list=[]
        return self.generic_visit(n)
    def visit_Assign(self, n):
        if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id=='slot_event':
            return None
        return self.generic_visit(n)
    def visit_Expr(self, n):
        if isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id=='copy_checkpoint':
            return None
        return self.generic_visit(n)
    def visit_With(self, n):
        n=self.generic_visit(n)
        if len(n.items)==1:
            e=n.items[0].context_expr
            if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id=='region':
                return n.body
        return n
original_methods=adapter_methods(original)
new_methods=adapter_methods(adapter)
for name,node in new_methods.items():
    assert ast.dump(StripAnnotations().visit(copy.deepcopy(node))) == ast.dump(
        StripAnnotations().visit(copy.deepcopy(original_methods[name]))), name

class Device:
    def __init__(self, name, index=None):
        if isinstance(name, Device):self.type,self.index=name.type,name.index;return
        parts=name.split(':');self.type=parts[0]
        self.index=int(parts[1]) if len(parts)>1 else index
    def __eq__(self, other):
        return isinstance(other, Device) and (self.type,self.index)==(other.type,other.index)
class SlotTensor:
    def __init__(self, device, size=2, broken=False):
        self._device=Device(device);self.size=size;self.broken=broken
    @property
    def device(self):
        assert active and active[-1]=='perf.pcv', 'annotation device read escaped exclusion'
        return self._device
    def __len__(self):return self.size
    def numel(self):
        assert active and active[-1]=='perf.pcv', 'annotation shape read escaped exclusion'
        return self.size
    def to(self, device):
        if self.broken:raise RuntimeError('upload failed')
        log.append(('upload-returned',))
        return self
current_device=0
fake_torch=NS(device=Device,Tensor=SlotTensor,cuda=NS(current_device=lambda:current_device))
flags._enabled=True
env.update(region=region, copy_checkpoint=checkpoint, torch=fake_torch, n=len,
           _nb=lambda t:len(t)*8, logger=NS(debug=lambda *a:None,warning=lambda *a:None),
           slot_event=None)
functions=[n for n in helper.body if isinstance(n, ast.FunctionDef)
           and n.name in ('slot_event','slot_needs_copy')]
exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])), '<slot helpers>', 'exec'),env)
class Metadata:
    def __init__(self,requests):self.requests=requests
env['LMCacheConnectorMetadata']=Metadata
states={}
for name,original_node in new_methods.items():
    node=copy.deepcopy(original_node)
    decorator=next(d for d in node.decorator_list if isinstance(d,ast.Call)
                   and isinstance(d.func,ast.Name) and d.func.id=='marked')
    states[name]=eval(compile(ast.Expression(decorator.args[1]),'<entry PCVs>','eval'),env)
    for loop in [n for n in ast.walk(node) if isinstance(n,ast.For)]:
        for i,item in enumerate(loop.body):
            if isinstance(item,ast.Expr) and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Name) and item.value.func.id=='copy_checkpoint':
                if isinstance(item.value.args[0],ast.Constant) and item.value.args[0].value=='waited':
                    loop.body=loop.body[:i+1]+[ast.Continue()]
                    break
    node.decorator_list=[];node.returns=None
    for arg in node.args.args:arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<adapter prefix>', 'exec'),env)

def request(device='cpu', load=True, save=True, mismatch=False, broken=False):
    return NS(req_id='r',token_ids=[1,2],slot_mapping=SlotTensor(device,3 if mismatch else 2,broken),
              load_spec=None if load is None else NS(can_load=load,vllm_cached_tokens=0),
              save_spec=None if save is None else NS(can_save=save))
requests=[[],[request()],[request('cuda:0')],[request(load=False,save=False)],
          [request(load=None,save=None)],[request(mismatch=True)],
          [request(),request('cuda:0'),request(load=False,save=False)]]
empty_request=request()
empty_request.token_ids=[]
empty_request.slot_mapping=SlotTensor('cpu',0)
requests.extend([[empty_request], [request(), empty_request]])
cases=0
with patch.dict(sys.modules, {'torch':fake_torch}):
    for current_device in (0,1):
        for name in new_methods:
            for reqs in requests:
                for role,layerwise,engine_present,attn in (
                    ('kv_both',False,True,True),('kv_consumer',False,True,True),
                    ('kv_producer',False,True,True),('kv_both',True,True,True),
                    ('kv_both',False,False,True),('kv_both',False,True,False)):
                    md=Metadata(reqs)
                    self=NS(_parent=NS(_get_connector_metadata=lambda:md),kv_caches={'layer':[]},
                        lmcache_engine=NS(lookup_unpin=lambda _:None) if engine_present else None,
                        kv_role=role,use_layerwise=layerwise,device=Device('cuda'),config=NS(),
                        _layerwise_save_storers={},_stats_monitor=NS(update_interval_vllm_hit_tokens=lambda _:None,
                        update_interval_prompt_tokens=lambda _:None))
                    args=(self,NS(attn_metadata={} if attn else None)) if name=='start_load_kv' else (self,)
                    log.clear()
                    pcvs=states[name](*args)
                    with region('lmc.vllm.'+name):env[name](*args)
                    waited=[e for e in log if e[0]=='waited']
                    expected=pcvs['load_uploads' if name=='start_load_kv' else 'store_uploads']
                    assert len(waited)==expected,(name,pcvs,log)
                    for i,e in enumerate(log):
                        if e[0]!='waited':continue
                        assert i>0 and log[i-1][0]=='upload-returned',log
                        assert e[2]=='lmc.vllm.'+name,log
                    cases+=1
    for kind in ('load','store'):
        flags._enabled=False
        assert env['slot_event'](kind, SlotTensor('cpu'), Device('cuda')) is None
        flags._enabled=True
    for name in new_methods:
        self.lmcache_engine=NS(lookup_unpin=lambda _:None)
        self.kv_role='kv_both';self.use_layerwise=False
        self._parent=NS(_get_connector_metadata=lambda:Metadata([request(broken=True)]))
        args=(self,NS(attn_metadata={})) if name=='start_load_kv' else (self,)
        log.clear()
        try:
            with region('lmc.vllm.'+name):env[name](*args)
        except RuntimeError:
            assert not log,log
        else:
            raise AssertionError('Upload failure must propagate without waited')
print(f'PASS: {cases} adapter entry-state cases, CUDA aliases, disabled measurement; native bodies unchanged')
