"""Ensure broad annotations preserve generator and coroutine behavior."""
import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path
import types
import sys

source = Path(__file__).parent / 'overlay/lmcache/_drperf.py'
tree = ast.parse(source.read_text())
selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in ('_ArchitectureSteps', 'active_marked')]
stack = []
@contextmanager
def region(name):
    assert not stack, stack
    stack.append(name)
    try:
        yield
    finally:
        assert stack.pop() == name

import functools
import inspect
ns = dict(region=region, _functools=functools, _inspect=inspect)
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), ns)
mark = ns['active_marked']

@mark('generator')
def generator():
    assert stack == ['generator']
    value = yield 2
    try:
        yield value + 3
    except ValueError:
        yield 7
    return 11
g = generator()
assert next(g) == 2 and not stack
assert g.send(4) == 7 and not stack
assert g.throw(ValueError()) == 7 and not stack
try:
    next(g)
except StopIteration as e:
    assert e.value == 11
else:
    raise AssertionError('return value lost')
assert not stack
closed = []
@mark('close')
def closeable():
    try:
        yield 1
    finally:
        closed.append(stack[:])
g = closeable(); next(g); g.close()
assert closed == [['close']] and not stack

@mark('async')
async def operation(i):
    assert stack == ['async']
    await asyncio.sleep(0)
    assert stack == ['async']
    return i * 2
@mark('cancel')
async def cancellable():
    await asyncio.Event().wait()
async def main():
    assert await asyncio.gather(*(operation(i) for i in range(20))) == list(range(0, 40, 2))
    task = asyncio.create_task(cancellable())
    await asyncio.sleep(0)
    assert not stack
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError('cancellation lost')
asyncio.run(main())
assert not stack
print('PASS: active-step annotations preserve interleaving, send/throw/close, return values and cancellation')

# Exercise the actual reviewed-scope wrappers independently of LMCache imports.
# State expressions run only inside PCV exclusion; disabled runs emit nothing.
from types import SimpleNamespace
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('region', 'marked')]
activity = []
class Context:
    def __init__(self, name):
        self.name, self.refine = name, False
    def waited_null_on_exit(self):
        assert perf.depth > 0
        self.refine = True
        return self
    def __enter__(self):
        activity.append(('enter', self.name)); return self
    def __exit__(self, *exc):
        if self.refine: activity.append(('null', self.name))
        activity.append(('exit', self.name)); return False
perf = SimpleNamespace(available=True, _enabled=True, _seen=set(), depth=0)
def enter(): perf.depth += 1
def leave(): perf.depth -= 1
perf._pcv_enter, perf._pcv_exit = enter, leave
ns = dict(_base_region=lambda name, states=None, **static: Context(name),
          _SCOPE_NULL_REASONS={'reviewed': 'Explicit manual review.'},
          _perf=perf, _functools=functools)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
state_calls = []
def states(fail):
    assert perf.depth > 0
    state_calls.append(fail)
    return dict(fail=fail)
@ns['marked']('reviewed', states)
def reviewed(fail):
    assert perf.depth == 0
    if fail: raise ValueError('preserved')
    return 42
assert reviewed(False) == 42
try: reviewed(True)
except ValueError: pass
else: raise AssertionError('exception suppressed')
assert activity == [('enter','reviewed'), ('null','reviewed'), ('exit','reviewed')]*2
activity.clear()
perf._enabled = False
assert reviewed(False) == 42 and not activity and len(state_calls) == 2
perf._enabled = True
with ns['region']('unreviewed'): pass
assert activity == [('enter','unreviewed'), ('exit','unreviewed')]
print('PASS: reviewed-scope opt-in, excluded PCVs, exception preservation, disabled/unreviewed controls')
