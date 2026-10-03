"""Async runtime observations; no dependency targets or predicates are inferred.

Hooks delimit logical synchronization calls, not coroutine send/throw steps.
async_region keeps physical instruction regions off suspended coroutines while
preserving a logical scope for coverage across their active steps.
"""
import asyncio
import contextvars
import functools
import itertools
import types

_ids = itertools.count(1)
_scope = contextvars.ContextVar('drperf_async_scope', default=None)
_inside = contextvars.ContextVar('drperf_async_wait', default=None)
_installed = []
_APIS = ((asyncio.Lock, 'acquire'), (asyncio.Semaphore, 'acquire'),
         (asyncio.Event, 'wait'), (asyncio.Condition, 'wait'),
         (asyncio.Queue, 'get'), (asyncio.Queue, 'put'))


def binding():
    import perfmark
    if perfmark._ext is None or not hasattr(perfmark._ext, 'runtime'):
        raise RuntimeError('Async wait capture requires rebuilt perfmark Python bindings (./build.sh)')
    return perfmark, perfmark._ext.runtime


def install():
    if _installed:
        return
    pm, emit = binding()
    for code, (cls, method) in enumerate(_APIS, 1):
        original = getattr(cls, method)
        name = 'asyncio.' + cls.__name__ + '.' + method

        def make(original, code, name):
            def start():
                task = asyncio.current_task()
                if not pm._ext.runtime_state()[0] or _inside.get() is task:
                    return None
                token = _inside.set(task)
                identity = next(_ids)
                emit(0, identity, code)
                return token, identity

            def finish(context, status):
                token, identity = context
                try:
                    if pm._ext.runtime_state()[1]:
                        emit(1, identity, code, status)
                    else:
                        # An observation-only region makes unmarked task waits
                        # visible without counting a suspended coroutine.
                        with pm.region(name):
                            emit(1, identity, code, status)
                finally:
                    _inside.reset(token)

            @functools.wraps(original)
            async def observed(self, *args, **kwargs):
                context = pm._ext.runtime_call(start)
                if context is None:
                    return await original(self, *args, **kwargs)
                status = 1
                try:
                    result = await original(self, *args, **kwargs)
                    status = 0
                    return result
                finally:
                    pm._ext.runtime_call(finish, context, status)
            return observed
        wrapper = make(original, code, name)
        setattr(cls, method, wrapper)
        _installed.append((cls, method, original, wrapper))


def uninstall():
    for cls, method, original, wrapper in reversed(_installed):
        if getattr(cls, method) is wrapper:
            setattr(cls, method, original)
    _installed.clear()


class async_region:
    """Use as a coroutine decorator, or await async_region(...).run(coro).

    Each active coroutine step is counted in a fresh physical region. Coverage
    retains one logical scope, so waited checkpoints may follow multiple awaits
    or be placed in the enclosing async region. No thread-local region is left
    open while another task runs.
    """
    def __init__(self, name, **state):
        self.name, self.state = name, state

    def __call__(self, function):
        @functools.wraps(function)
        async def wrapped(*args, **kwargs):
            return await self.run(function(*args, **kwargs))
        return wrapped

    async def run(self, awaitable):
        pm, emit = binding()
        task = asyncio.current_task()
        inherited = _scope.get()
        parent = inherited[1] if inherited and inherited[0] is task else 0
        scope = next(_ids)
        iterator = awaitable.__await__()
        return await self._drive(iterator, task, scope, parent,
                                 inherited[2] if parent else 0, pm, emit)

    @types.coroutine
    def _drive(self, iterator, task, scope, parent, grandparent, pm, emit):
        value, error = None, None
        while True:
            done = False
            try:
                with pm.region(self.name, **self.state):
                    token = _scope.set((task, scope, parent))
                    emit(2, scope, parent)
                    try:
                        try:
                            if error is None:
                                yielded = iterator.send(value)
                            else:
                                yielded = iterator.throw(error)
                        except StopIteration as stopped:
                            done, result = True, stopped.value
                    finally:
                        _scope.reset(token)
            finally:
                emit(2, parent, grandparent)
            if done:
                return result
            try:
                value = yield yielded
                error = None
            except BaseException as caught:
                error = caught
