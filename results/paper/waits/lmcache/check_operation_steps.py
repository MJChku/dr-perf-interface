"""Check async annotation preserves suspension, failures and cancellation."""
import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path
import sys
import types

source = Path(__file__).parent / 'overlay/lmcache/_drperf.py'
node = next(n for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == '_OperationSteps')
stack, publications = [], []
@contextmanager
def region(name):
    assert not stack, ('region leaked across coroutine suspension', stack)
    stack.append(name)
    try:
        yield
    finally:
        assert stack.pop() == name

def publish(*identity):
    assert stack
    publications.append((identity, stack[-1]))
sys.modules['perfmark'] = types.SimpleNamespace(event_publish=publish)
namespace = {'region': region}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
Steps = namespace['_OperationSteps']

async def main():
    async def operation(i):
        await asyncio.sleep(0)
        await asyncio.sleep(.001)
        return i * 2
    async def wrapped(i):
        return await Steps(operation(i), 'operation', (1, i))
    assert await asyncio.gather(*(wrapped(i) for i in range(20))) == list(range(0,40,2))
    assert len(publications) == 20
    async def fail():
        await asyncio.sleep(0)
        raise ValueError('expected')
    try:
        await Steps(fail(), 'failed', (2, 1))
    except ValueError:
        pass
    else:
        raise AssertionError('exception suppressed')
    async def wait():
        await asyncio.Event().wait()
    async def cancellable():
        return await Steps(wait(), 'cancelled', (3, 1))
    task = asyncio.create_task(cancellable())
    await asyncio.sleep(0)
    assert not stack
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError('cancellation suppressed')
    assert len(publications) == 20 and not stack

asyncio.run(main())
print('PASS: 20 interleaved operations; no regions span suspension; exceptions/cancellation publish nothing')
