"""Two independent logical waits in one coroutine; checkpoint omissions are deliberate."""
import argparse
import asyncio
from perfmark import async_region, region, event_publish, event_waited

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', type=int, choices=(0, 1, 2), default=2)
parser.add_argument('--parent', action='store_true')
args = parser.parse_args()

async def main():
    events = [asyncio.Event(), asyncio.Event()]
    async def release(index):
        await asyncio.sleep(.005 * (index+1))
        with region('producer.'+str(index+1)):
            event_publish(index+1, 1)
            events[index].set()
    async def consume():
        await events[0].wait()
        await events[1].wait()
        if not args.parent:
            for event in range(1, args.checkpoints+1): event_waited(event, 1)
    async def parent():
        await async_region('consumer', need=1).run(consume())
        if args.parent:
            for event in range(1, args.checkpoints+1): event_waited(event, 1)
    await asyncio.gather(async_region('request', need=1).run(parent()), release(0), release(1))

with region('capture'): pass
asyncio.run(main())
