"""Task waits, logical invocation coverage, and suspension-safe region scopes."""
import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'perfmark/python'), str(ROOT/'lib')]
import perfmark
import perfmark_asyncio as runtime
import event_model
import waits
from test_wait_coverage import Capture


class AsyncRuntime(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        runtime.uninstall()
        self.records = []
        self.patch = patch.object(perfmark._ext, 'runtime', self.record)
        self.patch.start()
        runtime.install()
        with perfmark.region('capture'): pass

    def tearDown(self):
        runtime.uninstall()
        self.patch.stop()
        self.assertEqual(perfmark._ext.runtime_state()[1], 0)

    def record(self, *args):
        self.records.append(args)

    async def test_two_waits_one_scope_without_leaking_into_publisher(self):
        e = asyncio.Event()
        async def publish():
            await asyncio.sleep(0)
            self.assertEqual(perfmark._ext.runtime_state()[1], 0)
            e.set()
        async def consume():
            await e.wait()
            await e.wait()
            return 7
        consumer, _ = await asyncio.gather(perfmark.async_region('consumer').run(consume()), publish())
        self.assertEqual(consumer, 7)
        self.assertEqual(len([r for r in self.records if r[0] == 0]), 2)
        self.assertEqual(len([r for r in self.records if r[0] == 1]), 2)
        scopes = {r[1] for r in self.records if r[0] == 2 and r[1]}
        self.assertEqual(len(scopes), 1)

    async def test_lock_suspension_and_cancellation(self):
        lock = asyncio.Lock()
        await lock.acquire()
        self.records.clear()
        task = asyncio.create_task(perfmark.async_region('cancelled').run(lock.acquire()))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.assertEqual(len([r for r in self.records if r[0] == 0]), 1)
        ends = [r for r in self.records if r[0] == 1]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0][3], 1)
        lock.release()

    async def test_nested_condition_reacquisition_is_one_logical_wait(self):
        condition = asyncio.Condition()
        async def waiter():
            async with condition:
                await condition.wait()
        async def publisher():
            await asyncio.sleep(0)
            async with condition: condition.notify()
        await asyncio.gather(perfmark.async_region('waiter').run(waiter()), publisher())
        # Two explicit lock acquires, one Condition.wait; internal reacquire
        # is implementation of that logical wait and has no second endpoint.
        self.assertEqual([r[2] for r in self.records if r[0] == 0].count(4), 1)
        self.assertEqual([r[2] for r in self.records if r[0] == 0].count(1), 2)

    async def test_nested_scope_and_independent_child_task(self):
        async def leaf(): await asyncio.sleep(0)
        async def parent():
            await perfmark.async_region('nested').run(leaf())
            await asyncio.create_task(perfmark.async_region('independent').run(leaf()))
        await perfmark.async_region('parent').run(parent())
        relationships = {(r[1], r[2]) for r in self.records if r[0] == 2 and r[1]}
        self.assertEqual(len(relationships), 3)
        self.assertEqual(sum(parent != 0 for _, parent in relationships), 1)

    async def test_supported_queue_semaphore_event_return_values(self):
        async def body():
            q = asyncio.Queue(1)
            await q.put(42)
            self.assertEqual(await q.get(), 42)
            sem = asyncio.Semaphore(1)
            self.assertTrue(await sem.acquire())
            sem.release()
            event = asyncio.Event(); event.set()
            self.assertTrue(await event.wait())
        await perfmark.async_region('primitives').run(body())
        self.assertEqual({r[2] for r in self.records if r[0] == 0}, {2,3,5,6})


class AsyncCoverage(unittest.TestCase):
    def scope(self, c, identity, parent=0):
        c.record('runtime_scope', identity, parent)

    def endpoint(self, c, end, identity):
        c.record('runtime_wait_end' if end else 'runtime_wait_begin', identity, 1,
                 runtimeStatus=0, potentialSyscalls=0, elapsedUs=0)

    def model(self, c):
        model = c.model()
        model['waits']['operations'] = waits.analyze(c.events)
        return model

    def test_waits_across_steps_need_two_credits_and_one_indicator_evaluation(self):
        for credits in (0, 1, 2):
            c = Capture(); c.publish(1); c.publish(2)
            with c.region('consumer'):
                self.scope(c, 100)
                self.endpoint(c, False, 10)
            with c.region('other_task'): pass
            with c.region('consumer'):
                self.scope(c, 100)
                self.endpoint(c, True, 10)
                self.endpoint(c, False, 11)
            with c.region('consumer'):
                self.scope(c, 100)
                self.endpoint(c, True, 11)
                for event in range(1, credits+1): c.waited(event)
            model = self.model(c)
            model['waitDeclarations'] = dict(version=1, claims=[dict(id='one', region='consumer',
                event='1', producer='publisher', indicator='True')])
            r = event_model.check(model)
            self.assertEqual((r['coverage']['obligations'],r['coverage']['covered']), (2, credits))
            claim = r['interfaceChecks']['claims'][0]
            self.assertEqual(claim['present']+claim['absent'], 1)
            self.assertEqual(claim['unexpectedAbsence'], int(credits == 0))

    def test_parent_credit_covers_child_across_suspension(self):
        c=Capture();c.publish()
        with c.region('parent'):
            self.scope(c,100)
            with c.region('child'):
                self.scope(c,200,100);self.endpoint(c,False,10)
        with c.region('parent'):
            self.scope(c,100)
            with c.region('child'):
                self.scope(c,200,100);self.endpoint(c,True,10)
        with c.region('parent'):
            self.scope(c,100);c.waited()
        r=event_model.check(self.model(c))
        self.assertEqual(r['coverage']['covered'],1)
        self.assertEqual(r['edges'][0]['coverageCredit']['region'],'child')

    def test_other_task_on_same_thread_cannot_cover(self):
        c=Capture();c.publish()
        with c.region('task'):
            self.scope(c,100);self.endpoint(c,False,10);self.endpoint(c,True,10)
        with c.region('task'):
            self.scope(c,200);c.waited()
        r=event_model.check(self.model(c))
        self.assertEqual(r['coverage']['covered'],0)

    def test_logical_descendant_publisher_rejected_across_steps(self):
        c=Capture()
        with c.region('parent'):
            self.scope(c,100)
            with c.region('child'):
                self.scope(c,200,100);c.record('declared_publish')
        with c.region('parent'):
            self.scope(c,100);c.waited()
        r=event_model.check(self.model(c))
        self.assertEqual(r['edges'][0]['status'],'violation')

    def test_unfinished_await_is_not_dischargeable(self):
        c=Capture();c.publish()
        with c.region('task'):
            self.scope(c,100);self.endpoint(c,False,10);c.waited()
        r=event_model.check(self.model(c))
        self.assertEqual((r['coverage']['obligations'],r['coverage']['covered']),(1,0))

    def test_native_layers_fold_only_inside_same_logical_task_operation(self):
        c=Capture()
        with c.region('task'):
            self.scope(c,100)
            begin=c.record('runtime_wait_begin',10,1,asyncScope='100')
            nested=c.record('completion',asyncScope='100')
        with c.region('other_task'):
            self.scope(c,200)
            other=c.record('completion',asyncScope='200')
        with c.region('task'):
            self.scope(c,100)
            c.record('runtime_wait_end',10,1,asyncScope='100',runtimeStatus=0)
            after=c.record('completion',asyncScope='100')
        model=self.model(c)
        ops={o['id']:o for o in model['waits']['operations']}
        self.assertIn('runtimeContainer',ops[nested['id']])
        self.assertNotIn('runtimeContainer',ops[other['id']])
        self.assertNotIn('runtimeContainer',ops[after['id']])
        r=event_model.check(model)
        self.assertEqual(r['coverage']['obligations'],3)
        self.assertEqual(r['coverage']['nestedNativeCalls'],1)

    def test_scope_cannot_change_entry_pcvs(self):
        c=Capture()
        with c.region('task'): self.scope(c,100)
        with c.region('task'):
            self.scope(c,100);c.stack[-1]['values']={'n':2}
        with self.assertRaisesRegex(ValueError,'PCVs'):
            event_model.check(self.model(c))

if __name__ == '__main__': unittest.main()
