"""Coverage budgets on complete invocation trees; independent of native identity."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import copy
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'lib'))
import event_model


class Capture:
    def __init__(self):
        self.clock = self.seq = 0
        self.stack = []
        self.traces, self.events, self.ops = [], [], []
        self.thread, self.group = '1', '0:1'

    @contextmanager
    def region(self, name):
        self.seq += 1
        call = dict(group=self.group, thread=self.thread, seq=self.seq, region=name, values={})
        self.traces.append(call)
        self.stack.append(call)
        yield
        self.seq += 1
        call['end'] = self.seq
        self.stack.pop()

    def record(self, kind, event=1, generation=1, **kwargs):
        self.clock += 3
        call = self.stack[-1]
        e = dict(id=str(len(self.events)), kind=kind, start=self.clock, end=self.clock+1,
            group=self.group, tid=self.thread, regionSeq=call['seq'], region=call['region'],
            api='sem_wait' if kind=='completion' else 'perfmark_event_'+kind.removeprefix('declared_'),
            object=str(event), aux=str(generation), returned=True, result=0, **kwargs)
        self.events.append(e)
        if kind=='completion': self.ops.append(e)
        return e

    def publish(self, event=1, generation=1):
        with self.region('publisher'):
            self.record('declared_publish', event, generation)

    def waited(self, event=1, generation=1):
        return self.record('declared_waited', event, generation)

    def sync(self, retries=1):
        for i in range(retries):
            self.record('completion')['result'] = 110 if i + 1 < retries else 0

    def model(self):
        counts = Counter(t['region'] for t in self.traces)
        return dict(validity={}, regions=[dict(id=n, states=[], calls=k,
            points=[dict(state=[], calls=k)]) for n,k in counts.items()],
            trace=dict(complete=True,events=self.traces),
            waits=dict(status='observed',events=self.events,operations=self.ops))

    def check(self):
        return event_model.check(self.model())


class CoverageBudget(unittest.TestCase):
    def test_old_capture_gx_lock_cannot_consume_application_wait_credit(self):
        c = Capture()
        c.publish()
        with c.region('consumer'):
            with c.region('emulator_call'):
                c.record('completion', callerModule='gx_cuda.so')
            with c.region('real_wait'):
                c.record('completion', callerModule='application.so')
            c.waited()
        model = c.model()  # Both operations stored, as in an older report.
        model['provenance'] = dict(runs=[dict(measurement=dict(
            pid=1, native_gx=True, excluded_cuda_module='gx_cuda.so'))])
        r = event_model.check(model)
        self.assertEqual((r['coverage']['obligations'], r['coverage']['covered']), (1, 1))
        self.assertEqual(r['edges'][0]['coverageCredit']['region'], 'real_wait')
        self.assertEqual(r['implementationSynchronization']['calls'], 1)
        self.assertEqual(len(model['waits']['operations']), 2)
        # A missing real checkpoint is still detected with the same scope.
        model['waits']['events'] = [e for e in model['waits']['events']
                                   if e['kind'] != 'declared_waited']
        r = event_model.check(model)
        self.assertEqual(r['coverage']['uncovered'], 1)
        self.assertEqual(r['unexplained'][0]['region'], 'real_wait')

    def test_native_caller_evidence_does_not_discharge_obligations(self):
        c = Capture()
        with c.region('A'):
            for _ in range(4):
                c.record('completion', callerModule='libpython.so', callerOffset='123')['result'] = 110
        r = self.counts(c, 1, 0)
        example = r['coverage']['regions'][0]['examples'][0]
        self.assertEqual(example['callers'], [dict(api='sem_wait', module='libpython.so',
                                                   offset='123', calls=4)])
        self.assertEqual(example['otherCallerCalls'], 0)

    def counts(self, capture, total, covered):
        r = capture.check()
        self.assertEqual((r['coverage']['obligations'], r['coverage']['covered'],
                          r['coverage']['uncovered']), (total,covered,total-covered))
        self.assertEqual(sum(e['coverageCredit'] is not None for e in r['edges']),covered)
        return r

    def test_retries_one_invocation_one_obligation(self):
        c=Capture();c.publish()
        with c.region('A'):
            with c.region('B'): c.sync(10)
            c.waited()
        r=self.counts(c,1,1)
        self.assertEqual(r['coverage']['nativeCalls'],10)
        self.assertEqual(r['unexplained'],[])

    def test_two_calls_require_two_credits(self):
        for credits in (0,1,2,3):
            c=Capture();c.publish()
            with c.region('A'):
                for _ in range(2):
                    with c.region('B'): c.sync(3)
                for _ in range(credits): c.waited()
            self.counts(c,2,min(2,credits))

    def test_parent_own_sync_not_merged_with_children(self):
        c=Capture();c.publish()
        with c.region('A'):
            c.sync()
            for _ in range(2):
                with c.region('B'): c.sync()
            c.waited();c.waited()
        self.counts(c,3,2)

    def test_deep_propagation_without_recursion_limit(self):
        c=Capture();c.publish()
        # Build a deep capture iteratively, like the checker itself.
        scopes=[]
        for _ in range(1200):
            scope=c.region('R');scope.__enter__();scopes.append(scope)
        c.sync(3)
        for scope in reversed(scopes[1:]): scope.__exit__(None,None,None)
        c.waited();scopes[0].__exit__(None,None,None)
        self.counts(c,1,1)

    def test_child_and_parent_cannot_reuse_credit(self):
        c=Capture();c.publish(1);c.publish(2)
        with c.region('A'):
            with c.region('B'): c.sync();c.waited(1)
            with c.region('C'): c.sync()
            c.waited(2)
        self.counts(c,2,2)

    def test_sibling_cannot_pay_another_siblings_obligation(self):
        c=Capture();c.publish()
        with c.region('A'):
            with c.region('B'): c.sync()
            with c.region('C'): c.waited()
        self.counts(c,1,0)

    def test_marker_before_sync_cannot_bank_credit(self):
        c=Capture();c.publish()
        with c.region('A'):
            c.waited()
            with c.region('B'): c.sync()
        self.counts(c,1,0)

    def test_later_sync_in_same_invocation_requires_later_checkpoint(self):
        c=Capture();c.publish()
        with c.region('A'):
            c.sync();c.waited();c.sync()
        self.counts(c,2,1)

    def test_credits_do_not_escape_root_or_thread(self):
        for other_thread in (False,True):
            c=Capture();c.publish()
            with c.region('A'): c.sync()
            if other_thread: c.thread='2'
            with c.region('A'): c.waited()
            self.counts(c,1,0)

    def test_shared_helper_covered_per_calling_context(self):
        c=Capture();c.publish(1);c.publish(2)
        for name,event in [('A',1),('C',2)]:
            with c.region(name):
                with c.region('B'): c.sync(4)
                c.waited(event)
        self.counts(c,2,2)

    def test_channel_reused_across_pairs_is_rejected_and_cannot_pay(self):
        c=Capture()
        for name,generation in [('A',1),('C',2)]:
            c.publish(1,generation)
            with c.region(name): c.sync();c.waited(1,generation)
        r=self.counts(c,2,0)
        self.assertEqual(r['status'],'violation')
        self.assertTrue(any('shared across region pairs' in v['reason'] for v in r['violations']))

    def test_same_pair_repeated_generations_are_allowed(self):
        c=Capture()
        for g in range(10):
            c.publish(1,g)
            with c.region('A'): c.sync();c.waited(1,g)
        self.assertEqual(self.counts(c,10,10)['status'],'ordered')

    def test_publication_region_changes_are_also_pair_conflicts(self):
        c=Capture()
        for name,g in [('P',1),('Q',2)]:
            with c.region(name): c.record('declared_publish',1,g)
            with c.region('B'): c.sync();c.waited(1,g)
        self.assertEqual(self.counts(c,2,0)['status'],'violation')

    def test_no_declarations_still_reports_missing_coverage(self):
        c=Capture()
        with c.region('B'): c.sync(5)
        self.assertEqual(self.counts(c,1,0)['unexplained'][0]['count'],1)

    def test_native_resolved_publisher_does_not_replace_annotation(self):
        c=Capture()
        with c.region('B'):
            c.sync();c.ops[-1]['producers']=['native-release']
        self.assertEqual(self.counts(c,1,0)['nativeResolved'],1)

    def test_failed_retries_count_once(self):
        c=Capture();c.publish()
        with c.region('A'):
            with c.region('B'):
                c.sync(4)
                for op in c.ops[:-1]: op['result']=110
            c.waited()
        self.counts(c,1,1)

    def test_incomplete_sync_cannot_be_covered(self):
        c=Capture();c.publish()
        with c.region('A'):
            c.sync();c.ops[-1].update(returned=False,end=0);c.waited()
        self.counts(c,1,0)

    def test_invalid_declarations_do_not_discharge(self):
        for mode in ('missing','late','duplicate','incomplete'):
            c=Capture()
            if mode!='missing' and mode!='late': c.publish()
            if mode=='duplicate': c.publish()
            with c.region('A'):
                c.sync(); marker=c.waited()
                if mode=='incomplete': marker['returned']=False
            if mode=='late': c.publish()
            self.counts(c,1,0)

    def test_excluded_annotation_regions_do_not_create_obligations(self):
        c=Capture();c.publish()
        with c.region('A'): c.sync();c.waited()
        m=c.model()
        op=dict(c.ops[0],region='perf.pcv',regionSeq=999,id='excluded')
        m['waits']['operations'].append(op)
        self.assertEqual(event_model.check(m)['coverage']['obligations'],1)
        op['region']='missing-application-region'
        with self.assertRaisesRegex(ValueError,'uncaptured'):
            event_model.check(m)

    def test_incomplete_capture_disables_coverage(self):
        c=Capture();c.publish()
        with c.region('A'): c.sync();c.waited()
        m=c.model();m['trace']['complete']=False
        self.assertEqual(event_model.check(m)['coverage']['status'],'unverified')

    def test_check_does_not_mutate_input(self):
        c=Capture();c.publish()
        with c.region('A'): c.sync();c.waited()
        m=c.model();before=copy.deepcopy(m)
        event_model.check(m)
        self.assertEqual(m,before)

    def test_budget_matches_independent_maximum_matching_on_nested_programs(self):
        # Enumerate possible credit assignments in small generated executions.
        # This checks the budget result, not its particular greedy allocation.
        rng=random.Random(715)
        for _ in range(80):
            c=Capture()
            for event in range(1,5): c.publish(event)
            def body(depth):
                name='R'+str(depth)
                with c.region(name):
                    for _ in range(rng.randrange(1,4)):
                        if depth<3 and rng.random()<.5: body(depth+1)
                        elif rng.random()<.5: c.sync(rng.randrange(1,4))
                        else: c.waited(depth+1)
            body(0)
            r=c.check();calls={t['seq']:t for t in c.traces}
            # Every successful operation terminates one generated retry chain.
            groups={o['id']:o for o in c.ops if o['result']==0}
            markers=[e for e in c.events if e['kind']=='declared_waited']
            options=[]
            for m in markers:
                parent=calls[m['regionSeq']]
                options.append([identity for identity,o in groups.items()
                    if parent['seq']<=o['regionSeq'] and calls[o['regionSeq']]['end']<=parent['end']
                    and o['end']<m['start']])
            # Independent augmenting-path maximum bipartite matching.
            assigned={}
            def augment(i,seen):
                for s in options[i]:
                    if s in seen: continue
                    seen.add(s)
                    if s not in assigned or augment(assigned[s],seen):
                        assigned[s]=i;return True
                return False
            expected=sum(augment(i,set()) for i in range(len(options)))
            self.assertEqual(r['coverage']['covered'],expected)


class NativeCoverage(unittest.TestCase):
    def test_compiled_examples_and_cli_exit_codes(self):
        import importlib.util
        import tempfile
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('check_coverage',root/'examples/waits/check_coverage.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix='coverage-',dir=root/'out') as tmp:
            self.assertEqual(len(module.run(tmp)),18)
