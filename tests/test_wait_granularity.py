"""Multiple wait obligations must survive annotation relocation."""
import unittest

from test_wait_coverage import Capture


class WaitGranularity(unittest.TestCase):
    def program(self, split, checkpoints, retries=1):
        c = Capture()
        c.publish(1)
        c.publish(2)
        with c.region('request'):
            for event in (1, 2):
                if split:
                    with c.region('wait_step'):
                        for i in range(retries):
                            c.record('completion', event=event)['result'] = 110 if i+1 < retries else 0
                else:
                    for i in range(retries):
                        c.record('completion', event=event)['result'] = 110 if i+1 < retries else 0
            for event in range(1, checkpoints + 1):
                c.waited(event)
        return c.check()

    def test_same_region_exposes_omitted_second_checkpoint(self):
        report = self.program(split=False, checkpoints=1)
        self.assertEqual(report['coverage']['obligations'], 2)
        self.assertEqual(report['coverage']['covered'], 1)
        self.assertEqual(report['coverage']['nativeCalls'], 2)
        self.assertEqual(report['coverage']['uncovered'], 1)
        self.assertEqual(len(report['edges']), 1)

    def test_split_region_exposes_an_omitted_second_checkpoint(self):
        report = self.program(split=True, checkpoints=1)
        self.assertEqual(report['coverage']['obligations'], 2)
        self.assertEqual(report['coverage']['covered'], 1)
        self.assertEqual(report['coverage']['uncovered'], 1)
        self.assertEqual(report['unexplained'][0]['region'], 'wait_step')

    def test_retries_stay_grouped_within_each_refined_invocation(self):
        report = self.program(split=True, checkpoints=1, retries=8)
        self.assertEqual(report['coverage']['nativeCalls'], 16)
        self.assertEqual(report['coverage']['obligations'], 2)
        self.assertEqual(report['coverage']['covered'], 1)
        self.assertEqual(report['coverage']['uncovered'], 1)

    def test_two_checkpoints_cover_the_refined_invocations(self):
        report = self.program(split=True, checkpoints=2, retries=8)
        self.assertEqual(report['coverage']['obligations'], 2)
        self.assertEqual(report['coverage']['covered'], 2)
        self.assertEqual(report['unexplained'], [])

    def test_two_successes_same_object_same_site_are_not_retries(self):
        c=Capture();c.publish()
        with c.region('request'):
            c.sync();c.sync();c.waited()
        report=c.check()
        self.assertEqual((report['coverage']['obligations'],report['coverage']['covered']),(2,1))

    def test_positive_socket_receive_results_are_successes(self):
        c=Capture()
        with c.region('request'):
            for _ in range(2):
                op=c.record('completion')
                op.update(kind='receive',api='recv',result=8)
        self.assertEqual(c.check()['coverage']['obligations'],2)

    def test_failed_attempts_at_different_sites_are_separate(self):
        c=Capture()
        with c.region('request'):
            for site in ('100','200'):
                c.record('completion',callerOffset=site)['result']=110
        self.assertEqual(c.check()['coverage']['obligations'],2)

    def test_caller_coverage_does_not_expose_an_unobserved_task_wait(self):
        c = Capture()
        c.publish(1)
        # Cooperative suspension with no captured native operation contributes
        # no obligation. A caller's native future wait does not expose the
        # suspended task's prerequisites by itself.
        with c.region('task_step'):
            pass
        with c.region('request'):
            c.sync()
            c.waited(1)
        report = c.check()
        self.assertEqual(report['coverage']['obligations'], 1)
        self.assertEqual(report['coverage']['covered'], 1)
        self.assertNotIn('task_step', [r['region'] for r in report['coverage']['regions']])


if __name__ == '__main__':
    unittest.main()
