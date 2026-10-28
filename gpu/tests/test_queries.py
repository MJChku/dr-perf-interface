"""GPU report queries use one metric and preserve composition/observed weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'lib'))
import gpu_backend as gpu
import gpu_queries as queries


def row(name,n,counts,**identity):
    return dict(region=name,pcvs={'n':n},metrics=counts,
        sites={'kernel__gxptx:bb0:cpp1:op':counts},**identity)


class Queries(unittest.TestCase):
    def ordinary(self):
        rows=[]
        for n in [1,2,3,3,3,3]:
            rows.append(row('reads',n,{'global_read_bytes':n*n,'fp32_flops':n}))
        for n in [1,2,3]:
            rows.append(row('flops',n,{'global_read_bytes':n,'fp32_flops':100*n}))
        return gpu.build_model(rows)

    def test_rank_one_metric_and_weight_observed_invocations(self):
        model=self.ordinary()
        read=queries.rank(model,metric='global_read_bytes',top='costly',topk=1)
        self.assertEqual(read['regions'][0]['name'],'reads')
        self.assertEqual(read['regions'][0]['score'],41)
        flop=queries.rank(model,metric='fp32_flops',top='costly',topk=1)
        self.assertEqual(flop['regions'][0]['name'],'flops')
        self.assertEqual(flop['regions'][0]['score'],600)
        unexplained=queries.rank(model,metric='global_read_bytes',top='unexplained')
        self.assertEqual(len(unexplained['regions']),1)
        self.assertEqual(unexplained['regions'][0]['score'],41)

    def test_parent_cost_exclusive_but_unfitted_child_term_unexplained(self):
        rows=[];token=0
        for n in [0,1,2,3]:
            token+=1;parent=token
            rows.append(row('parent',n,{},pid=1,token=token,parent=0))
            for _ in range(n*n):
                token+=1;rows.append(row('child',n,{'global_read_bytes':10},pid=1,token=token,parent=parent))
        model=gpu.build_model(rows)
        costly=queries.rank(model,metric='global_read_bytes',top='costly')
        self.assertEqual([(r['name'],r['score']) for r in costly['regions']],[('child',140)])
        unexplained=queries.rank(model,metric='global_read_bytes',top='unexplained')
        self.assertEqual([(r['name'],r['score']) for r in unexplained['regions']],[('parent',140)])
        self.assertIn('unexplained(F[child])',unexplained['regions'][0]['formula'])

    def test_metric_filter_retains_breakdown_without_mutating_source(self):
        model=self.ordinary();before=copy.deepcopy(model)
        view=queries.select(model,region='reads',metric='global_read_bytes')
        self.assertEqual(model,before)
        r=view['regions'][0]
        self.assertEqual(set(r['fits']),{'global_read_bytes'})
        self.assertTrue(r['fits']['global_read_bytes']['unexplainedByState'])
        self.assertEqual(set(r['sites']['kernel__gxptx:bb0']),{'global_read_bytes'})
        for o in r['observations']:self.assertEqual(set(o['metrics']),{'global_read_bytes'})
        text=queries.render(model,region='reads',metric='global_read_bytes')
        self.assertNotIn('fp32_flops',text)
        self.assertIn('Unexplained basic block',text)

    def test_selected_metric_quantifies_largest_same_state_spread(self):
        # State means are affine even though individual invocations differ.
        # The cross-state range is deliberately much larger than any spread.
        rows=[]
        for n, values in [(1,(9999998,10000002)), (2,(19999996,20000004)),
                          (3,(29999998,30000002))]:
            for value in values:
                rows.append(row('varying',n,{'global_read_bytes':value,'fp32_flops':n}))
        model=gpu.build_model(rows);before=copy.deepcopy(model)
        fit=model['regions'][0]['fits']['global_read_bytes']
        self.assertTrue(fit['accepted'])
        self.assertEqual(fit['unexplainedMean'],0)
        text=queries.render(model,metric='global_read_bytes')
        self.assertIn("global_read_bytes 19999996..20000004; mean 20000000 at PCVs {'n': 2}",text)
        self.assertIn('largest observed within-state range',text)
        self.assertEqual(text.count('Same-state invocation variation:'),1)
        self.assertNotIn('9999998..30000002',text)
        self.assertNotIn('Same-state invocation variation:',queries.render(model,metric='fp32_flops'))
        full=queries.render(model)
        self.assertIn('Same-state invocation variation: global_read_bytes\n',full)
        self.assertNotIn('largest observed within-state range',full)
        self.assertEqual(model,before)

    def test_incomplete_capture_is_explicit_in_compact_query(self):
        model=gpu.build_model([row('bad',1,{'global_read_bytes':16,'unmodeled_work_events':1})])
        text=queries.render(model,metric='global_read_bytes',top='costly')
        self.assertIn('INCOMPLETE CAPTURE',text)
        self.assertIn('unknown cost is not estimated',text)
        self.assertIn('INCOMPLETE resource capture',text)

    def test_validation_and_empty_unexplained_result(self):
        model=self.ordinary()
        for kwargs in [dict(metric=None,top='costly'),dict(metric='bogus',top='costly'),
                       dict(metric='fp32_flops',top='costly',topk=0),
                       dict(metric='fp32_flops',top='costly',region='missing')]:
            with self.assertRaises(ValueError):queries.rank(model,**kwargs)
        result=queries.rank(model,metric='fp32_flops',top='unexplained')
        self.assertEqual(result['regions'],[])

    def test_standalone_report_command_needs_no_gpu_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'profile.gpu.drperf.json';p.write_text(json.dumps(self.ordinary()))
            run=subprocess.run([str(ROOT/'bin/drperf'),'--gpu','--report',str(p),
                '--metric','global_read_bytes','--top','unexplained','--topk','1','--json'],
                text=True,capture_output=True)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads(run.stdout)
            self.assertEqual(result['schema'],'drperf.gpu.query.v1')
            self.assertEqual(result['regions'][0]['score'],41)


if __name__=='__main__':unittest.main()
