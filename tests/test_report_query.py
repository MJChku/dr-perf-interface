"""Report queries retain weighting, child semantics and static coverage scope."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import explorer
import derive
import report_query
import source_coverage


def region(name,calls,own,residual=0):
    return dict(id=name,name=name,states=[],calls=calls,points=[dict(state=[],calls=calls,observed=own)],
        regimes=[dict(coefficients=[],constant=own-residual,points=[dict(state=[],calls=calls,unexplained=residual)],
                      attribution=dict(coefficients=[],constant=[],unexplained=[]))])


def model():
    regions=[region('A',2,10),region('B',4,100,25),region('C',2,10)]
    def event(name,start,end):
        return dict(region=name,seq=start,end=end,values={},group='0',thread='1')
    trace=[event('A',1,2),event('A',3,8),event('B',4,5),event('B',6,7),
           event('C',9,12),event('B',10,11),event('C',13,16),event('B',14,15)]
    return dict(schema=explorer.SCHEMA,regions=regions,trace=dict(complete=True,events=trace),
        composition=dict(status='observed',regions=[
            dict(id='A',children=[dict(child='B',multiplicity=None)]),
            dict(id='B',children=[]),
            dict(id='C',children=[dict(child='B',multiplicity=dict(coefficients=[],constant='1'))])]))


class ReportQuery(unittest.TestCase):
    def test_integer_rendering_does_not_modify_fitted_values(self):
        self.assertEqual(derive.fmt(120.8),'121')
        self.assertEqual(derive.fmt(-12.6),'-13')
        self.assertEqual(derive.fmt(.0555),'0')
        m=model();m['regions'][0]['regimes'][0]['constant']=12.8
        detail=report_query.region_detail(m,'A',report_query.metrics(m))
        self.assertTrue(detail['formulas'][0].startswith('13 +'))
        self.assertEqual(detail['regimes'][0]['constant'],12.8)

    def test_full_cost_text_has_one_shared_legend_and_no_duplicate_composed_formulas(self):
        m=model()
        for r in m['regions']:
            for f in r['regimes']:
                f.update(unexplainedShare=0,dependent=[],range=[])
        text='\n'.join(explorer.cost_lines(m))
        self.assertEqual(text.count('Function breakdowns list'),1)
        self.assertEqual(text.count('  A = '),1)
        self.assertNotIn('composed interfaces (',text)

    def test_unresolved_child_full_cost_but_fitted_child_residual_is_not_inherited(self):
        rows={r['region']:r for r in report_query.metrics(model())}
        self.assertEqual(rows['A']['ownInstructions'],20)
        self.assertEqual(rows['A']['inclusiveInstructionsEstimate'],220)
        self.assertEqual(rows['A']['interfaceUnexplainedInstructionsEstimate'],200)
        self.assertEqual(rows['B']['ownUnexplainedInstructions'],100)
        self.assertEqual(rows['C']['interfaceUnexplainedInstructionsEstimate'],0)

    def test_missing_trace_does_not_make_parent_residual_zero(self):
        m=model();m['trace']['complete']=False
        rows={r['region']:r for r in report_query.metrics(m)}
        self.assertIsNone(rows['A']['interfaceUnexplainedInstructionsEstimate'])
        self.assertEqual(rows['B']['ownUnexplainedInstructions'],100)

    def test_call_weighting_and_nonzero_small_percent(self):
        m=model();r=m['regions'][0];r['calls']=101
        r['points']=[dict(state=[0],calls=100,observed=10),dict(state=[1],calls=1,observed=1000)]
        r['regimes'][0]['points']=[dict(state=[0],unexplained=1),dict(state=[1],unexplained=100)]
        row=report_query.metrics(m)[0]
        self.assertEqual(row['ownInstructions'],2000)
        self.assertEqual(row['ownUnexplainedInstructions'],200)
        self.assertEqual(report_query.percent(.00001),'<1%')

    def test_cli_read_only_queries_without_evidence_or_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'profile.drperf.json'
            m=model();m['waits']=dict(evidence=dict(path='missing-sidecar.gz'))
            path.write_text(json.dumps(m));before=path.read_bytes()
            def query(*args):
                return subprocess.run([str(ROOT/'bin/drperf'),'--report',str(path),*args],capture_output=True,text=True)
            ranked=query('--top','unexplained','--topk','1','--json')
            self.assertEqual(ranked.returncode,0,ranked.stderr)
            self.assertEqual(json.loads(ranked.stdout)['regions'][0]['region'],'A')
            costly=query('--top','costly','--json')
            self.assertEqual(json.loads(costly.stdout)['regions'][0]['region'],'B')
            stats=query('--stats','--json')
            self.assertEqual(json.loads(stats.stdout)['singleStateRegions'],3)
            details=query('--region','A')
            self.assertEqual(details.returncode,0,details.stderr)
            self.assertIn('unexplained(F[B])',details.stdout)
            self.assertIn('FUNCTION BREAKDOWN',details.stdout)
            self.assertEqual(query('--region','unknown').returncode,2)
            self.assertEqual(query('--topk','0').returncode,2)
            self.assertEqual(path.read_bytes(),before)

    def test_static_coverage_counts_nested_unexecuted_spans_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'app.py').write_text('# comment\nfrom perfmark import region\n\n'
                'with region("outer"):\n    x = 1\n    with region("inner"):\n        x += 1\n'
                '# outside\ny = 2\n')
            (root/'other.py').write_text('a = 1\n# no annotation\n')
            locations=explorer.source_locations(root,None)
            coverage=source_coverage.build(dict(regions=[],trace=dict(complete=True)),locations,root)
            self.assertEqual(coverage['lines']['total'],7)
            self.assertEqual(coverage['lines']['marked'],4)
            self.assertEqual(coverage['observed'],0)
            self.assertEqual(coverage['notObserved'],2)
            self.assertEqual(coverage['lines']['files'][0]['unmarkedRanges'],[[2,2],[9,9]])

    def test_decorator_and_c_spans_include_body_not_unmarked_functions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'app.py').write_text('@active_marked("async")\nasync def f():\n    return 1\n\nx = 2\n')
            (root/'app.c').write_text('/* comment */\nvoid f() {\n perfmark_begin("c", 0);\n x++;\n perfmark_end("c");\n}\n')
            coverage=source_coverage.marked_lines(root,explorer.source_locations(root,None))
            self.assertEqual(coverage['marked'],6)
            self.assertEqual(coverage['total'],9)

    def test_dynamic_python_name_is_marked_without_a_static_region_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'app.py').write_text('name = "runtime"\nwith region(name):\n    x = 1\ny = 2\n')
            locations=explorer.source_locations(root,None)
            self.assertEqual(dict(locations),{})
            coverage=source_coverage.marked_lines(root,locations)
            self.assertEqual((coverage['marked'],coverage['total']),(2,4))

if __name__=='__main__':unittest.main()
