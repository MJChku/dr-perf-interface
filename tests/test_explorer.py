"""Contract tests for exact observed relations and portable region exports."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import explorer


def event(region, value, seq, group='0:1', end=None, field='n'):
    return {'region':region,'values':{field:value},'seq':seq,'end':end or seq+1,
            'group':group,'thread':'1'}


class Relations(unittest.TestCase):
    def test_last_value_affine_and_evidence(self):
        records=[]
        for i,n in enumerate((3,9,21,8,30,44)):
            records += [event('source', n, 4*i+1),event('target', 7*n+2, 4*i+3)]
        relations,meta=explorer.relation_discovery(records)
        relation=next(r for r in relations if r['target']['region']=='target' and
                      r['terms'][0]['kind']=='last')
        self.assertEqual(relation['terms'][0]['coefficient'],7)
        self.assertEqual(relation['constant'],2)
        self.assertEqual(relation['evidence']['calls'],6)
        self.assertTrue(relation['evidence']['exact'])

    def test_runs_reset_history(self):
        records=[]
        for group in ('0:1','1:1'):
            total=0
            for i,n in enumerate((3,9,21,8,30,44)):
                total+=n
                records += [event('source',n,4*i+1,group),event('target',total,4*i+3,group)]
        relations,_=explorer.relation_discovery(records)
        relation=next(r for r in relations if r['target']['region']=='target' and
                      r['terms'][0]['kind']=='cum')
        self.assertEqual(relation['constant'],0)
        self.assertEqual(relation['evidence']['groups'],2)

    def test_large_integers_preserve_exactness(self):
        records=[]
        for i,n in enumerate((2**54+3,2**54+9,2**54+21,2**54+8,2**54+30,2**54+44)):
            records += [event('source',n,4*i+1),event('target',n+7,4*i+3)]
        relations,_=explorer.relation_discovery(records)
        r=next(r for r in relations if r['target']['region']=='target')
        self.assertEqual(r['constantExact'],'7')
        self.assertEqual(explorer.portable(2**54+3),str(2**54+3))

    def test_relationship_failure_is_not_hidden(self):
        records=[]
        for i,n in enumerate((3,9,21,8,30,44)):
            records += [event('source',n,4*i+1),event('target',7*n+(i==5),4*i+3)]
        relations,_=explorer.relation_discovery(records,max_candidates=1000)
        self.assertFalse(any(r['target']['region']=='target' for r in relations))

    def test_completed_values_exclude_inflight_calls(self):
        records=[event('source',10,1,end=20),event('source',3,2,end=3),event('target',3,4)]
        features,rows,_=explorer.feature_table(records)
        j=next(i for i,f in enumerate(features) if f['kind']=='cumend' and f['region']=='source')
        self.assertEqual(rows[2][j],3)

    def test_schema_alignment_and_missing_names(self):
        keys={0:{'region':'r','states':[('n',1),('m',2)]},
              1:{'region':'r','states':[('m',4),('n',3)]},
              2:{'region':'r','states':[('wrong',7),('m',2)]}}
        out=explorer.normalise_keys(keys)
        self.assertEqual(out[1]['states'],[('n',3),('m',4)])
        self.assertTrue(out[2]['overflow'])
        self.assertEqual(keys[1]['states'],[('m',4),('n',3)])

    def test_python_source_span_and_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.py'
            path.write_text('import perfmark\nwith perfmark.region("copy", bytes=8*n):\n    work()\n')
            found=explorer.source_locations(tmp,{'copy'})['copy'][0]
            self.assertEqual((found['line'],found['endLine']),(2,3))
            self.assertEqual(found['expressions'],{'bytes':'8*n'})
            self.assertEqual(len(found['sha256']),64)

    def test_trace_integrity_blocks_duplicate_sequences(self):
        records=[{'region':'r','state':{'n':1},'seq':1,'seq_end':2,'tid':1}]*2
        _,errors=explorer.canonical_traces(records,{'r'})
        self.assertTrue(errors)


class ExportIntegrity(unittest.TestCase):
    def test_all_region_nesting_matches_existing_reader(self):
        records=[
            {'region':'outer','state':{'n':3},'seq':1,'seq_end':14,'tid':1},
            {'region':'inner','state':{'n':2},'seq':2,'seq_end':9,'tid':1},
            {'region':'leaf','state':{'n':1},'seq':3,'seq_end':4,'tid':1},
            {'region':'inner','state':{'n':1},'seq':5,'seq_end':8,'tid':1},
            {'region':'leaf','state':{'n':1},'seq':6,'seq_end':7,'tid':1},
            {'region':'worker','state':{},'seq':10,'seq_end':11,'tid':2},
            {'region':'outer:derived','state':{},'seq':12,'seq_end':13,'tid':1},
            {'region':'outer','state':{'n':1},'seq':15,'seq_end':18,'tid':1},
            {'region':'leaf','state':{'n':1},'seq':16,'seq_end':17,'tid':1},
        ]
        fast=explorer.nesting_statistics(records)
        for name in {r['region'] for r in records}:
            self.assertEqual(fast[name],explorer.derive.nested_calls(records,name))

    def test_discovery_obeys_candidate_budget(self):
        records=[]
        for i,n in enumerate((3,9,21,8,30,44)):
            records += [event('source',n,4*i+1),event('target',7*n+2,4*i+3)]
        _,metadata=explorer.relation_discovery(records,max_candidates=1)
        self.assertLessEqual(metadata['candidatesTested'],1)
        self.assertTrue(metadata['truncated'])

    def test_missing_trace_is_explicit_and_local_sidecar_wins(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw';raw.mkdir()
            old=Path(tmp)/'stale.trace';old.write_text(json.dumps({'region':'wrong'})+'\n')
            runs={'path':str(raw),'runs':[{'file':'run.1.json','data':{'drperf':{'trace_file':str(old),'trace_records':1}}}]}
            records,errors=explorer.load_local_traces(runs)
            self.assertFalse(records);self.assertTrue(errors)
            local=raw/'run.1.json.trace';local.write_text(json.dumps({'region':'right'})+'\n')
            records,errors=explorer.load_local_traces(runs)
            self.assertEqual(records[0]['region'],'right');self.assertFalse(errors)

    def test_export_ignores_other_json_reports(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)
            (raw/'run.1.json').write_text(json.dumps({'drperf':{},'regions':[]}))
            (raw/'run.1.json.blocks').touch()
            (raw/'client.json').write_text('{}')
            (raw/'regions.drperf.json').write_text('{"schema":"drperf.explorer.v1"}')
            runs=explorer.load_raw_runs(raw)
            self.assertEqual([r['file'] for r in runs['runs']],['run.1.json'])


class SourceMapping(unittest.TestCase):
    def test_decorator_and_explicit_begin_end_spans(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.py'
            path.write_text('from perf import marked, begin, end\n@marked("decorated", lambda n: dict(pairs=n*n))\ndef fn(n):\n    work(n)\nbegin("manual", n=4)\nwork(4)\nend("manual")\n')
            found=explorer.source_locations(tmp,{'decorated','manual'})
            self.assertEqual((found['decorated'][0]['line'],found['decorated'][0]['endLine']),(2,4))
            self.assertEqual(found['decorated'][0]['expressions'],{'pairs':'n*n'})
            self.assertEqual((found['manual'][0]['line'],found['manual'][0]['endLine']),(5,7))

    def test_c_comment_like_strings_do_not_hide_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sample.c'
            path.write_text('const char *url="https://example/*";\nperfmark_begin("r", "n", n);\nwork(n);\nperfmark_end("r");\n/* perfmark_begin("wrong", "n", n); */\n')
            found=explorer.source_locations(tmp,{'r','wrong'})
            self.assertEqual((found['r'][0]['line'],found['r'][0]['endLine']),(2,4))
            self.assertNotIn('wrong',found)

    def test_trace_crossing_and_non_key_state_fields(self):
        records=[{'region':'r','state':{'n':1,'tag':'debug'},'seq':1,'seq_end':4,'tid':1},
                 {'region':'s','state':{'m':2},'seq':2,'seq_end':5,'tid':1}]
        events,errors=explorer.canonical_traces(records,{'r','s'},{'r':['n'],'s':['m']})
        self.assertEqual(events[0]['values'],{'n':1})
        self.assertIn('crossing region boundaries on one thread',errors)


class SchemaVariants(unittest.TestCase):
    def test_different_names_split_but_reordering_does_not(self):
        keys={0:{'region':'query','states':[('load',1),('parts',2)],'count':3},
              1:{'region':'query','states':[('parts',4),('load',2)],'count':3},
              2:{'region':'query','states':[('store',1),('parts',3)],'count':3}}
        records=[{'region':'query','nk':2,'state':{'load':1,'parts':2,'debug':'extra'}},
                 {'region':'query','nk':2,'state':{'store':1,'parts':3}}]
        rewritten,traces,metadata=explorer.split_schemas(keys,records)
        self.assertEqual(rewritten[0]['region'],rewritten[1]['region'])
        self.assertNotEqual(rewritten[0]['region'],rewritten[2]['region'])
        self.assertEqual(traces[0]['region'],rewritten[0]['region'])
        self.assertEqual(traces[1]['region'],rewritten[2]['region'])
        self.assertTrue(all(m['originalName']=='query' for m in metadata.values()))
        aligned=explorer.normalise_keys(rewritten)
        self.assertEqual(aligned[1]['states'],[('load',2),('parts',4)])
        self.assertFalse(any(row.get('overflow') for row in aligned.values()))


if __name__ == "__main__": unittest.main()
