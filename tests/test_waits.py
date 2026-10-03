"""Native synchronization evidence: conservative identities, hooks, and probes."""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import explorer
import runner
import waits


def e(kind, start, end, obj='1', **fields):
    return dict({'id':str(start),'kind':kind,'api':kind,'start':start,'end':end,
                 'object':obj,'aux':'0','group':'0:1','region':'r','regionSeq':1,
                 'tid':1,'returned':True,'result':0,'potentialSyscalls':0,'elapsedUs':0,
                 'processShared':False},**fields)


class WaitEvidence(unittest.TestCase):
    def test_native_stream_match_is_not_rendered_as_a_region_publication(self):
        events=[e('transfer',1,2,obj='10',region='mv.d2h'),
                e('stream_wait',3,4,obj='10',region='mv.sync')]
        operations=waits.analyze(events)
        self.assertEqual(operations[0]['dependency'],'observed-stream-prefix')
        report=dict(probe=False,events=events,operations=operations,regions=[],warnings=[])
        text='\n'.join(waits.lines(report))
        self.assertIn('observed-stream-prefix: 1',text)
        self.assertNotIn('published by',text)
        self.assertNotIn('mv.sync waits',text)
        self.assertEqual(operations[0]['producers'],['1'])  # Preserve raw evidence.

    def test_gx_implementation_scope_requires_recorded_mode_and_caller(self):
        records = [e('lock', 1, 2, callerModule='gx_cuda.so'),
                   e('lock', 3, 4, callerModule='python3.12', instructionExcluded=True),
                   e('completion', 5, 6, callerModule='application.so'),
                   e('lock', 7, 8),  # Older captures without origin must stay visible.
                   e('stream_wait', 9, 10, callerModule='gx_cuda.so'),
                   e('declared_waited', 11, 12, callerModule='gx_cuda.so')]
        original = copy.deepcopy(records)
        meta = dict(pid=1, native_gx=True, excluded_cuda_module='gx_cuda.so')
        selected, summary = waits.application_operations(records, [dict(measurement=meta)])
        self.assertEqual(selected, records[1:])
        self.assertEqual(summary['calls'], 1)
        self.assertEqual(records, original, 'Raw evidence must not be rewritten or removed')
        for override in ({'native_gx': False}, {'excluded_cuda_module': 'other.so'},
                         {'pid': 2}, {'native_gx': 'true'}):
            selected, summary = waits.application_operations(records,
                [dict(measurement=dict(meta, **override))])
            self.assertEqual(selected, records)
            self.assertEqual(summary['calls'], 0)

    def test_gx_scope_is_process_local_and_preserves_cuda_waits(self):
        records = [e('lock', 1, 2, callerModule='gx_cuda.so'),
                   e('lock', 3, 4, callerModule='gx_cuda.so', group='1:2'),
                   e('event_wait', 5, 6, callerModule='gx_cuda.so'),
                   e('device_wait', 7, 8, callerModule='gx_cuda.so'),
                   e('stream_dependency', 9, 10, callerModule='gx_cuda.so')]
        selected, summary = waits.application_operations(records,
            [dict(data=dict(drperf=dict(pid=1, native_gx=True, excluded_cuda_module='gx_cuda.so'))),
             dict(data=dict(drperf=dict(pid=2, native_gx=False, excluded_cuda_module='gx_cuda.so')))])
        self.assertEqual(selected, records[1:])
        self.assertEqual(summary['calls'], 1)

    def test_build_retains_gx_raw_evidence_and_only_budgets_application_calls(self):
        records = [e('lock', 1, 2, callerModule='gx_cuda.so'),
                   e('completion', 3, 4, callerModule='application.so'),
                   e('stream_wait', 5, 6, callerModule='gx_cuda.so')]
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'run.json.waits').write_text(
                ''.join(json.dumps(row)+'\n' for row in records))
            runs = dict(path=folder, runs=[dict(file='run.json', data=dict(drperf=dict(
                pid=1, waits_enabled=True, wait_records=len(records), wait_dropped=0,
                native_gx=True, excluded_cuda_module='gx_cuda.so')))])
            report = waits.build(runs, [dict(id='r', states=[])], [])
            self.assertEqual(len(report['events']), 3)
            self.assertEqual([o['kind'] for o in report['operations']], ['completion', 'stream_wait'])
            self.assertEqual(report['implementationSynchronization']['calls'], 1)
            self.assertTrue(any('native-GX implementation: 1' in line for line in waits.lines(report)))


    def lifetime(self):
        return [e('sem_init',1,2), e('sem_publish',4,5,region='producer'),
                e('completion',3,6,region='consumer'), e('sem_destroy',7,8)]

    def test_unique_publication_without_wait_annotations(self):
        op=waits.analyze(self.lifetime())[0]
        self.assertEqual(op['producers'],['4'])
        self.assertEqual(op['dependency'],'matched-completion')
        self.assertEqual(op['blocking'],'unknown')
        self.assertEqual(op['ordering'],'publication-overlapped-wait')

    def test_already_completed_is_still_semantic_dependency(self):
        records=self.lifetime(); records[2].update(start=6,end=7); records[-1].update(start=8,end=9)
        op=waits.analyze(records)[0]
        self.assertEqual(op['producers'],['4'])
        self.assertEqual(op['ordering'],'publication-returned-before-wait')

    def test_initial_tokens_missing_init_and_open_lifetime_are_unresolved(self):
        for change in ('initial','missing','open'):
            records=self.lifetime()
            if change=='initial': records[0]['aux']='1'
            if change=='missing': records=records[1:]
            if change=='open': records=records[:-1]
            self.assertFalse(waits.analyze(records)[0]['producers'],change)

    def test_multiple_publications_are_not_arbitrarily_assigned(self):
        records=self.lifetime()+[e('sem_publish',3,4,region='other')]
        self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_shared_semaphore_and_untracked_token_consumption_are_unresolved(self):
        records=self.lifetime(); records[0]['processShared']=True
        self.assertFalse(waits.analyze(records)[0]['producers'])
        records=self.lifetime()+[e('sem_consume',5,6)]
        self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_same_handle_in_different_process_cannot_satisfy_wait(self):
        records=self.lifetime(); records[1]['group']='1:1'
        self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_different_objects_do_not_form_all_to_all_dependencies(self):
        records=self.lifetime()+[e('sem_publish',3,4,obj='2',region='other')]
        self.assertEqual(waits.analyze(records)[0]['producers'],['4'])

    def test_dropped_records_disable_dependency_claims(self):
        self.assertFalse(waits.analyze(self.lifetime(),complete=False)[0]['producers'])

    def test_pending_other_object_does_not_hide_completed_dependency(self):
        records=self.lifetime()+[e('completion',9,0,obj='2',region='',returned=False)]
        ops=waits.analyze(records)
        self.assertEqual(ops[0]['producers'],['4'])
        self.assertFalse(ops[1]['producers'])
        records[-1]['object']='1'
        self.assertTrue(all(not op['producers'] for op in waits.analyze(records)))

    def test_pending_event_record_cannot_confirm_stream_dependency(self):
        records=[e('event_create',1,2,obj='40'),e('event_record',3,4,obj='40'),
                 e('event_record',5,0,obj='40',returned=False),
                 e('stream_dependency',6,7,obj='50',aux='40')]
        self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_background_shutdown_wait_is_retained_without_invalidating_capture(self):
        records=self.lifetime()+[e('completion',9,0,obj='2',region='',returned=False)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run.json.waits'
            def build():
                path.write_text(''.join(json.dumps(e)+'\n' for e in records))
                runs={'path':folder,'runs':[{'file':'run.json','data':{'drperf':{
                    'pid':1,'waits_enabled':True,'wait_records':len(records),'wait_dropped':0}}}]}
                return waits.build(runs,[{'id':'consumer','states':[]}],[])
            report=build()
            self.assertEqual(report['status'],'observed')
            self.assertEqual(len(report['pendingCalls']),1)
            self.assertEqual(report['operations'][0]['dependency'],'matched-completion')
            records[-1]['region']='consumer'
            self.assertEqual(build()['status'],'partial')

    def test_failed_wait_does_not_confirm_dependency(self):
        records=self.lifetime(); records[2]['result']=-1
        self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_receive_results_distinguish_data_zero_bytes_and_error(self):
        records=[e('receive',1,2,result=value) for value in (4,0,-1)]
        operations=waits.analyze(records)
        self.assertEqual([op['receiveStatus'] for op in operations],['data','zero-bytes','error'])
        self.assertTrue(waits.succeeded(records[0]))
        self.assertTrue(waits.succeeded(records[1]))
        self.assertFalse(waits.succeeded(records[2]))
        self.assertTrue(all(not op['producers'] for op in operations))

    def test_event_re_record_uses_latest_generation(self):
        records=[e('event_create',-2,-1),e('event_record',1,2,aux='10'),
                 e('event_record',3,4,aux='20'),e('event_wait',5,6)]
        op=waits.analyze(records)[0]
        self.assertEqual(op['producers'],['3'])
        self.assertEqual(op['stream'],'20')

    def test_event_missing_creation_and_ipc_are_unresolved(self):
        for creation in ([],[e('event_create',-1,0,aux='4')]):
            records=creation+[e('event_record',1,2),e('event_wait',3,4)]
            self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_stream_event_dependency_uses_event_argument_not_stream_handle(self):
        records=[e('event_create',1,2,obj='40'),e('event_record',3,4,obj='40',aux='60'),
                 e('stream_dependency',5,6,obj='50',aux='40')]
        op=waits.analyze(records)[0]
        self.assertEqual(op['producers'],['3'])
        self.assertEqual(op['stream'],'60')
        self.assertEqual(op['consumerStream'],'50')
        self.assertEqual(op['blocking'],'not-a-host-wait')
        records[0]['aux']='4'
        self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_event_reuse_and_overlapping_record_do_not_claim_old_producer(self):
        for middle in [e('event_create',3,4),e('event_destroy',3,4),e('event_record',3,7)]:
            records=[e('event_create',-2,-1),e('event_record',1,2),middle,e('event_wait',5,6)]
            self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_lock_and_condition_are_visible_without_false_producer(self):
        records=[e('lock',1,4,potentialSyscalls=1),e('condition',5,8),e('signal',6,7)]
        ops=waits.analyze(records)
        self.assertEqual(len(ops),2)
        self.assertTrue(all(r['blocking']=='unknown' and not r['producers'] for r in ops))

    def test_stream_wait_covers_observed_copies_on_same_stream_only(self):
        records=[e('transfer',1,2,obj='40',region='copy_a'),
                 e('transfer',3,4,obj='41',region='copy_b'),
                 e('stream_wait',5,6,obj='40'),e('stream_wait',7,8,obj='40')]
        ops=waits.analyze(records)
        self.assertEqual(ops[0]['producers'],['1'])
        self.assertEqual(ops[0]['dependency'],'observed-stream-prefix')
        self.assertFalse(ops[1]['producers'])

    def test_stream_default_handles_and_reuse_without_create_are_unresolved(self):
        for handle in ('0','1','2','40'):
            records=[e('transfer',1,2,obj=handle),e('stream_wait',5,6,obj=handle)]
            if handle=='40': records.append(e('stream_destroy',3,4,obj=handle))
            self.assertFalse(waits.analyze(records)[0]['producers'])

    def test_wait_capture_defaults_on_and_explicit_opt_out_is_preserved(self):
        self.assertEqual(runner.wait_options({}), ['-waits'])
        self.assertEqual(runner.wait_options({'DRPERF_WAITS':'0'}), [])
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runner.build_env()['DRPERF_WAITS'], '1')
        with patch.dict(os.environ, {'DRPERF_WAITS':'0'}, clear=True):
            self.assertEqual(runner.build_env()['DRPERF_WAITS'], '0')

    def test_delay_options_require_explicit_probe(self):
        with self.assertRaises(ValueError): runner.wait_options({'DRPERF_WAIT_DELAY_MS':'100'})
        with self.assertRaises(ValueError): runner.wait_options({'DRPERF_WAITS':'yes'})
        self.assertEqual(runner.wait_options({'DRPERF_WAITS':'1'}),['-waits'])
        self.assertEqual(runner.wait_options({'DRPERF_WAITS':'1','DRPERF_MAX_WAIT_RECORDS':'500000'}),
                         ['-waits','-max_wait_records','500000'])
        for value in ('0','-1','bad','10000001'):
            with self.assertRaises(ValueError):
                    runner.wait_options({'DRPERF_WAITS':'1','DRPERF_MAX_WAIT_RECORDS':value})

    def test_probe_validation_keeps_legacy_and_checks_each_requested_target(self):
        hit=e('declared_publish',1,2,region='A',injectedDelayMs=10)
        legacy=[{'measurement':{'pid':1,'wait_delay_ms':10}}]
        result=waits.validate_delay_probes([hit],legacy)
        self.assertTrue(result['requested']); self.assertTrue(result['actual'])
        self.assertFalse(result['warnings'])
        def run(pid,target,count):
            return {'measurement':{'pid':pid,'wait_delay_ms':10,'wait_delay_region':target,
                'wait_delay_kind':'event','wait_delay_injections':count}}
        hit.update(group='0:1')
        result=waits.validate_delay_probes([hit],[run(1,'A',1),run(2,'B',0)])
        self.assertEqual(sum('did not execute' in w for w in result['warnings']),1)
        self.assertIn('region B',next(w for w in result['warnings'] if 'did not execute' in w))


class NativeWaits(unittest.TestCase):
    def run_model(self, program, folder, env):
        with patch.dict(os.environ,{'DRPERF_WAITS':'1','DRPERF_FOLLOW_THREADS':'0',
                                   'DRPERF_WAIT_DELAY_MS':'0','DRPERF_WAIT_DELAY_REGION':'',**env}):
            rc,log,_=runner.run(program,str(folder),timeout=60)
        self.assertEqual(rc,0,log)
        model=explorer.build_model(folder,discover=False)
        self.assertFalse(model['validity']['errors'])
        self.assertFalse(model['validity']['traceErrors'])
        self.assertEqual(model['waits']['status'],'observed')
        return model

    def test_unmarked_mutex_housekeeping_is_aggregated(self):
        with tempfile.TemporaryDirectory(prefix='waits-unmarked-',dir=ROOT/'out') as temp:
            folder=Path(temp); source=folder/'app.c'; program=folder/'app'
            source.write_text('#include <pthread.h>\n#include "perfmark.h"\n'
                              'int main(void) { pthread_mutex_t m=PTHREAD_MUTEX_INITIALIZER;'
                              'perfmark_begin("start","n",1); perfmark_end("start");'
                              'for(int i=0;i<1000;i++){pthread_mutex_lock(&m);pthread_mutex_unlock(&m);}'
                              'perfmark_begin("inside","n",1);pthread_mutex_lock(&m);'
                              'pthread_mutex_unlock(&m);perfmark_end("inside");return 0;}')
            subprocess.run(['gcc','-O2','-pthread',str(source),'-I'+str(ROOT/'perfmark'),
                            '-L'+str(ROOT/'build'),'-lperfmark','-Wl,-rpath,'+str(ROOT/'build'),
                            '-o',str(program)],check=True)
            measured=self.run_model([str(program)],folder/'raw',{'DRPERF_MAX_WAIT_RECORDS':'16'})
            self.assertEqual(measured['provenance']['runs'][0]['measurement']['max_wait_records'],16)
            report=measured['waits']
            self.assertGreaterEqual(report['unattributedOperations']['pthread_mutex_lock'],1000)
            locks=[r for r in report['operations'] if r['api']=='pthread_mutex_lock']
            self.assertEqual(len(locks),1)
            self.assertEqual(locks[0]['region'],'inside')
            with patch.dict(os.environ,{'DRPERF_WAITS':'1','DRPERF_MAX_WAIT_RECORDS':'1'}):
                rc,log,_=runner.run([str(program)],str(folder/'limited'),timeout=60)
            self.assertEqual(rc,0,log)
            limited=explorer.build_model(folder/'limited',discover=False)['waits']
            self.assertEqual(limited['status'],'partial')
            self.assertTrue(all(not op['producers'] for op in limited['operations']))

    def test_semantic_completion_counts_and_delay_probe(self):
        with tempfile.TemporaryDirectory(prefix='waits-',dir=ROOT/'out') as temp:
            folder=Path(temp); program=folder/'app'
            subprocess.run(['gcc','-O2','-pthread',str(ROOT/'examples/waits/demo.c'),
                            '-I'+str(ROOT/'perfmark'),'-L'+str(ROOT/'build'),'-lperfmark',
                            '-Wl,-rpath,'+str(ROOT/'build'),'-o',str(program)],check=True)
            for label,delay in [('baseline','0'),('probe','150')]:
                model=self.run_model([str(program)],folder/label,
                                     {'DRPERF_WAIT_DELAY_REGION':'decode','DRPERF_WAIT_DELAY_MS':delay})
                report=model['waits']
                row=next(r for r in report['regions'] if r['region']=='consume' and r['api']=='sem_wait')
                self.assertEqual((row['calls'],row['matched'],row['unresolved']),(6,6,0))
                self.assertEqual(row['countFormula']['coefficients'],['1'])
                self.assertEqual(row['countFormula']['constant'],'0')
                self.assertEqual(report['probe'],label=='probe')
                self.assertEqual(report['probeRequested'],label=='probe')
                capture=model['provenance']['runs'][0]['measurement']
                self.assertEqual(capture['wait_delay_region'],'decode' if label=='probe' else '')
                self.assertEqual(capture['wait_delay_kind'],'semaphore')
                self.assertEqual(capture['wait_delay_ms'],int(delay))
                self.assertEqual(capture['wait_delay_injections'] > 0,label=='probe')
                metadata=next(r for r in report['regions'] if r['region']=='metadata')
                self.assertEqual(metadata['kind'],'lock')
                self.assertEqual(metadata['matched'],0)
                if label=='probe':
                    sems=[r for r in report['operations'] if r['api']=='sem_wait']
                    self.assertTrue(any(r.get('ordering')=='publication-overlapped-wait' for r in sems))
                    self.assertGreater(sum(r['potentialSyscalls'] for r in sems),0)

    def test_cuda_wait_observation_survives_instruction_exclusion_and_handle_reuse(self):
        with tempfile.TemporaryDirectory(prefix='waits-cuda-',dir=ROOT/'out') as temp:
            folder=Path(temp); lib=folder/'wait_cuda.so'; program=folder/'app'
            source=str(ROOT/'tests/waits_cuda.c')
            subprocess.run(['gcc','-O2','-fPIC','-shared','-pthread','-DEMULATOR',source,'-o',str(lib)],check=True)
            subprocess.run(['gcc','-O2',source,str(lib),'-I'+str(ROOT/'perfmark'),
                            '-L'+str(ROOT/'build'),'-lperfmark','-Wl,-rpath,'+str(ROOT/'build'),
                            '-o',str(program)],check=True)
            model=self.run_model([str(program)],folder/'raw',{'DRPERF_EXCLUDE_CUDA_MODULE':lib.name})
            events={r['id']:r for r in model['waits']['events']}
            syncs=[r for r in model['waits']['operations'] if r['api']=='cudaEventSynchronize']
            self.assertEqual(len(syncs),2)
            self.assertTrue(all(r['instructionExcluded'] for r in syncs))
            self.assertEqual([events[r['producers'][0]]['region'] for r in syncs],['download','upload'])
            self.assertEqual([r['stream'] for r in syncs],['42','43'])
            stream=next(r for r in model['waits']['operations'] if r['api']=='cudaStreamSynchronize')
            self.assertEqual(stream['dependency'],'observed-stream-prefix')
            self.assertEqual([events[p]['region'] for p in stream['producers']],['copy'])
            self.assertTrue(stream['instructionExcluded'])
            # Nested sem_wait inside the CUDA API is not double-counted.
            self.assertFalse(any(r['api']=='sem_wait' for r in model['waits']['operations']))
            keys,slots=runner.blocks_of_set(runner.load_runs(str(folder/'raw')))
            self.assertEqual(sum(c for k in keys.values() for b,c in k['vec'].items() if slots[b][0]==lib.name),0)

    def test_unannotated_socket_receive_is_reported_under_ordinary_python_region(self):
        with tempfile.TemporaryDirectory(prefix='waits-python-',dir=ROOT/'out') as temp:
            folder=Path(temp); app=folder/'app.py'
            app.write_text('import socket, perfmark\na,b=socket.socketpair()\nb.send(b"x")\n'
                           'with perfmark.region("load", n=1):\n assert a.recv(1)==b"x"\na.close(); b.close()\n')
            model=self.run_model([sys.executable,str(app)],folder/'raw',{})
            rows=[r for r in model['waits']['regions'] if r['region']=='load' and r['kind']=='receive']
            self.assertTrue(rows)
            self.assertEqual(sum(r['calls'] for r in rows),1)
            self.assertEqual(sum(r['matched'] for r in rows),0)

if __name__=='__main__': unittest.main()
