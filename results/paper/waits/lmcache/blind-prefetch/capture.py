import json,os,sys
from pathlib import Path
root=Path('/home/ubuntu/drperf')
sys.path.insert(0,str(root/'lib'))
import runner,explorer,event_model
p=Path('/w/lmc-waits/blind-prefetch')
folder=p/(sys.argv[1] if len(sys.argv)>1 else 'initial')
serializer=sys.argv[2] if len(sys.argv)>2 else 'single'
folder.mkdir(exist_ok=True)
os.environ['LD_LIBRARY_PATH']=str(root/'build')+':'+os.environ.get('LD_LIBRARY_PATH','')
os.environ.update(DRPERF_WAITS='1',DRPERF_FOLLOW_THREADS='0',DRPERF_MAX_WAIT_RECORDS='200000')
rc,log,_=runner.run([sys.executable,str(p/'workload.py'),'--capture','--serializer',serializer,'--output',str(folder/'timing.json')],str(folder/'raw'),timeout=240)
(folder/'capture.log').write_text(log)
assert rc==0,log[-2000:]
declarations=json.loads((p/'wait-interfaces.json').read_text())
m=explorer.build_model(folder/'raw',discover=False,wait_declarations=declarations)
explorer.write_model(m,folder/'profile.drperf.json')
(folder/'feedback.txt').write_text('\n'.join(event_model.lines(m['eventModel']))+'\n')
print((folder/'feedback.txt').read_text(),flush=True)
for f in folder.rglob('*'):
 if f.is_file(): f.chmod(0o644)
