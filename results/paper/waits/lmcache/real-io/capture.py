import os,sys
from pathlib import Path
root=Path('/w/lmc-waits/real-io-drperf')
sys.path.insert(0,str(root/'lib'))
import runner
os.environ["LD_LIBRARY_PATH"]=str(root/"build")+":"+os.environ.get("LD_LIBRARY_PATH","")
os.environ.update(DRPERF_WAITS='1',DRPERF_FOLLOW_THREADS='0',DRPERF_MAX_WAIT_RECORDS='200000',DRPERF_WAIT_DELAY_MS='0')
for mode in ['before','after']:
 folder=Path('/w/lmc-waits/real-io-capture')/mode
 folder.mkdir(parents=True,exist_ok=True)
 for f in (folder/'raw').glob('run.*'): f.unlink()
 rc,log,_=runner.run([sys.executable,'/w/lmc-waits/lmcache_real_io.py','--patched','/w/lmc-waits/lm_connector_after.py','--output',str(folder/'timing.json'),'--sizes','3,16','--repeats','2','--mode',mode,'--capture'],str(folder/'raw'),timeout=180)
 (folder/'capture.log').write_text(log)
 print(mode,rc,log[-1000:],flush=True)
 assert rc==0
