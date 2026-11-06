"""Bounded CPU training, export and state-preserving (not bit-exact) resume."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from fullhouse_runtime import ROOT, SOURCE, validate_source

def main():
    validate_source()
    run=ROOT/'artifacts/fullhouse-training'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    python=ROOT.parent/'.venvs/fullhouse/bin/python'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    results={'source_commit':validate_source()['commit'],'device':'cpu','resume_semantics':'models and reservoirs retained; RNG restarts; not bit-exact', 'stages':[]}
    for total in (6,12):
        folder=run/str(total);folder.mkdir()
        if total==12:shutil.copyfile(run/'6/latest.pt',folder/'latest.pt')
        command=[str(python),'-m','training.train','--iterations',str(total),'--traversals','8','--buffer-capacity','2000',
                 '--strategy-buffer-capacity','2000','--batch-size','64','--train-steps','2','--strategy-train-steps','2',
                 '--checkpoint-every','6','--checkpoint-dir',str(folder),'--workers','1','--device','cpu',
                 '--no-stack-randomization','--save-resume-buffers','--opponent-exploration-decay-iters','12','--seed','42']
        if total==12:command+=['--resume']
        started=time.monotonic();peak=0;failure=None
        with (folder/'train.log').open('w') as log:
            p=subprocess.Popen(command,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                try:
                    status=Path(f'/proc/{p.pid}/status').read_text()
                    peak=max(peak,next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0))
                except FileNotFoundError:pass
                if time.monotonic()-started>600 or peak>20*1024**3:
                    failure='time_or_rss_limit';os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                time.sleep(.25)
        row=dict(total_iterations=total,command=command,returncode=p.returncode,failure=failure,peak_rss_bytes=peak,elapsed_seconds=time.monotonic()-started)
        results['stages'].append(row)
        (run/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        if p.returncode or failure:raise SystemExit('Training failed: '+str(folder/'train.log'))
        # Inspection runs in the same bounded CPU environment and checks real state.
        code='''import torch,json,sys,numpy as np
from pathlib import Path
p=Path(sys.argv[1]);c=torch.load(p/'latest.pt',map_location='cpu',weights_only=False)
assert c['iteration']==int(p.name)
assert len(c['buffers'])==6 and 'strategy_buffer' in c
assert all(torch.isfinite(v).all() for v in c['net'].values())
assert all(torch.isfinite(v).all() for v in c['strategy_net'].values())
assert c['strategy_buffer_size']>=128 and c['buffer_total']>=128
models=list(p.glob('*.npz'));assert models
for model in models:
 with np.load(model,allow_pickle=False) as m:assert all(np.isfinite(m[k]).all() for k in m.files)
print(json.dumps({'iteration':c['iteration'],'buffer_sizes':c['buffer_sizes'],'strategy_buffer_size':c['strategy_buffer_size'],'exports':[str(x) for x in models]}))
'''
        checked=subprocess.check_output([str(python),'-c',code,str(folder)],cwd=SOURCE,env=env,text=True,timeout=60)
        row['inspection']=json.loads(checked)
        row['checkpoint_sha256']=hashlib.sha256((folder/'latest.pt').read_bytes()).hexdigest()
        (run/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(checked,flush=True)
    print(run,flush=True)

if __name__=='__main__':main()
