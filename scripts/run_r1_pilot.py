"""Corrected 12-iteration CPU campaign plus exact split-resume comparison."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from fullhouse_runtime import ROOT
from build_fullhouse_r1 import build

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--average-sampler',choices=['importance','fixed-opponent'],default='importance')
    args=p.parse_args()
    source=build();python=ROOT.parent/'.venvs/fullhouse/bin/python'
    out=ROOT/'artifacts/r1-training'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    manifest={'revision':'r1-cpu-v2','average_sampler':args.average_sampler,'patch_sha256':hashlib.sha256((ROOT/'patches/fullhouse-r1.patch').read_bytes()).hexdigest(),
              'support_sha256':hashlib.sha256((ROOT/'native/fullhouse-r1/support.py').read_bytes()).hexdigest(),'stages':[]}
    for name,extra in [('continuous',[]),('split',['--stop-after','6']),('resumed',['--resume']),('repeat',[])]:
        folder=out/name;folder.mkdir()
        if name=='resumed':shutil.copyfile(out/'split/latest.pt',folder/'latest.pt')
        cmd=[str(python),'-m','training.train','--iterations','12','--traversals','8','--buffer-capacity','2000','--strategy-buffer-capacity','2000',
             '--train-steps','2','--strategy-train-steps','2','--batch-size','64','--workers','1','--device','cpu','--seed','42',
             '--average-sampler',args.average_sampler,'--checkpoint-dir',str(folder),'--checkpoint-every','6','--save-resume-buffers','--no-stack-randomization',
             '--opponent-exploration-start','0','--opponent-exploration-end','0',*extra]
        started=time.monotonic();peak=0
        with (folder/'train.log').open('w') as f:
            p=subprocess.Popen(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                try:
                    status=Path(f'/proc/{p.pid}/status').read_text();peak=max(peak,next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0))
                except FileNotFoundError:pass
                if peak>20*1024**3 or time.monotonic()-started>600:
                    os.killpg(p.pid,signal.SIGKILL);p.wait();raise RuntimeError('Pilot resource limit')
                time.sleep(.1)
        manifest['stages'].append(dict(name=name,command=cmd,returncode=p.returncode,peak_rss_bytes=peak,elapsed_seconds=time.monotonic()-started))
        (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if p.returncode:raise RuntimeError('See '+str(folder/'train.log'))
    # Trusted, small locally produced checkpoint objects.
    import torch
    import numpy as np
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    ck=[torch.load(out/n/'latest.pt',weights_only=False,map_location='cpu') for n in ('continuous','resumed','repeat')]
    assert equal(ck[0],ck[1]),'Split resume differs'
    assert equal(ck[0],ck[2]),'Same seed rerun differs'
    before=torch.load(out/'split/latest.pt',weights_only=False,map_location='cpu')
    assert any(not torch.equal(before['net'][k],ck[0]['net'][k]) for k in before['net'])
    assert all(torch.isfinite(v).all() for v in ck[0]['net'].values())
    assert ck[0]['strategy_buffer_size']>0
    manifest['verified']={'exact_cpu_resume':True,'same_seed_repeat':True,'actual_parameter_update':True,'average_strategy_samples':ck[0]['strategy_buffer_size']}
    manifest['model']=str(out/'continuous/iter_00012.npz')
    (out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out,flush=True)

if __name__=='__main__':main()
