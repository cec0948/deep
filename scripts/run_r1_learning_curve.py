"""Bounded three-seed CPU learning curve with common-deal evaluations."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
from fullhouse_runtime import ROOT
from build_fullhouse_r1 import build

PYTHON=ROOT.parent/'.venvs/fullhouse/bin/python'

def run(cmd,cwd,log,timeout):
    start=time.monotonic();peak=0
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    with log.open('w') as f:
        p=subprocess.Popen(list(map(str,cmd)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                try:
                    status=Path(f'/proc/{p.pid}/status').read_text()
                    peak=max(peak,next((int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')),0))
                except FileNotFoundError:pass
                if peak>20*1024**3 or time.monotonic()-start>timeout:raise TimeoutError(str(log))
                time.sleep(.2)
        finally:
            if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
    result=dict(command=list(map(str,cmd)),returncode=p.returncode,elapsed_seconds=time.monotonic()-start,peak_parent_rss_bytes=peak,log=str(log))
    if p.returncode:raise RuntimeError('See '+str(log))
    return result

def main():
    source=build()
    out=ROOT/'artifacts/r1-learning-curve'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    print(out,flush=True)
    state=dict(training=[],evaluations=[],complete=False,seeds=[42,43,44],evaluation_deal_seed=77110600,
        patch_sha256=hashlib.sha256((ROOT/'patches/fullhouse-r1.patch').read_bytes()).hexdigest(),
        support_sha256=hashlib.sha256((ROOT/'native/fullhouse-r1/support.py').read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),strength_verified=False)
    def save():(out/'results.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for seed in state['seeds']:
        folder=out/f'seed-{seed}';folder.mkdir()
        cmd=[PYTHON,'-m','training.train','--iterations',100,'--traversals',32,'--buffer-capacity',20000,
             '--strategy-buffer-capacity',20000,'--train-steps',16,'--strategy-train-steps',16,'--batch-size',128,
             '--workers',1,'--device','cpu','--seed',seed,'--checkpoint-dir',folder,'--checkpoint-every',25,
             '--save-resume-buffers','--no-stack-randomization','--opponent-exploration-start',0,'--opponent-exploration-end',0]
        result=run(cmd,source,folder/'train.log',600);result['seed']=seed
        state['training'].append(result);save();print(f'Trained seed {seed}: {result["elapsed_seconds"]:.1f}s',flush=True)
    candidates=[('public',None),('r1-12',ROOT/'artifacts/r1-training/20261106T014513666340Z/continuous/iter_00012.npz')]
    candidates += [(f'seed-{seed}-iter-{iteration}',out/f'seed-{seed}'/f'iter_{iteration:05d}.npz') for seed in state['seeds'] for iteration in (25,50,100)]
    baseline=None
    for name,model in candidates:
        cmd=[PYTHON,ROOT/'scripts/evaluate_fullhouse.py','--deals',100,'--deal-seed',state['evaluation_deal_seed'],'--arms','fullhouse','--timeout',180]
        if model:cmd+=['--model',model]
        result=run(cmd,ROOT,out/f'eval-{name}.log',200)
        directory=Path((out/f'eval-{name}.log').read_text().strip().splitlines()[-1])
        evaluation=json.loads((directory/'results.json').read_text());assert evaluation['complete']
        rows=[json.loads(x) for x in (directory/'hands.jsonl').read_text().splitlines()]
        blocks={deal:statistics.mean(r['chips'] for r in rows if r['deal']==deal) for deal in range(100)}
        if baseline is None:baseline=blocks
        diffs=[blocks[d]-baseline[d] for d in blocks]
        result.update(name=name,directory=str(directory),summary=evaluation['summary']['fullhouse'],
            paired_delta_vs_public_bb100=statistics.mean(diffs),paired_delta_ci95_half_width=1.96*statistics.stdev(diffs)/10)
        state['evaluations'].append(result);save();print(f'Evaluated {name}: {result["summary"]}',flush=True)
    state['complete']=True;save();print(out,flush=True)

if __name__=='__main__':main()
