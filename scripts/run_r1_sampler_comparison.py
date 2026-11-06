"""Compare fixed-opponent sampling to the recorded equal-iteration R1 baseline."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
from fullhouse_runtime import ROOT
from build_fullhouse_r1 import build
from run_r1_learning_curve import PYTHON,run

BASELINE=ROOT/'artifacts/r1-learning-curve/20261106T015348055280Z'

def main():
    source=build();old=json.loads((BASELINE/'results.json').read_text());assert old['complete']
    out=ROOT/'artifacts/r1-sampler'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True);print(out,flush=True)
    state=dict(complete=False,baseline_run=str(BASELINE),seeds=[42,43,44],training=[],evaluations=[],deal_seed=88110600,
        baseline_manifest_sha256=hashlib.sha256((BASELINE/'results.json').read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [ROOT/'patches/fullhouse-r1.patch',ROOT/'native/fullhouse-r1/support.py',Path(__file__),ROOT/'scripts/evaluate_fullhouse.py']},strength_verified=False)
    def save():(out/'results.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    for seed in state['seeds']:
        folder=out/f'seed-{seed}';folder.mkdir()
        # Same iterations, hand trajectories, replay capacities and optimizer budget.
        cmd=[PYTHON,'-m','training.train','--iterations',100,'--traversals',32,'--buffer-capacity',20000,
            '--strategy-buffer-capacity',20000,'--train-steps',16,'--strategy-train-steps',16,'--batch-size',128,
            '--workers',1,'--device','cpu','--seed',seed,'--checkpoint-dir',folder,'--checkpoint-every',25,
            '--save-resume-buffers','--no-stack-randomization','--opponent-exploration-start',0,'--opponent-exploration-end',0,
            '--average-sampler','fixed-opponent']
        result=run(cmd,source,folder/'train.log',600);result['seed']=seed
        state['training'].append(result);save();print('Trained',seed,result['elapsed_seconds'],flush=True)
    candidates=[('public',None)]
    for seed in state['seeds']:
        candidates.extend([(f'importance-{seed}',BASELINE/f'seed-{seed}/iter_00100.npz'),(f'fixed-{seed}',out/f'seed-{seed}/iter_00100.npz')])
    blocks={}
    for name,model in candidates:
        cmd=[PYTHON,ROOT/'scripts/evaluate_fullhouse.py','--deals',100,'--deal-seed',state['deal_seed'],'--arms','fullhouse','--timeout',180]
        if model:cmd+=['--model',model]
        result=run(cmd,ROOT,out/f'eval-{name}.log',200)
        directory=Path((out/f'eval-{name}.log').read_text().strip().splitlines()[-1]);evaluation=json.loads((directory/'results.json').read_text());assert evaluation['complete']
        rows=[json.loads(x) for x in (directory/'hands.jsonl').read_text().splitlines()]
        assert len(rows)==600 and len({(r['deal'],r['hero']) for r in rows})==600
        assert all(sum(r['utilities'])==0 for r in rows)
        blocks[name]=[statistics.mean(r['chips'] for r in rows if r['deal']==d) for d in range(100)]
        result.update(name=name,directory=str(directory),summary=evaluation['summary']['fullhouse'])
        state['evaluations'].append(result);save();print('Evaluated',name,result['summary'],flush=True)
    state['paired_fixed_minus_importance']={}
    for seed in state['seeds']:
        diffs=[a-b for a,b in zip(blocks[f'fixed-{seed}'],blocks[f'importance-{seed}'])]
        state['paired_fixed_minus_importance'][str(seed)]=dict(bb100=statistics.mean(diffs),ci95_half_width=1.96*statistics.stdev(diffs)/10)
    state['complete']=True;save();print(out,flush=True)

if __name__=='__main__':main()
