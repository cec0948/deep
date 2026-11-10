"""Recompute every completed evaluation from raw hands and paired deal blocks."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import statistics

from fullhouse_v3_common import ROOT,atomic_json,digest


def interval(values):
    return dict(mean=statistics.mean(values),ci95_half_width=1.96*statistics.stdev(values)/len(values)**.5)


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
    cfg=json.loads((folder/'config.json').read_text());output=ROOT/'reports/compact-v4'/folder.name;output.mkdir(parents=True,exist_ok=True)
    rows={};hands=0
    for path in sorted((folder/'evaluation').glob('*/results.json')):
        r=json.loads(path.read_text());blocks=[[None]*6 for _ in range(r['deals'])];count=0
        raw=path.parent/'hands.jsonl.gz'
        with gzip.open(raw,'rt') as stream:
            for line in stream:
                h=json.loads(line);deal=h['deal'];hero=h['hero']
                if not 0<=deal<r['deals'] or not 0<=hero<6 or blocks[deal][hero] is not None:raise ValueError('Duplicate or out-of-range hand')
                if h['seed']!=cfg['evaluation_deal_seed']+deal or sum(h['utilities'])!=0 or h['chips']!=h['utilities'][hero]:raise ValueError('Invalid hand payoff or seed')
                blocks[deal][hero]=h['chips'];count+=1
        if count!=r['hands'] or any(None in b for b in blocks):raise ValueError('Incomplete evaluation')
        means=[statistics.mean(b) for b in blocks]
        if len(means)!=len(r['deal_blocks']) or any(abs(x-y)>1e-9 for x,y in zip(means,r['deal_blocks'])):raise ValueError('Block mismatch')
        metrics=interval(means)
        if abs(metrics['mean']-r['bb_per_100'])>1e-9 or abs(metrics['ci95_half_width']-r['ci95_half_width'])>1e-9:raise ValueError('Score/CI mismatch')
        if digest(Path(r['candidate']))!=r['candidate_sha256']:raise ValueError('Evaluated model changed')
        key=path.parent.name;rows[key]=r;hands+=count
        dest=output/'evaluation'/key;dest.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest/'results.json');shutil.copyfile(raw,dest/raw.name)
    comparisons={}
    for seed in cfg['seeds']:
        for kind in ('current','mixture'):
            for field in ('noregrets','public'):
                lo=f'{field}-seed-{seed}-{kind}-30';hi=f'{field}-seed-{seed}-{kind}-60'
                if lo in rows and hi in rows:comparisons[f'{field}-seed-{seed}-{kind}-60-minus-30']=interval([x-y for x,y in zip(rows[hi]['deal_blocks'],rows[lo]['deal_blocks'])])
        for field in cfg['evaluation_fields']:
            baseline=f'{field}-old-v3-adv42'
            for kind in ('current','mixture'):
                candidate=f'{field}-seed-{seed}-{kind}-60'
                if baseline in rows and candidate in rows:comparisons[candidate+'-minus-old-v3']=interval([x-y for x,y in zip(rows[candidate]['deal_blocks'],rows[baseline]['deal_blocks'])])
    scores={name:{k:r[k] for k in ('bb_per_100','ci95_half_width','hands','candidate_type','candidate_sha256','elapsed_seconds','peak_sampled_rss_bytes')} for name,r in rows.items()}
    expected=36
    summary=dict(complete=len(rows)==expected,completed_evaluations=len(rows),planned_evaluations=expected,total_hands=hands,
                 scores=scores,paired_comparisons=comparisons,config=cfg,summarizer_sha256=digest(Path(__file__)),
                 limitations=['Nominal deal-block intervals; no adjustment for multiple comparisons','Fixed trained models and two-model opponent pool',
                              'Three learning seeds are not independent card evaluation sets','No professional strength or automatic promotion'])
    atomic_json(output/'summary.json',summary);shutil.copyfile(folder/'config.json',output/'config.json');shutil.copyfile(folder/'source-manifest.json',output/'source-manifest.json')
    print(json.dumps(dict(complete=summary['complete'],evaluations=len(rows),hands=hands,scores=scores,paired_comparisons=comparisons),indent=2))


if __name__=='__main__':main()
