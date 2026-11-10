"""Independent conditional-root samples under a frozen six-player policy.

Measures label sensitivity, not policy strength or convergence. No model update.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from fullhouse_v3_common import ROOT,atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=32);args=p.parse_args()
    if args.repetitions<4:p.error('At least four repetitions required')
    source=ROOT/'artifacts/fullhouse-v3/20261109T034714104421Z/source';sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from training.network import DuelingAdvantageNet
    from root_regret_replay import RootReplay,collect_root_replay,hand_classes,card_pairs
    torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=False)
    checkpoint=source.parent/'seed-42/latest.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=DuelingAdvantageNet();net.load_state_dict(ck['net']);net.eval();del ck
    report=dict(complete=False,checkpoint_sha256=digest(checkpoint),script_sha256=digest(Path(__file__)),
                collector_sha256=digest(ROOT/'scripts/root_regret_replay.py'),repetitions=args.repetitions,seeds=[1701,1702],
                limitations=['Frozen last-iteration policy, not cumulative-regret training','Natural card-class weighted TV and uniform-class TV are sensitivity diagnostics, not confidence intervals or poker strength'],streams=[])
    samples=[];started=time.monotonic()
    for seed in report['seeds']:
        rng=np.random.default_rng(seed);replay=RootReplay();previous=np.zeros((169,5));rows=[]
        for repetition in range(args.repetitions):
            collect_root_replay(net,rng,replay,1,1)
            state=replay.state_dict()
            if len(state['features'])!=169 or not np.all(state['count']==repetition+1):raise AssertionError('Incomplete balanced class coverage')
            rows.append(state['numerator']-previous);previous=state['numerator'].copy()
            if (repetition+1)%4==0:print(seed,'repetitions',repetition+1,'seconds',round(time.monotonic()-started,1),flush=True)
        samples.append(np.asarray(rows));features=state['features']
        report['streams'].append(dict(seed=seed,**replay.diagnostics()))
        atomic_json(args.output/'results.json',report)
    def policy(target):
        positive=np.maximum(target,0)*features[:,46:51]
        mass=positive.sum(1,keepdims=True)
        legal=features[:,46:51].astype(float)
        return np.divide(positive,mass,out=legal/legal.sum(1,keepdims=True),where=mass>0)
    weights=np.array([len(card_pairs(h)) for h in hand_classes()],float);weights/=weights.sum()
    aa=int(np.flatnonzero((features[:,10]==1)&(features[:,7]>.99))[0])
    report['sensitivity']=[]
    for n in sorted(set([4,16,args.repetitions])):
        if n>args.repetitions:continue
        targets=[s[:n].mean(0) for s in samples];policies=[policy(t) for t in targets]
        tv=np.abs(policies[0]-policies[1]).sum(1)/2
        report['sensitivity'].append(dict(observations_per_class=n,natural_weighted_tv=float(tv@weights),uniform_tv=float(tv.mean()),
                                           aa_regrets=[t[aa].tolist() for t in targets],aa_fold=[float(p[aa,0]) for p in policies]))
    np.savez_compressed(args.output/'samples.npz',features=features,stream_1701=samples[0],stream_1702=samples[1])
    report['samples_sha256']=digest(args.output/'samples.npz');report['complete']=True;report['seconds']=time.monotonic()-started
    atomic_json(args.output/'results.json',report);print(json.dumps(report['sensitivity']),flush=True)


if __name__=='__main__':main()
