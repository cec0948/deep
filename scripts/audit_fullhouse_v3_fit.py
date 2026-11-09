"""Descriptive fit diagnostics over each saved replay, without extra training."""
import argparse
import json
from pathlib import Path
import sys

from fullhouse_v3_common import ROOT, digest, atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('folder',nargs='?',type=Path)
    p.add_argument('--iterations',nargs='+',type=int,default=[60,120])
    args=p.parse_args()
    folder=(args.folder or Path(json.loads((ROOT/'artifacts/fullhouse-v3/current.json').read_text())['folder'])).resolve()
    state=json.loads((folder/'status.json').read_text());cfg=state['config']
    sys.path.insert(0,str(folder/'source'))
    import numpy as np
    import torch
    from training.network import AverageStrategyNet
    from training.r1_support import strategy_strata,weight_diagnostics
    from train_fullhouse_v3 import verify
    verify(folder,state);torch.set_num_threads(1)
    report=ROOT/'reports/fullhouse-v3'/folder.name
    for seed in cfg['seeds']:
        for iteration in args.iterations:
            path=folder/f'seed-{seed}'/f'stage_{iteration:05d}.full.pt'
            if not path.exists():path=folder/f'seed-{seed}/latest.pt'
            # The main writer replaces latest.pt atomically, never in place.
            with path.open('rb') as f:
                ck=torch.load(f,map_location='cpu',weights_only=False)
            if ck['iteration']!=iteration or ck['r1_config']['seed']!=seed:
                raise ValueError(f'Requested saved replay is unavailable: {seed}/{iteration}')
            if ck['r1_config']['revision']!=cfg['revision']:raise ValueError('Revision mismatch')
            output=report/f'seed-{seed}'/f'fit-{iteration:05d}.json'
            if output.exists():
                # Historical stage files and final latest.pt are immutable at audit time.
                saved=json.loads(output.read_text())
                if saved['checkpoint_sha256']!=digest(path):raise ValueError('Audited checkpoint changed')
                print(seed,iteration,'existing audit verified',flush=True);continue
            b=ck['strategy_buffer'];n=b['count'];x=b['features'][:n];y=b['advantages'][:n]
            w=b['iterations'][:n].astype(float)/cfg['horizon']*b['sample_weights'][:n]
            if not np.isfinite(w).all() or (w<0).any() or not w.max()>0:raise ValueError('Invalid weights')
            if not np.allclose(y.sum(1),1,atol=1e-5) or (y<0).any():raise ValueError('Invalid policy targets')
            net=AverageStrategyNet();net.load_state_dict(ck['strategy_net']);net.eval()
            ce=np.empty(n);kl=np.empty(n);tv=np.empty(n)
            with torch.no_grad():
                for start in range(0,n,4096):
                    end=min(start+4096,n);inputs=torch.from_numpy(x[start:end])
                    pred=net.get_strategy(inputs,inputs[:,46:51]).numpy()
                    if not np.isfinite(pred).all() or not np.allclose(pred.sum(1),1,atol=1e-5):raise ValueError('Invalid network output')
                    target=y[start:end].astype(float)
                    ce[start:end]=-(target*np.log(np.maximum(pred,1e-12))).sum(1)
                    entropy=-(target*np.log(np.maximum(target,1e-12))).sum(1)
                    kl[start:end]=ce[start:end]-entropy
                    tv[start:end]=np.abs(target-pred).sum(1)/2
            groups=strategy_strata(x);strata={};normalized=w/w.max()
            for group in np.unique(groups):
                mask=groups==group;weights=normalized[mask]
                if not weights.sum():continue
                strata[str(int(group))]=dict(rows=int(mask.sum()),
                    original_mass=float(weights.sum()/normalized.sum()),
                    weighted_cross_entropy=float(np.average(ce[mask],weights=weights)),
                    weighted_row_kl=float(np.average(kl[mask],weights=weights)),
                    weighted_row_tv=float(np.average(tv[mask],weights=weights)),
                    weights=weight_diagnostics(w[mask]))
            result=dict(seed=seed,iteration=iteration,rows=n,checkpoint=str(path.relative_to(ROOT)),
                checkpoint_sha256=digest(path),source_hashes=state['source_hashes'],
                global_weights=weight_diagnostics(w),
                original_weighted_row_kl=float(np.average(kl,weights=normalized)),
                stratified_weighted_row_kl=float(np.mean([v['weighted_row_kl'] for v in strata.values()])),
                strata=strata,probability_floor=1e-12,
                limitations=['In-sample fit to the saved replay; not a holdout or strength test',
                             'Row KL includes disagreement among iteration targets even at an identical input',
                             'Different iterations/seeds have different replay targets; their losses are not a controlled fitting comparison',
                             'Stratification reweights states; it does not improve within-stratum collector ESS'])
            output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,result)
            print(seed,iteration,'ESS',result['global_weights']['ess'],'stratified row KL',result['stratified_weighted_row_kl'],flush=True)
            del ck,net,b,x,y


if __name__=='__main__':main()
