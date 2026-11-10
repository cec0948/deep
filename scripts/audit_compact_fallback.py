"""Describe all-negative regret predictions on a saved replay; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from fullhouse_v3_common import ROOT,atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    folder=a.folder.resolve();cfg=json.loads((folder/'config.json').read_text());sys.path[:0]=[str(folder/'source'),str(folder/'tools')]
    import numpy as np
    import torch
    from compact_regret import CompactRegretNet,BBCompactPredictions
    torch.set_num_threads(1);rows=[]
    for seed in cfg['seeds']:
        checkpoint=folder/f'seed-{seed}/latest.pt'
        with checkpoint.open('rb') as stream:
            sha=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
            ck=torch.load(stream,map_location='cpu',weights_only=False)
        net=CompactRegretNet();net.load_state_dict(ck['net']);net.eval()
        x=np.concatenate([b['features'] for b in ck['buffers']]);y=np.concatenate([b['advantages'] for b in ck['buffers']]);w=np.concatenate([b['iterations'] for b in ck['buffers']])
        street=np.argmax(x[:,17:21],1);rng=np.random.default_rng(1903);stats=[]
        for s in range(4):
            ids=np.flatnonzero(street==s);ids=rng.choice(ids,min(2048,len(ids)),replace=False)
            inputs=torch.from_numpy(x[ids]);legal=inputs[:,46:51]
            with torch.no_grad():score=net(inputs,legal).numpy();pred=BBCompactPredictions(net)(inputs,legal).numpy()
            no_positive=np.max(np.where(legal.numpy()>0,score,-np.inf),1)<=0
            maximum=np.max(np.where(legal.numpy()>0,score,-np.inf),1)
            best=np.argmax(np.where(legal.numpy()>0,score,-np.inf),1)
            mass=w[ids].astype(float)
            stats.append(dict(street=s,sampled_rows=len(ids),weighted_all_negative_fraction=float(np.average(no_positive,weights=mass)),
                              near_zero_max_fraction=float(np.mean(np.abs(maximum)<1e-6)),
                              best_negative_action_counts=np.bincount(best[no_positive],minlength=5).tolist(),
                              mse=float(np.average(np.square(pred-y[ids]).mean(1),weights=mass)),
                              zero_mse=float(np.average(np.square(y[ids]).mean(1),weights=mass))))
        rows.append(dict(seed=seed,iteration=ck['iteration'],checkpoint_sha256=sha,streets=stats))
    result=dict(rows=rows,limitation='Training replay distribution and finite labels, not a held-out game-strength test')
    atomic_json(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
