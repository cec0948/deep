"""Verify final checkpoints, all pre-update snapshots, and hand-wise selection."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

from fullhouse_v3_common import ROOT,atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();folder=a.folder.resolve()
    cfg=json.loads((folder/'config.json').read_text());manifest=json.loads((folder/'source-manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(folder/name)!=sha:raise ValueError('Frozen source changed')
    sys.path[:0]=[str(folder/'source'),str(folder/'tools')]
    import numpy as np
    import torch
    from compact_regret import ARRAY_MAP
    from fullhouse_v4_runtime import FullhouseV4
    torch.set_num_threads(1);seeds=[]
    for seed in cfg['seeds']:
        path=folder/f'seed-{seed}';ck=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
        if ck['iteration']!=cfg['iterations'] or ck['source_manifest']!=manifest or ck['config']!=cfg:raise ValueError('Checkpoint mismatch')
        r=ck['root_replay'];expected_count=cfg['root_repetitions_first']+(cfg['iterations']-1)*cfg['root_repetitions']
        if len(r['features'])!=169 or not np.all(r['count']==expected_count):raise ValueError('Missing conditional root samples')
        np.testing.assert_allclose(r['mass'],cfg['iterations']*(cfg['iterations']+1)/2,atol=1e-10)
        mixture=json.loads((path/f'mixture_{cfg["iterations"]:05d}.json').read_text())
        if mixture['models']!=ck['snapshots']:raise ValueError('Mixture/checkpoint mismatch')
        if [m['weight'] for m in mixture['models']]!=list(range(1,cfg['iterations']+1)):raise ValueError('Wrong mixture weights')
        for m in mixture['models']:
            if digest(path/m['file'])!=m['sha256']:raise ValueError('Changed snapshot')
        for t in range(cfg['checkpoint_every'],cfg['iterations'],cfg['checkpoint_every']):
            if digest(path/f'current_{t:05d}.npz')!=digest(path/f'policy_{t+1:05d}.npz'):raise ValueError('Policy was saved after, rather than before, its update')
        with np.load(path/f'current_{cfg["iterations"]:05d}.npz') as data:
            for key,name in ARRAY_MAP.items():
                np.testing.assert_array_equal(data[name],ck['net']['main.'+key].numpy())
                np.testing.assert_array_equal(data['opening_'+name],ck['net']['opening.'+key].numpy())
        aa=np.flatnonzero((r['features'][:,10]==1)&(r['features'][:,7]>.99))[0];x=r['features'][aa];average=np.zeros(5)
        for m in mixture['models']:
            policy=FullhouseV4(path/m['file'],folder/'source')
            average+=m['weight']*policy.lookup._model_strategy(x,x[46:51],policy.opening_model)
        average/=sum(m['weight'] for m in mixture['models'])
        seeds.append(dict(seed=seed,iteration=ck['iteration'],root_observations=int(r['count'].sum()),minimum_class_observations=int(r['count'].min()),
                          snapshots=len(mixture['models']),mixture_aa_fold=float(average[0]),checkpoint_sha256=digest(path/'latest.pt')))
    mixture_hands=0
    for result in sorted((folder/'evaluation').glob('*/results.json')):
        record=json.loads(result.read_text())
        if record['candidate_type']!='snapshot_mixture':continue
        spec=json.loads(Path(record['candidate']).read_text());total=sum(m['weight'] for m in spec['models'])
        with gzip.open(result.parent/'hands.jsonl.gz','rt') as stream:
            for line in stream:
                h=json.loads(line);seed=int.from_bytes(hashlib.sha256(f'compact-mixture:{record["deal_seed"]}:{h["deal"]}:{h["hero"]}'.encode()).digest()[:8],'big')
                if seed!=h['hand_seed']:raise ValueError('Wrong independent policy seed')
                draw=random.Random(seed).randrange(total)
                for m in spec['models']:
                    draw-=m['weight']
                    if draw<0:expected=m['iteration'];break
                if h['policy_iteration']!=expected:raise ValueError('Policy changed or was sampled with wrong weights')
                mixture_hands+=1
    result=dict(passed=True,seeds=seeds,verified_mixture_hands=mixture_hands,verifier_sha256=digest(Path(__file__)))
    atomic_json(folder/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
