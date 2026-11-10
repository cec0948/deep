"""Deployment mixture semantics and complete deterministic training resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def test_hand_mixture_matches_own_reach_average():
    from kuhn_validation import own_reach,policy_value,OWNER
    rng=np.random.default_rng(717)
    profiles=[rng.dirichlet([1,1],size=12) for _ in range(3)]
    weights=np.array([1,2,3],float);weights/=weights.sum()
    mass=sum(w*own_reach(p) for w,p in zip(weights,profiles))
    average=sum(w*own_reach(p)[:,None]*p for w,p in zip(weights,profiles))/mass[:,None]
    fixed=rng.dirichlet([1,1],size=12)
    def player0(p):
        p=p.copy();p[OWNER==1]=fixed[OWNER==1];return policy_value(p)
    expected=sum(w*player0(p) for w,p in zip(weights,profiles))
    assert abs(expected-player0(average))<1e-12
    arithmetic=sum(w*p for w,p in zip(weights,profiles))
    assert abs(player0(arithmetic)-expected)>.001


def test_runtime_mixture_selects_by_time_and_keeps_one_policy_per_hand():
    import hashlib
    from unittest.mock import patch
    from fullhouse_v4_runtime import SnapshotMixture
    class Policy:
        def __init__(self,path,source):self.path=path.name
        def decide(self,state,seed):return dict(path=self.path)
    class Draw:
        def __init__(self,seed):self.seed=seed
        def randrange(self,total):assert total==6;return self.seed
    with tempfile.TemporaryDirectory() as name:
        folder=Path(name);models=[]
        for t in (1,2,3):
            path=folder/f'policy_{t}.npz';path.write_bytes(bytes([t]))
            models.append(dict(iteration=t,weight=t,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest=folder/'mixture.json';manifest.write_text(json.dumps(dict(model_type='compact_snapshot_mixture',models=models)))
        mix=SnapshotMixture(manifest)
        with patch('fullhouse_v4_runtime.FullhouseV4',Policy),patch('fullhouse_v4_runtime.random.Random',Draw):
            selected=[]
            for seed in range(6):
                mix.begin_hand(seed);selected.append(mix.active_iteration)
                first=mix.decide({'cards':['Ac','Ad']},1)
                second=mix.decide({'cards':['2c','7d']},999)
                assert first['mixture_iteration']==second['mixture_iteration']
            assert selected==[1,2,2,3,3,3]


def test_continuous_and_resumed_compact_training_match():
    from prepare_compact_run import prepare
    import torch
    cfg=json.loads((ROOT/'configs/fullhouse-compact-training.json').read_text())
    cfg.update(verification_fixture=True,seeds=[42],iterations=12,traversals=4,buffer_capacity=3000,
               main_steps=4,opening_steps=4,opening_max_steps=4,root_repetitions_first=1,root_repetitions=1,checkpoint_every=6)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts',prefix='compact-resume-') as name:
        folder=Path(name);continuous=prepare(folder/'continuous',cfg);split=prepare(folder/'split',cfg)
        def run(path,*extra):
            p=subprocess.run([sys.executable,'scripts/train_compact_selfplay.py','--run-dir',str(path),'--seed','42',*extra],cwd=ROOT,
                             env=env,capture_output=True,text=True,timeout=180)
            assert p.returncode==0,p.stdout+'\n'+p.stderr
        run(continuous);run(split,'--through','6');run(split,'--resume')
        a=torch.load(continuous/'seed-42/latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(split/'seed-42/latest.pt',map_location='cpu',weights_only=False)
        def equal(a,b):
            if torch.is_tensor(a):assert torch.equal(a,b)
            elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a:equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b):equal(x,y)
            else:assert a==b
        equal(a,b)
        mixture=json.loads((split/'seed-42/mixture_00012.json').read_text())
        assert [r['weight'] for r in mixture['models']]==list(range(1,13))
        with np.load(split/'seed-42/policy_00001.npz') as first:
            assert np.all(first['trunk_w0']==0) and np.all(first['opening_trunk_w0']==0)
        assert min(a['root_replay']['count'])==12
