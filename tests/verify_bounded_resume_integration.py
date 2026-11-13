"""Small fixture: one update after a legacy checkpoint equals compact recovery.

This never reads or trains the user's production240 candidate. Its private
fixture and comparison proof are written under the supplied new audit folder.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from compact_lightweight_backup import equal_state
from fullhouse_v3_common import PYTHON,atomic_json,digest
from prepare_compact_run import prepare
from train_compact_bounded import commit_checkpoint,load_current


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'artifacts/continuation-study/20261112T070916841250Z/campaign/20261112T070916841250Z'
    cfg=json.loads((parent/'config.json').read_text())
    cfg.update(verification_fixture=True,seeds=[42],iterations=12,stages=[6,12],traversals=8,
        buffer_capacity=3000,main_steps=8,batch_size=64,opening_steps=8,opening_max_steps=8,
        root_repetitions=1,root_repetitions_first=1,checkpoint_every=6,seconds_per_training_worker=300)
    legacy=prepare(output/'legacy',cfg)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def worker(script,folder,through,resume=False):
        command=[str(PYTHON),str(ROOT/'scripts'/script),'--run-dir',str(folder),'--seed','42','--through',str(through)]
        if resume:command.append('--resume')
        with (output/f'{script}-{through}.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    worker('train_compact_selfplay.py',legacy,6)
    before=torch.load(legacy/'seed-42/latest.pt',map_location='cpu',weights_only=False)
    bounded=output/'bounded';bounded.mkdir()
    for name in ['source','tools']:shutil.copytree(legacy/name,bounded/name)
    newcfg=dict(cfg,initial_iteration=6,stages=[7],evaluation_iterations=[6,7],deployment_average='current_only_no_history')
    manifest=json.loads((legacy/'source-manifest.json').read_text());manifest['driver_sha256']=digest(ROOT/'scripts/train_compact_bounded.py')
    atomic_json(bounded/'config.json',newcfg);atomic_json(bounded/'source-manifest.json',manifest)
    before.pop('snapshots');state=dict(before,config=newcfg,source_manifest=manifest)
    folder=bounded/'seed-42';folder.mkdir();(folder/'worker.lock').touch();(folder/'diagnostics.jsonl').touch()
    commit_checkpoint(bounded,42,state,lambda p:shutil.copyfile(legacy/'seed-42/current_00006.npz',p),{6,7})
    atomic_json(folder/'status.json',dict(status='complete',iteration=6,through=6))
    worker('train_compact_selfplay.py',legacy,7,True);worker('train_compact_bounded.py',bounded,7)
    expected=torch.load(legacy/'seed-42/latest.pt',map_location='cpu',weights_only=False);actual,_=load_current(bounded,42)
    for key in ['net','buffers','root_replay','rng']:
        if not equal_state(expected[key],actual[key]):raise AssertionError('Next update differs: '+key)
    with np.load(legacy/'seed-42/current_00007.npz',allow_pickle=False) as a,np.load(bounded/'seed-42/current_00007.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    assert not list(folder.glob('policy_*.npz')) and not list(folder.glob('mixture_*.json'))
    proof=dict(passed=True,continuation_equivalence_passed=True,verification_fixture=True,
        production_candidates_trained=False,seed=42,initial_iteration=6,through_iteration=7,
        next_network_parameters_identical=True,next_replay_identical=True,next_root_statistics_identical=True,
        next_rng_identical=True,export_arrays_identical=True,no_historical_policy_files=True,
        bounded_checkpoint_directories=len(list((folder/'checkpoints').glob('iter-*'))))
    atomic_json(output/'result.json',proof);print(json.dumps(proof,indent=2));return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
