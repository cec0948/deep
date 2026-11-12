"""Continuation must preserve learned state, time weights and paired precision."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def assert_same(a,b):
    if torch.is_tensor(a):
        assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:
        assert a==b


def test_forked_training_matches_continuous_and_keeps_parent():
    from extend_compact_run import extend
    from prepare_compact_run import prepare
    from fullhouse_v3_common import digest
    cfg=json.loads((ROOT/'configs/fullhouse-compact-training.json').read_text())
    cfg.update(verification_fixture=True,seeds=[42],iterations=6,traversals=4,buffer_capacity=3000,
               main_steps=4,opening_steps=4,opening_max_steps=4,root_repetitions_first=1,
               root_repetitions=1,checkpoint_every=6)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(folder,*args):
        result=subprocess.run([sys.executable,'scripts/train_compact_selfplay.py','--run-dir',str(folder),
                               '--seed','42',*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts',prefix='compact-extension-test-') as name:
        folder=Path(name);parent=prepare(folder/'parent',cfg);run(parent)
        (parent/'launch.json').write_text(json.dumps(dict(status='complete')))
        old={str(p.relative_to(parent)):digest(p) for p in parent.rglob('*') if p.is_file()}
        extended=extend(parent,folder/'extended',12,190261110)
        assert old=={str(p.relative_to(parent)):digest(p) for p in parent.rglob('*') if p.is_file()}
        before=torch.load(parent/'seed-42/latest.pt',map_location='cpu',weights_only=False)
        copied=torch.load(extended/'seed-42/latest.pt',map_location='cpu',weights_only=False)
        for key in ('net','buffers','root_replay','rng','snapshots'):
            assert_same(before[key],copied[key])
        run(extended,'--resume')
        continuous=prepare(folder/'continuous',dict(cfg,iterations=12));run(continuous)
        a=torch.load(continuous/'seed-42/latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(extended/'seed-42/latest.pt',map_location='cpu',weights_only=False)
        for key in ('net','buffers','root_replay','rng','snapshots','iteration','source_manifest'):
            assert_same(a[key],b[key])
        assert [m['weight'] for m in b['snapshots']]==list(range(1,13))
        assert old=={str(p.relative_to(parent)):digest(p) for p in parent.rglob('*') if p.is_file()}


def test_evaluation_plan_covers_both_stages_and_all_fields():
    from run_compact_evaluation import evaluation_jobs
    cfg=json.loads((ROOT/'configs/fullhouse-compact-training.json').read_text())
    assert len(evaluation_jobs(ROOT/'artifacts/example',cfg))==36
    cfg.update(iterations=120,evaluation_iterations=[60,120],evaluation_all_stages_fields=True)
    jobs=evaluation_jobs(ROOT/'artifacts/example',cfg)
    assert len(jobs)==len({j['id'] for j in jobs})==42
    assert len([j for j in jobs if j.get('iteration')==60 and j['field']=='mixed'])==6


def test_retraining_without_lost_baseline_preserves_all_paired_fields():
    from run_compact_evaluation import evaluation_jobs
    cfg=json.loads((ROOT/'configs/fullhouse-compact-training.json').read_text())
    cfg.update(evaluation_baselines=['public'],evaluation_all_stages_fields=True)
    jobs=evaluation_jobs(ROOT/'artifacts/example',cfg)
    assert len(jobs)==len({j['id'] for j in jobs})==39
    assert not any('old-v3' in j['id'] for j in jobs)
    for iteration in (30,60):
        for seed in cfg['seeds']:
            for kind in ('current','mixture'):
                assert {j['field'] for j in jobs if j.get('iteration')==iteration and j.get('seed')==seed and j.get('kind')==kind}==set(cfg['evaluation_fields'])


def test_paired_progress_does_not_treat_shared_cards_as_independent_seeds():
    from compact_progress import progress
    cfg=dict(evaluation_iterations=[60,120],evaluation_fields=['noregrets','public','mixed'],
             evaluation_deals=4,seeds=[342,343,344])
    rows={};differences=[0.,100.,0.,100.]
    for field in cfg['evaluation_fields']:
        for kind in ('current','mixture'):
            for seed in cfg['seeds']:
                rows[f'{field}-seed-{seed}-{kind}-60']=dict(deal_blocks=[0.]*4)
                rows[f'{field}-seed-{seed}-{kind}-120']=dict(deal_blocks=differences)
    result=progress(rows,cfg)
    expected=1.96*statistics.stdev(differences)/2
    assert abs(result['changes']['public/current']['ci95_half_width']-expected)<1e-12
    assert result['recommendation']=='pause_and_analyze'


def test_automatic_extensions_keep_the_predeclared_stage_guards():
    from continue_compact_learning import extension_allowed
    decision={'conditions':dict(public_lower_above_zero=True,at_least_two_public_seeds_improve=True,
                               mixed_lower_above_minus_25=False,noregrets_lower_above_minus_50=False)}
    assert extension_allowed(60,decision)
    assert not extension_allowed(120,decision)
    decision['conditions'].update(mixed_lower_above_minus_25=True,noregrets_lower_above_minus_50=True)
    assert extension_allowed(120,decision)
    decision['conditions']['at_least_two_public_seeds_improve']=False
    assert not extension_allowed(60,decision)
    assert not extension_allowed(180,decision)
