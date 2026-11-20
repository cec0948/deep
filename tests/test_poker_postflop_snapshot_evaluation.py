from collections import Counter
from contextlib import closing
import gzip
import json
from pathlib import Path
import random
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_poker_postflop_snapshots as study
from fullhouse_v3_common import atomic_json, digest


def test_public_comparator_does_not_add_an_opponent_family_or_duplicate_cards():
    cfg, fields, jobs = study.fixed_design()
    assert len(jobs) == len({j['id'] for j in jobs}) == 75
    assert sum(j['label']==study.EXTRA for j in jobs) == 3
    assert all(j['field']=='public' for j in jobs if j['label']==study.EXTRA)
    assert fields['public']['deals'] == 20000 and fields['varied']['role']=='holdout'
    assert sum(fields[j['field']]['deals']*6 for j in jobs)==2196000
    for f in ('calling-station','tight-value','pressure'):
        assert fields[f]['family']=='rules'


def synthetic_mixture_of_identical_postflop_models(dest, parent, source):
    """Pin four copies of the genuine parent postflop model for a parity fixture."""
    dest.mkdir()
    iterations = [486,492,498,504]
    pre = parent['preflop']
    shutil.copyfile(pre['path'],dest/'preflop.npz')
    atomic_json(dest/'plan.json',dict(format=study.snapshots.FORMAT,seed=342,arm='parity',
        iterations=iterations,preflop_sha256=pre['sha256'],source_features_sha256=digest(Path(source)/'bot/features.py')))
    for i in iterations:
        slot=dest/f'iter-{i:05d}'
        slot.mkdir()
        model=slot/'current.npz'
        shutil.copyfile(parent['postflop']['path'],model)
        atomic_json(slot/'receipt.json',dict(iteration=i,plan_sha256=digest(dest/'plan.json'),
            export_sha256=parent['postflop']['sha256'],export_bytes=model.stat().st_size,
            start_sha256='test',worker_pid=123,worker_start_token='test-run'))
    study.snapshots.finalize(dest)
    return dict(path=str(dest/'mixture.json'),sha256=digest(dest/'mixture.json'))


@pytest.mark.skipif(not study.panel.DOSE_PLAN.is_file(),reason='Frozen source environment is required for absolute trace parity')
def test_one_private_draw_and_terminal_replay_match_genuine_fixed_policy(tmp_path,monkeypatch):
    source_plan=study.read(study.panel.DOSE_PLAN)
    ref=source_plan['policies']['steps0']['342']
    source=source_plan['source']
    mixture=synthetic_mixture_of_identical_postflop_models(tmp_path/'mixture',study.read(ref['path']),source)
    out=tmp_path/'population'
    out.mkdir()
    (out/'evaluation').mkdir()
    monkeypatch.setattr(study,'OUTPUT',out)
    field='public'
    plan=dict(fields={field:dict(deal_seed=2037338220,deals=2)},source=source,
        policies={study.BASE:{'342':ref},study.MIX:{'342':mixture}},frozen_files={str(study.SCRIPT):digest(study.SCRIPT)})
    atomic_json(out/'evaluation-plan.json',plan)
    _,strategies,opps,op_model,deps=study.panel.frozen_inputs('steps0',342,field)
    traces={}
    for label in (study.BASE,study.MIX):
        job=dict(field=field,seed=342,label=label,id=label)
        with study.ready_engine(op_model,out) as eng:
            study.one_cell(job,plan,eng,strategies,opps,deps)
        cell=out/'evaluation'/label
        r=study.baseline_eval.verify_results(cell,replay=True)
        assert r['verification']['hands']==12 and r['verification']['terminal_payoffs_replayed']
        with gzip.open(cell/'hands.jsonl.gz','rt') as stream:
            traces[label]=[json.loads(line) for line in stream]
    observed=Counter()
    for a,b in zip(traces[study.BASE],traces[study.MIX],strict=True):
        coin=random.Random(b'scheduled-postflop-choice-v1\0'+a['hand_seed'].to_bytes(8,'big'))
        iteration=(486,492,498,504)[coin.randrange(4)]
        assert b.pop('snapshot_iteration') == iteration
        assert a == b
        observed[str(iteration)] += 1
    assert r['snapshot_hand_counts']==observed


def test_nominal_interval_pairs_before_averaging():
    result=study.nominal(study.np.full(20,3.))
    assert result==dict(mean=3.,paired_ci95=[3.,3.])
