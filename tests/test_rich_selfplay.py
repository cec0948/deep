from pathlib import Path
import json
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compact_lightweight_backup import equal_state
from fullhouse_v3_common import digest
from train_rich_selfplay import load_current,run,validate_config,validate_fitting_report
from rich_poker_observation import RICH_DIM

SOURCE=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/source'
TEACHER=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/seed-342/current_00360.npz'


def configuration():
    return dict(verification_fixture=True,seeds=[342],stages=[1,2],iterations=2,
        traversals=1,buffer_capacity=64,main_steps=2,batch_size=16,learning_rate=.001,
        checkpoint_every=1,seconds_per_training_worker=60,width=16,source=str(SOURCE),
        teachers={'342':dict(path=str(TEACHER),sha256=digest(TEACHER))},
        frozen_files={str(TEACHER):digest(TEACHER)},parent_iteration=360)


def prepare(folder,config):
    folder.mkdir();(folder/'config.json').write_text(json.dumps(config))


def test_new_family_checkpoint_resume_preserves_next_model_replay_and_rng(tmp_path):
    if not TEACHER.exists():pytest.skip('Requires existing360teacher asset')
    config=configuration();whole=tmp_path/'whole';split=tmp_path/'split'
    prepare(whole,config);prepare(split,config)
    assert run(whole,342,2)['complete']
    assert run(split,342,1)['complete']
    assert run(split,342,2)['complete']
    a,path=load_current(whole/'seed-342',config,342,RICH_DIM)
    b,_=load_current(split/'seed-342',config,342,RICH_DIM)
    assert equal_state(a,b)
    assert a['iteration']==2 and a['parent_iteration']==360
    assert a['teacher_knowledge_inherited'] and not a['parent_rng_inherited']
    assert not a['legacy_exact_resume'] and not a['average_policy_available']
    proof=json.loads((split/'seed-342/warm-start-verification.json').read_text())
    assert proof['passed'] and proof['teacher_probability_byte_identical'] and proof['samples']>=32
    assert (split/'seed-342/policy-current.json').exists()
    descriptor=json.loads((split/'seed-342/policy-current.json').read_text())
    assert descriptor['iteration']==2 and descriptor['format']=='rich-policy-export-v1'
    assert len(list((split/'seed-342/checkpoints').glob('iter-*')))==3
    (path/'current.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError):load_current(whole/'seed-342',config,342,RICH_DIM)


def test_production_budget_cannot_be_silently_extended():
    if not TEACHER.exists():pytest.skip('Requires existing360teacher asset')
    cfg=configuration();cfg['verification_fixture']=False
    with pytest.raises(ValueError,match='predeclared'):validate_config(cfg,342,1)


def test_fitting_gate_uses_initial_teacher_not_monotonic_previous_model():
    cfg={'fitting_max_prior_mse_ratio':1.05}
    validate_fitting_report(dict(eligible_rows=5,steps=256,teacher_prior_mse_bb2=10.,
                                  replay_mse_after_bb2=10.4,replay_mse_before_bb2=2.),cfg)
    with pytest.raises(ValueError,match='consistency failed'):
        validate_fitting_report(dict(eligible_rows=5,steps=256,teacher_prior_mse_bb2=10.,
                                    replay_mse_after_bb2=10.6),cfg)
    with pytest.raises(ValueError):
        validate_fitting_report(dict(eligible_rows=5,steps=256,teacher_prior_mse_bb2=10.,
                                    replay_mse_after_bb2=float('nan')),cfg)
