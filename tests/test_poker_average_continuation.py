import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_poker_average_continuation import validate_config,summarize,jobs,LABELS,SEEDS,FIELDS
from train_poker_average_continuation import record_average_quality


def test_projection_error_is_reported_but_not_a_strength_or_stop_decision():
    result=record_average_quality(dict(weighted_replay_tv=.17,weighted_replay_kl=.2))
    assert result['within_original_quality_reference'] is False
    assert result['weighted_replay_tv']==.17
    assert record_average_quality(dict(weighted_replay_tv=.1,weighted_replay_kl=.1))['within_original_quality_reference']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,1.1])
def test_invalid_projection_output_still_stops(value):
    with pytest.raises((FloatingPointError,ValueError)):
        record_average_quality(dict(weighted_replay_tv=value,weighted_replay_kl=.1))


def test_declared_recovery_horizon_and_budget():
    cfg=json.loads((ROOT/'configs/poker-average-continuation-20261117.json').read_text())
    validate_config(cfg)
    assert cfg['iterations']==420 and cfg['resume_iterations']=={'342':366,'343':366,'344':372}
    assert cfg['main_steps']==1024 and cfg['batch_size']==512 and cfg['buffer_capacity']==100000
    assert cfg['average_fit_steps']==8192 and cfg['average_width']==128 and not cfg['average_normalize']
    assert len(jobs())==36
    for key,value in [('iterations',480),('main_steps',256),('auxiliary_quality_error_stops_main',True),
                      ('primary_candidate','average361_420'),('automatic_retry',True)]:
        with pytest.raises(ValueError):validate_config(dict(cfg,**{key:value}))


def test_main_policy_is_preselected_and_weak_average_does_not_replace_it():
    rows={}
    for label in LABELS:
        for seed in SEEDS:
            for field in FIELDS:
                delta=10 if label=='current420' else -100 if label=='average361_420' else 0
                rows[f'{field}-seed-{seed}-{label}']=dict(
                    deal_blocks=[v+delta for v in [-3.,-1.,1.,3.]],opponents={'same':True})
    result=summarize(rows,deals=4)
    assert result['adoption_against_original120']['original_four_conditions_pass']
    assert not result['average_diagnostic_conditions']['original_four_conditions_pass']
    assert result['current_minus_original120']['public']['mean']==10
    assert result['independent_card_blocks']==4 and not result['model_promoted']
    with pytest.raises(ValueError):summarize(dict(list(rows.items())[:-1]),deals=4)
