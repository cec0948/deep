import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from validate_sampling_objective import information_metadata,verify_objective,collect_replay,validate_config


def settings():
    return dict(game='leduc_poker',method='importance-corrected-balanced',seed=81,iterations=2,
        traversals_per_player=16,advantage_steps=4,batch_size=16,memory_capacity=2000,
        network_layers=[8,8],learning_rate=.001,max_seconds=120,sampling_draws=60000,
        sampling_seed=2026111281,probe_seeds=[181,182,183],gradient_relative_tolerance=1e-9,
        loss_relative_tolerance=1e-9,monte_carlo_sigma=8,negative_control_min_relative_error=1e-4)


def test_full_leduc_metadata_preserves_players_rounds_and_legal_actions():
    pyspiel=pytest.importorskip('pyspiel')
    records=information_metadata(pyspiel.load_game('leduc_poker',{'players':2}))
    assert len(records)==936
    assert {player for _,_,player in records.values()}=={0,1}
    assert {group//6 for _,group,_ in records.values()}=={0,1}
    assert all(mask.shape==(3,) and mask.sum()>=2 for mask,_,_ in records.values())


def test_actual_official_replay_obeys_raw_gradient_identity_with_conditional_compression():
    pytest.importorskip('pyspiel');import torch
    torch.set_num_threads(1)
    cfg=settings();data,collection=collect_replay(cfg)
    assert collection['illegal_nonzero_regret_targets']==0
    assert collection['retained_rows_per_player'][0]>0 and collection['retained_rows_per_player'][1]>0
    report,proof=verify_objective(data,cfg)
    assert report['passed'],report['gate']
    assert proof['conditional_variance_constant']>0
    assert len(proof['mass'])<len(data['weights'])
    assert all(row['missing_scale_squared_relative_gradient_error']>1e-4 for row in report['gradient_checks'])


def test_exact_gate_cannot_be_relaxed_by_configuration():
    cfg=settings();validate_config(cfg)
    cfg['gradient_relative_tolerance']=.1
    with pytest.raises(ValueError,match='tolerance'):
        validate_config(cfg)
