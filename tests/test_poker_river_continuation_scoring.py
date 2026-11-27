import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from score_poker_river_continuation import adoption, equal_cell_summary, summarize_cells


def test_equal_opponent_and_seed_weight_is_independent_of_cell_frequency():
    cells={(field,seed): ([1.,3.] if field=='x' else [-1.,-1.,-1.,-1.])
           for field in ('x','y') for seed in (342,343,344)}
    proofs={key:{'observed_contribution_bb_per_100':1. if key[0]=='x' else -2.} for key in cells}
    estimate=summarize_cells(cells,proofs)
    assert estimate['mean']==pytest.approx(.5)
    assert estimate['standard_error']==pytest.approx(math.sqrt(3)/6)
    assert estimate['deals']==18
    assert estimate['by_opponent']['y']['mean']==-1.
    assert estimate['by_seed']['344']['mean']==pytest.approx(.5)
    assert estimate['observed_contribution_bb_per_100']==pytest.approx(-.5)


def test_require_both_incremental_reached_lower_and_dealt_gain_positive():
    assert adoption({'lower':.01},.001)
    assert not adoption({'lower':.1},-.03)
    assert not adoption({'lower':0.},1.)
    assert not adoption({'lower':.1},0.)
    with pytest.raises(ValueError):
        equal_cell_summary([[1.]])


def test_matched_attribution_separates_repeated_preference_from_coefficient_change():
    from score_poker_river_continuation import CALL, FOLD, expected_shift
    from scipy.special import logit
    import numpy as np
    state={'amount_owed':100,'pot':300}
    before={CALL:.4,FOLD:.5,('raise',300):.1}
    features=np.zeros(10)
    settings=dict(center=[0.]*10,scale=[1.]*10,coefficients=[0.]*10)
    old=dict(settings,intercept=float(logit(.36)))
    new=dict(settings,intercept=float(logit(.16)))
    old_shift=expected_shift(state,old,before,features)
    new_shift=expected_shift(state,new,before,features)
    assert old_shift==.125
    assert new_shift==-.1
    assert new_shift-old_shift==pytest.approx(-.225)
    assert old_shift + (new_shift-old_shift)==pytest.approx(new_shift)
    assert expected_shift(state,dict(settings,intercept=float(logit(.27))),before,features)==0.
    assert expected_shift(state,new,before,None)==0.
