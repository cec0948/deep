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
