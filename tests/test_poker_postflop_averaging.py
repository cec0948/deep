from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_poker_postflop_averaging import matched_first_round


def example():
    return dict(traverser=0, traversals=256, new_regret_rows=9840, root_rows=676,
                fit_rows_by_street=[128924,175751,206512], fit_draws_by_street=[131001,131486,130729],
                fit_draws=393216, reference_collection={'hands':512, 'decision_nodes':5877},
                input_weights={'preflop':'fixed','general':'parent'}, output_weights={'preflop':'fixed','general':'old'})


def test_first_fit_requires_matching_sampling_but_distinct_updates():
    control = example()
    changed = deepcopy(control)
    with pytest.raises(ValueError):
        matched_first_round(changed, control)
    changed['output_weights']['general'] = 'smaller-step'
    matched_first_round(changed, control)
    changed['fit_draws_by_street'][0] += 1
    changed['fit_draws_by_street'][1] -= 1
    with pytest.raises(ValueError):
        matched_first_round(changed, control)
