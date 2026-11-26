from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from score_poker_street_progress import interval


def test_independent_fixed_seed_deals_do_not_count_between_seed_offsets_as_sampling_noise():
    # Each fixed policy sees distinct deals. Its level is averaged once, equally by seed.
    result = interval({'342': [1, 3], '343': [-11, -7], '344': [50, 50, 50, 50]})
    assert result['mean'] == pytest.approx((2-9+50)/3)
    assert result['standard_error'] == pytest.approx((1**2+2**2)**.5/3)
    assert result['lower'] == pytest.approx(result['mean']-1.96*result['standard_error'])


def test_rejects_passing_action_or_future_card_replicates_as_independent_roots():
    with pytest.raises(ValueError):
        interval({str(s):[[0,1],[1,0]] for s in (342,343,344)})
