from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_poker_flop_rollback import gap_recovery


@pytest.mark.parametrize('ratio', [-1, 0, 1, 2])
def test_fieller_accounts_for_paired_covariance_and_does_not_clip(ratio):
    old_gap = np.array([1., 2., 3., 4., 5.])
    estimate = gap_recovery(old_gap, ratio * old_gap)
    assert estimate['fraction'] == ratio
    assert estimate['paired_fieller_ci95'] == pytest.approx([ratio, ratio])


def test_near_zero_gap_has_no_bounded_interval():
    estimate = gap_recovery([-.1, 0., .1, .02], [1., 3., 2., 4.])
    assert estimate['fraction'] > 1
    assert estimate['denominator_ci95'][0] < 0 < estimate['denominator_ci95'][1]
    assert estimate['paired_fieller_ci95'] is None


def test_nonpositive_gap_has_no_recoverable_shortfall():
    assert gap_recovery([-1., 1.], [1., 2.])['fraction'] is None
