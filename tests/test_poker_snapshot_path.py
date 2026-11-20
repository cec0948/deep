from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_poker_snapshot_path import ratio_by_card


def test_assignment_size_weighting_and_within_card_correlation():
    counts = np.array([0, 1, 2, 3])
    gains = np.array([0, 6, -2, 2])
    mean, error = ratio_by_card(gains, counts)
    assert mean == 1
    # Four independent card clusters; count-weighted gains, including empty clusters.
    assert error == pytest.approx(np.sqrt((0 + 25 + 16 + 1) / 3) / 2 / 1.5)
    # Adding fixed copies of every seat payoff must not shrink uncertainty.
    assert ratio_by_card(gains * 3, counts * 3) == pytest.approx((mean, error))


def test_zero_assignment_rejected():
    with pytest.raises(ValueError):
        ratio_by_card(np.zeros(4), np.zeros(4))
