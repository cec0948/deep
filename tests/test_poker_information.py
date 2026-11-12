from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_poker_information import (ROOT, prove_witnesses, summarize_replay,
                                     suit_isomorphic, validate_replay)


def valid_rows():
    x = np.zeros((3, 51), np.float32)
    x[:, 17] = 1.; x[:, 21] = 1.; x[:, 46:48] = 1.; x[:, 27] = .03
    x[2, 7] = 1.
    return x, np.array([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [3, 0, 0, 0, 0]], np.float32), np.array([1, 3, 2], np.float32)


def test_suit_equivalence_rejects_rank_suit_witness():
    assert suit_isomorphic(['As', 'Kd'], ['2s', '7d', 'Th'], ['Ac', 'Kh'], ['2c', '7h', 'Td'])
    assert not suit_isomorphic(['As', 'Kd'], ['2s', '7d', 'Th'], ['As', 'Kd'], ['2h', '7d', 'Ts'])


def test_replay_duplicate_count_and_conditional_variance():
    x, y, w = valid_rows()
    result = summarize_replay(x, y, w)
    g = result['by_stratum']['0']
    assert g['rows'] == 3 and g['unique_feature_vectors'] == 2
    assert g['rows_in_repeated_feature_groups'] == 2
    # Weighted mean of the duplicate input is -0.5, residual weighted SSE=3.
    assert g['weighted_within_feature_target_variance'] == pytest.approx(3 / 6 / 5)
    assert result['malformed_rows'] == 0


@pytest.mark.parametrize('bad', ['weight', 'nonfinite', 'street', 'legal'])
def test_malformed_replay_is_rejected(bad):
    x, y, w = valid_rows()
    if bad == 'weight': w[0] = 0
    if bad == 'nonfinite': y[1, 0] = float('nan')
    if bad == 'street': x[0, 18] = 1
    if bad == 'legal': x[0, 46:51] = 0
    with pytest.raises(ValueError): validate_replay(x, y, w)


def test_real_engine_histories_and_cards_have_identical_model_inputs():
    parent = ROOT / 'artifacts/compact-v4/20261112T032318825281Z'
    if not (parent / 'seed-342/current_00120.npz').exists():
        pytest.skip('Requires restored frozen120 policies and trusted engine assets')
    cases = prove_witnesses(parent)
    assert len(cases) == 2
    assert all(c['engine_accepted_both'] and c['byte_identical_features'] for c in cases)
    history, cards = cases
    assert history['states'][0]['players'] == history['states'][1]['players']
    assert history['states'][0]['action_log'] != history['states'][1]['action_log']
    assert history['same_card_observation']
    assert not cards['same_card_observation']
    assert not cards['card_observations_globally_suit_isomorphic']
    assert all(v['byte_identical'] for c in cases for v in c['policy_outputs'].values())
