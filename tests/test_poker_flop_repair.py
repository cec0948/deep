from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from poker_flop_repair import HIDDEN, INPUT_DIM, MODELS, first_flop_decision, improved_distribution


def test_first_flop_decision_resets_after_preflop_but_never_after_a_flop_raise():
    events = [{'action': 'call', 'seat': 2}, {'action': 'deal'}, {'action': 'check', 'seat': 1}]
    state = {'street': 'flop', 'seat_to_act': 2, 'action_log': events}
    assert first_flop_decision(state)
    events.extend([{'action': 'check', 'seat': 2}, {'action': 'raise', 'seat': 4}])
    assert not first_flop_decision(state)
    assert not first_flop_decision(dict(state, street='turn'))


def test_repair_requires_both_critics_and_caps_shift_to_a_previously_unused_action():
    features = np.zeros(INPUT_DIM)
    weights = dict(center=features, scale=np.ones(INPUT_DIM),
                   w1=np.zeros((MODELS, HIDDEN, INPUT_DIM)), b1=np.zeros((MODELS, HIDDEN)),
                   w2=np.zeros((MODELS, 5, HIDDEN)), b2=np.zeros((MODELS, 5)))
    baseline = np.array([1., 0.])
    heads = [1, 3]
    weights['b2'][0, 3] = 20
    weights['b2'][1, 3] = -1
    probs, shift = improved_distribution(features, heads, baseline, weights)
    assert shift == 0
    np.testing.assert_equal(probs, baseline)

    weights['b2'][1, 3] = 1
    probs, shift = improved_distribution(features, heads, baseline, weights)
    assert shift == .25
    np.testing.assert_equal(probs, [.75, .25])
