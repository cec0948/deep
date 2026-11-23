from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from poker_opponent_population import TreePolicy, assignments


class Returning:
    def __init__(self, label):
        self.label = label

    def strategy(self, features, legal):
        return self.label


def test_routing_preserves_frozen_preflop_and_traverser_for_every_dealer():
    slots = np.array([-1,300,360,420,0,300])
    visits = Counter()
    policy = TreePolicy(Returning('current'), {i:Returning(i) for i in (300,360,420)}, visits)
    for dealer in range(6):
        policy.by_position = np.roll(slots, -dealer)
        for seat, expected in enumerate(slots):
            features = np.zeros(51)
            features[21+(seat-dealer)%6] = 1
            for street in range(4):
                features[17:21] = 0
                features[17+street] = 1
                action = policy.strategy(features, None)
                assert action == ('current' if not street or expected<=0 else expected)
    assert visits == Counter({-1:18, 0:18, 300:36, 360:18, 420:18})


def test_population_draws_are_private_to_the_game_tree_and_repeatable():
    slots = assignments(342,481,256,0)
    assert slots.dtype == np.dtype('<i2')
    assert (slots[:,0] == -1).all()
    assert set(slots[:,1:].flat) == {0,300,360,420}
    assert np.array_equal(slots,assignments(342,481,256,0))
    assert not np.array_equal(slots,assignments(343,481,256,0))
    with pytest.raises(ValueError):
        assignments(342,481,256,1)
