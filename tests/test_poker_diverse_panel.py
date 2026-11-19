"""Seat balance and paired randomization for the frozen opponent panel."""
from collections import Counter
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_poker_diverse_panel import FIELDS, MIXED, VARIED, lineup, play


def test_varied_rotates_clockwise_roles_without_hero_identity_or_weights():
    for card in range(50, 55):
        assert set(lineup('varied', card)) == set(VARIED)
        for hero in range(6):
            opponents = [lineup('varied', card)[(actor-hero) % 6 - 1]
                         for actor in range(6) if actor != hero]
            assert set(opponents) == set(VARIED)
    for slot in range(5):
        assert set(lineup('varied', card)[slot] for card in range(50, 55)) == set(VARIED)
    assert lineup('mixed', 1) == MIXED
    for field in set(FIELDS)-{'varied','mixed'}:
        assert lineup(field, 2) == (field,)*5
    with pytest.raises(ValueError):
        lineup('unknown', 0)


def test_two_frozen_candidates_share_deals_and_public_history_randomness():
    class FakePolicy:
        def decide(self, state, seed):
            self.draws.append(seed)
            return dict(ok=True, action={'kind':'check'}, distribution=[{'probability': 1.}])
        def __init__(self):
            self.draws = []
    class Engine:
        def call(self, request):
            if request['op'] == 'reset':
                self.actor = 0
            else:
                assert request['action'] == {'seat': self.actor, 'kind':'check'}
                self.actor += 1
            if self.actor == 6:
                return dict(terminal=True, utilities=[0]*6)
            return dict(seat_to_act=self.actor, terminal=False, street='flop',
                        action_log=['check']*self.actor, can_check=True, raise_bounds=None)
    before, after = FakePolicy(), FakePolicy()
    for hero in range(6):
        for candidate in (before, after):
            row = play(Engine(), candidate, {'public':FakePolicy()}, lineup('public', 90),
                       90, hero, Counter(), time.monotonic()+2)
            assert row['chips'] == 0 and row['steps'] == 6 and row['hero'] == hero
    assert len(set(before.draws)) == 6
    assert before.draws == after.draws
