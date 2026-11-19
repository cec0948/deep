"""Analytic checks and six-seat engine fixtures for the opt-in turn search."""
from pathlib import Path
import multiprocessing
import random
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fullhouse_runtime import Bridge
import poker_multivalued_turn as search
from poker_multivalued_turn import TurnSearchPolicy, biased_distribution, solve_choices
from poker_public_state import PublicState
from poker_river_runtime import legal_menu


def turn_request():
    actions = [{'seat': 3, 'kind': 'call'}]
    actions.extend({'seat': seat, 'kind': 'fold'} for seat in (4, 5, 0, 1))
    actions.extend({'seat': seat, 'kind': 'check'} for seat in (2, 2, 3))
    return dict(op='replay', hero=2, button=0, hole=['As', 'Kd'],
                board=['2c', '7d', 'Jh', '9c'], actions=actions)


@pytest.mark.parametrize('passive', ['call', 'check'])
def test_bias_preserves_support_and_relative_raise_sizes(passive):
    actions = [{'kind': 'fold'}, {'kind': passive},
               {'kind': 'raise', 'to': 200}, {'kind': 'raise', 'to': 500}]
    probabilities = np.array([.20, .40, .10, .30])
    if passive == 'check':
        actions.pop(0)
        probabilities = probabilities[1:]
        probabilities = probabilities / probabilities.sum()
    rows = [dict(action=a, probability=p) for a, p in zip(actions, probabilities)]
    for mode, targets in enumerate((set(), {'fold'}, {'call', 'check'}, {'raise'})):
        expected = probabilities * np.array([5 if a['kind'] in targets else 1 for a in actions])
        actual = biased_distribution(rows, mode)
        np.testing.assert_allclose(actual, expected / expected.sum())
    assert [row['probability'] for row in rows] == pytest.approx(probabilities)


def test_solve_choices_cannot_see_the_other_hand_or_secret_mode():
    # One player sees a fair strong/weak hand and may check or bet. If facing
    # a bet, the other player observes the bet but cannot see which hand made it.
    # Checking earns +1/-1. A fold to the bet pays +1, a call pays +2/-2.
    matrices = np.zeros((2, 1, 2, 4, 4))
    for hand, strength in enumerate((1., -1.)):
        matrices[hand, 0, 0] = .5 * strength
        matrices[hand, 0, 1, :, :2] = .5
        matrices[hand, 0, 1, :, 2:] = strength
    hero, opponent, gap = solve_choices(matrices, iterations=2500)
    np.testing.assert_allclose(hero.sum(axis=(1, 2)), 1.)
    np.testing.assert_allclose(opponent.sum(axis=2), 1.)
    assert hero[0, 1].sum() > .97
    assert hero[1, 1].sum() == pytest.approx(1 / 3, abs=.04)
    assert opponent[0, 1, 2:].sum() == pytest.approx(2 / 3, abs=.04)
    value = np.einsum('ham,van,hvamn->', hero, opponent, matrices)
    assert value == pytest.approx(1 / 3, abs=.025)
    assert 0 <= gap < .07


def test_both_players_choose_private_continuation_at_the_leaf():
    # A biased matching-pennies payoff forces each player away from uniform
    # play. Modes 2/3 duplicate 0/1 so the allocation within each pair is free.
    types = [0, 1, 0, 1]
    matrix = np.array([[3., -1.], [-1., 1.]])[np.ix_(types, types)]
    hero, opponent, gap = solve_choices(matrix.reshape(1, 1, 1, 4, 4), iterations=2500)
    for profile in (hero, opponent):
        assert profile[0, 0, ::2].sum() == pytest.approx(1 / 3, abs=.04)
    assert 0 <= gap < .08


class UniformPolicy:
    def decide(self, state, seed):
        actions = list(legal_menu(state).values())
        return dict(action=actions[seed % len(actions)], ok=True,
                    distribution=[dict(action=a, probability=1 / len(actions)) for a in actions])


def test_six_seat_turn_fixture_and_three_way_fallback():
    request = turn_request()
    # Keep seat 4 in the hand and check through the flop with all three seats.
    multiway = dict(request, actions=[dict(a) for a in request['actions']])
    multiway['actions'][1]['kind'] = 'call'
    multiway['actions'].append({'seat': 4, 'kind': 'check'})
    bridge = Bridge()
    try:
        turn = bridge.call(request)
        assert turn['street'] == 'turn' and turn['pot'] == 250
        assert [p['seat'] for p in turn['players'] if not p['is_folded']] == [2, 3]
        state = bridge.call(multiway)
    finally:
        bridge.close()
    assert state['street'] == 'turn'
    assert [p['seat'] for p in state['players'] if not p['is_folded']] == [2, 3, 4]
    base = UniformPolicy()
    policy = TurnSearchPolicy(base, seconds=5)
    policy.begin_hand('multiway-fixture')
    result = policy.decide(state, 29)
    expected = base.decide(state, 29)
    assert result['distribution'] == expected['distribution']
    assert result['action'] == expected['action']


def test_root_range_decks_ignore_actual_hero_cards_and_replay_public_state():
    bridge = Bridge()
    try:
        state = bridge.call(turn_request())
        other = bridge.call(dict(turn_request(), hole=['3h', '3s']))
    finally:
        bridge.close()
    public = PublicState.from_bridge(state).public_only()
    other_public = PublicState.from_bridge(other).public_only()
    assert public == other_public
    ranges = [([('Ah', 'Kh')], [1.]), ([('3s', '3h')], [1.])]
    deck, classes = search._world_deck(public, [2, 3], ranges, random.Random(91))
    assert (deck, classes) == search._world_deck(other_public, [2, 3], ranges, random.Random(91))
    assert len(set(deck)) == 52
    assert deck[4:8] == ['Ah', 'Kh', '3s', '3h']
    assert deck[12:16] == list(public.board)
    rules = search.Rules()
    try:
        tree = search._Rollouts(rules, public, deck, UniformPolicy(), random.Random(45))
        assert PublicState.from_bridge(tree.states[()]).public_only() == public
        assert tree.states[()]['your_cards'] == ['Ah', 'Kh']
        # Six-seat chip accounting also runs on the immediate all-in branch.
        value = tree.payoff({'kind': 'raise', 'to': 9900}, 2, (0, 2), time.monotonic() + 2)
        assert np.isfinite(value) and -100 <= value <= 500
    finally:
        rules.close()


def test_expired_worker_returns_blueprint_and_exits(monkeypatch):
    def stall(*args):
        time.sleep(3.)

    bridge = Bridge()
    try:
        state = bridge.call(turn_request())
    finally:
        bridge.close()
    candidate = TurnSearchPolicy(UniformPolicy(), seconds=5)
    candidate.begin_hand('deadline-fixture')
    monkeypatch.setattr(search, '_worker', stall)
    children = {p.pid for p in multiprocessing.active_children()}
    start = time.monotonic()
    result = candidate.decide(state, 42, deadline=start + .35)
    assert result['turn_search']['reason'] == 'hard_deadline'
    assert not result['turn_search']['used']
    assert result['action'] == candidate.blueprint.decide(state, 42)['action']
    assert time.monotonic() - start < 1.2
    assert {p.pid for p in multiprocessing.active_children()} <= children


@pytest.mark.parametrize('mode', [1, 2, 3])
def test_following_river_uses_chosen_hero_mode_and_hand_reset_clears_it(monkeypatch, mode):
    bridge = Bridge()
    req = turn_request()
    try:
        before = bridge.call(req)
        through = [{'seat': seat, 'kind': 'check'} for seat in (2, 3, 2)]
        through.append({'seat': 3, 'kind': 'raise', 'to': 125})
        river = bridge.call(dict(req, board=req['board'] + ['Qs'], actions=req['actions'] + through))
    finally:
        bridge.close()

    def solved(public, blueprint, deadline):
        actions = list(legal_menu(public.to_observation()).values())
        joint = np.zeros((3, len(actions), 4))
        check = next(i for i, action in enumerate(actions) if action['kind'] == 'check')
        joint[:, check, mode] = 1.
        return dict(used=True, reason='search_complete', actions=actions,
                    joint=joint.tolist(), root_bucket_counts=[8, 8, 8])

    monkeypatch.setattr(search, '_search', solved)
    base = UniformPolicy()
    policy = TurnSearchPolicy(base)
    policy.begin_hand('continuation-fixture')
    assert policy.decide(before, 42)['action'] == {'kind': 'check'}
    reply = policy.decide(river, 43)
    base_rows = base.decide(river, 43)['distribution']
    expected = biased_distribution(base_rows, mode)
    assert [row['action'] for row in reply['distribution']] == [row['action'] for row in base_rows]
    assert [row['probability'] for row in reply['distribution']] == pytest.approx(expected)
    policy.begin_hand('different-fixture')
    assert policy.decide(river, 43)['distribution'] == base_rows
