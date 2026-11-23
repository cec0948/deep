import copy
from contextlib import closing
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))

from evaluate_average_baseline import _validate_reply
from fullhouse_runtime import Bridge
import poker_solverlike_opponent as solver


def test_play_matches_upstream_on_historical_card_fixtures():
    bot = solver.SolverLikeOpponent()
    streets = set()
    kinds = set()
    with closing(Bridge()) as engine:
        for card in range(2038400000, 2038400128):
            state = engine.call(dict(op='reset', seed=card, button=0))
            while not state['terminal']:
                streets.add(state['street'])
                reply = _validate_reply(state, bot.decide(state, seed=1))
                actual = reply['action']
                assert reply['distribution'] == [dict(action=actual, probability=1.0)]
                bounds = state['raise_bounds']
                # All feasible upstream actions must match exactly on original
                # bridge observations, without changing its deterministic mix.
                if bounds is not None:
                    raw = bot.upstream.decide(state)
                    kinds.add(raw['action'])
                    if raw['action'] == 'raise':
                        expected = dict(kind='raise', to=raw['amount'])
                    elif raw['action'] == 'all_in':
                        expected = dict(kind='raise', to=bounds[1])
                    elif raw['action'] == 'fold' and state['can_check']:
                        expected = dict(kind='check')
                    elif raw['action'] == 'call' and state['can_check']:
                        expected = dict(kind='check')
                    else:
                        expected = dict(kind=raw['action'])
                    assert actual == expected
                state = engine.call(dict(op='step', action=dict(seat=state['seat_to_act'], **actual)))
            assert sum(state['utilities']) == 0
    assert streets == {'preflop', 'flop', 'turn', 'river'}
    assert {'call', 'check', 'raise', 'fold'}.issubset(kinds)


def test_action_log_and_closed_betting_are_translated():
    bot = solver.SolverLikeOpponent()
    with closing(Bridge()) as engine:
        state = engine.call(dict(op='reset', seed=19, button=0))
    state['your_cards'] = ['As', 'Ah']
    assert bot.decide(state, 1)['action'] == dict(kind='raise', to=250)
    assert bot.decide(state, 9999) == bot.decide(dict(state, hidden_opponent_cards=['Ks', 'Kh']), 1)
    raised = copy.deepcopy(state)
    raised.update(current_bet=300, amount_owed=300, min_raise_to=500, raise_bounds=[500, 10000])
    raised['action_log'].append(dict(action='raise', seat=2, amount=300))
    assert bot.decide(raised, 1)['action'] == dict(kind='raise', to=900)
    # An all-in or oversized raise with no raise rights is an all-in call.
    closed = dict(raised, current_bet=10000, amount_owed=10000, min_raise_to=None, raise_bounds=None)
    assert bot.decide(closed, 1)['action'] == dict(kind='call')


def test_source_identity_must_match(monkeypatch):
    monkeypatch.setattr(solver, 'MODEL_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='Unexpected upstream'):
        solver.SolverLikeOpponent()


def test_policy_errors_surface(monkeypatch):
    bot = solver.SolverLikeOpponent()
    with closing(Bridge()) as engine:
        state = engine.call(dict(op='reset', seed=19, button=0))

    def unavailable(_state):
        raise RuntimeError('unexpected policy failure')

    monkeypatch.setattr(bot.upstream, '_decide_preflop', unavailable)
    with pytest.raises(RuntimeError, match='unexpected policy failure'):
        bot.decide(state, 1)
