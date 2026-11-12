"""Behavior baseline boundaries, hidden-info isolation and raw verification."""
import copy
import gzip
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

from fullhouse_runtime import Bridge
from poker_reference_opponents import (PROFILES, ReferenceOpponent, checked_action,
                                       distribution, observation)
from evaluate_poker_reference import hand_seed, verify_raw


@pytest.fixture
def engine():
    bridge = Bridge()
    try:
        yield bridge
    finally:
        bridge.close()


@pytest.mark.parametrize('profile', PROFILES)
def test_observation_and_randomness_do_not_use_unavailable_information(engine, profile):
    state = engine.call(dict(op='reset', seed=98765, button=4))
    original = copy.deepcopy(state)
    a = ReferenceOpponent(profile)
    first = a.decide(state, 1234)
    state['opponent_private_cards'] = [['As', 'Ah']]*5
    state['future_board'] = ['Ks', 'Qs', 'Js', 'Ts', '9s']
    state['deck'] = ['bad-secret-placeholder']
    state['utilities'] = [50000,-10000,-10000,-10000,-10000,-10000]
    assert a.decide(state, 1234) == first
    for key, value in original.items():
        assert state[key] == value


@pytest.mark.parametrize('profile', PROFILES)
def test_reference_actions_are_accepted_by_real_engine_through_complete_hands(engine, profile):
    player = ReferenceOpponent(profile)
    streets = set(); decisions = 0
    for deal in range(20):
        state = engine.call(dict(op='reset', seed=887700+deal, button=deal % 6))
        step = 0
        while not state['terminal']:
            streets.add(state['street'])
            reply = player.decide(state, deal*1000+step)
            assert abs(sum(x['probability'] for x in reply['distribution'])-1) < 1e-12
            for row in reply['distribution']:
                checked_action(observation(state), row['action'])
            if profile == 'calling-station':
                assert reply['action']['kind'] in ('call', 'check')
            state = engine.call(dict(op='step', action=dict(reply['action'], seat=state['seat_to_act'])))
            step += 1; decisions += 1
            assert step <= 512
        assert sum(state['utilities']) == 0
    assert decisions > 100
    if profile == 'calling-station':
        assert streets == {'preflop','flop','turn','river'}


def test_short_allin_raise_and_unavailable_raise_are_respected(engine):
    state = engine.call(dict(op='reset', seed=123, button=0))
    # Boundary fixture uses the same concrete legal-bound contract as the engine.
    state.update(your_stack=50, your_bet_this_street=100, amount_owed=20,
                 can_check=False, raise_bounds=[150,150])
    rows, _ = distribution('pressure', state)
    assert [r['action']['to'] for r in rows if r['action']['kind']=='raise'] == [150]
    state['raise_bounds'] = None
    assert all(r['action']['kind'] != 'raise' for r in distribution('pressure',state)[0])
    state['amount_owed'] = 0; state['can_check'] = True
    assert all(r['action']['kind'] == 'check' for r in distribution('tight-value',state)[0])


def test_raw_verifier_rejects_changed_payoffs_and_missing_hands(tmp_path):
    rows = []
    for deal in range(2):
        for hero in range(6):
            payoffs = [0]*6; payoffs[hero]=100; payoffs[(hero+1)%6]=-100
            rows.append(dict(deal=deal,hero=hero,seed=500+deal,chips=100,utilities=payoffs,
                             steps=12,hand_seed=hand_seed(500,deal,hero)))
    path = tmp_path/'hands.jsonl.gz'
    def write(values):
        with gzip.open(path,'wt') as stream:
            for row in values:
                stream.write(json.dumps(row)+'\n')
    write(rows)
    assert verify_raw(path,2,500)==[100,100]
    write(rows[:-1])
    with pytest.raises(ValueError,match='Incomplete'):
        verify_raw(path,2,500)
    rows[0]['chips']=999
    write(rows)
    with pytest.raises(ValueError,match='payoff'):
        verify_raw(path,2,500)
