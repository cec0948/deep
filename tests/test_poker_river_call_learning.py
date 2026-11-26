from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_poker_river_call import calibrate
from poker_river_call_learning import FEATURES, RiverCallPolicy, eligible, features, passive_update, uniform_equity


def river():
    stacks = (10_000, 9_950, 9_900, 9_700, 10_000, 10_000)
    return dict(street='river', terminal=False, can_check=False, seat_to_act=2, pot=450, amount_owed=200,
                players=[dict(seat=s, is_folded=s not in (2,3), is_all_in=False, stack=chips,
                              bet_this_street=200 if s==3 else 0) for s,chips in enumerate(stacks)],
                action_log=[dict(action='deal'), dict(action='check',seat=2), dict(action='raise',seat=3)],
                your_cards=['Ac','Ah'], community_cards=['2h','3s','4d','8c','Ks'])


def test_only_the_first_eligible_river_bet_and_public_features_are_used():
    state = river()
    assert eligible(state)
    x = features(state)
    assert len(x) == len(FEATURES) and x[4] == 1
    extra = dict(state, opponent_cards=['8h','8s'], showdown_share=0.)
    np.testing.assert_array_equal(x, features(extra))
    prior = [dict(action='check',seat=3), dict(action='raise',seat=2), dict(action='raise',seat=3)]
    assert not eligible(dict(state, action_log=[dict(action='deal'), *prior]))
    assert not eligible(dict(state, pot=451))
    assert not eligible(dict(state, amount_owed=400))


def test_passive_redistribution_matches_claimed_mass_and_retains_seeded_raises():
    bet = dict(action={'kind':'raise','to':1200}, probability=.3)
    reply = dict(action=bet['action'], distribution=[dict(action={'kind':'fold'},probability=.5),
                                                   dict(action={'kind':'call'},probability=.2),bet])
    updated = passive_update(reply, 'call', 52)
    assert updated['action'] == bet['action'] and updated['distribution'][-1] == bet
    assert [row['probability'] for row in updated['distribution']] == [.375,.325,.3]
    assert reply['distribution'][0]['probability'] == .5


def test_protected_boards_keep_call_even_when_originally_no_fold_to_redirect():
    state = dict(river(), community_cards=['As','Ks','Qs','Js','Ts'])
    assert eligible(state)
    assert uniform_equity(tuple(state['your_cards']),tuple(state['community_cards'])) == .5
    parent_reply = dict(distribution=[dict(action={'kind':'call'},probability=1.)], action={'kind':'call'})
    class Parent:
        def decide(self, state, seed):
            return parent_reply
    model = object.__new__(RiverCallPolicy)
    model.parent = Parent()
    model.model = {}
    # If the floor guard misses this board, the unfitted calibrator raises.
    assert model.decide(state,11) is parent_reply


def test_fixed_logit_offset_can_learn_lower_equity_from_biased_showdowns():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(1000,len(FEATURES)))
    y = 1./(1.+np.exp(-(x[:,0]-1.2)))
    model, report = calibrate(x,y)
    assert abs(model['intercept']+1.2)<.0001
    assert report['training_regularized_objective']<report['uniform_offset_training_objective']
