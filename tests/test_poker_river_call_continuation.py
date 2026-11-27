from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_poker_diverse_river_calls import continue_fit
from poker_river_call_diverse import DiverseRiverCallPolicy
from poker_river_call_learning import FEATURES
from test_poker_river_call_learning import river


def test_residual_fit_starts_at_parent_and_keeps_its_feature_basis():
    rng = np.random.default_rng(815)
    n, d = 500, len(FEATURES)
    x = rng.normal(size=(n,d))
    parent = dict(center=[.4]*d, scale=[.8]*d, coefficients=[.1]*d, intercept=-.6)
    base = x[:,0]+parent['intercept']+((x-.4)/.8)@np.asarray(parent['coefficients'])
    # These synthetic labels obey the same win/tie/loss support as actual showdowns.
    y = rng.binomial(1,expit(base-.7))
    weights = np.r_[np.repeat(.5/100,100),np.repeat(.5/400,400)]
    model, metrics = continue_fit(parent,x,y,weights)
    expected_start = weights @ (np.logaddexp(0.,base)-y*base)
    assert abs(metrics['source_training_objective']-expected_start) < 1.e-12
    assert metrics['fitted_training_objective'] < expected_start
    assert metrics['initial_residual_norm'] == 0. and metrics['final_residual_norm'] > 0.
    assert model['center'] == parent['center'] and model['scale'] == parent['scale']


def test_parent_sample_guard_and_no_trade_margin(monkeypatch):
    state = river()
    reply = dict(action={'kind':'raise','to':1000},distribution=[
        dict(action={'kind':'fold'},probability=.4),dict(action={'kind':'call'},probability=.2),
        dict(action={'kind':'raise','to':1000},probability=.4)])

    class Parent:
        def decide(self,state,seed):
            return reply

    policy = object.__new__(DiverseRiverCallPolicy)
    policy.parent = Parent()
    policy.center = policy.coef = np.zeros(len(FEATURES))
    policy.scale = np.ones(len(FEATURES))
    monkeypatch.setattr('poker_river_call_diverse.features',lambda state: np.zeros(len(FEATURES)))
    odds = state['amount_owed']/(state['pot']+state['amount_owed'])
    policy.intercept = np.log(odds/(1.-odds))
    assert policy.decide(state,10) is reply
    policy.intercept = 20.
    after = policy.decide(state,10)
    assert after['action'] == reply['action'] and after['distribution'][2] == reply['distribution'][2]
    np.testing.assert_allclose([r['probability'] for r in after['distribution']],[.3,.3,.4])
    protected = dict(state,community_cards=['As','Ks','Qs','Js','Ts'])
    assert policy.decide(protected,10) is reply
