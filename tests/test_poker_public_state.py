"""Public-information boundaries, exact event reconstruction and marginal ranges."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fullhouse_runtime import Bridge
from poker_public_state import (PublicState,Range,decision_prefixes,infer_range,observe_for_hole)


@pytest.fixture
def engine():
    value=Bridge()
    try:yield value
    finally:value.close()


def heads_up_flop(engine, hole=('As','Kd'), board=('2s','7d','Th')):
    return engine.call(dict(op='replay',hero=1,button=0,hole=list(hole),board=list(board),actions=[
        dict(seat=3,kind='fold'),dict(seat=4,kind='fold'),dict(seat=5,kind='fold'),dict(seat=0,kind='fold'),
        dict(seat=1,kind='call'),dict(seat=2,kind='check')]))


def test_real_bridge_roundtrip_and_exact_commitments(engine):
    raw=engine.call(dict(op='reset',seed=113,button=0))
    for index in range(18):
        state=PublicState.from_bridge(raw)
        assert state.to_observation()==raw
        assert state.pot==sum(p.hand_commit for p in state.players)
        assert PublicState.from_json(json.loads(json.dumps(state.to_json())))==state
        action=dict(seat=raw['seat_to_act'],kind='check' if raw['can_check'] else 'call')
        if index==0:action=dict(seat=raw['seat_to_act'],kind='raise',to=357)
        raw=engine.call(dict(op='step',action=action))
        if raw['terminal']:break


def test_actual_raise_totals_and_incremental_calls_preserved(engine):
    raw=engine.call(dict(op='reset',seed=3,button=0))
    raw=engine.call(dict(op='step',action=dict(seat=3,kind='raise',to=357)))
    raw=engine.call(dict(op='step',action=dict(seat=4,kind='call')))
    state=PublicState.from_bridge(raw)
    assert state.events[-2].amount==357 and state.events[-2].to==357
    assert state.events[-2].paid==357
    assert state.events[-1].amount==357 and state.events[-1].paid==357
    assert state.players[3].hand_commit==357
    bad=copy.deepcopy(raw);bad['action_log'][-1]['amount']=457
    with pytest.raises(ValueError,match='incremental'):PublicState.from_bridge(bad)


def test_short_allin_bounds_match_trusted_contract(engine):
    raw=engine.call(dict(op='reset',seed=32,button=0))
    for action in [dict(seat=3,kind='raise',to=9900),dict(seat=4,kind='raise',to=10000),dict(seat=5,kind='call')]:
        raw=engine.call(dict(op='step',action=action))
        state=PublicState.from_bridge(raw)
        assert state.to_observation()==raw
        assert state.min_raise_increment==9800


def test_suit_identity_preserves_known_51_feature_collision_witness(engine):
    a=PublicState.from_bridge(heads_up_flop(engine))
    b=PublicState.from_bridge(heads_up_flop(engine,board=('2h','7d','Ts')))
    assert a.canonical_suit_key()!=b.canonical_suit_key()
    swapped=a.to_observation();mapping=dict(zip('cdhs','dhsc'))
    swapped['your_cards']=[c[0]+mapping[c[1]] for c in swapped['your_cards']]
    swapped['community_cards']=[c[0]+mapping[c[1]] for c in swapped['community_cards']]
    assert PublicState.from_bridge(swapped).canonical_suit_key()==a.canonical_suit_key()
    reversed_hole=a.to_observation();reversed_hole['your_cards'].reverse()
    assert PublicState.from_bridge(reversed_hole).canonical_suit_key()==a.canonical_suit_key()


def test_history_is_not_reduced_to_current_numeric_totals(engine):
    raw=heads_up_flop(engine)
    state=PublicState.from_bridge(raw)
    assert tuple(e.seat for e in state.events if e.kind=='fold')==(3,4,5,0)
    assert state.events[-1].kind=='deal'
    assert len(state.to_json()['action_log'])==9
    changed=copy.deepcopy(raw);changed['action_log'][2]['seat']=4
    with pytest.raises(ValueError,match='actor'):PublicState.from_bridge(changed)


@pytest.mark.parametrize('mutation',[
    lambda s:s.update(opponent_hole=['Ac','Ad']),
    lambda s:s['players'][0].update(hole=['Ac','Ad']),
    lambda s:s['action_log'][2].update(cards=['Ac','Ad']),
    lambda s:s.update(your_cards=['As','As']),
    lambda s:s.update(your_cards=['2s','Ad']),
    lambda s:s.update(community_cards=['2s','7d','Th','Ac']),
    lambda s:s.update(pot=s['pot']+1),
    lambda s:s.update(amount_owed=s['amount_owed']+1),
    lambda s:s['players'][0].update(stack=9999),
    lambda s:s.update(can_check=1),
])
def test_reject_inconsistent_or_leaking_observations(engine,mutation):
    raw=heads_up_flop(engine);mutation(raw)
    with pytest.raises(ValueError):PublicState.from_bridge(raw)


def test_hypothetical_hand_adapter_uses_only_prefix_public_cards(engine):
    state=PublicState.from_bridge(heads_up_flop(engine))
    prefixes=decision_prefixes(state)
    assert len(prefixes)==6
    for prefix,event in prefixes:
        assert prefix.hero_hole is None and 'your_cards' not in prefix.to_observation()
        assert prefix.board==() and prefix.street=='preflop'
        raw=observe_for_hole(prefix,event.seat,('Ac','Ad'))
        assert raw['your_cards']==['Ac','Ad']
        assert raw['seat_to_act']==event.seat
        assert raw['community_cards']==[]
        assert raw['action_log']==state.to_observation()['action_log'][:event.index]
        assert PublicState.from_bridge(raw).to_observation()==raw
    with pytest.raises(ValueError,match='actor'):observe_for_hole(state,2,('Ac','Ad'))
    with pytest.raises(ValueError,match='collides'):observe_for_hole(state,1,('2s','Ad'))


def test_uniform_range_conditions_on_board_only_and_joint_solver_handles_collisions():
    own=Range.uniform(('2s','7d','Th','3c','4c'))
    assert len(own.combos)==1081
    assert ('Kd','As') in own.combos or ('As','Kd') in own.combos
    opponent_view=own.condition_on_blockers(('As','Kd'))
    assert len(opponent_view.combos)==990
    assert all(not set(h)&{'As','Kd'} for h in opponent_view.combos)
    assert len(own.combos)==1081


def test_action_likelihood_updates_have_no_actual_hero_card_dependency(engine):
    a=heads_up_flop(engine)
    b=heads_up_flop(engine,hole=('Jc','Td'))
    prior=Range.from_weights([('Ac','Ad'),('Jc','Jd')],[1,1],provenance='test_public_prior')
    calls=[]
    def likelihood(observation,event):
        assert observation['community_cards']==[]
        assert observation['seat_to_act']==1 and event.kind=='call'
        calls.append(tuple(observation['your_cards']))
        return 1. if observation['your_cards'][0][0]=='A' else 1/3
    ra=infer_range(a,1,likelihood,prior)
    rb=infer_range(b,1,likelihood,prior)
    assert ra==rb and ra.weights==pytest.approx((.75,.25))
    assert calls==[('Ac','Ad'),('Jc','Jd')]*2
    assert 'marginal_policy_factorization_not_exact_joint_posterior' in ra.provenance
    assert 'folded_player_card_removal_not_conditioned' in ra.provenance


def test_zero_invalid_or_unsupported_likelihood_never_silently_becomes_uniform(engine):
    state=heads_up_flop(engine);prior=Range.from_weights([('Ac','Ad')],[1],provenance='test')
    with pytest.raises(ValueError,match='zero likelihood'):infer_range(state,1,lambda o,e:0.,prior)
    with pytest.raises(ValueError,match='finite probability'):infer_range(state,1,lambda o,e:float('nan'),prior)
    def unsupported(observation,event):raise ValueError('off-tree raise unsupported')
    with pytest.raises(ValueError,match='off-tree'):infer_range(state,1,unsupported,prior)


def test_bad_ranges_rejected():
    for combos,weights in [([('As','As')],[1]),([('As','Kd'),('Kd','As')],[1,1]),([('As','Kd')],[-1]),
                           ([('As','Kd')],[float('inf')]),([('As','Kd')],[0]),
                           ([('As','Kd'),('Ac','Ad')],[1e308,1e308])]:
        with pytest.raises(ValueError):Range.from_weights(combos,weights,provenance='test')
