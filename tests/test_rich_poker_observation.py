from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/source'
sys.path[:0] = [str(ROOT/'scripts'), str(SOURCE)]
from rich_poker_observation import (RICH_DIM, OLD_DIM, MAX_EVENTS, CARDS, CARD_INDEX,
    HISTORY_OFFSET, KIND_INDEX, encode_game, encode_bridge, history_start, append_action)
from fullhouse_runtime import Bridge
from poker_public_state import PublicState
from training import game


def legacy(legal=None):
    x = np.zeros(51, np.float32)
    x[46:51] = np.array([0,1,0,0,0] if legal is None else legal)
    return x


def material(public):
    """Independent public ledger -> simulation-array adapter for parity tests."""
    state = np.zeros(59, np.int32)
    for p in public.players:
        state[p.seat] = p.stack
        state[6+p.seat] = p.bet_this_street
        state[12+p.seat] = p.hand_commit
        state[18+p.seat] = p.is_folded
        state[24+p.seat] = p.is_all_in
    state[36] = public.pot; state[37] = public.current_bet
    state[40] = public.seat_to_act; state[41] = ('preflop','flop','turn','river').index(public.street)
    state[43] = public.dealer; state[44] = 6; state[52] = public.min_raise_increment
    holes = np.zeros((6,2), np.int8); holes[public.seat_to_act] = [CARD_INDEX[c] for c in public.hero_hole]
    deck = np.zeros(52, np.int8); deck[12:12+len(public.board)] = [CARD_INDEX[c] for c in public.board]
    history = tuple((('preflop','flop','turn','river').index(e.street),
                     e.seat if e.seat is not None else -1,KIND_INDEX[e.kind],e.paid,e.to or 0)
                    for e in public.events)
    return state, deck, holes, history


def test_rich_shape_legacy_prefix_and_hidden_information_invariance():
    state, deck, holes = game.new_hand(6, 0, np.random.default_rng(7))
    seat = int(state[40]); legal = game.legal_actions(state,seat); old = legacy(legal)
    history = history_start(state)
    a = encode_game(state,deck,holes,seat,legal,history,old)
    assert a.shape == (RICH_DIM,) and RICH_DIM == 2885 and a.dtype == np.float32
    assert np.array_equal(a[:OLD_DIM],old)
    changed_holes = holes.copy(); changed_holes[[i for i in range(6) if i != seat]] = 51
    changed_deck = deck.copy(); changed_deck[12:] = deck[12:][::-1]
    b = encode_game(state,changed_deck,changed_holes,seat,legal,history,old)
    assert np.array_equal(a,b)
    # The default legacy-prefix computation must also use only observable
    # cards, rather than trusting a pre-supplied feature fixture alone.
    actual=encode_game(state,deck,holes,seat,legal,history)
    changed=encode_game(state,changed_deck,changed_holes,seat,legal,history)
    assert np.array_equal(actual,changed)
    with pytest.raises(ValueError,match='current actor'):
        encode_game(state,deck,holes,(seat+1)%6,legal,history,old)


def test_history_is_immutable_and_preserves_incremental_calls_and_deals():
    state,deck,holes = game.new_hand(6,0,np.random.default_rng(12)); first=history_start(state)
    child=state.copy(); game.step(child,deck,holes,1)
    branch=append_action(first,state,child,1)
    assert len(first)==2 and len(branch)==3 and first==branch[:2]
    assert branch[-1]==(0,3,KIND_INDEX['call'],100,0)
    state=child; history=branch
    while int(state[41]) == 0 and not state[45]:
        child=state.copy(); game.step(child,deck,holes,1)
        history=append_action(history,state,child,1);state=child
    assert history[-1]==(1,-1,KIND_INDEX['deal'],0,0)
    seat=int(state[40]); legal=game.legal_actions(state,seat)
    encoded=encode_game(state,deck,holes,seat,legal,history,legacy(legal))
    mutated=deck.copy(); mutated[15:]=mutated[15:][::-1]
    assert np.array_equal(encoded,encode_game(state,mutated,holes,seat,legal,history,legacy(legal)))


def test_history_overflow_is_rejected_instead_of_truncated():
    state,deck,holes=game.new_hand(6,0,np.random.default_rng(4));seat=int(state[40]);legal=game.legal_actions(state,seat)
    event=(0,3,KIND_INDEX['check'],0,0)
    with pytest.raises(ValueError,match='truncation forbidden'):
        encode_game(state,deck,holes,seat,legal,(event,)*(MAX_EVENTS+1),legacy(legal))


def test_bridge_and_array_observations_preserve_known_old_collisions():
    from audit_poker_information import witness_requests
    bridge=Bridge()
    try:
        for case in witness_requests():
            encoded=[]
            for request in case['requests']:
                observed=bridge.call(request);public=PublicState.from_bridge(observed)
                state,deck,holes,history=material(public);old=legacy()
                by_bridge=encode_bridge(observed,old)
                by_game=encode_game(state,deck,holes,public.seat_to_act,old[46:51],history,old)
                assert np.array_equal(by_bridge,by_game)
                encoded.append(by_bridge)
            assert np.array_equal(encoded[0][:51],encoded[1][:51])
            assert not np.array_equal(encoded[0][51:],encoded[1][51:])
    finally:bridge.close()


def test_hidden_fields_and_illegal_legacy_features_are_rejected():
    bridge=Bridge()
    try:
        observed=bridge.call(dict(op='replay',hero=3,button=0,hole=['As','Kd'],board=[],actions=[]))
        with pytest.raises(ValueError,match='unknown/hidden'):
            encode_bridge(dict(observed,opponent_holes=[['Ac','Ad']]),legacy())
        bad=legacy();bad[3]=np.nan
        with pytest.raises(ValueError,match='Finite legacy'):
            encode_bridge(observed,bad)
    finally:bridge.close()
