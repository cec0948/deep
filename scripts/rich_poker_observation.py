"""Lossless public-history side input for a new six-player policy family.

The first51 values retain the legacy teacher interface. Everything thereafter
uses only the acting player's cards, visible board and public chip/history data.
The fixed100BB, six-seat rules contract bounds amounts and history; overflow is
an error, never a silently truncated observation.
"""
from __future__ import annotations

import importlib
import numbers

import numpy as np

from poker_public_state import PublicState

OLD_DIM = 51
MAX_EVENTS = 515
EVENT_WIDTH = 5
CARD_DIM = 208
PLAYER_DIM = 30
CONTEXT_DIM = 21
CARD_OFFSET = OLD_DIM
PLAYER_OFFSET = CARD_OFFSET + CARD_DIM
CONTEXT_OFFSET = PLAYER_OFFSET + PLAYER_DIM
HISTORY_OFFSET = CONTEXT_OFFSET + CONTEXT_DIM
RICH_DIM = HISTORY_OFFSET + MAX_EVENTS * EVENT_WIDTH
SCHEMA = 'rich-public-poker-observation-v1'
AMOUNT_SCALE = 60000.
KINDS = ('small_blind', 'big_blind', 'fold', 'check', 'call', 'raise', 'deal')
KIND_INDEX = {name: index for index, name in enumerate(KINDS)}
STREETS = ('preflop', 'flop', 'turn', 'river')
VISIBLE = (0, 3, 4, 5)
CARDS = tuple(rank+suit for suit in 'cdhs' for rank in '23456789TJQKA')
CARD_INDEX = {name: index for index, name in enumerate(CARDS)}

# Frozen training.game contract; keeping constants here avoids importing a
# source tree merely to encode a public JSON observation.
S_STACKS, S_BETS, S_INVESTED, S_FOLDED, S_ALLIN = 0, 6, 12, 18, 24
S_POT, S_CURBET, S_TOACT, S_STREET, S_DEALER = 36, 37, 40, 41, 43
S_NPLAYERS, S_TERMINAL, S_LAST_FULL_RAISE = 44, 45, 52


def _integer(value, label, low=0, high=60000):
    if not isinstance(value, numbers.Integral) or isinstance(value, (bool, np.bool_)):
        raise ValueError('Integer required: '+label)
    result = int(value)
    if not low <= result <= high:
        raise ValueError('Out-of-contract '+label)
    return result


def _history(history):
    if not isinstance(history, (tuple, list)) or len(history) > MAX_EVENTS:
        raise ValueError('Public history exceeds515events; truncation forbidden')
    rows = []
    for event in history:
        if not isinstance(event, (tuple, list)) or len(event) != EVENT_WIDTH:
            raise ValueError('Expected(street,actor,kind,paid,raise-to)event')
        street = _integer(event[0], 'event street', high=3)
        actor = _integer(event[1], 'event actor', low=-1, high=5)
        kind = _integer(event[2], 'event kind', high=len(KINDS)-1)
        paid = _integer(event[3], 'event paid')
        to = _integer(event[4], 'event raise-to')
        if (kind == KIND_INDEX['deal']) != (actor == -1):
            raise ValueError('Only public deal markers may omit an actor')
        if kind == KIND_INDEX['deal'] and (paid or to):
            raise ValueError('Deal markers carry no hidden cards or chips')
        if kind != KIND_INDEX['raise'] and to != 0:
            raise ValueError('Only raises carry a raise-to total')
        rows.append((street, actor, kind, paid, to))
    return tuple(rows)


def _legacy(values):
    x = np.asarray(values, dtype=np.float32)
    if x.shape != (OLD_DIM,) or not np.isfinite(x).all():
        raise ValueError('Finite legacy51vector required')
    if not np.isin(x[46:51], [0., 1.]).all() or not x[46:51].any():
        raise ValueError('Legacy legal-action mask required')
    return x


def _encode(old51, hole, board, players, dealer, actor, street, pot,
            current_bet, min_raise, owed, history):
    old51 = _legacy(old51); history = _history(history)
    hole = tuple(_integer(v, 'own card', high=51) for v in hole)
    board = tuple(_integer(v, 'visible board card', high=51) for v in board)
    if len(hole) != 2 or len(set(hole+board)) != len(hole+board):
        raise ValueError('Own/visible cards must be different')
    street = _integer(street, 'street', high=3)
    if len(board) != VISIBLE[street]:
        raise ValueError('Visible board/street mismatch')
    dealer = _integer(dealer, 'dealer', high=5)
    actor = _integer(actor, 'actor', high=5)
    x = np.zeros(RICH_DIM, dtype=np.float32); x[:OLD_DIM] = old51
    x[CARD_OFFSET+np.asarray(hole)] = 1.
    if street:
        x[CARD_OFFSET+52+np.asarray(board[:3])] = 1.
    if street >= 2:
        x[CARD_OFFSET+104+board[3]] = 1.
    if street >= 3:
        x[CARD_OFFSET+156+board[4]] = 1.
    if len(players) != 6:
        raise ValueError('Six public player records required')
    for seat, row in enumerate(players):
        if len(row) != 5:
            raise ValueError('Expected public stack/bet/invested/folded/allin')
        numeric = [_integer(v, 'public chips')/AMOUNT_SCALE for v in row[:3]]
        flags = [_integer(v, 'public player flag', high=1) for v in row[3:]]
        x[PLAYER_OFFSET+seat*5:PLAYER_OFFSET+(seat+1)*5] = numeric+flags
    c = CONTEXT_OFFSET
    x[c+dealer] = 1.; x[c+6+actor] = 1.; x[c+12+street] = 1.
    x[c+16:c+20] = [_integer(v, 'public context chips')/AMOUNT_SCALE
                    for v in (pot, current_bet, min_raise, owed)]
    x[c+20] = len(history)/MAX_EVENTS
    if history:
        values = np.asarray(history, dtype=np.float32)
        values[:, 0] /= 3.
        values[:, 1] = (values[:, 1]+1.)/6.
        values[:, 2] = (values[:, 2]+1.)/len(KINDS)
        values[:, 3:] /= AMOUNT_SCALE
        x[HISTORY_OFFSET:HISTORY_OFFSET+len(history)*EVENT_WIDTH] = values.ravel()
    return x


def encode_game(state, deck, hole_cards, seat, legal, history, legacy_features=None):
    """Encode one nonterminal frozen Python-game decision without hidden leakage."""
    state = np.asarray(state)
    if state.ndim != 1 or len(state) != 59 or state[S_TERMINAL] or state[S_NPLAYERS] != 6:
        raise ValueError('A live six-player frozen game state is required')
    seat = _integer(seat, 'acting seat', high=5)
    if seat != state[S_TOACT]:
        raise ValueError('Only the current actor may observe its private cards')
    legal = np.asarray(legal, dtype=np.float32)
    if legal.shape != (5,) or not np.isin(legal, [0., 1.]).all() or not legal.any():
        raise ValueError('A nonempty five-action mask is required')
    if legacy_features is None:
        features = importlib.import_module('bot.features')
        legacy_features = features.encode_state(state, hole_cards, deck, seat, legal,
                                                 features.load_preflop_equity())
    old51 = _legacy(legacy_features)
    if not np.array_equal(old51[46:51], legal):
        raise ValueError('Legacy/public legal mask mismatch')
    street = int(state[S_STREET])
    players = [(int(state[S_STACKS+i]), int(state[S_BETS+i]),
                int(state[S_INVESTED+i]), int(state[S_FOLDED+i]), int(state[S_ALLIN+i]))
               for i in range(6)]
    return _encode(old51, np.asarray(hole_cards)[seat],
                   np.asarray(deck)[12:12+VISIBLE[street]], players,
                   int(state[S_DEALER]), seat, street, int(state[S_POT]),
                   int(state[S_CURBET]), int(state[S_LAST_FULL_RAISE]),
                   max(0, int(state[S_CURBET]-state[S_BETS+seat])), history)


def encode_bridge(observation, old51):
    """Identical rich layout from a validated trusted-engine public observation."""
    public = PublicState.from_bridge(observation)
    history = tuple((STREETS.index(event.street),
                     -1 if event.seat is None else event.seat,
                     KIND_INDEX[event.kind], event.paid, event.to or 0)
                    for event in public.events)
    players = [(p.stack, p.bet_this_street, p.hand_commit, int(p.is_folded),
                int(p.is_all_in)) for p in public.players]
    return _encode(old51, [CARD_INDEX[c] for c in public.hero_hole],
                   [CARD_INDEX[c] for c in public.board], players, public.dealer,
                   public.seat_to_act, STREETS.index(public.street), public.pot,
                   public.current_bet, public.min_raise_increment,
                   public.current_bet-public.players[public.seat_to_act].bet_this_street,
                   history)


def history_start(initial_state):
    state = np.asarray(initial_state)
    if (state.shape != (59,) or state[S_TERMINAL] or state[S_NPLAYERS] != 6
            or state[S_STREET] != 0 or state[S_POT] != 150
            or np.any(state[S_FOLDED:S_FOLDED+6])):
        raise ValueError('History must start at the fixed six-seat blind posting')
    dealer = int(state[S_DEALER]); sb, bb = (dealer+1)%6, (dealer+2)%6
    if state[S_INVESTED+sb] != 50 or state[S_INVESTED+bb] != 100:
        raise ValueError('Unexpected blind commitments')
    return ((0, sb, KIND_INDEX['small_blind'], 50, 0),
            (0, bb, KIND_INDEX['big_blind'], 100, 0))


def append_action(history, before, after, action):
    """Return a NEW branch history; calls store paid chips, raises street totals."""
    history = _history(history); before = np.asarray(before); after = np.asarray(after)
    action = _integer(action, 'abstract action', high=4)
    actor, street = int(before[S_TOACT]), int(before[S_STREET])
    owed = max(0, int(before[S_CURBET]-before[S_BETS+actor]))
    stack = int(before[S_STACKS+actor])
    if action == 0:
        kind, paid, to = 'fold', 0, 0
    elif action == 1:
        kind, paid, to = ('call' if owed else 'check'), min(owed, stack), 0
    else:
        paid = (stack if action == 4 else
                min(stack, owed+round((.5 if action == 2 else 1.)*(int(before[S_POT])+owed))))
        kind, to = 'raise', int(before[S_BETS+actor])+paid
    result = history+((street, actor, KIND_INDEX[kind], paid, to),)
    if not after[S_TERMINAL]:
        for new_street in range(street+1, int(after[S_STREET])+1):
            result += ((new_street, -1, KIND_INDEX['deal'], 0, 0),)
    return _history(result)
