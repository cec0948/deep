"""Evaluate one forced flop action on matched cards with a frozen continuation policy."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import random

from poker_public_state import CARDS
from poker_reference_opponents import checked_action
from poker_river_runtime import action_key, legal_menu


LINEUPS = ('self480', 'public', 'tight-value')
POLICIES = ('480', '486', '492', '498', '504')
BUTTON = 0


def _nonce(*parts):
    record = ':'.join(map(str, ('flop-action-values-v1', *parts))).encode('ascii')
    return int.from_bytes(hashlib.sha256(record).digest()[:8], 'big')


def source_cards(bridge, deal_seed):
    """Opening six check/calls disclose the original private hands to the simulator only."""
    state = bridge.call(dict(op='reset', seed=deal_seed, button=BUTTON))
    holes = {}
    for _ in range(6):
        if state.get('terminal') or state['street'] != 'preflop':
            raise ValueError('Incomplete source deal')
        seat = state['seat_to_act']
        if seat in holes:
            raise ValueError('Preflop traversal repeated a seat')
        holes[seat] = state['your_cards']
        action = {'seat': seat, 'kind': 'check' if state['can_check'] else 'call'}
        state = bridge.call(dict(op='step', action=action))
    if state.get('street') != 'flop' or set(holes) != set(range(6)):
        raise ValueError('Original deal failed to reach the flop')
    cards = tuple(c for seat in range(6) for c in holes[seat]) + tuple(state['community_cards'])
    if len(cards) != 15 or len(set(cards)) != 15 or not set(cards).issubset(CARDS):
        raise ValueError('Invalid private cards or flop')
    return cards


def future_deck(fixed, deal_seed, replicate):
    """Retain the twelve original hole cards and flop; uniformly draw the remaining deck."""
    remaining = [card for card in CARDS if card not in fixed]
    if len(remaining) != 37 or len(fixed) != 15:
        raise ValueError('Expected twelve original hole cards and a three-card flop')
    random.Random(_nonce('deck', deal_seed, replicate)).shuffle(remaining)
    return [*fixed, *remaining]


def _actor_view(state):
    # The explicit-deck rules oracle also emits a public per-seat cumulative chip
    # commitment. Policies were evaluated on Bridge's observation layout.
    return dict(state, players=[{k:v for k,v in row.items() if k != 'hand_commit'}
                                for row in state['players']])


def _support(sample):
    legal = sample['legal_actions']
    root = sample['root_observation']
    if (root.get('terminal') or root['street'] != 'flop' or root['seat_to_act'] != sample['hero']
            or root['dealer'] != BUTTON):
        raise ValueError('Sample is not a decision by the acting flop hero')
    keys = [action_key(a) for a in legal]
    if len(set(keys)) != len(legal) or set(keys) != set(legal_menu(root)):
        raise ValueError('Mismatched or duplicate concrete legal actions')
    active = set()
    if set(sample['policies']) != set(POLICIES):
        raise ValueError('Incomplete frozen root strategies')
    for label in POLICIES:
        probabilities = sample['policies'][label]['probabilities']
        if (len(probabilities) != len(legal) or any(not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-7)):
            raise ValueError('Invalid root probability vector')
        active.update(i for i, p in enumerate(probabilities) if p > 0)
    return sorted(active)


def continuation_action(policy, view, nonce):
    # Profiles implement the same mixed strategy with different internal RNG
    # conventions. Use the actual legal distribution and a common uniform draw.
    distribution = policy.decide(view, nonce)['distribution']
    rows = sorted(distribution, key=lambda row: _action_order(row['action']))
    weights = [row['probability'] for row in rows]
    total = math.fsum(weights)
    if (not rows or any(not math.isfinite(p) or p < 0 for p in weights)
            or not math.isclose(total, 1.0, abs_tol=1e-7)):
        raise ValueError('Invalid continuation strategy')
    u = ((nonce >> 11) / (1 << 53)) * total
    mass = 0.
    for row in rows:
        mass += row['probability']
        if u < mass:
            return checked_action(view, row['action'])
    # For rounding at the last ulp, return the last supported action.
    return checked_action(view, next(row['action'] for row in reversed(rows) if row['probability'] > 0))


def _action_order(action):
    return (('fold', 'check', 'call', 'raise').index(action['kind']), action.get('to', 0))


def _play(rules, sample, deck, forced, parent, opponent, replicate, record):
    hero, deal_seed = sample['hero'], sample['deal_seed']
    state = rules.call(dict(op='reset', deck=deck, button=BUTTON))
    for action in sample['prefix']:
        state = rules.call(dict(op='step', action=action))
    if state['terminal'] or _actor_view(state) != sample['root_observation']:
        raise ValueError('Sampled root changed under reconstruction or future-card redraw')
    action = dict(checked_action(state, forced), seat=hero)
    trace = [action] if record else None
    state = rules.call(dict(op='step', action=action))
    ordinals = Counter()
    for _ in range(512 - len(sample['prefix'])):
        if state['terminal']:
            utilities = state['utilities']
            if (len(utilities) != 6 or any(type(u) is not int for u in utilities) or sum(utilities) != 0
                    or (forced['kind'] == 'fold' and utilities[hero] != sample['root_observation']['your_stack'] - 10000)):
                raise ValueError('Invalid six-seat settlement or fold payoff')
            return utilities, trace
        actor = state['seat_to_act']
        key = (state['street'], actor)
        nonce = _nonce('action', deal_seed, hero, replicate, *key, ordinals[key])
        ordinals[key] += 1
        view = _actor_view(state)
        decision = continuation_action(parent if actor == hero else opponent, view, nonce)
        action = dict(decision, seat=actor)
        if record:
            trace.append(action)
        state = rules.call(dict(op='step', action=action))
    raise ValueError('Excessively long forced branch')


def rollout_state(sample, bridge, rules, parent, opponents, repeats=8):
    """Return per-lineup utilities and replayable traces for replicate zero.

    The hero follows parent after the one forced root decision in every lineup.
    Every branch gets the same private deal, fresh runout and seat/street-indexed
    action randomness. Returned utilities are whole-hand chips (100 chips / BB).
    """
    if set(opponents) != set(LINEUPS) or opponents['self480'] is not parent:
        raise ValueError('Expected three designated lineups and the matched self-play policy')
    if type(repeats) is not int or repeats < 1:
        raise ValueError('Expected a positive number of rollout replicates')
    active = _support(sample)
    header = {key: sample[key] for key in ('state_id', 'seed', 'deal_seed', 'hero')}
    legal = sample['legal_actions']
    values = [dict(header, lineup=field, legal_actions=legal, repeats=repeats,
                   action_payoffs_chips=[[] if i in active else None for i in range(len(legal))])
              for field in LINEUPS]
    traces = []
    fixed = source_cards(bridge, sample['deal_seed'])
    for rep in range(repeats):
        deck = future_deck(fixed, sample['deal_seed'], rep)
        for row in values:
            lineup = row['lineup']
            for i in active:
                utilities, actions = _play(rules, sample, deck, legal[i], parent, opponents[lineup], rep, rep == 0)
                row['action_payoffs_chips'][i].append(utilities[sample['hero']])
                if rep == 0:
                    traces.append(dict(header, lineup=lineup, action_index=i, replicate=rep, deck=deck,
                                       button=BUTTON, prefix=sample['prefix'], actions=actions, utilities=utilities))
    return values, traces
