"""Transparent, untrained opponents for six-seat poker regression checks.

These behavior baselines are independent of the two learned opponents used by
the existing evaluation. They are deliberately simple, exploitable policies,
not independent evidence of professional strength. Scores below are heuristics,
not equity estimates. Only the acting player's observation enters a decision.
"""
import hashlib
import json
import random


PROFILES = ('calling-station', 'tight-value', 'pressure')
SPEC = {
    'version': 'reference-opponents-v1',
    'calling-station': 'Always checks or calls; never folds or raises.',
    'tight-value': 'Folds weak holdings to a price, calls moderate holdings, value raises strong holdings.',
    'pressure': 'Calls a broad range and frequently makes pot-sized pressure raises, including fixed-rate bluffs.',
    'limitations': ['Untrained, exploitable behavioral baselines; no claim of strength',
                    'Hand score is a fixed heuristic, not showdown equity or an opponent range estimate',
                    'No learned model, original hero features, or private opponent cards are used'],
}

_OBSERVATION_KEYS = ('your_cards', 'community_cards', 'street', 'pot', 'your_stack',
                     'your_bet_this_street', 'amount_owed', 'can_check', 'raise_bounds')
_RANKS = '23456789TJQKA'
_CATEGORY = {'High Card': .20, 'Pair': .47, 'Two Pair': .68,
             'Trips': .80, 'Straight': .86, 'Flush': .90,
             'Full House': .95, 'Quads': .98, 'Straight Flush': 1.0}


def observation(state):
    """Discard all fields except explicitly allowed acting-player information."""
    if state.get('terminal'):
        raise ValueError('Cannot act in a terminal state')
    result = {key: state[key] for key in _OBSERVATION_KEYS}
    hole, board = result['your_cards'], result['community_cards']
    cards = hole + board
    if (len(hole) != 2 or len(board) not in (0, 3, 4, 5)
            or len(set(cards)) != len(cards)
            or any(len(c) != 2 or c[0] not in _RANKS or c[1] not in 'cdhs' for c in cards)):
        raise ValueError('Invalid observable cards')
    for name in ('pot', 'your_stack', 'your_bet_this_street', 'amount_owed'):
        if type(result[name]) is not int or result[name] < 0:
            raise ValueError('Invalid public chip amount')
    if type(result['can_check']) is not bool or result['can_check'] != (result['amount_owed'] == 0):
        raise ValueError('Inconsistent check/call observation')
    bounds = result['raise_bounds']
    if bounds is not None and (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(x) is not int for x in bounds) or bounds[0] <= 0
            or bounds[0] > bounds[1]
            or bounds[1] != result['your_stack'] + result['your_bet_this_street']):
        raise ValueError('Invalid engine raise bounds')
    return result


def hand_score(obs):
    """A reproducible made-hand/draw heuristic, explicitly not poker equity."""
    hole, board = obs['your_cards'], obs['community_cards']
    ranks = sorted((_RANKS.index(c[0]) + 2 for c in hole), reverse=True)
    if not board:
        high, low = ranks
        if high == low:
            return min(.98, .55 + .03 * high)
        score = .15 + .025 * high + .012 * low
        score += .06 if hole[0][1] == hole[1][1] else 0
        score += .04 if high - low <= 1 else -.03 if high - low > 4 else 0
        return min(.95, max(0., score))
    import eval7
    category = eval7.handtype(eval7.evaluate([eval7.Card(c) for c in hole + board]))
    score = _CATEGORY[category]
    board_ranks = [_RANKS.index(c[0]) + 2 for c in board]
    if category == 'High Card':
        score += .008 * ranks[0]
    elif category == 'Pair':
        # A public-board pair alone is weaker than pairing a private card.
        private_pair = ranks[0] == ranks[1] or any(r in board_ranks for r in ranks)
        score += .08 if private_pair else -.12
    if len(board) < 5:
        cards = hole + board
        if max(sum(c[1] == suit for c in cards) for suit in 'cdhs') == 4:
            score += .09
        known_ranks = {_RANKS.index(c[0]) + 2 for c in cards}
        if 14 in known_ranks:
            known_ranks.add(1)
        if any(len(known_ranks.intersection(range(low, low + 5))) == 4 for low in range(1, 11)):
            score += .07
    return min(1., score)


def checked_action(obs, action):
    """Validate against the trusted engine's concrete legal-action boundary."""
    kind = action.get('kind')
    if kind == 'raise':
        bounds = obs['raise_bounds']
        if set(action) != {'kind', 'to'} or bounds is None or type(action['to']) is not int:
            raise ValueError('Invalid raise')
        if not bounds[0] <= action['to'] <= bounds[1]:
            raise ValueError('Raise outside engine bounds')
    elif set(action) != {'kind'} or kind not in (('check',) if obs['can_check'] else ('fold', 'call')):
        raise ValueError('Illegal passive action')
    return action


def distribution(profile, state):
    if profile not in PROFILES:
        raise ValueError('Unknown reference opponent')
    obs = observation(state)
    passive = {'kind': 'check' if obs['can_check'] else 'call'}
    if profile == 'calling-station':
        return [dict(action=passive, probability=1.)], dict(hand_score=None, pot_odds=None)
    score = hand_score(obs)
    price = min(obs['amount_owed'], obs['your_stack'])
    pot_odds = price / max(1, obs['pot'] + price)
    if profile == 'tight-value':
        threshold = (.47 if not obs['community_cards'] else .36) + .40 * pot_odds
        fold = 0. if obs['can_check'] else .95 if score < threshold else .05
        raise_probability = .72 if score >= .76 else .15 if score >= .64 else 0.
        fraction = .66
    else:
        fold = 0. if obs['can_check'] else .12 if score < .35 and pot_odds > .32 else .02
        raise_probability = .65 if score >= .68 else .38
        fraction = 1.
    if obs['raise_bounds'] is None:
        raise_probability = 0.
    raise_probability = min(raise_probability, 1. - fold)
    candidates = [(passive, 1. - fold - raise_probability)]
    if fold:
        candidates.append(({'kind': 'fold'}, fold))
    if raise_probability:
        low, high = obs['raise_bounds']
        to = obs['your_bet_this_street'] + obs['amount_owed'] + round(fraction * (obs['pot'] + obs['amount_owed']))
        candidates.append(({'kind': 'raise', 'to': max(low, min(high, to))}, raise_probability))
    result = [dict(action=checked_action(obs, action), probability=probability)
              for action, probability in candidates if probability > 0.]
    if abs(sum(x['probability'] for x in result) - 1.) > 1e-12:
        raise AssertionError('Invalid probability normalization')
    return result, dict(hand_score=score, pot_odds=pot_odds)


class ReferenceOpponent:
    def __init__(self, profile):
        if profile not in PROFILES:
            raise ValueError('Unknown reference opponent')
        self.profile = profile

    def decide(self, state, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError('Nonnegative integer action seed required')
        probabilities, metrics = distribution(self.profile, state)
        # Separate random streams for opponent profiles and candidate actions.
        key = hashlib.sha256(f'poker-reference-v1:{self.profile}:{seed}'.encode()).digest()
        rng = random.Random(int.from_bytes(key[:8], 'big'))
        action = rng.choices([x['action'] for x in probabilities],
                             [x['probability'] for x in probabilities])[0]
        return dict(action=action, distribution=probabilities, profile=self.profile,
                    diagnostics=metrics, strength_verified=False)
