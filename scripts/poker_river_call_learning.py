"""Calibrate river showdown equity against a reached bet, using only the hero's view."""
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import math
import random

import eval7
import numpy as np

from controlled_poker_refit import read
from fullhouse_v3_common import digest
from poker_flop_repair import _ref
from poker_river_dominance import DECK, STARTING_STACK, RiverDominancePolicy, closing_call_floor


FORMAT = 'selfplay-river-call-learning-v1'
FEATURES = ('uniform_logit', 'call_pot_odds', 'log_pot_bb', 'opponent_all_in', 'hero_checked_river',
            'earlier_opponent_raises', 'earlier_hero_raises', 'earlier_hero_calls',
            'board_distinct_ranks', 'board_max_suit_count')
MASS = .25
PENALTY = .02


def river_events(state):
    actions = state['action_log']
    for i in reversed(range(len(actions))):
        if actions[i]['action'] == 'deal':
            return actions[:i], actions[i+1:]
    raise ValueError('Missing river deal in public action record')


def eligible(state):
    if state['street'] != 'river' or state['terminal'] or state['can_check']:
        return False
    players = state['players']
    active = [p for p in players if not p['is_folded']]
    if len(players) != 6 or len(active) != 2:
        return False
    hero = next(p for p in active if p['seat'] == state['seat_to_act'])
    opponent = next(p for p in active if p is not hero)
    owed = state['amount_owed']
    if (not 0 < owed <= hero['stack'] or opponent['bet_this_street'] - hero['bet_this_street'] != owed
            or hero['stack']-owed != opponent['stack']):
        return False
    contributions = [STARTING_STACK-p['stack'] for p in players]
    if max(contributions) > STARTING_STACK-opponent['stack'] or sum(contributions) != state['pot']:
        return False
    _, river = river_events(state)
    wagers = [event['seat'] for event in river if event['action'] in ('bet', 'raise')]
    return (wagers == [opponent['seat']]
            and all(e['action'] == 'check' for e in river if e.get('seat') == hero['seat']))


@lru_cache(maxsize=16384)
def uniform_equity(hole, board):
    hole = [eval7.Card(c) for c in hole]
    board = [eval7.Card(c) for c in board]
    known = set(hole+board)
    if len(known) != 7 or len(hole) != 2 or len(board) != 5:
        raise ValueError('Seven distinct visible river cards required')
    rank = eval7.evaluate(board+hole)
    total = 0.
    for a, b in combinations((c for c in DECK if c not in known), 2):
        other = eval7.evaluate(board+[a,b])
        total += float(rank > other) + .5*float(rank == other)
    return total / 990.


def features(state):
    hero = state['seat_to_act']
    opponent = next(p for p in state['players'] if not p['is_folded'] and p['seat'] != hero)
    board = state['community_cards']
    e = min(.9995, max(.0005, uniform_equity(tuple(sorted(state['your_cards'])), tuple(sorted(board)))))
    offset = math.log(e/(1.-e))
    before, river = river_events(state)
    def count_prior(seat, actions):
        return sum(event.get('seat') == seat and event['action'] in actions for event in before)
    return np.asarray([offset, state['amount_owed']/(state['pot']+state['amount_owed']),
                       math.log1p(state['pot']/100.), float(opponent['is_all_in']),
                       float(any(e.get('seat') == hero and e['action'] == 'check' for e in river)),
                       count_prior(opponent['seat'], ('bet','raise')), count_prior(hero, ('bet','raise')),
                       count_prior(hero, ('call',)), len({c[:-1] for c in board}),
                       max(Counter(c[-1] for c in board).values())], dtype=np.float64)


def passive_update(reply, prefer, seed):
    opposite = 'fold' if prefer == 'call' else 'call'
    probs = {row['action']['kind']: row['probability'] for row in reply['distribution']
             if row['action']['kind'] in ('call','fold')}
    shift = MASS * probs.get(opposite, 0.)
    if shift == 0.:
        return reply
    probs[opposite] -= shift
    probs[prefer] = probs.get(prefer, 0.) + shift
    dist = [dict(action={'kind': k}, probability=p) for k,p in probs.items() if p>0.]
    dist.extend(row for row in reply['distribution'] if row['action']['kind'] not in probs)
    # Couple to the baseline sample, retaining the exact seeded raise decision.
    action = reply['action']
    if action['kind'] == opposite and random.Random(f'river-call-passive:{seed}').random() < MASS:
        action = {'kind': prefer}
    return dict(reply, distribution=dist, action=action, policy_source='learned_river_call',
                passive_probability_shift=shift)


class RiverCallPolicy:
    def __init__(self, manifest):
        path = Path(manifest).resolve()
        self.digest = digest(path)
        spec = read(path)
        self.model = None
        if spec['format'] != FORMAT:
            self.parent = RiverDominancePolicy(path)
            return
        base = _ref(spec['baseline'], path.parent)
        if read(base)['seed'] != spec['seed']:
            raise ValueError('River call policy seed differs from the selected parent')
        self.parent = RiverDominancePolicy(base)
        model = read(_ref(spec['model'], path.parent))
        size = len(FEATURES)
        if (model['format'] != FORMAT or tuple(model['features']) != FEATURES
                or model['penalty'] != PENALTY or model['passive_shift'] != MASS):
            raise ValueError('Unexpected river call model specification')
        self.center, self.scale, self.coef = [np.asarray(model[k], dtype=np.float64)
                                             for k in ('center','scale','coefficients')]
        self.intercept = float(model['intercept'])
        if (any(v.shape != (size,) or not np.isfinite(v).all() for v in (self.center,self.scale,self.coef))
                or np.any(self.scale < .1) or not math.isfinite(self.intercept)):
            raise ValueError('Invalid learned showdown calibration')
        self.model = model

    def decide(self, state, seed):
        parent = self.parent.decide(state, seed)
        if self.model is None or not eligible(state) or parent.get('call_over_fold_minimum_chips', 0)>0:
            return parent
        # The existing guard may have correctly omitted a floor when there was
        # no fold to redirect. The learned layer must also leave those hands alone.
        if closing_call_floor(state)>0:
            return parent
        x = features(state)
        logit = x[0]+self.intercept+self.coef @ ((x-self.center)/self.scale)
        odds = state['amount_owed']/(state['pot']+state['amount_owed'])
        margin = logit-math.log(odds/(1.-odds))
        if not np.isfinite(margin):
            raise ValueError('Nonfinite river call decision')
        if margin == 0:
            return parent
        return passive_update(parent, 'call' if margin>0 else 'fold', seed)


if __name__ == '__main__':
    from serve_poker_flop_repair import main
    main(RiverCallPolicy, 'river-call-learning')
