"""Training-only conditional root rollouts; deployment remains a neural policy.

For each UTG private-card class, sample the other cards uniformly conditional on
that class. Enumerate legal root actions and follow the frozen current policy
thereafter. Each child payoff is an unbiased Monte Carlo value of that action.
Equal repetitions per class/iteration preserve the intended time-weighted mean
at each root infoset. These samples do not update downstream regrets.
"""
from itertools import combinations

import numpy as np


def hand_classes():
    return ([(r, r, False) for r in range(13)]
            + [(hi, lo, suited) for hi in range(13) for lo in range(hi) for suited in (False, True)])


def card_pairs(hand):
    hi, lo, suited = hand
    if hi == lo:
        return [(13*a+hi, 13*b+lo) for a, b in combinations(range(4), 2)]
    return [(13*a+hi, 13*b+lo) for a in range(4) for b in range(4) if (a == b) == suited]


def conditional_deck(cards, seat, rng):
    if len(set(cards)) != 2 or not 0 <= seat < 6:
        raise ValueError('Two distinct private cards and a six-player seat required')
    deck = np.empty(52, np.int8)
    indices = [2*seat, 2*seat+1]
    deck[indices] = cards
    remaining = np.array([c for c in range(52) if c not in cards], np.int8)
    slots = np.ones(52, bool); slots[indices] = False
    deck[slots] = rng.permutation(remaining)
    return deck


def rollout_action_values(state, deck, hole, hero, policy, rng, game, encode):
    """Return legal-action values in BB; all future actors follow policy."""
    legal = game.legal_actions(state, hero)
    values = np.zeros(5, np.float64)
    for action in np.flatnonzero(legal):
        child = state.copy(); game.step(child, deck, hole, int(action)); steps = 0
        while not child[game.S_TERMINAL]:
            seat = int(child[game.S_TOACT]); mask = game.legal_actions(child, seat)
            x = encode(child, hole, deck, seat, mask)
            p = policy.strategy(x, mask.astype(np.float32)).astype(float)
            if not np.isfinite(p).all() or (p < 0).any() or not p.sum() > 0:
                raise FloatingPointError('Invalid rollout policy')
            p *= mask; p /= p.sum()
            a = int(rng.choice(5, p=p))
            game.step(child, deck, hole, a); steps += 1
            if steps > 512: raise RuntimeError('Root rollout failed to terminate')
        values[action] = child[game.S_PAYOFFS+hero]/100.
    return values


class RootReplay:
    """Exact sufficient statistics across all root observations, without eviction."""
    def __init__(self):
        self.rows = {}

    def add(self, features, regret, iteration):
        x = np.asarray(features, np.float32); y = np.asarray(regret, np.float64)
        if x.shape != (51,) or y.shape != (5,) or not np.isfinite(x).all() or not np.isfinite(y).all() or iteration < 1:
            raise ValueError('Invalid root observation')
        key = x.tobytes(); weight = float(iteration)
        if key not in self.rows:
            self.rows[key] = [x.copy(), np.zeros(5), np.zeros(5), 0., 0., 0]
        row = self.rows[key]
        row[1] += weight*y; row[2] += weight*y*y
        row[3] += weight; row[4] += weight*weight; row[5] += 1

    def state_dict(self):
        rows = list(self.rows.values())
        return dict(features=np.asarray([r[0] for r in rows], np.float32).reshape(-1,51),
                    numerator=np.asarray([r[1] for r in rows], np.float64).reshape(-1,5),
                    second_numerator=np.asarray([r[2] for r in rows], np.float64).reshape(-1,5),
                    mass=np.asarray([r[3] for r in rows], np.float64),
                    squared_mass=np.asarray([r[4] for r in rows], np.float64),
                    count=np.asarray([r[5] for r in rows], np.int64))

    def load_state_dict(self, state):
        n = len(state['features'])
        shapes = dict(features=(n,51),numerator=(n,5),second_numerator=(n,5),mass=(n,),squared_mass=(n,),count=(n,))
        for name, shape in shapes.items():
            if state[name].shape != shape or not np.isfinite(state[name]).all():
                raise ValueError('Invalid root replay array: '+name)
        self.rows = {}
        for x, num, second, mass, squares, count in zip(state['features'], state['numerator'], state['second_numerator'], state['mass'], state['squared_mass'], state['count']):
            if mass <= 0 or count <= 0 or squares <= 0 or not np.isfinite(num).all() or not np.isfinite(second).all():
                raise ValueError('Invalid root replay checkpoint')
            self.rows[x.tobytes()] = [x.copy(), num.copy(), second.copy(), float(mass), float(squares), int(count)]
        if len(self.rows) != len(state['features']): raise ValueError('Duplicate root replay input')

    def means(self):
        state = self.state_dict()
        return state['features'], state['numerator']/state['mass'][:,None], state['mass']

    def diagnostics(self):
        state = self.state_dict()
        if not len(state['count']): return dict(classes=0, observations=0)
        return dict(classes=len(state['count']), observations=int(state['count'].sum()),
                    minimum_observations=int(state['count'].min()), median_observations=float(np.median(state['count'])),
                    minimum_weight_ess=float(np.min(state['mass']**2/state['squared_mass'])))


def collect_root_replay(net, rng, replay, iteration, repetitions, hero=0):
    from training import game as g
    from training.traverse_fast import NumpyPolicy, _encode_fast
    policy = NumpyPolicy(net)
    dealer = (hero-3) % 6
    count = 0
    for hand in hand_classes():
        pairs = card_pairs(hand)
        for _ in range(repetitions):
            cards = pairs[int(rng.integers(0,len(pairs)))]; deck = conditional_deck(cards,hero,rng)
            class DeckRng:
                def permutation(self, n):
                    if n != 52: raise ValueError('Unexpected deal request')
                    return deck.copy()
            state, dealt, hole = g.new_hand(6,dealer,DeckRng())
            if state[g.S_TOACT] != hero: raise AssertionError('Expected initial UTG node')
            legal = g.legal_actions(state,hero); x = _encode_fast(state,hole,dealt,hero,legal)
            sigma = policy.strategy(x,legal.astype(np.float32)).astype(float); sigma /= sigma.sum()
            values = rollout_action_values(state,dealt,hole,hero,policy,rng,g,_encode_fast)
            regret = (values-float(sigma@values))*legal
            replay.add(x,regret,iteration); count += 1
    return count
