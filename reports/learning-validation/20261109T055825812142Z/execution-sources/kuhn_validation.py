"""Exact two-player Kuhn oracle and an adapter to the actual V3 CFR components.

The oracle enumerates all 64 pure strategies per player, respecting information
sets. The learner calls the production external traversal, reservoirs, neural
networks and average collector. Only game rules and features are substituted.
Complete 12-infoset features deliberately avoid poker abstraction error.
"""
from contextlib import ExitStack, contextmanager
from itertools import permutations, product
from unittest.mock import patch

import numpy as np

HISTORIES = ('', 'c', 'b', 'cb')
DEALS = tuple(permutations(range(3), 2))
OWNER = np.repeat([0, 1, 1, 0], 3)
NEXT = {('', 0): 'c', ('', 1): 'b', ('c', 0): 'cc', ('c', 1): 'cb',
        ('b', 0): 'bf', ('b', 1): 'bc', ('cb', 0): 'cbf', ('cb', 1): 'cbc'}


def terminal_value(history, cards):
    if history in ('bf', 'cbf'):
        return 1. if history == 'bf' else -1.
    if history in ('cc', 'bc', 'cbc'):
        stake = 1. if history == 'cc' else 2.
        return stake if cards[0] > cards[1] else -stake
    return None


def policy_value(profile):
    def visit(history, cards):
        value = terminal_value(history, cards)
        if value is not None: return value
        owner = len(history) % 2
        index = HISTORIES.index(history)*3 + cards[owner]
        return sum(profile[index, a]*visit(NEXT[history, a], cards) for a in (0, 1))
    return sum(visit('', cards) for cards in DEALS)/len(DEALS)


def exact_metrics(profile):
    profile = np.asarray(profile, float)
    if profile.shape != (12, 2) or (profile < -1e-10).any() or not np.allclose(profile.sum(1), 1):
        raise ValueError('Invalid Kuhn policy')
    value = policy_value(profile)
    best = []
    for player in (0, 1):
        indices = np.flatnonzero(OWNER == player)
        values = []
        for actions in product((0, 1), repeat=len(indices)):
            deviation = profile.copy()
            deviation[indices] = np.eye(2)[np.asarray(actions)]
            values.append((1 if player == 0 else -1)*policy_value(deviation))
        best.append(max(values))
    nash_conv = best[0]+best[1]
    return dict(value_player0=value, best_response_values=best, nash_conv=nash_conv,
                exploitability=nash_conv/2,
                definition='half the sum of unilateral best-response improvements, in antes/hand')


def equilibrium():
    # Standard alpha=1/3 member of Kuhn's equilibrium family. Rows J,Q,K.
    aggressive = np.array([1/3, 0, 1, 1/3, 0, 1, 0, 1/3, 1, 0, 2/3, 1])
    return np.stack([1-aggressive, aggressive], 1)


def regret_matching(regrets):
    positive = np.maximum(regrets, 0.)
    totals = positive.sum(1, keepdims=True)
    return np.divide(positive, totals, out=np.full_like(positive, .5), where=totals > 0)


def tabular_reference(iterations=5000):
    """Full-tree, simultaneous, linearly weighted CFR; no neural code used."""
    regrets = np.zeros((12, 2)); total = np.zeros((12, 2))
    for t in range(1, iterations+1):
        profile = regret_matching(regrets)
        delta = np.zeros((12, 2))
        def visit(history, cards, reach):
            terminal = terminal_value(history, cards)
            if terminal is not None: return terminal
            player = len(history) % 2
            index = HISTORIES.index(history)*3 + cards[player]
            values = []
            for action in (0, 1):
                child_reach = reach.copy(); child_reach[player] *= profile[index, action]
                values.append(visit(NEXT[history, action], cards, child_reach))
            values = np.asarray(values)
            value = profile[index] @ values
            delta[index] += reach[1-player]*(1 if player == 0 else -1)*(values-value)/len(DEALS)
            total[index] += t*reach[player]*profile[index]/len(DEALS)
            return value
        for cards in DEALS: visit('', cards, np.ones(2))
        regrets += t*delta
    average = total/total.sum(1, keepdims=True)
    return dict(iterations=iterations, profile=average.tolist(), **exact_metrics(average))


def features():
    x = np.zeros((12, 51), np.float32)
    x[np.arange(12), np.arange(12)] = 1.
    x[:, 17] = 1.
    x[np.arange(12), 21+OWNER] = 1.
    x[:, 27] = .015
    x[:, 46:48] = 1.
    return x


def own_reach(profile):
    reach = np.ones(12)
    reach[9:12] = profile[0:3, 0]
    return reach


def buffer_average(buffer):
    n = len(buffer)
    indices = np.argmax(buffer.features[:n, :12], 1)
    weights = buffer.iterations[:n].astype(float)*buffer.sample_weights[:n]
    mass = np.bincount(indices, weights=weights, minlength=12)
    num = np.stack([np.bincount(indices, weights=weights*buffer.advantages[:n, a], minlength=12)
                    for a in (0, 1)], 1)
    return np.divide(num, mass[:, None], out=np.full((12, 2), .5), where=mass[:, None] > 0), mass


@contextmanager
def game_adapter():
    from training import game as g
    all_features = features()
    legal = np.array([1, 1, 0, 0, 0], bool)
    def new_hand(n_players, dealer, rng, starting_stacks=None):
        state = np.zeros(g.STATE_SIZE, np.int32)
        state[g.S_NPLAYERS] = 2
        cards = rng.permutation(3)[:2]
        return state, None, cards
    def encode(state, cards, deck, seat, legal_mask):
        return all_features[state[g.S_STREET]*3+cards[seat]].copy()
    def step(state, deck, cards, action):
        if state[g.S_TERMINAL] or action not in (0, 1): raise ValueError('Invalid Kuhn action')
        history = NEXT[HISTORIES[state[g.S_STREET]], action]
        value = terminal_value(history, cards)
        if value is not None:
            state[g.S_TERMINAL] = 1
            state[g.S_PAYOFFS:g.S_PAYOFFS+2] = [round(100*value), round(-100*value)]
        else:
            state[g.S_STREET] = HISTORIES.index(history)
            state[g.S_TOACT] = len(history) % 2
    with ExitStack() as stack:
        for module in ('training.game', 'training.traverse_fast'):
            stack.enter_context(patch(module+'.new_hand', new_hand))
            stack.enter_context(patch(module+'.legal_actions', lambda state, seat: legal.copy()))
            stack.enter_context(patch(module+'.step', step))
        stack.enter_context(patch('training.traverse_fast._encode_fast', encode))
        yield new_hand


def collect_regrets(net, traverser, buffer, iteration, traversals, rng, new_hand):
    from training.traverse_fast import NumpyPolicy, traverse_external
    policy = NumpyPolicy(net)
    xs, ys = [], []
    for _ in range(traversals):
        state, deck, cards = new_hand(2, 0, rng)
        traverse_external(state, deck, cards, traverser, policy, xs, ys, [], [], [], rng)
    buffer.add_batch(np.asarray(xs, np.float32), np.asarray(ys, np.float32), iteration)
    return len(xs)


def traversal_oracle():
    """Enumerate every random branch of actual external sampling at uniform play."""
    from training.network import DuelingAdvantageNet
    from training.traverse_fast import NumpyPolicy, traverse_external
    from training import game as g
    import torch
    net = DuelingAdvantageNet()
    with torch.no_grad():
        for parameter in net.parameters(): parameter.zero_()
    policy = NumpyPolicy(net)
    sums = np.zeros((12, 2)); counts = np.zeros(12)
    with game_adapter():
        for player in (0, 1):
            for cards in DEALS:
                for draws in product((.25, .75), repeat=2):
                    class Rng:
                        def __init__(self): self.values = iter(draws)
                        def random(self): return next(self.values)
                    state = np.zeros(g.STATE_SIZE, np.int32)
                    state[g.S_NPLAYERS] = 2
                    xs, ys = [], []
                    traverse_external(state, None, np.asarray(cards), player, policy, xs, ys, [], [], [], Rng())
                    for x, y in zip(xs, ys):
                        i = int(np.argmax(x[:12])); sums[i] += y[:2]; counts[i] += 1
    expected = np.array([[-.375, .375]]*3 + [[-.25, .25]]*3
                        + [[.5, -.5], [-.5, .5], [-1.5, 1.5]]*2)
    observed = sums/counts[:, None]
    np.testing.assert_allclose(observed, expected, atol=1e-7)
    return dict(infosets=12, maximum_error=float(np.max(np.abs(observed-expected))),
                uniform_expected_regrets=expected.tolist(), observed=observed.tolist())
