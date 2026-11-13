"""Range-conditioned CFR for an explicitly bounded two-player river game.

This is a research subgame solver, not a six-player equilibrium claim. Both
private ranges are solved together. Only public board blockers are removed
from each range; collisions BETWEEN the two hands are removed jointly.
Utilities are continuation chips, centered by half the existing pot so they
sum to zero. Historical money in the pot is sunk at this root.
"""
from dataclasses import dataclass, field
import math
import time

import eval7
import numpy as np


DECK = tuple(r + s for r in '23456789TJQKA' for s in 'cdhs')
CARD_INDEX = {c: n for n, c in enumerate(DECK)}


def canonical_hand(hand):
    if len(hand) != 2 or any(c not in CARD_INDEX for c in hand) or hand[0] == hand[1]:
        raise ValueError('A private hand must contain two different standard cards')
    return tuple(sorted(hand, key=CARD_INDEX.__getitem__))


@dataclass(frozen=True)
class RiverGame:
    board: tuple
    pot: int
    stacks: tuple
    street_bets: tuple = (0, 0)
    to_act: int = 0
    min_raise: int = 100
    bet_fractions: tuple = (0.5, 1.0)
    max_raises: int = 2
    checks: int = 0

    def __post_init__(self):
        for name in ('board', 'stacks', 'street_bets', 'bet_fractions'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if (len(self.board) != 5 or len(set(self.board)) != 5
                or any(c not in CARD_INDEX for c in self.board)):
            raise ValueError('Exactly five different public board cards are required')
        if (type(self.pot) is not int or self.pot <= 0 or self.to_act not in (0, 1)
                or len(self.stacks) != 2 or len(self.street_bets) != 2
                or any(type(x) is not int or x < 0 for x in self.stacks + self.street_bets)
                or self.pot < sum(self.street_bets) or self.stacks[self.to_act] <= 0):
            raise ValueError('Invalid heads-up river chip state')
        if (type(self.min_raise) is not int or self.min_raise <= 0
                or type(self.max_raises) is not int or not 0 <= self.max_raises <= 3
                or self.checks not in (0, 1)):
            raise ValueError('Invalid raise/check budget')
        if not self.bet_fractions or any(not math.isfinite(x) or x <= 0 for x in self.bet_fractions):
            raise ValueError('Positive finite bet fractions required')
        if self.street_bets[self.to_act] > self.street_bets[1-self.to_act]:
            raise ValueError('Acting player cannot already lead the outstanding bet')
        effective = min(b+s for b, s in zip(self.street_bets, self.stacks))
        if max(self.street_bets) > effective:
            raise ValueError('Existing unmatched short-all-in/side-pot state is unsupported')
        if self.checks and self.street_bets[0] != self.street_bets[1]:
            raise ValueError('A prior check cannot accompany an outstanding bet')
        if effective == max(self.street_bets) and self.street_bets[0] == self.street_bets[1]:
            raise ValueError('No remaining live betting decision')
        if (self.max_raises and max(self.street_bets) < effective < max(self.street_bets)+self.min_raise
                and self.street_bets[self.to_act]+self.stacks[self.to_act] > effective):
            raise ValueError('Effective cap below a legal non-all-in raise is unsupported')


@dataclass(frozen=True)
class Action:
    kind: str
    to: int | None = None

    def as_dict(self):
        return {'kind': self.kind, **({'to': self.to} if self.to is not None else {})}


@dataclass
class _Node:
    player: int | None
    history: tuple
    bets: tuple
    actions: tuple = ()
    children: list = field(default_factory=list)
    folded: int | None = None
    terminal: bool = False
    regret: object = None
    average: object = None
    strategy: object = None


def _tree(game):
    effective = min(b+s for b, s in zip(game.street_bets, game.stacks))
    nodes, leaves = [], []

    def build(player, bets, checks, raises, increment, locked, history=()):
        node = _Node(player, history, bets); nodes.append(node)
        owed = max(bets)-bets[player]
        actions = [Action('fold')] if owed else []
        actions.append(Action('call' if owed else 'check'))
        if not locked and raises < game.max_raises and effective > max(bets):
            pot_after_call = game.pot + sum(b-game.street_bets[i] for i, b in enumerate(bets)) + owed
            amounts = {min(effective, max(bets)+max(increment, round(f*pot_after_call)))
                       for f in game.bet_fractions}
            amounts.add(effective)
            actions += [Action('raise', n) for n in sorted(amounts) if n > max(bets)]
        node.actions = tuple(actions)
        for action in node.actions:
            past = history + ((player, action.kind, action.to),)
            if action.kind == 'fold':
                child = _Node(None, past, bets, folded=player, terminal=True)
                leaves.append(child)
            elif action.kind == 'call' or (action.kind == 'check' and checks):
                final = list(bets); final[player] = max(bets)
                child = _Node(None, past, tuple(final), terminal=True); leaves.append(child)
            elif action.kind == 'check':
                child = build(1-player, bets, 1, raises, increment, locked, past)
            else:
                final = list(bets); final[player] = action.to
                delta = action.to-max(bets)
                child = build(1-player, tuple(final), 0, raises+1, max(increment, delta), delta < increment, past)
            node.children.append(child)
        return node

    root = build(game.to_act, game.street_bets, game.checks, 0, game.min_raise, False)
    return root, nodes, leaves


def terminal_value(game, bets, folded, showdown_sign):
    """Player-zero centered continuation utility; player one gets its negation."""
    c0, c1 = (bets[i]-game.street_bets[i] for i in (0, 1))
    if folded == 0:
        return -game.pot/2-c0
    if folded == 1:
        return game.pot/2+c1
    return showdown_sign*(game.pot+c0+c1)/2+(c1-c0)/2


def _range(hands, weights, board):
    if len(hands) != len(weights) or not len(hands):
        raise ValueError('Nonempty equally sized hand/weight arrays required')
    out = tuple(canonical_hand(h) for h in hands)
    if len(set(out)) != len(out) or any(set(h).intersection(board) for h in out):
        raise ValueError('Duplicate private combinations or public card collision')
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0) or not w.sum() > 0:
        raise ValueError('Range weights must be finite, nonnegative and have positive mass')
    return out, w / w.sum()


@dataclass
class RiverSolution:
    game: RiverGame
    hands: tuple
    actions: tuple
    probabilities: np.ndarray
    action_values: np.ndarray | None
    iterations: int
    elapsed_seconds: float
    setup_seconds: float
    expected_values: tuple | None
    best_response_values: tuple | None
    nash_conv: float | None
    exploitability: float | None
    compatible_pairs: int
    chance_normalizer: float
    terminal_nodes: int
    stopped_by_time: bool
    supported_hands: np.ndarray | None = None
    scope: str = 'Fixed two-player river action abstraction with supplied ranges; no full-game or six-player strength guarantee'

    def distribution(self, hand):
        key = canonical_hand(hand)
        try: index = self.hands.index(key)
        except ValueError as error: raise ValueError('Actual hand is outside the public-conditioned solved range') from error
        if self.supported_hands is not None and not self.supported_hands[index]:
            raise ValueError('Actual hand has zero compatible supplied-range mass')
        return [{'action': a.as_dict(), 'probability': float(p)}
                for a, p in zip(self.actions, self.probabilities[index]) if p > 0]

    def summary(self):
        return dict(iterations=self.iterations, elapsed_seconds=self.elapsed_seconds,
                    setup_seconds=self.setup_seconds, expected_values=self.expected_values,
                    best_response_values=self.best_response_values, nash_conv=self.nash_conv,
                    exploitability=self.exploitability, compatible_pairs=self.compatible_pairs,
                    chance_normalizer=self.chance_normalizer, terminal_nodes=self.terminal_nodes,
                    stopped_by_time=self.stopped_by_time, actions=[a.as_dict() for a in self.actions],
                    chance_model='Normalized product of the supplied two marginal ranges with exact joint card blockers', scope=self.scope)


class _Solver:
    def __init__(self, game, hands0, weights0, hands1, weights1):
        self.game = game
        self.hands = []; self.weights = []
        for hands, weights in [(hands0, weights0), (hands1, weights1)]:
            h, w = _range(hands, weights, set(game.board)); self.hands.append(h); self.weights.append(w)
        masks = [np.array([sum(1 << CARD_INDEX[c] for c in h) for h in hands], dtype=np.uint64)
                 for hands in self.hands]
        self.compat = ((masks[0][:, None] & masks[1][None, :]) == 0).astype(np.float64)
        self.normalizer = float(self.weights[0] @ self.compat @ self.weights[1])
        if not self.normalizer > 0:
            raise ValueError('The two ranges have no compatible private-card assignment')
        board = [eval7.Card(c) for c in game.board]
        ranks = [np.array([eval7.evaluate(board+[eval7.Card(c) for c in h]) for h in hands]) for hands in self.hands]
        self.sign = ((ranks[0][:, None] > ranks[1][None, :]).astype(np.float64)
                     - (ranks[0][:, None] < ranks[1][None, :]).astype(np.float64)) * self.compat
        self.root, self.nodes, self.leaves = _tree(game)
        self.leaf_index = {id(n): t for t, n in enumerate(self.leaves)}
        for node in self.nodes:
            shape = (len(self.hands[node.player]), len(node.actions))
            node.regret = np.zeros(shape); node.average = np.zeros(shape)
            node.strategy = np.full(shape, 1/shape[1])

    def _profile(self, average=False):
        for node in self.nodes:
            source = node.average if average else np.maximum(node.regret, 0)
            mass = source.sum(1, keepdims=True)
            node.strategy = np.divide(source, mass, out=np.full_like(source, 1/source.shape[1]), where=mass > 0)

    def _forward(self, node, r0, r1, leaf_reaches, own_reaches):
        if node.terminal:
            t = self.leaf_index[id(node)]; leaf_reaches[0][:, t] = r0; leaf_reaches[1][:, t] = r1; return
        own_reaches[id(node)] = r0 if node.player == 0 else r1
        for a, child in enumerate(node.children):
            self._forward(child, r0*node.strategy[:, a] if node.player == 0 else r0,
                          r1*node.strategy[:, a] if node.player == 1 else r1, leaf_reaches, own_reaches)

    def _terminals(self):
        reaches = [np.empty((len(h), len(self.leaves))) for h in self.hands]; own = {}
        self._forward(self.root, np.ones(len(self.hands[0])), np.ones(len(self.hands[1])), reaches, own)
        other1 = reaches[1]*self.weights[1][:, None]; other0 = reaches[0]*self.weights[0][:, None]
        sign0 = self.sign @ other1; mass0 = self.compat @ other1
        sign1 = self.sign.T @ other0; mass1 = self.compat.T @ other0
        v0 = np.empty_like(sign0); v1 = np.empty_like(sign1)
        for t, node in enumerate(self.leaves):
            c0, c1 = (node.bets[i]-self.game.street_bets[i] for i in (0, 1))
            if node.folded is not None:
                value = terminal_value(self.game, node.bets, node.folded, 0)
                v0[:, t] = value*mass0[:, t]; v1[:, t] = -value*mass1[:, t]
            else:
                halfpot = (self.game.pot+c0+c1)/2; offset = (c1-c0)/2
                v0[:, t] = halfpot*sign0[:, t]+offset*mass0[:, t]
                v1[:, t] = -halfpot*sign1[:, t]-offset*mass1[:, t]
        return (v0, v1), own

    def _backward(self, node, terminals, own, iteration=None, br=False, root_actions=None):
        if node.terminal:
            t = self.leaf_index[id(node)]; return terminals[0][:, t], terminals[1][:, t]
        children = [self._backward(c, terminals, own, iteration, br, root_actions) for c in node.children]
        a0 = np.column_stack([c[0] for c in children]); a1 = np.column_stack([c[1] for c in children])
        if node is self.root and root_actions is not None:
            root_actions.append(a0 if node.player == 0 else a1)
        if node.player == 0:
            v0 = a0.max(1) if br else (a0*node.strategy).sum(1); v1 = a1.sum(1)
            action_values, value = a0, v0
        else:
            v0 = a0.sum(1); v1 = a1.max(1) if br else (a1*node.strategy).sum(1)
            action_values, value = a1, v1
        if iteration is not None:
            node.regret += iteration*(action_values-value[:, None])
            node.average += iteration*own[id(node)][:, None]*node.strategy
        return v0, v1

    def iterate(self, iteration):
        self._profile(); terminals, own = self._terminals(); self._backward(self.root, terminals, own, iteration)

    def evaluate(self):
        self._profile(average=True); terminals, own = self._terminals(); actions=[]
        values = self._backward(self.root, terminals, own, root_actions=actions)
        best = self._backward(self.root, terminals, own, br=True)
        expected = tuple(float(w@v/self.normalizer) for w, v in zip(self.weights, values))
        br_values = tuple(float(w@v/self.normalizer) for w, v in zip(self.weights, best))
        if not math.isclose(sum(expected), 0, abs_tol=1e-8):
            raise ArithmeticError('River continuation utilities failed zero-sum check')
        # Each player's best response is computed separately against the fixed
        # opponent average profile: own maxima do not enter opponent values.
        nash_conv = max(0., sum(br_values))
        actor=self.game.to_act
        mass=(self.compat@self.weights[1] if actor==0 else self.compat.T@self.weights[0])
        conditional=np.divide(actions[0],mass[:,None],out=np.zeros_like(actions[0]),where=mass[:,None]>0)
        return expected,br_values,nash_conv,conditional


def solve_river(game, hands0, weights0, hands1, weights1, *, iterations=1000, time_budget=None, compute_diagnostics=True):
    """Solve all private hands, with a soft in-process deadline.

    Matrix kernels cannot be pre-empted inside Python. A caller needing a hard
    protocol deadline must run this in a bounded worker process and retain its
    baseline fallback. Setup and final average-profile evaluation count toward
    elapsed_seconds. With too little time, returns a legal uniform fallback.
    """
    if type(iterations) is not int or iterations < 0 or iterations > 1000000:
        raise ValueError('Invalid bounded CFR iteration count')
    if time_budget is not None and (not math.isfinite(time_budget) or time_budget < 0 or time_budget > 60):
        raise ValueError('Time budget must be finite and between zero and60seconds')
    began=time.monotonic(); solver=_Solver(game,hands0,weights0,hands1,weights1); setup=time.monotonic()-began
    deadline=math.inf if time_budget is None else began+time_budget
    finished=0; max_iteration_seconds=0.; stopped=False
    for iteration in range(1,iterations+1):
        # Reserve enough time for one final average-profile/BR calculation.
        if time.monotonic()+max(0.002,2.5*max_iteration_seconds)>=deadline:
            stopped=True; break
        tick=time.monotonic();solver.iterate(iteration);finished=iteration
        max_iteration_seconds=max(max_iteration_seconds,time.monotonic()-tick)
    expected=br_values=nash_conv=conditional=None
    solver._profile(average=True)
    if compute_diagnostics and (time.monotonic()+max_iteration_seconds < deadline or time_budget is None):
        expected,br_values,nash_conv,conditional=solver.evaluate()
    actor = game.to_act
    supported = (solver.weights[actor] > 0) & ((solver.compat @ solver.weights[1]) > 0
                   if actor == 0 else (solver.compat.T @ solver.weights[0]) > 0)
    return RiverSolution(game,solver.hands[game.to_act],solver.root.actions,solver.root.strategy.copy(),conditional,
        finished,time.monotonic()-began,setup,expected,br_values,nash_conv,
        nash_conv/2 if nash_conv is not None else None,int(solver.compat.sum()),solver.normalizer,
        len(solver.leaves),stopped,supported)
