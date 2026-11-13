"""Independent small-range oracle for the vectorized river CFR evaluator.

The oracle enumerates compatible private pairs for profile EV and recursively
optimizes one player's whole information set for each best response. It never
uses the solver's forward/backward counterfactual evaluation implementation.
"""
from pathlib import Path
import sys

import eval7
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from river_subgame_solver import RiverGame, _Solver


def test_profile_ev_and_information_respecting_best_response_with_blockers():
    game = RiverGame(('2c', '3d', '7h', '9s', 'Jc'), 500, (1000, 800),
                     street_bets=(0, 100), to_act=0, min_raise=100,
                     bet_fractions=(.5,), max_raises=1)
    solver = _Solver(game,
                     [('As', 'Ad'), ('4s', '5s'), ('Ks', 'Kd')], [.6, .3, .1],
                     [('Ks', 'Kd'), ('Ah', 'Qd'), ('As', '6h')], [.2, .7, .1])
    assert int(solver.compat.sum()) == 7
    assert abs(solver.normalizer - .92) < 1e-12

    ranks = [[eval7.evaluate([eval7.Card(c) for c in (*game.board, *h)])
              for h in hands] for hands in solver.hands]

    def terminal(node, i, j):
        paid0 = node.bets[0] - game.street_bets[0]
        paid1 = node.bets[1] - game.street_bets[1]
        if node.folded == 0:
            return -game.pot / 2 - paid0
        if node.folded == 1:
            return game.pot / 2 + paid1
        share = (1. if ranks[0][i] > ranks[1][j] else
                 .5 if ranks[0][i] == ranks[1][j] else 0.)
        return share * (game.pot + paid0 + paid1) - paid0 - game.pot / 2

    def profile_ev(node, i, j):
        if node.terminal:
            return terminal(node, i, j)
        hand = i if node.player == 0 else j
        return sum(node.strategy[hand, action] * profile_ev(child, i, j)
                   for action, child in enumerate(node.children))

    def best_response(node, hero, hand, opponent_reach):
        if node.terminal:
            return sum(probability * (terminal(node, hand, other) if hero == 0 else
                                      -terminal(node, other, hand))
                       for other, probability in enumerate(opponent_reach))
        if node.player == hero:
            # Choose one action for the hero hand across ALL hidden opponent
            # hands, rather than cheating by maximizing each private deal.
            return max(best_response(child, hero, hand, opponent_reach)
                       for child in node.children)
        return sum(best_response(child, hero, hand,
                                 opponent_reach * node.strategy[:, action])
                   for action, child in enumerate(node.children))

    rng = np.random.default_rng(9124)
    for _ in range(20):
        for node in solver.nodes:
            node.average = rng.random(node.average.shape)
        expected, best, gap, _ = solver.evaluate()
        numerator = sum(solver.weights[0][i] * solver.weights[1][j] *
                        profile_ev(solver.root, i, j)
                        for i, hand0 in enumerate(solver.hands[0])
                        for j, hand1 in enumerate(solver.hands[1])
                        if not set(hand0).intersection(hand1))
        assert abs(expected[0] - numerator / solver.normalizer) < 1e-10
        assert abs(sum(expected)) < 1e-10
        for hero in (0, 1):
            values = []
            for hand, cards in enumerate(solver.hands[hero]):
                opponent_reach = np.array([
                    weight if not set(cards).intersection(other) else 0.
                    for other, weight in zip(solver.hands[1 - hero],
                                             solver.weights[1 - hero])])
                values.append(best_response(solver.root, hero, hand, opponent_reach))
            reference = np.dot(solver.weights[hero], values) / solver.normalizer
            assert abs(best[hero] - reference) < 1e-10
        assert abs(gap - max(0., sum(best))) < 1e-10
