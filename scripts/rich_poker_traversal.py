"""Fresh external-sampling traversal with full public-history observations.

Uses the existing frozen pure-Python rules engine. Hidden cards are available
to terminal-payoff simulation, but never to the acting policy's observation.
"""
from __future__ import annotations

from collections import Counter
import importlib

import numpy as np

from rich_poker_observation import (RICH_DIM, append_action, encode_game,
                                    history_start, S_TOACT, S_STREET, S_TERMINAL)


def run_traversals(n_traversals, traverser, policy, buffers, iteration, rng, *,
                   equity_tables=None, strategy_buffers=None, max_nodes=200000):
    """Add raw-BB instantaneous regret samples to the traverser's bounded buffer.

    Optional strategy samples carry own reach at traverser-visited infosets.
    They are diagnostics, not a claim of an unbiased six-player average policy.
    """
    if (type(n_traversals) is not int or not 1 <= n_traversals <= 4096
            or type(traverser) is not int or not 0 <= traverser < 6
            or type(iteration) is not int or iteration < 1
            or type(max_nodes) is not int or not 1 <= max_nodes <= 10000000):
        raise ValueError('Invalid bounded rich traversal configuration')
    game = importlib.import_module('training.game')
    features = importlib.import_module('bot.features')
    equity = equity_tables if equity_tables is not None else features.load_preflop_equity()
    stats = Counter(); street_counts = Counter(); total_value = 0.

    def walk(state, deck, holes, history, own_reach):
        stats['nodes'] += 1
        if stats['nodes'] > max_nodes:
            raise RuntimeError('Rich traversal node budget exhausted; no partial iteration retry')
        if state[S_TERMINAL]:
            stats['terminals'] += 1
            if sum(int(state[game.S_PAYOFFS+i]) for i in range(6)) != 0:
                raise ValueError('Terminal chip conservation failed')
            return float(state[game.S_PAYOFFS+traverser])/100.
        seat = int(state[S_TOACT]); legal = game.legal_actions(state, seat)
        old = features.encode_state(state, holes, deck, seat, legal, equity)
        x = encode_game(state, deck, holes, seat, legal, history, old)
        sigma = np.asarray(policy.strategy(x, legal.astype(np.float32)), dtype=float)
        if (sigma.shape != (5,) or not np.isfinite(sigma).all() or np.any(sigma < 0)
                or np.any(sigma[~legal] != 0) or not np.isclose(sigma.sum(), 1., atol=1e-6)):
            raise ValueError('Rich policy returned an invalid legal strategy')
        sigma /= sigma.sum()
        stats['policy_calls'] += 1; stats['max_history_events'] = max(stats['max_history_events'], len(history))

        def child(action, reach):
            after = state.copy(); game.step(after, deck, holes, action)
            branch = append_action(history, state, after, action)
            return walk(after, deck, holes, branch, reach)

        if seat != traverser:
            action = int(rng.choice(5, p=sigma))
            return child(action, own_reach)
        if strategy_buffers is not None and own_reach > 0:
            strategy_buffers[seat].add(x, sigma.astype(np.float32), iteration,
                                       sample_weight=float(own_reach))
            stats['strategy_rows'] += 1
        values = np.zeros(5, dtype=np.float64)
        for action in np.flatnonzero(legal):
            values[action] = child(int(action), own_reach*float(sigma[action]))
        value = float(np.dot(sigma, values))
        advantages = np.where(legal, values-value, 0.).astype(np.float32)
        if not np.isfinite(advantages).all():
            raise ValueError('Nonfinite counterfactual regret')
        buffers[seat].add(x, advantages, iteration)
        stats['regret_rows'] += 1; street_counts[str(int(state[S_STREET]))] += 1
        return value

    for _ in range(n_traversals):
        dealer = int(rng.integers(0, 6))
        state, deck, holes = game.new_hand(6, dealer, rng)
        total_value += walk(state, deck, holes, history_start(state), 1.)
    return dict(traversals=n_traversals, traverser=traverser, iteration=iteration,
                feature_dim=RICH_DIM, mean_return_bb=total_value/n_traversals,
                new_rows_by_seat={str(seat):stats['regret_rows'] if seat == traverser else 0 for seat in range(6)},
                streets=dict(street_counts), **dict(stats),
                observation_scope='Acting hole, visible board, full public history; no opponent holes or future board',
                strategy_sample_scope='Traverser-visited own-reach diagnostics only; no six-player average guarantee')


run_traversals_rich = run_traversals
