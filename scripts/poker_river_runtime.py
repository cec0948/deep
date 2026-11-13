"""Opt-in river research policy. No registry change, training or deployment.

The base policy remains available before any search. Beliefs are explicitly an
independent-marginal blueprint approximation, not the true six-player posterior.
"""
from __future__ import annotations

import os
if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import select
import time

import numpy as np

from fullhouse_runtime import Bridge, Fullhouse, validate_request
from fullhouse_v4_runtime import FullhouseV4
from poker_public_state import PublicState, Range, decision_prefixes, observe_for_hole
from river_subgame_solver import RiverGame, solve_river


class SearchUnavailable(ValueError):
    pass


def action_key(action):
    return (action['kind'], action.get('to'))


def legal_menu(state):
    """Exactly the existing blueprint's concrete menu, including zero-mass acts."""
    call = {'kind': 'check' if state['can_check'] else 'call'}
    actions = [call]
    if not state['can_check']:
        actions.append({'kind': 'fold'})
    bounds = state['raise_bounds']
    if bounds:
        lo, hi = bounds
        for fraction in (.5, 1.):
            amount = state['your_bet_this_street'] + state['amount_owed'] + round(
                fraction * (state['pot'] + state['amount_owed']))
            actions.append({'kind': 'raise', 'to': max(lo, min(hi, amount))})
        actions.append({'kind': 'raise', 'to': hi})
    return {action_key(a): a for a in actions}


def legal_action(state, action):
    kind = action.get('kind')
    if kind == 'raise':
        bounds = state.get('raise_bounds')
        return (bounds is not None and type(action.get('to')) is int
                and bounds[0] <= action['to'] <= bounds[1])
    return kind == ('check' if state['can_check'] else 'call') or (
        kind == 'fold' and not state['can_check'])


class RiverSearchPolicy:
    """Current-policy baseline plus an explicitly budgeted two-active-player solve."""

    def __init__(self, model, source, *, seconds=5., opponent_policy=None,
                 tremble=.01, max_raises=2):
        if not 0 < seconds <= 10 or not 0 <= tremble < 1:
            raise ValueError('Research decision budget must be in (0,10] seconds')
        self.base = FullhouseV4(model, source)
        self.opponent = opponent_policy or self.base
        self.seconds, self.tremble, self.max_raises = seconds, tremble, max_raises
        self.model, self.digest = self.base.model, self.base.digest
        self.model_type = 'bounded_river_research'
        self._strategy_cache = {}
        self._context = None

    def begin_hand(self, hand_seed):
        """Clear previously chosen strategy likelihoods at a hand boundary."""
        self._strategy_cache.clear()
        self._context = None

    @staticmethod
    def _history_key(public):
        raw = json.dumps(public.public_only().to_json(), sort_keys=True,
                         separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()

    def _ranges(self, state, seats, deadline):
        prefixes = decision_prefixes(state)
        ranges = []
        metadata = []
        calls = 0
        for seat in seats:
            prior = Range.uniform(state['community_cards'])
            weights = np.asarray(prior.weights, dtype=float).copy()
            model = self.base if seat == state['seat_to_act'] else self.opponent
            known_own_policy = seat == state['seat_to_act']
            search_history = 0
            events = [(prefix, event) for prefix, event in prefixes if event.seat == seat]
            for prefix, event in events:
                observed = event.to_engine_action()
                key = action_key(observed)
                # Menu is independent of the hypothetical private cards.
                probe = observe_for_hole(prefix, seat, prior.combos[0])
                cached = self._strategy_cache.get(self._history_key(prefix)) if known_own_policy else None
                menu = ({action_key(a): a for a in cached['actions']}
                        if cached else legal_menu(probe))
                if cached:
                    search_history += 1
                    positions = {frozenset(h): i for i, h in enumerate(cached['hands'])}
                if key not in menu:
                    raise SearchUnavailable('off_tree_history_not_modeled')
                for index, combo in enumerate(prior.combos):
                    if index % 16 == 0 and time.monotonic() >= deadline:
                        raise SearchUnavailable('range_deadline')
                    if weights[index] == 0:
                        continue
                    if cached:
                        probabilities = cached['probabilities'][positions[frozenset(combo)]]
                        probability = sum(float(p) for a, p in zip(cached['actions'], probabilities)
                                          if action_key(a) == key)
                    else:
                        observation = observe_for_hole(prefix, seat, combo)
                        reply = model.decide(observation, 0)
                        probability = sum(row['probability'] for row in reply['distribution']
                                          if action_key(row['action']) == key)
                        calls += 1
                    epsilon = 0. if known_own_policy else self.tremble
                    likelihood = ((1-epsilon)*probability + epsilon/len(menu))
                    weights[index] *= likelihood
                total = float(weights.sum())
                if not math.isfinite(total) or total <= 0:
                    raise SearchUnavailable('zero_or_nonfinite_range_mass')
                weights /= total
            provenance = {
                'source': 'public_history_blueprint_likelihood',
                'seat': seat, 'events': len(events),
                'tremble': 0. if known_own_policy else self.tremble,
                'previous_search_decisions_replayed': search_history,
                'model_sha256': model.digest,
                'folded_player_removal_marginalized_exactly': False,
                'independent_marginal_approximation': True,
                'conditioned_on_actual_hero_hole': False,
            }
            metadata.append(provenance)
            ranges.append(Range.from_weights(prior.combos, weights,
                                            provenance=(json.dumps(provenance, sort_keys=True),)))
        return ranges, calls, metadata

    def decide(self, state, seed):
        """Enforce the whole decision deadline outside the numerical worker."""
        started = time.monotonic()
        baseline = self.base.decide(state, seed)
        context_key = (state['dealer'], state['seat_to_act'], tuple(state['your_cards']),
                       tuple(state['community_cards']))
        if context_key != self._context:
            self._strategy_cache.clear()
            self._context = context_key
        active = [p for p in state['players'] if not p['is_folded']]
        if (state.get('street') != 'river' or len(active) != 2
                or any(p['is_all_in'] or p['stack'] <= 0 for p in active)):
            return dict(baseline, river_search=dict(attempted=False, used=False,
                reason='outside_two_active_player_river', requested_seconds=self.seconds,
                elapsed_seconds=time.monotonic()-started, strength_verified=False))
        context = multiprocessing.get_context('fork')
        receive, send = context.Pipe(duplex=False)
        # Reserve parent IPC/termination/serialization time inside the user budget.
        worker_budget = max(.001, self.seconds-(time.monotonic()-started)-.15)
        process = context.Process(target=self._worker,
                                  args=(send, state, seed, worker_budget), daemon=True)
        process.start()
        send.close()
        reply = None
        try:
            wait = max(0., self.seconds-(time.monotonic()-started)-.10)
            if receive.poll(wait):
                try:
                    reply = receive.recv()
                except EOFError:
                    pass
        finally:
            receive.close()
            if process.is_alive():
                process.kill()
            process.join(timeout=.05)
        if reply is None or reply.get('worker_error'):
            reason = 'hard_deadline' if reply is None else reply['worker_error']
            return dict(baseline, river_search=dict(attempted=True, used=False,
                reason=reason, requested_seconds=self.seconds,
                elapsed_seconds=time.monotonic()-started, strength_verified=False))
        cached = reply.pop('_search_strategy', None)
        if cached is not None and reply['river_search']['used']:
            self._strategy_cache[cached['key']] = cached
        reply['river_search'].update(elapsed_seconds=time.monotonic()-started,
                                     requested_seconds=self.seconds)
        return reply

    def _worker(self, connection, state, seed, budget):
        try:
            connection.send(self._decide_local(state, seed, budget))
        except BaseException as error:
            connection.send({'worker_error': type(error).__name__+': '+str(error)})
        finally:
            connection.close()

    def _decide_local(self, state, seed, budget):
        started = time.monotonic()
        deadline = started + budget
        baseline = self.base.decide(state, seed)
        diagnostic = dict(attempted=False, used=False, requested_seconds=self.seconds,
                          strength_verified=False, full_game_equilibrium_guarantee=False)

        def finish(reason, reply=None):
            result = dict(baseline if reply is None else reply)
            diagnostic.update(reason=reason, elapsed_seconds=time.monotonic()-started)
            result.update(river_search=diagnostic, strength_verified=False)
            return result

        if state.get('street') != 'river':
            return finish('outside_two_active_player_river')
        active = [p for p in state['players'] if not p['is_folded']]
        if len(active) != 2 or any(p['is_all_in'] or p['stack'] <= 0 for p in active):
            return finish('outside_two_active_player_river')
        diagnostic['attempted'] = True
        try:
            public_state = PublicState.from_bridge(state)
            active.sort(key=lambda p: p['seat'])
            seats = [p['seat'] for p in active]
            actor = seats.index(state['seat_to_act'])
            ranges, calls, metadata = self._ranges(state, seats, deadline)
            actual = set(state['your_cards'])
            if not any(set(combo) == actual and weight > 0
                       for combo, weight in zip(ranges[actor].combos, ranges[actor].weights)):
                raise SearchUnavailable('actual_hand_zero_range_mass')
            diagnostic.update(range_policy_calls=calls,
                              range_seconds=time.monotonic()-started,
                              ranges=metadata)
            remaining = deadline-time.monotonic()
            if remaining <= .01:
                raise SearchUnavailable('range_deadline')
            increment = public_state.min_raise_increment
            last = state.get('action_log', [])[-1:]
            checks = int(bool(state['can_check'] and last and last[0].get('action') == 'check'))
            game = RiverGame(board=tuple(state['community_cards']), pot=state['pot'],
                             stacks=tuple(p['stack'] for p in active),
                             street_bets=tuple(p['bet_this_street'] for p in active),
                             to_act=actor, min_raise=increment, max_raises=self.max_raises,
                             checks=checks)
            solved = solve_river(game, ranges[0].combos, ranges[0].weights,
                                 ranges[1].combos, ranges[1].weights,
                                 time_budget=max(.001, remaining-.03))
            if solved.iterations < 2:
                raise SearchUnavailable('insufficient_solver_iterations')
            # A dedicated adapter below keeps the solver's mathematical policy
            # separate from the external concrete-action protocol.
            distribution = root_distribution(solved, state['your_cards'])
            if any(not legal_action(state, row['action']) for row in distribution):
                raise SearchUnavailable('solver_action_outside_trusted_engine_bounds')
            if time.monotonic() > deadline+.02:
                raise SearchUnavailable('decision_deadline')
            probability = [row['probability'] for row in distribution]
            if (not probability or any(not math.isfinite(v) or v < 0 for v in probability)
                    or not math.isclose(sum(probability), 1., abs_tol=1e-6)):
                raise SearchUnavailable('invalid_solver_distribution')
            selected = random.Random(seed).choices(distribution, weights=probability)[0]['action']
            diagnostic.update(used=True, solver_iterations=solved.iterations,
                              solver_seconds=solved.elapsed_seconds,
                              solver_setup_seconds=solved.setup_seconds,
                              compatible_private_pairs=solved.compatible_pairs,
                              subgame_exploitability_chips=solved.exploitability,
                              abstraction='two_player_river_bounded_raise_menu',
                              belief_scope='Approximate blueprint marginals; exact pairwise card blockers')
            reply = dict(baseline, action=selected, distribution=distribution,
                         policy_source='range_conditioned_river_search',
                         _search_strategy=dict(key=self._history_key(public_state),
                             hands=solved.hands, actions=[a.as_dict() for a in solved.actions],
                             probabilities=solved.probabilities))
            return finish('completed', reply)
        except (SearchUnavailable, ValueError, KeyError, TypeError) as error:
            return finish(str(error))


def root_distribution(solved, hole):
    """Convert solver root policy to the existing action/distribution protocol."""
    return solved.distribution(hole)


def timed_replay(engine, request, deadline):
    """Bound trusted replay as part of the JSONL request's total time budget."""
    engine.p.stdin.write(json.dumps(dict(request, op='replay'))+'\n')
    engine.p.stdin.flush()
    if not select.select([engine.p.stdout], [], [], max(0., deadline-time.monotonic()))[0]:
        engine.p.kill()
        engine.p.wait(timeout=.1)
        raise RuntimeError('Trusted replay exceeded decision deadline')
    line = engine.p.stdout.readline()
    if not line:
        raise RuntimeError('Trusted replay stopped')
    result = json.loads(line)
    if not result.get('ok'):
        raise ValueError(result.get('error', 'Trusted replay failed'))
    return result['state']


def protocol_request(raw):
    """A fresh hand identifier is required for cross-request belief memory."""
    if not isinstance(raw, dict):
        raise ValueError('JSON object required')
    request = dict(raw)
    hand = request.pop('hand_id', None)
    if not ((type(hand) is str and 1 <= len(hand) <= 128)
            or (type(hand) is int and 0 <= hand < 2**64)):
        raise ValueError('Explicit hand_id string or unsigned integer required')
    validate_request(request)
    return request, (type(hand).__name__, hand)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--opponent-model', type=Path)
    parser.add_argument('--seconds', type=float, choices=(5., 10.), default=5.)
    args = parser.parse_args()
    opponent = Fullhouse(args.opponent_model) if args.opponent_model else None
    policy = RiverSearchPolicy(args.model, args.source, seconds=args.seconds,
                               opponent_policy=opponent)
    engine = Bridge()
    previous_hand = None
    try:
        for line in __import__('sys').stdin:
            started = time.monotonic()
            try:
                if len(line) > 1048576:
                    raise ValueError('Request exceeds1MiB')
                request, hand = protocol_request(json.loads(line))
                if hand != previous_hand:
                    policy.begin_hand(hand)
                    previous_hand = hand
                state = timed_replay(engine, request, started+args.seconds-.05)
                available = args.seconds-(time.monotonic()-started)-.03
                if available <= .15:
                    reply = policy.base.decide(state, request.get('seed', 0))
                    reply['river_search'] = dict(attempted=False, used=False,
                        reason='replay_exhausted_budget', requested_seconds=args.seconds)
                else:
                    policy.seconds = available
                    try:
                        reply = policy.decide(state, request.get('seed', 0))
                    finally:
                        policy.seconds = args.seconds
                reply['river_search'].update(requested_seconds=args.seconds,
                    protocol_elapsed_seconds=time.monotonic()-started,
                    latency_scope='JSON parsing, trusted replay, belief update, solve and IPC; excludes service startup and network')
            except (ValueError, TypeError, KeyError, RuntimeError) as error:
                reply = dict(ok=False, error=str(error))
            print(json.dumps(reply, allow_nan=False), flush=True)
            if engine.p.poll() is not None:
                break
    finally:
        engine.close()


if __name__ == '__main__':
    main()
