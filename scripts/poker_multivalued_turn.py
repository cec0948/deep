"""Opt-in turn search on a six-seat table once exactly two players remain.

A depth-one subgame lets the actor select a legal root action. Both players
then choose a continuation simultaneously, before the next action or river
card is observed. Each player sees its own coarse private-hand class and
the public action. Values come from terminal rollouts of four modifications
of one frozen blueprint. The selected action and private continuation are
drawn jointly and the hero follows that continuation through the hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import random
import subprocess
import sys
import time
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import eval7
import numpy as np

from fullhouse_runtime import Bridge, ROOT
from poker_preflop_specialist import evaluator_policy
from poker_public_state import CARDS, PublicState, decision_prefixes, observe_for_hole
from poker_river_runtime import action_key, legal_menu, protocol_request, timed_replay

MODES = ('blueprint', 'fold', 'check_call', 'raise')
RULES = ROOT / '.cache/noregrets/target/release/poker-rules'


def biased_distribution(rows, mode):
    """Bias the legal concrete distribution, including all raise sizes."""
    if mode not in range(4):
        raise ValueError('unknown_continuation')
    weights = []
    for row in rows:
        kind = row['action']['kind']
        targeted = ((mode == 1 and kind == 'fold') or
                    (mode == 2 and kind in ('check', 'call')) or
                    (mode == 3 and kind == 'raise'))
        weights.append(float(row['probability']) * (5 if targeted else 1))
    total = sum(weights)
    if not math.isfinite(total) or total <= 0 or any(w < 0 for w in weights):
        raise ValueError('invalid_blueprint_probabilities')
    return [w / total for w in weights]


def _regret_policy(regrets, axis):
    positive = np.maximum(regrets, 0.)
    mass = positive.sum(axis=axis, keepdims=True)
    return np.divide(positive, mass, out=np.full_like(positive, 1 / positive.shape[axis]), where=mass > 0.)


def solve_choices(payoffs, iterations=160):
    """Regret matching on the chance-weighted, restricted information sets.

    The actor chooses a root action and a private continuation. The opponent
    observes only that root action before its own continuation choice. The
    final gap bounds unilateral improvement only in this estimated subgame.
    """
    u = np.asarray(payoffs, dtype=float)
    if (u.ndim != 5 or not all(u.shape) or u.shape[-2:] != (4, 4)
            or not np.isfinite(u).all() or type(iterations) is not int or iterations < 1):
        raise ValueError('invalid_leaf_payoffs_or_iterations')
    hero, villain, actions, _, _ = u.shape
    hr = np.zeros((hero, actions * 4))
    vr = np.zeros((villain, actions, 4))
    xsum, ysum = np.zeros_like(hr), np.zeros_like(vr)
    for _ in range(iterations):
        x, y = _regret_policy(hr, 1), _regret_policy(vr, 2)
        hv = np.einsum('hvaij,vaj->hai', u, y).reshape(hr.shape)
        vv = -np.einsum('hvaij,hai->vaj', u, x.reshape(hero, actions, 4))
        hr += hv - (x * hv).sum(axis=1, keepdims=True)
        vr += vv - (y * vv).sum(axis=2, keepdims=True)
        xsum += x
        ysum += y
    x, y = (xsum / iterations).reshape(hero, actions, 4), ysum / iterations
    upper = np.einsum('hvaij,vaj->hai', u, y).reshape(hero, -1).max(axis=1).sum()
    lower = np.einsum('hvaij,hai->vaj', u, x).min(axis=2).sum()
    return x, y, float(max(0., upper - lower))


def hand_bucket(board, hole):
    category = eval7.handtype(eval7.evaluate([eval7.Card(c) for c in (*board, *hole)]))
    # This symmetric, deliberately coarse abstraction uses only visible cards.
    return {'High Card': 0, 'Pair': 1}.get(category, 2)


class Rules(Bridge):
    def __init__(self):
        self.p = subprocess.Popen([str(RULES)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def _range(public, seat, blueprint, rng, deadline):
    events = [(p, e) for p, e in decision_prefixes(public) if e.seat == seat]
    pool = list(combinations([card for card in CARDS if card not in public.board], 2))
    hands = rng.sample(pool, min(len(pool), 128))
    masses = [1.] * len(hands)
    # Each active player's own observed actions reweight its public prior.
    for prefix, event in events:
        observed = action_key(event.to_engine_action())
        menu = legal_menu(observe_for_hole(prefix, seat, hands[0]))
        if observed not in menu:
            raise ValueError('observed_raise_outside_blueprint_menu')
        for i, hole in enumerate(hands):
            if time.monotonic() >= deadline:
                raise TimeoutError('range_deadline')
            reply = blueprint.decide(observe_for_hole(prefix, seat, hole), 0)
            p = sum(row['probability'] for row in reply['distribution'] if action_key(row['action']) == observed)
            masses[i] *= .99 * p + .01 / len(menu)
    if not all(math.isfinite(w) and w >= 0 for w in masses) or sum(masses) <= 0:
        raise ValueError('invalid_sampled_range')
    return hands, masses


def _world_deck(public, seats, ranges, rng):
    # Condition the PRODUCT of the two ranges on card compatibility. Redraw
    # both on collision so the first seat is not given an unconditioned prior.
    for _ in range(10000):
        holes = [rng.choices(hands, weights=mass)[0] for hands, mass in ranges]
        if set(holes[0]).isdisjoint(holes[1]):
            break
    else:
        raise ValueError('sampled_ranges_are_incompatible')
    fixed = dict(enumerate(public.board, 12))
    for seat, hole in zip(seats, holes):
        fixed.update(enumerate(hole, seat * 2))
    remaining = [card for card in CARDS if card not in fixed.values()]
    rng.shuffle(remaining)
    deck = [fixed[i] if i in fixed else remaining.pop() for i in range(52)]
    return deck, [hand_bucket(public.board, h) for h in holes]


class _Rollouts:
    """Trusted terminal payoff. Cache a simulated state only within one world."""

    def __init__(self, rules, public, deck, blueprint, rng):
        self.rules, self.public, self.deck, self.blueprint = rules, public, deck, blueprint
        self.history = [e.to_engine_action() for e in public.events if e.kind in ('fold', 'check', 'call', 'raise')]
        root = self._restore(())
        if PublicState.from_bridge(root).public_only() != public:
            raise ValueError('synthetic_world_public_replay_mismatch')
        self.states = {(): root}
        self.actions = {}
        self.live = ()
        # Common random numbers for all 4x4 continuations and all root actions.
        self.tape = [rng.random() for _ in range(128)]

    def _restore(self, path):
        state = self.rules.call(dict(op='reset', button=self.public.dealer, deck=self.deck))
        for action in self.history:
            state = self.rules.call(dict(op='step', action=action))
        for seat, kind, to in path:
            state = self.rules.call(dict(op='step', action=dict(seat=seat, kind=kind, **({'to': to} if to is not None else {}))))
        return state

    def after(self, path, action):
        seat = self.states[path]['seat_to_act']
        child = path + ((seat, *action_key(action)),)
        if child not in self.states:
            if self.live != path:
                self._restore(path)
            self.states[child] = self.rules.call(dict(op='step', action=dict(action, seat=seat)))
            self.live = child
        return child

    def payoff(self, action, hero, modes, deadline):
        path = self.after((), action)
        while not self.states[path]['terminal']:
            if time.monotonic() >= deadline:
                raise TimeoutError('rollout_deadline')
            state = self.states[path]
            mode = modes[int(state['seat_to_act'] != hero)]
            key = (path, mode)
            if key not in self.actions:
                # Only the acting player's information is exposed by the oracle.
                if path not in self.actions:
                    self.actions[path] = self.blueprint.decide(state, 0)['distribution']
                rows = self.actions[path]
                weights = biased_distribution(rows, mode)
                draw = self.tape[len(path) - 1]
                selected = rows[-1]['action']
                for row, weight in zip(rows, weights):
                    draw -= weight
                    if draw < 0:
                        selected = row['action']
                        break
                self.actions[key] = selected
            path = self.after(path, self.actions[key])
        utility = self.states[path]['utilities']
        if len(utility) != 6 or sum(utility) != 0:
            raise ValueError('invalid_terminal_payoff')
        return utility[hero] / 100.


def _search(public, blueprint, deadline):
    key = json.dumps(public.to_json(), sort_keys=True, separators=(',', ':')).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))
    hero = public.seat_to_act
    seats = [hero, next(p.seat for p in public.players if not p.is_folded and p.seat != hero)]
    ranges = [_range(public, seat, blueprint, rng, deadline) for seat in seats]
    actions = list(legal_menu(public.to_observation()).values())
    payoffs = np.zeros((3, 3, len(actions), 4, 4))
    counts = Counter()
    worlds = 0
    hero_effect = opponent_effect = 0
    rules = Rules()
    try:
        while worlds < 96 and time.monotonic() + .06 < deadline:
            deck, (h, v) = _world_deck(public, seats, ranges, rng)
            tree = _Rollouts(rules, public, deck, blueprint, rng)
            block = np.empty((len(actions), 4, 4))
            try:
                for a, i, j in product(range(len(actions)), range(4), range(4)):
                    block[a, i, j] = tree.payoff(actions[a], hero, (i, j), deadline)
            except TimeoutError:
                break
            # Same deck and action RNG across modes: a payoff change here
            # reflects a continuation choice, rather than reshuffled cards.
            hero_effect += int(np.any(np.ptp(block, axis=1) > 0, axis=1).sum())
            opponent_effect += int(np.any(np.ptp(block, axis=2) > 0, axis=1).sum())
            payoffs[h, v] += block
            counts[h] += 1
            worlds += 1
    finally:
        rules.close()
    if worlds < 8:
        return dict(used=False, reason='insufficient_complete_worlds', complete_worlds=worlds)
    x, _, gap = solve_choices(payoffs / worlds)
    return dict(used=True, reason='search_complete', actions=actions, joint=x.tolist(),
                root_bucket_counts=[counts[i] for i in range(3)], complete_worlds=worlds,
                estimated_restricted_game_gap_bb=gap, terminal_profile_rollouts=worlds * len(actions) * 16,
                root_action_worlds=worlds * len(actions), hero_mode_changed_payoff=hero_effect,
                opponent_mode_changed_payoff=opponent_effect)


def _worker(channel, public, blueprint, deadline):
    try:
        channel.send(_search(public, blueprint, deadline))
    except Exception as exc:
        channel.send(dict(used=False, reason=f'{type(exc).__name__}: {exc}'))
    finally:
        channel.close()


class TurnSearchPolicy:
    """Research decisions. Outside eligible turns and on deadline, use blueprint."""

    def __init__(self, blueprint, seconds=5):
        if seconds not in (5, 10):
            raise ValueError('choose a 5 or 10 second decision budget')
        self.blueprint, self.seconds = blueprint, seconds
        self.hand_id = None
        self.commitments = {}

    def begin_hand(self, hand_id):
        """Call once at every boundary with an explicit identifier."""
        if hand_id is None:
            raise ValueError('a hand identifier is required')
        self.hand_id = hand_id
        self.commitments.clear()

    @staticmethod
    def _past(public):
        return tuple((event.seat, event.kind, event.to) for event in public.events
                     if event.kind in ('fold', 'check', 'call', 'raise'))

    def _commit(self, public, hole, action, mode):
        actor = public.seat_to_act
        prefix = self._past(public) + ((actor, *action_key(action)),)
        self.commitments[actor] = (frozenset(hole), public.board, prefix, mode)

    def decide(self, state, seed, *, deadline=None):
        started = time.monotonic()
        deadline = min(started + self.seconds, deadline) if deadline is not None else started + self.seconds
        public = PublicState.from_bridge(state).public_only()
        baseline = self.blueprint.decide(state, seed)
        active = [p for p in public.players if not p.is_folded]
        info = dict(used=False, reason='outside_two_active_turn', strength_verified=False,
                    horizon='one_root_action', continuation_modes=MODES, requested_seconds=self.seconds)
        previous = self.commitments.get(public.seat_to_act)
        if previous:
            hole, board, prefix, mode = previous
            if (frozenset(state['your_cards']) == hole and public.board[:len(board)] == board
                    and self._past(public)[:len(prefix)] == prefix):
                rows = baseline['distribution']
                weights = biased_distribution(rows, mode)
                info.update(used=True, reason='committed_continuation', elapsed_seconds=time.monotonic() - started)
                return dict(baseline, action=random.Random(seed).choices([r['action'] for r in rows], weights)[0],
                            distribution=[dict(row, probability=p) for row, p in zip(rows, weights)],
                            policy_source='experimental_turn_continuation', turn_search=info)
            del self.commitments[public.seat_to_act]
        eligible = public.street == 'turn' and len(active) == 2 and all(p.stack > 0 for p in active)
        if eligible and self.hand_id is None:
            info['reason'] = 'explicit_hand_id_required_for_continuation'
        if eligible and self.hand_id is not None:
            ctx = multiprocessing.get_context('fork')
            receive, send = ctx.Pipe(duplex=False)
            child = ctx.Process(target=_worker, args=(send, public, self.blueprint, deadline - .32), daemon=True)
            child.start()
            send.close()
            try:
                info['reason'] = 'hard_deadline'
                if receive.poll(max(0., deadline - time.monotonic() - .12)):
                    try:
                        info.update(receive.recv())
                    except EOFError:
                        info['reason'] = 'worker_exited'
            finally:
                receive.close()
                if child.is_alive():
                    child.kill()
                child.join(timeout=.06)
        if info['used']:
            bucket = hand_bucket(public.board, state['your_cards'])
            joint = np.asarray(info.pop('joint')[bucket], dtype=float)
            actions = info.pop('actions')
            if info['root_bucket_counts'][bucket] < 3:
                info.update(used=False, reason='insufficient_actual_bucket_worlds')
            else:
                choices = list(product(range(len(actions)), range(4)))
                action_index, mode = random.Random(seed).choices(choices, weights=joint.ravel())[0]
                action = actions[action_index]
                self._commit(public, state['your_cards'], action, mode)
                rows = [dict(action=a, probability=float(p)) for a, p in zip(actions, joint.sum(axis=1))]
                baseline = dict(baseline, distribution=rows, action=action, policy_source='experimental_multivalued_turn')
        info['elapsed_seconds'] = time.monotonic() - started
        return dict(baseline, turn_search=info)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--seconds', type=int, choices=(5, 10), default=5)
    args = p.parse_args()
    policy = TurnSearchPolicy(evaluator_policy(args.model, args.source), args.seconds)
    engine = Bridge()
    try:
        for line in sys.stdin:
            started = time.monotonic()
            try:
                request, hand_id = protocol_request(json.loads(line))
                if hand_id != policy.hand_id:
                    policy.begin_hand(hand_id)
                state = timed_replay(engine, request, started + args.seconds - .10)
                reply = policy.decide(state, request.get('seed', 0), deadline=started + args.seconds)
                reply['turn_search']['protocol_elapsed_seconds'] = time.monotonic() - started
            except (KeyError, ValueError, RuntimeError) as exc:
                reply = dict(ok=False, error=str(exc))
            print(json.dumps(reply, allow_nan=False), flush=True)
            if engine.p.poll() is not None:
                break
    finally:
        engine.close()


if __name__ == '__main__':
    main()
