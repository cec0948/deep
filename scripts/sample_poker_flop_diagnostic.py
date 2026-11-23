"""Collect the first reached flop decision of a rotating hero under frozen 480 self-play."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply, policy_dependency_hashes, semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_capture_postflop_snapshots import ScheduledPostflopMixture
from poker_preflop_specialist import evaluator_policy
from poker_river_runtime import action_key, legal_menu
from regret_partition import base_strata


STUDY = ROOT / 'artifacts/preflop-followup/480-v1/postflop-averaging-20261120'
SOURCE_PLAN = STUDY / 'population/evaluation-plan.json'
SOURCE_SHA = '87927530b0cfbd8e049769fef29dcdc8a02640869ef189799c1e91e676c6ae07'
SLOTS = ('480', '486', '492', '498', '504')


def load_policies(seed):
    verify_files({SOURCE_PLAN: SOURCE_SHA})
    plan = read(SOURCE_PLAN)
    if seed not in plan['seeds']:
        raise ValueError('Unknown fixed training seed')
    source = plan['source']
    refs = [plan['policies'][label][str(seed)] for label in ('specialist480', 'mixture504')]
    dependencies = {ref['path']: ref['sha256'] for ref in refs}
    verify_files(dependencies)
    parent = evaluator_policy(refs[0]['path'], source)
    mixture = ScheduledPostflopMixture(refs[1]['path'], source)
    if parent.policies['preflop'].digest != mixture.preflop.digest:
        raise ValueError('Self-play and snapshot policies have different preflop weights')
    slots = [str(item['iteration']) for item in read(refs[1]['path'])['postflop']]
    if slots != list(SLOTS[1:]):
        raise ValueError('Unexpected frozen snapshot order')
    candidates = dict(zip(SLOTS, [parent.policies['postflop'], *mixture.postflop], strict=True))
    dependencies.update(policy_dependency_hashes(parent, source))
    dependencies.update(mixture.dependency_hashes)
    for path, sha in dependencies.items():
        if plan['frozen_files'].get(path) != sha:
            raise ValueError(f'Policy dependency differs from verified study: {path}')
    dependencies[str(SOURCE_PLAN)] = SOURCE_SHA
    dependencies[str(Path(__file__).resolve())] = digest(__file__)
    return parent, candidates, dependencies


def abstract_indices(state, legal_actions):
    """Map each valid NN output to a concrete menu index; two sizes may coincide."""
    keys = [action_key(action) for action in legal_actions]
    call = ('check' if state['can_check'] else 'call', None)
    indexes = [None if state['can_check'] else keys.index(('fold', None)), keys.index(call), None, None, None]
    bounds = state['raise_bounds']
    if bounds is not None:
        low, high = bounds
        owed = state['amount_owed']
        previous = state['your_bet_this_street'] + owed
        pot_after_call = state['pot'] + owed
        for i, amount in enumerate((previous + round(.5 * pot_after_call), previous + pot_after_call, high), 2):
            indexes[i] = keys.index(('raise', max(low, min(high, amount))))
    return indexes


def policy_at_root(policy, state, draw, legal_actions, indices):
    reply = _validate_reply(state, policy.decide(state, draw))
    keys = [action_key(action) for action in legal_actions]
    probabilities = np.zeros(len(keys))
    for item in reply['distribution']:
        probabilities[keys.index(action_key(item['action']))] += item['probability']
    chosen = keys.index(action_key(reply['action']))
    if probabilities[chosen] <= 0:
        raise ValueError('Selected root action has zero policy mass')
    x, mask = policy.features.encode_state_dict(state, policy.lookup._equity_tables)
    model = policy.lookup._models[0]  # roots are postflop, so the opening expert cannot apply
    raw = policy.lookup._forward_single(x, mask.astype(np.float32), model)
    scale = float(model['main_scales'][int(base_strata(x[None])[0])])
    if not np.isfinite(raw).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid postflop action regret or scale')
    # Verify that mapping the abstract predictions back to concrete probabilities
    # independently reconstructs the deployed policy, including all-negative fallback.
    scores = np.maximum(raw, 0.0) * mask
    masses = scores / scores.sum() if scores.sum() > 0.0 else mask / mask.sum()
    concrete = np.zeros(len(keys))
    for abstract, mass in enumerate(masses):
        if indices[abstract] is not None:
            concrete[indices[abstract]] += mass
    if concrete.sum() <= 0 or not np.allclose(concrete / concrete.sum(), probabilities, atol=1e-6, rtol=0):
        raise ValueError('Diagnostic action scores disagree with production decisions')
    return dict(probabilities=probabilities.tolist(), chosen_index=chosen,
                abstract_regrets_bb=[float(raw[i]) * scale if mask[i] else None for i in range(len(raw))])


def root_key(seed, deal_seed, hero, prefix, state):
    blob = json.dumps([seed, deal_seed, hero, prefix, state], sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


def first_root(engine, parent, candidates, seed, deal_seed, hero):
    state = engine.call(dict(op='reset', seed=deal_seed, button=0))
    prefix = []
    while not state['terminal'] and state['street'] in ('preflop', 'flop'):
        if len(prefix) >= 512:
            raise ValueError('Nonterminating hand in sampler')
        actor = state['seat_to_act']
        draw = semantic_seed(deal_seed, hero, state)
        if state['street'] == 'flop' and actor == hero:
            legal = list(legal_menu(state).values())
            if len(legal) >= 2:
                indices = abstract_indices(state, legal)
                records = {slot: policy_at_root(policy, state, draw, legal, indices)
                           for slot, policy in candidates.items()}
                return dict(state_id=root_key(seed, deal_seed, hero, prefix, state), seed=seed,
                            deal_seed=deal_seed, hero=hero, prefix=prefix, root_observation=state,
                            legal_actions=legal, abstract_to_concrete_index=indices, policies=records)
        action = _validate_reply(state, parent.decide(state, draw))['action']
        concrete = dict(seat=actor, **action)
        prefix.append(concrete)
        state = engine.call(dict(op='step', action=concrete))
    return None


def sample_seed(seed, card_start, target, max_scanned, output):
    if not 0 < target <= max_scanned or not 0 <= card_start < card_start + max_scanned <= 2**32:
        raise ValueError('Invalid bounded fresh-card range or sample size')
    output = Path(output) / f'seed-{seed}'
    output.mkdir(parents=True, exist_ok=False)
    parent, candidates, dependencies = load_policies(seed)
    verify_files(dependencies)
    by_seat = Counter()
    agree = Counter()
    count = 0
    temp = output / 'states.jsonl.partial'
    with closing(Bridge()) as engine, temp.open('x') as stream:
        for attempt in range(max_scanned):
            hero = attempt % 6
            row = first_root(engine, parent, candidates, seed, card_start + attempt, hero)
            if row is None:
                continue
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
            by_seat[hero] += 1
            records = row['policies']
            for slot in SLOTS[1:]:
                agree[slot] += records[slot]['chosen_index'] == records['480']['chosen_index']
            count += 1
            if count == target:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if count != target:
        raise ValueError(f'Fixed scan cap reached with {count}/{target} qualifying flops')
    verify_files(dependencies)
    raw = output / 'states.jsonl'
    temp.rename(raw)
    summary = dict(completed_utc=utc(), seed=seed, first_card=card_start, last_scanned=card_start+attempt,
                   attempts=attempt+1, accepted=count, first_eligible_states_by_hero=by_seat,
                   same_sampled_root_choice_as_480=agree, reaches_from='six identical frozen specialist480 policies',
                   outcome_used_for_selection=False, frozen_dependencies=dependencies, source_plan_sha256=SOURCE_SHA,
                   policy_model_sha256={slot: policy.digest for slot, policy in candidates.items()},
                   states_sha256=digest(raw))
    atomic_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('seed', type=int)
    ap.add_argument('card_start', type=int)
    ap.add_argument('target', type=int)
    ap.add_argument('max_scanned', type=int)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    sample_seed(args.seed, args.card_start, args.target, args.max_scanned, args.output)
