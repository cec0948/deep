"""Check training deal provenance, action targets, model routing and saved-label settlement."""
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import json
import math
import random

import numpy as np

from audit_poker_flop_diagnostic import load_training_engine, replay
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_flop_action_values import _actor_view, continuation_action, source_cards
from poker_multivalued_turn import Rules
from poker_preflop_specialist import StreetPolicy, evaluator_policy
from poker_public_state import CARDS
from poker_river_runtime import action_key
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges


STUDY = ROOT/'artifacts/preflop-followup/480-v1/flop-repair-20261126'
REPORT = ROOT/'reports'/STUDY.relative_to(ROOT/'artifacts')
PARENT = STUDY.parent/'opponent-diversity-20261123'
SEEDS = (342, 343, 344)
SAMPLED_POSITIONS = (0, 341, 682, 1023)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources(design):
    ranges = {key: item['range'] for key, item in prior_card_ranges(STUDY).items()}
    original_count = len(ranges)
    stage = read(PARENT/'stage-ready.json')
    plan = read(PARENT/'training-plan.json')
    verify_files({REPORT.parent/'pool-extension-528-20261126/status.json': design['source_study_sha256'],
                  PARENT/'stage-ready.json': '8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9',
                  PARENT/'training-plan.json': 'bc3526a3bd5b96d9f5edccff8414c70fd480db22a962cb8aa09f2423b1e5427d'})
    for seed in SEEDS:
        sid = str(seed)
        d = design['seed_sources'][sid]
        start, end = d['train_card_start'], d['train_card_exclusive_limit']
        check_card_range(start, end-start, ranges)
        ranges[sid] = (start, end)
        ref = stage['policies']['specialist480'][sid]
        require(ref == plan['parent_policies'][sid], 'Different 480 policy in source manifest')
        verify_files({ref['path']: d['parent_policy_sha256']})
        spec = read(ref['path'])
        for street in ('preflop', 'postflop'):
            verify_files({spec[street]['path']: d[street+'_sha256']})
        for t, ref in plan['historical_policies'][sid].items():
            verify_files({ref['path']: d['historical_selfplay_sha256'][t]})
    return original_count, plan


def nonce(*parts):
    payload = ':'.join(str(i) for i in ('flop-action-values-v1', *parts)).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')


def repair_menus(state):
    """Map five abstract outputs to distinct concrete actions in first-seen order."""
    owed = state['amount_owed']
    check_call = ('check' if state['can_check'] else 'call', None)
    fold = None if state['can_check'] else ('fold', None)
    raises = [None, None, None]
    if state['raise_bounds'] is not None:
        lo, hi = state['raise_bounds']
        price = state['your_bet_this_street'] + owed
        pot = state['pot'] + owed
        raises = [('raise', max(lo, min(hi, price+round(portion*pot)))) for portion in (.5, 1)]
        raises.append(('raise', hi))
    concrete = list(dict.fromkeys(k for k in [check_call, fold, *raises] if k is not None))
    abstract = [None if k is None else concrete.index(k) for k in [fold, check_call, *raises]]
    return concrete, abstract


def request_at_root(row):
    state = row['root_observation']
    return dict(hero=row['hero'], button=state['dealer'], hole=state['your_cards'],
                board=state['community_cards'], actions=row['prefix'], seed=0)


def check_row(row, seed, start, end, engine):
    hero, deal = row['hero'], row['deal_seed']
    require(row['seed'] == seed and start <= deal < end and hero == (deal-start)%6,
            'Invalid seed, training deal or rotated actor')
    state = row['root_observation']
    rebuilt = engine.call(dict(op='replay', **request_at_root(row)))
    require(rebuilt == state and state['street'] == 'flop' and state['seat_to_act'] == hero,
            'Training observation not reconstructable from actor-visible request')
    log = state['action_log']
    boundaries = [i for i,e in enumerate(log) if e['action'] == 'deal']
    require(len(boundaries) == 1 and not any(e.get('seat') == hero for e in log[boundaries[0]+1:]),
            'Actor has previously played on this flop')
    stages = [480, 480, 480, 300, 360, 420]
    lineup = [480 if s == hero else stages[random.Random(nonce('opponent',seed,deal,s)).randrange(6)]
              for s in range(6)]
    require(row['opponents_by_seat'] == lineup, 'Incorrect self-play lineup')
    acts, mapping = repair_menus(state)
    require([action_key(a) for a in row['legal_actions']] == acts
            and row['abstract_to_concrete_index'] == mapping, 'Action menu or clamped size map differs')
    ps = row['parent_probabilities']
    pays = row['action_payoffs_chips']
    require(len(acts) == len(ps) == len(pays) and len(acts) >= 2
            and all(math.isfinite(p) and 0 <= p <= 1 for p in ps)
            and math.isclose(math.fsum(ps), 1.0, abs_tol=1e-7), 'Invalid base distribution')
    for a, samples in zip(acts, pays, strict=True):
        require(len(samples) == 4 and all(type(n) is int and -10000 <= n <= 50000 for n in samples),
                'Missing or invalid training payout')
        if a[0] == 'fold':
            require(all(n == state['your_stack']-10000 for n in samples), 'Incorrect fold utility')
    return acts, mapping


def check_branches(row, engine, rules, game, models):
    deal, hero = row['deal_seed'], row['hero']
    fixed = list(source_cards(engine, deal))
    branches = 0
    for rep in range(4):
        unrevealed = [card for card in CARDS if card not in fixed]
        random.Random(nonce('deck', deal, rep)).shuffle(unrevealed)
        deck = [*fixed, *unrevealed]
        for i, move in enumerate(row['legal_actions']):
            state = rules.call(dict(op='reset', deck=deck, button=0))
            for a in row['prefix']:
                state = rules.call(dict(op='step', action=a))
            require(_actor_view(state) == row['root_observation'], 'Root changed on common runout')
            actions = [dict(move, seat=hero)]
            state = rules.call(dict(op='step', action=actions[0]))
            counts = Counter()
            while not state['terminal']:
                s = state['seat_to_act']
                key = (state['street'],s)
                draw = nonce('action',deal,hero,rep,*key,counts[key])
                counts[key] += 1
                p = models[row['opponents_by_seat'][s]]
                a = dict(continuation_action(p, _actor_view(state), draw), seat=s)
                actions.append(a)
                require(len(actions) <= 512, 'Malformed branch')
                state = rules.call(dict(op='step', action=a))
            trace = dict(hero=hero, deck=deck, button=0, prefix=row['prefix'], actions=actions,
                         utilities=state['utilities'])
            require(replay(game,trace) == row['action_payoffs_chips'][i][rep],
                    'Saved action label disagrees with independent training rules')
            branches += 1
    return branches


def collect_checks(seed, design, source_plan, engine, rules, game):
    sid = str(seed)
    d = design['seed_sources'][sid]
    source = ROOT/source_plan['campaign']/'source'
    parent = StreetPolicy(source_plan['parent_policies'][sid]['path'], source)
    models = {480: parent}
    for stage, ref in source_plan['historical_policies'][sid].items():
        models[int(stage)] = evaluator_policy(ref['path'], source)
    folder = STUDY/f'seed-{seed}'
    receipt = read(folder/'collection-complete.json')
    path = folder/'positions.jsonl.gz'
    verify_files({path: receipt['position_ledger_sha256']})
    start, end = d['train_card_start'], d['train_card_exclusive_limit']
    prev = start-1
    counts = Counter()
    lineups = Counter()
    with gzip.open(path, 'rt') as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            acts, mapping = check_row(row, seed, start, end, engine)
            require(prev < row['deal_seed'], 'Duplicate or unordered training deal')
            prev = row['deal_seed']
            p = parent.decide(row['root_observation'], 0)
            masses = {action_key(r['action']):r['probability'] for r in p['distribution']}
            require(np.allclose([masses.get(a, 0) for a in acts], row['parent_probabilities'],
                                rtol=0, atol=1e-7), 'Parent probabilities changed')
            if i in SAMPLED_POSITIONS:
                counts['independent_payoffs'] += check_branches(row, engine, rules, game, models)
            counts.update(positions=1, distinct_action_labels=len(acts),
                          zero_mass_branches=4*row['parent_probabilities'].count(0),
                          clamp_duplicate_heads=sum(k is not None for k in mapping)-len(acts))
            lineups.update(stage for seat,stage in enumerate(row['opponents_by_seat'])
                          if seat != row['hero'])
    require(counts['positions'] == receipt['positions'] == 1024
            and counts['distinct_action_labels'] == receipt['action_labels']
            and 4*counts['distinct_action_labels'] == receipt['terminal_rollouts']
            and counts['zero_mass_branches'] == receipt['zero_parent_mass_rollouts']
            and prev == receipt['last_card'] and start == receipt['first_card']
            and receipt['opponents_by_seat_assignment'] == {str(k):v for k,v in lineups.items()},
            'Collection counters or opponent assignments differ from ledger')
    return dict(counts, ledger_sha256=digest(path), restored_source_sha256=d['parent_policy_sha256'])


def reference_policy(policy, state, menu, mapped, ps):
    """Check the deployed probabilities without invoking its inference helpers."""
    post = policy.base.policies['postflop']
    x,_ = post.features.encode_state_dict(state, post.lookup._equity_tables)
    heads = [mapped.index(k) for k in range(len(menu))]
    b = np.zeros(5)
    b[heads] = ps
    w = policy.weights
    x = np.clip((np.r_[x,b]-w['center'])/w['scale'], -6., 6.)
    pending = []
    for n in range(2):
        h = np.maximum(w['w1'][n] @ x + w['b1'][n], 0.)
        pending.append((w['w2'][n] @ h + w['b2'][n])[heads])
    scores = np.asarray(pending)
    best = max(range(len(menu)), key=lambda j:(scores[:,j].mean(),
                   -('fold','check','call','raise').index(menu[j][0]), -(menu[j][1] or 0)))
    alpha = .25 * np.clip((np.min(scores[:,best]-scores@ps)-.05)/.25, 0., 1.)
    distribution = (1.-alpha)*ps
    distribution[best] += alpha
    return float(alpha), distribution


def policy_checks(seed):
    from poker_flop_repair import FlopRepairPolicy
    folder = STUDY/f'seed-{seed}'
    policy = FlopRepairPolicy(folder/'policy.json')
    counts = Counter()
    max_mass = 0.
    with gzip.open(folder/'positions.jsonl.gz','rt') as stream:
        for i,line in enumerate(stream):
            row = json.loads(line)
            state = row['root_observation']
            menu, mapped = repair_menus(state)
            ps = np.asarray(row['parent_probabilities'])
            mass, expected = reference_policy(policy, state, menu, mapped, ps)
            result = _validate_reply(state, policy.decide(state,i))
            actual = {action_key(r['action']):r['probability'] for r in result['distribution']}
            require(set(actual).issubset(menu) and np.allclose(
                [actual.get(k,0.) for k in menu], expected, rtol=1e-5, atol=1e-6
            ), 'Loaded head inference disagrees with independently reconstructed probabilities')
            require(0 <= mass <= .25 and np.abs(ps-expected).sum()*.5 <= .25+1e-7,
                    'Repair exceeds its probability budget')
            counts['root_distributions_checked'] += 1
            counts['nonzero_adjustments'] += mass>0
            max_mass = max(max_mass, mass)
    require(counts['root_distributions_checked']==1024 and counts['nonzero_adjustments']>0,
            'Repair did not change any registered root')
    complete = {k:int(v) for k,v in counts.items()}
    return dict(complete, maximum_repair_mass=max_mass, manifest_sha256=digest(folder/'policy.json'))


def main(mode):
    design = read(STUDY/'training-design.json')
    verify_files({REPORT/'training-design.json': digest(STUDY/'training-design.json')})
    count, plan = sources(design)
    field = 'action-label-audit' if mode=='labels' else 'policy-integrity-audit'
    if mode=='labels':
        game = load_training_engine()
        with closing(Bridge()) as engine, closing(Rules()) as rules:
            checks = {str(s):collect_checks(s, design, plan, engine, rules, game) for s in SEEDS}
    else:
        checks = {str(s):policy_checks(s) for s in SEEDS}
    result = dict(verified_utc=utc(), complete=True, scope=field,
                  training_design_sha256=digest(STUDY/'training-design.json'),
                  existing_evaluation_ranges_checked=count, seeds=checks, competitive_hands=0)
    for folder in (STUDY, REPORT):
        atomic_json(folder/f'{field}.json', result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('labels','policy'))
    main(parser.parse_args().mode)
