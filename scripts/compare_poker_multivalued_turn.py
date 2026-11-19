"""Paired five-second search pilot at the first eligible turn of fresh public-table deals."""
import argparse
from collections import Counter
import json
from math import log, sqrt
import os
from pathlib import Path
import random
from statistics import mean, median
from time import monotonic

from evaluate_average_baseline import check_dependencies, policy_dependency_hashes, semantic_seed, utc
from fullhouse_runtime import Bridge, Fullhouse, ROOT
from fullhouse_v3_common import atomic_json, digest
from poker_multivalued_turn import RULES, TurnSearchPolicy
from poker_preflop_specialist import evaluator_policy
from poker_public_state import PublicState
from run_compact_refit_probe import check_card_range
from run_poker_candidate_confirmation import prior_card_ranges


STUDY = ROOT/'artifacts/preflop-followup/480-v1/population/turn-search'
PARENT = ROOT/'artifacts/preflop-followup/480-v1/preflop-dose/evaluation-plan.json'
PARENT_SHA = '642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6'
DESIGN = dict(first_deal=2038300000, maximum_scanned=12000, selected_cards=18, seed=342,
              button=0, seconds_per_root=5, action_rng_seed=818255920, field='public',
              hero='scan position modulo six; first eligible decision per distinct deal',
              inclusion='turn, acting hero, exactly two nonfolded players with chips behind',
              difference='search-with-committed-continuation minus frozen steps0 whole-hand BB on selected spots',
              fallback='single attempt per spot; use frozen blueprint to termination on fallback',
              opponent='five fixed public Fullhouse copies; one surviving opponent at measured spot')


def register():
    check_dependencies({str(PARENT): PARENT_SHA})
    parent = json.loads(PARENT.read_text())
    model = parent['policies']['steps0'][str(DESIGN['seed'])]
    policy = evaluator_policy(model['path'], parent['source'])
    opponent = Fullhouse()
    hashes = policy_dependency_hashes(policy, parent['source'])
    hashes[str(PARENT)] = PARENT_SHA
    hashes[model['path']] = model['sha256']
    hashes[str(opponent.model)] = parent['opponents']['public']['public']
    for path in [Path(__file__).resolve(), Path(__file__).with_name('poker_multivalued_turn.py').resolve(),
                 Path(__file__).with_name('poker_river_runtime.py').resolve(), RULES]:
        hashes[str(path)] = digest(path)
    check_dependencies(hashes)
    previous = {key: item['range'] for key, item in prior_card_ranges(STUDY).items()}
    base = STUDY.parent
    for study in (base, base/'public-confirmation'):
        for field, spec in json.loads((study/'evaluation-plan.json').read_text())['fields'].items():
            previous[f'{study}/{field}'] = [spec['deal_seed'], spec['deal_seed']+spec['deals']]
    check_card_range(DESIGN['first_deal'], DESIGN['maximum_scanned'], previous)
    STUDY.mkdir(exist_ok=False)
    atomic_json(STUDY/'config.json', dict(registered=utc(), design=DESIGN, model=model, source=parent['source'],
                evaluation_deal_seed=DESIGN['first_deal'], evaluation_deals=DESIGN['maximum_scanned'],
                dependencies=hashes))


def step(engine, action):
    return engine.call({'op':'step', 'action':action})


def restore(engine, card, actions):
    state = engine.call({'op':'reset', 'seed':card, 'button':DESIGN['button']})
    for action in actions:
        if state['terminal'] or state['seat_to_act'] != action['seat']:
            raise ValueError('Incorrect replay actor')
        state = step(engine, action)
    return state


def standard_action(candidate, public, hero, nonce, state):
    player = state['seat_to_act']
    result = (candidate if player == hero else public).decide(state, semantic_seed(nonce, hero, state))
    return dict(result['action'], seat=player)


def eligible(state, hero):
    if state['terminal'] or state['street'] != 'turn' or state['seat_to_act'] != hero:
        return False
    obs = PublicState.from_bridge(state)
    active = [s for s in obs.players if not s.is_folded]
    return len(active) == 2 and all(s.stack > 0 for s in active)


def choose_spots(engine, candidate, public):
    selected = []
    rng = random.Random(DESIGN['action_rng_seed'])
    for index in range(DESIGN['maximum_scanned']):
        nonce = rng.getrandbits(64)
        hero = index % 6
        card = DESIGN['first_deal']+index
        state = restore(engine, card, [])
        history = []
        while not state['terminal']:
            if eligible(state, hero):
                selected.append(dict(card=card, hero=hero, nonce=nonce, history=history))
                break
            action = standard_action(candidate, public, hero, nonce, state)
            history.append(action)
            state = step(engine, action)
        if len(selected) == DESIGN['selected_cards']:
            return index+1, selected
    raise ValueError('Eligible fixed-card target not reached')


def branch(engine, row, candidate, public, search=None):
    card, hero, nonce = (row[k] for k in ('card', 'hero', 'nonce'))
    state = restore(engine, card, row['history'])
    if not eligible(state, hero):
        raise ValueError('Unmatched or ineligible prefix')
    actions = row['history'].copy()
    root = None
    root_info = None
    followups = 0
    while not state['terminal']:
        if search and state['seat_to_act'] == hero and (root is None or root_info['used']):
            rng = semantic_seed(nonce, hero, state)
            # Only own hand and public history may reach the search policy.
            observation = PublicState.from_bridge(state).to_observation()
            started = monotonic()
            result = search.decide(observation, rng)
            latency = monotonic()-started
            if root is None:
                root_info = dict(result['turn_search'], measured_seconds=latency)
            elif result['turn_search']['reason'] != 'committed_continuation':
                raise ValueError('Solver did not follow its sampled continuation')
            else:
                followups += 1
            action = dict(result['action'], seat=hero)
        else:
            action = standard_action(candidate, public, hero, nonce, state)
        if root is None:
            root = action
        actions.append(action)
        if len(actions) > 512:
            raise ValueError('Action count exceeded game bound')
        state = step(engine, action)
    audited = restore(engine, card, actions)
    if (not audited['terminal'] or audited['utilities'] != state['utilities']
            or sum(audited['utilities']) != 0 or len(audited['utilities']) != 6):
        raise ValueError('Trusted terminal replay disagreed')
    return dict(root_action=root, utility_bb=audited['utilities'][hero]/100, suffix=actions[len(row['history']):],
                followup_continuation_decisions=followups, search=root_info)


def run():
    cfg = json.loads((STUDY/'config.json').read_text())
    if cfg['design'] != DESIGN:
        raise ValueError('Preregistered design changed')
    check_dependencies(cfg['dependencies'])
    with (STUDY/'launch.json').open('x') as f:
        json.dump(dict(utc=utc(), config_sha256=digest(STUDY/'config.json')), f)
    candidate = evaluator_policy(cfg['model']['path'], cfg['source'])
    public = Fullhouse()
    search = TurnSearchPolicy(candidate, DESIGN['seconds_per_root'])
    engine = Bridge()
    try:
        scanned, spots = choose_spots(engine, candidate, public)
        atomic_json(STUDY/'selected.json', dict(utc=utc(), scanned=scanned, spots=spots))
        rows = []
        for spot in spots:
            search.begin_hand(spot['nonce'])
            baseline = branch(engine, spot, candidate, public)
            proposed = branch(engine, spot, candidate, public, search)
            if not proposed['search']['used'] and baseline['suffix'] != proposed['suffix']:
                raise ValueError('Fallback diverged from the paired blueprint')
            rows.append(dict(card=spot['card'], hero=spot['hero'], baseline=baseline, proposed=proposed,
                             difference_bb=proposed['utility_bb']-baseline['utility_bb']))
            atomic_json(STUDY/'progress.json', dict(completed=len(rows), selected=len(spots), utc=utc()))
    finally:
        engine.close()
    check_dependencies(cfg['dependencies'])
    differences = [r['difference_bb'] for r in rows]
    times = [r['proposed']['search']['measured_seconds'] for r in rows]
    used = [r for r in rows if r['proposed']['search']['used']]
    average = mean(differences)
    # Each payoff is within [-100,+500] BB at a six-handed 100 BB table;
    # paired outcomes on the same deal therefore lie in [-600,+600].
    radius = 1200*sqrt(log(40)/(2*len(rows)))
    raw = STUDY/'paired-branches.json'
    atomic_json(raw, rows)
    summary = dict(completed_utc=utc(), design=DESIGN, scanned=scanned, independent_eligible_deals=len(rows),
                   completed_searches=len(used), root_action_changes=sum(r['baseline']['root_action']!=
                   r['proposed']['root_action'] for r in rows), committed_followup_decisions=sum(
                   r['proposed']['followup_continuation_decisions'] for r in rows),
                   solver_reasons=dict(Counter(r['proposed']['search']['reason'] for r in rows)),
                   baseline_mean_bb=mean(r['baseline']['utility_bb'] for r in rows),
                   search_mean_bb=mean(r['proposed']['utility_bb'] for r in rows),
                   paired_change_bb_per_eligible_spot=average,
                   paired_changes_bb=differences,
                   conservative_95_hoeffding_bb=[max(-600,average-radius),min(600,average+radius)],
                   interval_scope='Bound for independent bounded selected deals under this sampling protocol; no model-seed uncertainty.',
                   measured_seconds=dict(median=median(times), maximum=max(times), exceeded_budget=sum(
                   t>DESIGN['seconds_per_root'] for t in times)), scope='Selected heads-up turn spots, fixed seed 342, public opponents',
                   selected_sha256=digest(STUDY/'selected.json'), raw_sha256=digest(raw),
                   config_sha256=digest(STUDY/'config.json'), implementation_sha256=cfg['dependencies'][str(
                       Path(__file__).with_name('poker_multivalued_turn.py').resolve())],
                   all_branches_terminal_replayed=True, model_promoted=False)
    atomic_json(STUDY/'summary.json', summary)
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    os.nice(5)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register','run'])
    args = parser.parse_args()
    if args.action == 'register':
        register()
    else:
        run()
