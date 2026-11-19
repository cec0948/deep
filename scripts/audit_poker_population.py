"""Recompute population and public-confirmation results from the per-hand action ledgers."""
import argparse
from contextlib import closing
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from controlled_poker_refit import read
from evaluate_average_baseline import hand_seed, semantic_seed
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import ReferenceOpponent
from rich_poker_runtime import RichPolicy
from run_poker_candidate_confirmation import prior_card_ranges


STUDY = ROOT/'artifacts/preflop-followup/480-v1/population'
CONFIG = ROOT/'configs/poker-population-20261119.json'
PARENT = STUDY.parent/'preflop-dose'
RICH = ROOT/'artifacts/rich-selfplay/20261113T075620400993Z/stage-ready.json'
RICH_SHA = '8970ce73ceabe645ab6c98bd5419d3e23ed19c9b82edf0a8700820471e8ff4e6'
LABELS = ('steps0', 'steps256')
SEEDS = (342, 343, 344)
VARIED = ('public', 'noregrets', 'rich96', 'tight-value', 'pressure')
RULES = ('calling-station', 'tight-value', 'pressure')
SAMPLE_STRIDE = 313
sha = lru_cache(maxsize=None)(digest)


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def close_to(actual, expected, detail):
    check(np.shape(actual) == np.shape(expected)
          and np.allclose(actual, expected, rtol=0, atol=1e-8), detail)


def matched_role(field, card, hero, actor):
    if actor == hero:
        return 'hero'
    distance = (actor-hero) % 6
    if field == 'mixed':
        return 'public' if distance % 2 else 'noregrets'
    if field == 'varied':
        return VARIED[(distance-1+card) % len(VARIED)]
    return field


def expected_identities(job, parent, replication):
    field, seed = job['field'], job['seed']
    fixed = parent['opponents']['public']
    if replication:
        return fixed, {'public'}
    roles = {matched_role(field, 0, 0, actor) for actor in range(1, 6)}
    ids = {}
    for role in roles:
        if role == 'rich96':
            check(sha(RICH) == RICH_SHA, 'Frozen rich96 stage changed')
            ids[role] = read(RICH)['policies']['rich96'][str(seed)]['sha256']
        elif role in RULES:
            ids[role] = dict(profile=role, source_sha256=sha(ROOT/'scripts/poker_reference_opponents.py'))
        else:
            ids[role] = fixed[role]
    return ids, roles


def sample_policies(roles, ref, source, seed, ids):
    bots = {'hero': evaluator_policy(ref['path'], source)}
    if 'public' in roles:
        bots['public'] = Fullhouse()
        check(bots['public'].digest == ids['public'], 'Public opponent identity changed')
    if 'rich96' in roles:
        historical = read(RICH)['policies']['rich96'][str(seed)]
        check(sha(Path(historical['path'])) == ids['rich96'], 'Historical opponent identity changed')
        bots['rich96'] = RichPolicy(historical['path'], None)
    for role in roles.intersection(RULES):
        bots[role] = ReferenceOpponent(role)
    bridge_model = None
    if 'noregrets' in roles:
        registry = read(ROOT/'artifacts/selfplay/current.json')
        bridge_model = ROOT/registry['model']
        check(sha(bridge_model) == ids['noregrets'], 'NoRegrets opponent identity changed')
    return bots, bridge_model


def replay_decisions(row, field, engine, bots):
    card, hero = row['seed'], row['hero']
    state = engine.call(dict(op='reset', seed=card, button=0))
    for bot in bots.values():
        if hasattr(bot, 'begin_hand'):
            bot.begin_hand(hand_seed(card, hero))
    for logged in row['actions']:
        actor = state['seat_to_act']
        role = matched_role(field, card, hero, actor)
        rng = semantic_seed(card, hero, state)
        reply = (engine.call(dict(op='blueprint', seed=rng)) if role == 'noregrets'
                 else bots[role].decide(state, rng))
        action = dict(reply['action'], seat=actor)
        check(action == logged, f'{field}/{card}/{hero}: {role} action differs')
        state = engine.call(dict(op='step', action=action))
    check(state['terminal'] and state['utilities'] == row['utilities'], 'Replayed terminal payout differs')


def audit_cell(folder, spec, job, plan, parent, replication):
    result = read(folder/'results.json')
    check(all(result[key] is True for key in ('complete','raw_hands_verified','terminal_payoffs_replayed')),
          f'{folder.name}: incomplete evaluation')
    label, seed, field = job['label'], job['seed'], job['field']
    ref = plan['policies'][label][str(seed)]
    start, deals = spec['deal_seed'], spec['deals']
    hands = deals*6
    expected = dict(candidate_sha256=ref['sha256'], candidate=ref['path'], field=field,
                    deal_seed=start, deals=deals, hands=hands, big_blind_chips=100)
    check(all(result.get(key) == value for key, value in expected.items()), f'{folder.name}: wrong model or cards')
    check(result['verification']['hands'] == hands and result['verification']['terminal_payoffs_replayed']
          and result['verification']['chip_conservation_verified'], 'Missing full evaluation replay')
    status = read(folder/'status.json')
    check(status['phase'] == 'complete' and status['verified_hands'] == hands
          and status['results_sha256'] == sha(folder/'results.json'), 'Missing final cell checksum')
    raw = folder/'hands.jsonl.gz'
    check(sha(raw) == result['raw_hands_sha256'] and sha(folder/'provenance.json') == result['provenance_sha256'],
          'Raw ledger or provenance checksum differs')
    prov = read(folder/'provenance.json')
    ids, roles = expected_identities(job, parent, replication)
    check(prov['dependency_hashes'] == result['dependency_hashes'] and all(sha(Path(k)) == v for
          k, v in prov['dependency_hashes'].items()) and all(prov[k] == result[k] for k in
          ('candidate','candidate_sha256','field','deals','deal_seed','hands','opponents')), 'Source identity mismatch')
    check(ids == result['opponents'] and sha(Path(ref['path'])) == ref['sha256'], 'Opponent or candidate mismatch')
    if not replication:
        check((prov['label'], prov['training_seed']) == (label, seed)
              and prov['first_deal_clockwise_lineup'] == [matched_role(field,start,0,a) for a in range(1,6)],
              'Wrong policy label/seed or first table composition')
    bots, bridge_model = sample_policies(roles, ref, plan['source'], seed, ids)
    block_sums = np.zeros(deals, dtype=np.int64)
    rows = decisions = sampled_hands = sampled_decisions = 0
    with closing(Bridge(bridge_model)) as engine, gzip.open(raw, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            block, hero = divmod(rows, 6)
            card = start+block
            check(block < deals and (block, hero, card, hand_seed(card,hero)) ==
                  (row['deal'], row['hero'], row['seed'], row['hand_seed']), 'Card/seat ledger coverage differs')
            pay = row['utilities']
            check(len(pay) == 6 and all(type(v) is int for v in pay) and sum(pay) == 0
                  and type(row['chips']) is int and row['chips'] == pay[hero], 'Invalid raw hand utilities')
            actions = row['actions']
            check(isinstance(actions, list) and 0 < len(actions) == row['steps'] <= 512, 'Invalid raw action coverage')
            block_sums[block] += pay[hero]
            if block % SAMPLE_STRIDE == 0 or block == deals-1:
                replay_decisions(row, field, engine, bots)
                sampled_hands += 1
                sampled_decisions += len(actions)
            rows += 1
            decisions += len(actions)
    check(rows == hands and decisions == result['verification']['concrete_decisions_replayed'],
          'Ledger row or decision count differs')
    blocks = block_sums/6
    close_to(blocks, result['deal_blocks'], 'Per-deal means differ')
    close_to(blocks.mean(), result['bb_per_100'], 'Hand-weighted return differs')
    evidence = dict(raw_sha256=sha(raw), result_sha256=sha(folder/'results.json'), hands=rows,
                    decisions=decisions, policy_replay_hands=sampled_hands, policy_replay_decisions=sampled_decisions)
    return blocks, evidence


def interval(blocks, z):
    width = z*float(blocks.std(ddof=1))/math.sqrt(len(blocks))
    mean = float(blocks.mean())
    return [mean-width, mean+width]


def audit_roles(fields, scores, reported):
    for role, row in reported.items():
        subset = [f for f in fields if fields[f]['role'] == role]
        families = {fields[f]['family'] for f in subset}
        groups = [[f for f in subset if fields[f]['family'] == family] for family in families]
        check(set(row['field_weights']) == set(subset), 'Summary roles differ')
        for group in groups:
            for field in group:
                close_to(row['field_weights'][field], 1/(len(groups)*len(group)), 'Family weights differ')
        for label in LABELS:
            positive = {f for f in subset if scores[f][label] > 0}
            check(positive == set(row['breadth'][label]['positive_fields']), 'Winning-table count differs')
        for cap in (50, 100, 200):
            entry = row['family_balanced_capped_bb100'][str(cap)]
            values = {}
            for label in LABELS:
                values[label] = sum(sum(min(scores[f][label], cap) for f in group)/len(group)
                                    for group in groups)/len(groups)
                close_to(values[label], entry[label], 'Aggregate cap or weight differs')
            close_to(values['steps256']-values['steps0'], entry['paired_change']['mean'],
                     'Family-balanced capped change differs')


def audit(study, panel_summary):
    cfg = read(CONFIG)
    registration = read(study/'registration.json')
    check(sha(CONFIG) == registration['config_sha256'], 'Study design changed')
    check(tuple(cfg['labels']) == LABELS and tuple(cfg['seeds']) == SEEDS, 'Unexpected candidates or training seeds')
    prior = prior_card_ranges(study)
    seen = [item['range'] for item in prior.values()]
    for key in ('public_confirmation', 'panel_cards', 'holdout_cards'):
        start, count = (cfg[key][k] for k in ('deal_seed', 'deals'))
        end = start+count
        check(all(end <= lo or hi <= start for lo, hi in seen), 'Original card ranges overlap')
        seen.append((start, end))
    parent = read(PARENT/'evaluation-plan.json')
    check(sha(PARENT/'evaluation-plan.json') == cfg['parent_plan_sha256'], 'Parent study changed')
    final = dict(study=cfg['study'], design_sha256=sha(CONFIG), sampled_card_stride=SAMPLE_STRIDE,
                 unique_card_blocks=sum(cfg[k]['deals'] for k in ('public_confirmation','panel_cards','holdout_cards')),
                 new_selfplay=0, new_fitting=0)
    for phase, folder, summary_file in [('panel', study, panel_summary),
                                         ('public', study/'public-confirmation', study/'public-confirmation'/'summary.json')]:
        plan = read(folder/'evaluation-plan.json')
        summary = read(summary_file)
        check(registration[f'{phase}_plan_sha256'] == sha(folder/'evaluation-plan.json')
              and plan['design'] == cfg and plan['policies'] == {k:parent['policies'][k] for k in LABELS}
              and all(sha(Path(k)) == v for k,v in plan['frozen_files'].items()), 'Frozen design mismatch')
        by_field, proofs = {}, {}
        fields = plan['fields']
        z = NormalDist().inv_cdf(1-.05/(2*(3*len(fields) if phase == 'panel' else 1)))
        expected_jobs = {f'{f}-seed-{s}-{l}': dict(id=f'{f}-seed-{s}-{l}', field=f, seed=s, label=l)
                         for f in fields for l in LABELS for s in SEEDS}
        check({j['id']:j for j in plan['jobs']} == expected_jobs and len(expected_jobs) == len(plan['jobs']),
              'Planned cells do not cover all candidates, fields, and seeds')
        for field, spec in fields.items():
            scores = {}
            for label in LABELS:
                matrix = []
                for seed in SEEDS:
                    name = f'{field}-seed-{seed}-{label}'
                    blocks, proof = audit_cell(folder/'evaluation'/name, spec, expected_jobs[name], plan, parent, phase=='public')
                    proofs[name] = proof
                    matrix.append(blocks)
                    reported_hash = (summary['cell_result_hashes'][name] if phase == 'panel'
                                     else summary['raw_checks'][name]['result'])
                    check(proof['result_sha256'] == reported_hash, 'Summary cell checksum differs')
                scores[label] = np.asarray(matrix)
            diffs = scores['steps256'] - scores['steps0']
            blocks = diffs.mean(axis=0)
            row = {label: float(matrix.mean()) for label,matrix in scores.items()}
            row.update(delta=float(blocks.mean()), paired_interval=interval(blocks,z),
                       by_training_seed={str(seed): {label:float(matrix[i].mean()) for label,matrix in scores.items()}
                                         for i,seed in enumerate(SEEDS)})
            if phase == 'panel':
                reported = summary['fields'][field]
                for label in LABELS:
                    close_to(row[label], reported[label]['mean'], 'Absolute panel result differs')
                    close_to(interval(scores[label].mean(axis=0),z), [reported[label][s] for s in ('lower','upper')],
                             'Absolute panel interval differs')
                close_to(row['paired_interval'], [reported['paired_change'][s] for s in ('lower','upper')],
                         'Panel paired interval differs')
                close_to(row['delta'], reported['paired_change']['mean'], 'Panel change differs')
            else:
                close_to(row['paired_interval'], summary['paired_ci95'], 'Public interval differs')
                close_to(row['delta'], summary['difference'], 'Public change differs')
                close_to(diffs.mean(axis=1), summary['differences_by_seed'], 'Public seed results differ')
                for label in LABELS:
                    close_to(row[label], summary['scores'][label], 'Public score differs')
            by_field[field] = row
        if phase == 'panel':
            check(set(summary['roles']) == {v['role'] for v in fields.values()}, 'Missing role summary')
            audit_roles(fields, by_field, summary['roles'])
        totals = {key:sum(p[key] for p in proofs.values()) for key in
                  ('hands','decisions','policy_replay_hands','policy_replay_decisions')}
        check(totals['hands'] == plan['planned_hands'], 'Total scheduled hands differ')
        final[phase] = dict(summary_sha256=sha(summary_file), evaluation_plan_sha256=sha(folder/'evaluation-plan.json'),
                            cells=len(proofs), totals=totals, statistics=by_field, ledgers=proofs)
    return final


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-summary', type=Path, default=STUDY/'summary.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(STUDY, args.panel_summary)
    if args.output:
        atomic_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))
