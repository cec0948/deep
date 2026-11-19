"""Reconstruct the frozen street-swap comparison from all three policies' hand logs."""
from contextlib import closing, ExitStack
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from statistics import NormalDist

import numpy as np

from audit_poker_warm_preflop import (
    DEALS, FIELDS, SEATS, SEEDS, START, STUDY as ORIGINAL, FirstActionDifferences, check, check_estimate,
)
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import atomic_json, digest


SWAP = ORIGINAL.parent / 'preflop-swap'
BASE, HYBRID, WARM = 'specialist480', 'hybrid492pre480post', 'warm492'
LABELS = (BASE, HYBRID, WARM)
CONTRASTS = ('hybrid_minus_480', '492_minus_hybrid')
SOURCE_SUMMARY = 'b1f9a3bf1721d87c535955b1f4e038ee111a3f8589b8e4db7a6d2b0fb76b7c9c'
SOURCE_AUDIT = '78ef335bb0895535c0e67f38d621b6480148032325a04c63411ffe23579d044a'


def check_design(outcome, plans, original):
    previous, current = plans[BASE], plans[HYBRID]
    design = read(SWAP/'design.json')
    verify_files({ORIGINAL/'summary.json': SOURCE_SUMMARY, ORIGINAL/'final-audit.json': SOURCE_AUDIT,
                  ORIGINAL/'evaluation-plan.json': original['evaluation_plan_sha256'],
                  SWAP/'evaluation-plan.json': outcome['evaluation_plan_sha256']})
    for plan in (previous, current):
        check((plan['card_start'], plan['deals']) == (START, DEALS), 'Different card range')
        verify_files(plan['frozen_files'])
    check(current['opponents'] == previous['opponents'] and current['source'] == previous['source'],
          'Source or opponent identities differ')
    check(design['preflop_from'] == WARM and design['postflop_from'] == BASE
          and design['contrast_order'] == list(CONTRASTS) and design['independent_new_cards'] is False
          and outcome['independent_new_cards'] is False, 'Changed street-swap design')
    expected_jobs = {(HYBRID, seed, field) for seed in SEEDS for field in FIELDS}
    check(set(current['policies']) == {HYBRID} and set(previous['policies']) == {BASE, WARM}
          and len(current['jobs']) == 9 and expected_jobs == {
              (j['label'], j['seed'], j['field']) for j in current['jobs']}, 'Unexpected new evaluation jobs')
    check(outcome['completed'] and outcome['source_summary_sha256'] == SOURCE_SUMMARY
          and outcome['source_audit_sha256'] == SOURCE_AUDIT
          and (outcome['new_cells'], outcome['reused_cells'], outcome['new_hands'], outcome['reused_hands'])
          == (9, 18, 1080000, 2160000) and outcome['independent_card_blocks'] == DEALS
          and outcome['new_fitting_jobs'] == 0 and outcome['new_selfplay_iterations'] == 0
          and outcome['model_promoted'] is False, 'Incomplete or different evaluation')
    check(outcome['reused_result_sha256'] == original['result_sha256']
          and set(outcome['result_sha256']) == {f'{f}-seed-{s}-{HYBRID}' for _, s, f in expected_jobs},
          'Missing or substituted result records')
    source_verification = read(ORIGINAL/'training-verification.json')
    check(digest(ORIGINAL/'stage-ready.json') == source_verification['stage_sha256']
          and source_verification['training_restore_and_initial_policy_matched'], 'Different training stage')
    stage = read(ORIGINAL/'stage-ready.json')
    parents = read(ORIGINAL/'plan.json')['parent_policies']
    check(stage['training_complete'] and stage['iteration'] == 492, 'Unverified 492-round stage')
    identities = {}
    for seed in map(str, SEEDS):
        refs = {label: plan['policies'][label][seed] for label, plan in plans.items()}
        check(refs[BASE] == parents[seed] and refs[WARM] == stage['arms'][WARM][seed]['policy'],
              'Wrong starting or completed candidate')
        models = {}
        for label, ref in refs.items():
            verify_files({ref['path']: ref['sha256']})
            spec = read(ref['path'])
            check(spec['format'] == 'preflop-street-policy-v1', 'Different street routing format')
            models[label] = {}
            for street in ('preflop', 'postflop'):
                asset = spec[street]
                verify_files({asset['path']: asset['sha256']})
                models[label][street] = asset['sha256']
        check(models[HYBRID] == dict(preflop=models[WARM]['preflop'], postflop=models[BASE]['postflop'])
              and all(models[BASE][s] != models[WARM][s] for s in ('preflop', 'postflop')),
              'Frozen hybrid does not isolate the specified network swap')
        identities[seed] = models
    return identities


def checked_cell(label, seed, field, plan, hashes):
    name = f'{field}-seed-{seed}-{label}'
    folder = (SWAP if label == HYBRID else ORIGINAL) / 'evaluation' / name
    verify_files({folder/'results.json': hashes[name]})
    result = read(folder/'results.json')
    raw = folder/'hands.jsonl.gz'
    verify_files({raw: result['raw_hands_sha256'], folder/'provenance.json': result['provenance_sha256'],
                  BRIDGE: result['engine_sha256']})
    provenance = read(folder/'provenance.json')
    ref = plan['policies'][label][str(seed)]
    for record in (result, provenance):
        check(record['candidate'] == ref['path'] and record['candidate_sha256'] == ref['sha256']
              and record['opponents'] == plan['opponents'][field] and record['candidate_type'] == 'preflop-street-policy-v1'
              and (record['field'], record['deals'], record['hands'], record['deal_seed'])
              == (field, DEALS, DEALS*SEATS, START), 'Cell identity mismatch: '+name)
    verification = result['verification']
    check(result['complete'] and result['raw_hands_verified'] and result['source_provenance_verified']
          and result['terminal_payoffs_replayed'] and result['big_blind_chips'] == 100
          and verification['terminal_payoffs_replayed'] and verification['aggregate_recomputed']
          and verification['chip_conservation_verified'] and verification['hands'] == DEALS*SEATS
          and verification['card_blocks'] == DEALS, 'Unverified cell: '+name)
    deps = dict(result['dependency_hashes'])
    check(deps == provenance['dependency_hashes'], 'Dependency manifest mismatch: '+name)
    verify_files(deps)
    spec = read(ref['path'])
    for asset in (ref, spec['preflop'], spec['postflop']):
        check(deps.pop(asset['path']) == asset['sha256'], 'Unbound model weights: '+name)
    protocol = (deps, result['evaluator_sha256'], result['runtime_sha256'],
                result['source_features_sha256'], provenance['semantic_rng'])
    return raw, result, protocol


def aligned_blocks(cells, traces):
    check(cells[0][2] == cells[1][2] == cells[2][2], 'Different game or opponent RNG protocols')
    chips = np.empty((len(cells), DEALS, SEATS), dtype=np.int64)
    count = 0
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, 'rb')) for path, _, _ in cells]
        for index, lines in enumerate(zip_longest(*streams)):
            deal, hero = divmod(index, SEATS)
            check(deal < DEALS and None not in lines, 'Unmatched card/seat triplet')
            card = START + deal
            expected_seed = int.from_bytes(hashlib.sha256(
                f'rich-evaluation-hand:{card}:{hero}'.encode()).digest()[:8], 'big')
            rows = [json.loads(line) for line in lines]
            for i, row in enumerate(rows):
                check((row['deal'], row['hero'], row['seed'], row['hand_seed']) == (deal, hero, card, expected_seed),
                      'Different card, seat, or hand RNG')
                utilities = row['utilities']
                check(len(utilities) == SEATS and all(type(x) is int for x in utilities)
                      and sum(utilities) == 0 and row['chips'] == utilities[hero]
                      and len(row['actions']) == row['steps'], 'Invalid raw payoff or action record')
                chips[i, deal, hero] = row['chips']
            traces[CONTRASTS[0]].add(rows[0], rows[1])
            traces[CONTRASTS[1]].add(rows[1], rows[2])
            count += 1
    check(count == DEALS*SEATS, 'Missing raw hand records')
    means = chips.mean(axis=2)
    for blocks, (_, result, _) in zip(means, cells, strict=True):
        np.testing.assert_allclose(blocks, result['deal_blocks'], rtol=0, atol=1e-8)
        np.testing.assert_allclose(blocks.mean(), result['bb_per_100'], rtol=0, atol=1e-8)
    return means


def audit():
    outcome, original = (read(root/'summary.json') for root in (SWAP, ORIGINAL))
    original_plan = read(ORIGINAL/'evaluation-plan.json')
    plans = {BASE: original_plan, HYBRID: read(SWAP/'evaluation-plan.json'), WARM: original_plan}
    identities = check_design(outcome, plans, original)
    z = NormalDist().inv_cdf(1 - 0.05/(2*len(FIELDS)))
    diagnostics = {key: {} for key in CONTRASTS}
    estimates = {key: {} for key in CONTRASTS}
    preflop_moves = {}
    with closing(Bridge()) as engine:
        for field in FIELDS:
            traces = {name: FirstActionDifferences(engine) for name in CONTRASTS}
            cards = []
            for seed in SEEDS:
                cells = [checked_cell(label, seed, field, plans[label],
                                      outcome['result_sha256'] if label == HYBRID else original['result_sha256'])
                         for label in LABELS]
                cards.append(aligned_blocks(cells, traces))
            scores = np.asarray(cards)  # fixed seed, policy, then original card
            for i, label in enumerate(LABELS):
                check_estimate(scores[:, i].mean(axis=0), outcome['absolute_scores'][label][field])
            deltas = np.diff(scores, axis=1)
            total = scores[:, 2] - scores[:, 0]
            np.testing.assert_allclose(deltas.sum(axis=1), total, rtol=0, atol=1e-8)
            check_estimate(total.mean(axis=0), original['trained_minus_frozen_parent'][field])
            for i, key in enumerate(CONTRASTS):
                by_seed = deltas[:, i]
                paired = by_seed.mean(axis=0)
                comparison = outcome['comparisons'][key][field]
                check_estimate(paired, comparison)
                seed_means = by_seed.mean(axis=1)
                check([r['seed'] for r in comparison['per_seed']] == list(SEEDS), 'Different training seeds')
                np.testing.assert_allclose(seed_means, [r['mean'] for r in comparison['per_seed']], rtol=0, atol=1e-8)
                simultaneous = outcome['simultaneous_95'][key][field]
                half = z*paired.std(ddof=1)/math.sqrt(DEALS)
                lower, upper = paired.mean()-half, paired.mean()+half
                np.testing.assert_allclose([lower, upper], [simultaneous['lower'], simultaneous['upper']],
                                           rtol=0, atol=1e-8)
                groups, moves = traces[key].results(paired)
                if key == 'hybrid_minus_480':
                    preflop_moves[field] = moves
                forbidden = ('postflop',) if i == 0 else ('opening', 'other_preflop')
                check(all(groups[street]['paired_hands'] == 0 for street in forbidden),
                      f'{field}/{key}: hero first differed on a street with identical weights')
                diagnostics[key][field] = groups
                estimates[key][field] = dict(mean=float(paired.mean()), seed_means=seed_means.tolist(),
                                             simultaneous_lower=float(lower), simultaneous_upper=float(upper))
    return dict(checked_utc=utc(), summary_sha256=digest(SWAP/'summary.json'),
                source_summary_sha256=SOURCE_SUMMARY, source_audit_sha256=SOURCE_AUDIT,
                new_cells=9, reused_cells=18, raw_hand_records_reaggregated=3240000,
                matched_card_seat_seed_field_triplets=1080000, independent_card_blocks=DEALS,
                first_card=START, last_card=START+DEALS-1, fixed_training_seeds=list(SEEDS),
                per_seed_street_network_sha256=identities, terminal_action_replay_verified_by_source_cells=True,
                raw_hand_and_provenance_hashes_verified=True, common_opponents_and_action_rng_protocol=True,
                absolute_and_paired_intervals_recomputed=True, per_card_per_seed_additivity_verified=True,
                simultaneous_interval_family='Three opponent fields within each prespecified contrast',
                first_hero_difference_agrees_with_changed_street=True, first_action_divergence=diagnostics,
                nonopening_preflop_first_action_change_hybrid_minus_480=preflop_moves,
                first_action_scope='Terminal changes per 100 of all hands, grouped at the first differing action. '
                                   'Opponent-first pairs are included in all statistics.',
                paired_estimates=estimates)


if __name__ == '__main__':
    atomic_json(SWAP/'final-audit.json', audit())
