"""Recompute the 480/492 paired scores and locate the first differing betting decision."""
from collections import Counter
from contextlib import closing
from itertools import zip_longest
from statistics import NormalDist
import gzip
import hashlib
import json
import math

import numpy as np

from controlled_poker_refit import read, utc
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest


STUDY = ROOT / 'artifacts/preflop-followup/480-v1/warm-start'
SEEDS = (342, 343, 344)
FIELDS = ('public', 'mixed', 'noregrets')
POLICIES = ('specialist480', 'warm492')
START = 2037600000
DEALS = 20000
SEATS = 6


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_estimate(values, estimate):
    mean = values.mean()
    half = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    np.testing.assert_allclose(
        [mean, half, mean-half, mean+half],
        [estimate[k] for k in ('mean', 'ci95_half_width', 'lower', 'upper')],
        rtol=0, atol=1e-8)


class FirstActionDifferences:
    """Account for every pair, including ones where an opponent first differed."""
    def __init__(self, bridge):
        self.bridge = bridge
        state = bridge.call(dict(op='reset', seed=START, button=0))
        self.nodes = {(): (state['street'], state['seat_to_act'])}
        self.pairs = Counter()
        self.changed_payoffs = Counter()
        self.preflop_changes = Counter()
        self.preflop_chips = Counter()
        self.chips = {k: np.zeros(DEALS, dtype=np.int64) for k in (
            'opening', 'other_preflop', 'postflop', 'same_trace',
            'opponent_first_after_hero_fold', 'opponent_first_other')}

    def hero_group(self, prefix, hero):
        # Street and acting seat are determined by the public bets and fixed stacks.
        # Only cache histories up to the end of preflop.
        key = tuple((a['seat'], a['kind'], a.get('to')) for a in prefix)
        for i in range(1, len(key)+1):
            cached = self.nodes.get(key[:i])
            if cached is None:
                state = self.bridge.call(dict(op='reset', seed=START, button=0))
                for j, action in enumerate(prefix, 1):
                    check(state['seat_to_act'] == action['seat'], 'Illegal common betting history')
                    state = self.bridge.call(dict(op='step', action=action))
                    check(not state['terminal'], 'Divergence after a terminal hand')
                    self.nodes[key[:j]] = (state['street'], state['seat_to_act'])
                    if state['street'] != 'preflop':
                        return 'postflop'
                break
            if cached[0] != 'preflop':
                return 'postflop'
        check(self.nodes[key] == ('preflop', hero), 'Wrong actor at the common preflop decision')
        return 'other_preflop' if prefix else 'opening'

    def add(self, before, after):
        hero = before['hero']
        delta = after['chips'] - before['chips']
        group = 'same_trace'
        for i, (old, new) in enumerate(zip_longest(before['actions'], after['actions'])):
            if old == new:
                continue
            check(old and new and old['seat'] == new['seat'], 'Invalid paired action order')
            prefix = before['actions'][:i]
            if old['seat'] == hero:
                group = self.hero_group(prefix, hero)
                if group == 'other_preflop':
                    change = old['kind'] + '_to_' + new['kind']
                    if old['kind'] == new['kind'] == 'raise':
                        change = 'raise_larger' if new['to'] > old['to'] else 'raise_smaller'
                    self.preflop_changes[change] += 1
                    self.preflop_chips[change] += delta
            else:
                folded = any(a['seat'] == hero and a['kind'] == 'fold' for a in prefix)
                group = 'opponent_first_after_hero_fold' if folded else 'opponent_first_other'
            break
        if group == 'same_trace':
            check(before['utilities'] == after['utilities'], 'Identical actions produced different payoffs')
        if group == 'opponent_first_after_hero_fold':
            check(delta == 0, 'Opponent action changed the payoff of a folded hero')
        self.pairs[group] += 1
        self.changed_payoffs[group] += delta != 0
        self.chips[group][before['deal']] += delta

    def results(self, expected):
        repeats = SEATS * len(SEEDS)
        check(sum(self.pairs.values()) == DEALS*repeats, 'Incomplete first-difference groups')
        np.testing.assert_allclose(sum(self.chips.values())/repeats, expected, rtol=0, atol=1e-9)
        check(sum(self.preflop_chips.values()) == self.chips['other_preflop'].sum()
              and sum(self.preflop_changes.values()) == self.pairs['other_preflop'],
              'Preflop actions did not recover the preflop total')
        groups = {k: dict(paired_hands=self.pairs[k], nonzero_hero_payoff_changes=self.changed_payoffs[k],
                          contribution_bb100=float(v.mean()/repeats)) for k, v in self.chips.items()}
        moves = {k: dict(paired_hands=n, contribution_bb100=self.preflop_chips[k]/(DEALS*repeats))
                 for k, n in sorted(self.preflop_changes.items())}
        return groups, moves


def verified_cell(plan, summary, seed, field, label):
    name = f'{field}-seed-{seed}-{label}'
    folder = STUDY/'evaluation'/name
    result = read(folder/'results.json')
    provenance = read(folder/'provenance.json')
    raw = folder/'hands.jsonl.gz'
    check(digest(folder/'results.json') == summary['result_sha256'][name]
          and digest(raw) == result['raw_hands_sha256']
          and digest(folder/'provenance.json') == result['provenance_sha256']
          and digest(BRIDGE) == result['engine_sha256'], 'Changed evaluation evidence: '+name)
    policy = plan['policies'][label][str(seed)]
    check(policy['path'] == result['candidate'] == provenance['candidate']
          and digest(policy['path']) == result['candidate_sha256'] == policy['sha256'] == provenance['candidate_sha256'],
          'Changed candidate: '+name)
    spec = read(policy['path'])
    policy_files = {policy['path']}
    for street in ('preflop', 'postflop'):
        ref = spec[street]
        check(digest(ref['path']) == ref['sha256'] == result['dependency_hashes'][ref['path']],
              'Different street model: '+name)
        policy_files.add(ref['path'])
    check(result['dependency_hashes'] == provenance['dependency_hashes']
          and result['opponents'] == provenance['opponents'] == plan['opponents'][field]
          and (result['field'], result['deal_seed'], result['hands'], result['big_blind_chips'])
          == (field, START, DEALS*SEATS, 100), 'Wrong game or opponents: '+name)
    check(result['complete'] and result['raw_hands_verified'] and result['terminal_payoffs_replayed']
          and result['verification']['terminal_payoffs_replayed'] and result['verification']['aggregate_recomputed'],
          'Cell did not finish its terminal replay: '+name)
    dependencies = {k: v for k, v in result['dependency_hashes'].items() if k not in policy_files}
    return raw, result, (dependencies, provenance['semantic_rng'])


def paired_cards(cells, divergences):
    (left_path, left, left_rules), (right_path, right, right_rules) = cells
    check(left_rules == right_rules, 'Different opponent assets, engine or action RNG protocol')
    chips = np.empty((2, DEALS, SEATS), dtype=np.int64)
    count = 0
    with gzip.open(left_path, 'rb') as first, gzip.open(right_path, 'rb') as second:
        for index, pair in enumerate(zip_longest(first, second)):
            deal, hero = divmod(index, SEATS)
            check(deal < DEALS and None not in pair, 'Unmatched card/seat pair')
            card = START+deal
            hand_seed = int.from_bytes(hashlib.sha256(f'rich-evaluation-hand:{card}:{hero}'.encode()).digest()[:8], 'big')
            rows = [json.loads(line) for line in pair]
            for i, row in enumerate(rows):
                check((row['deal'], row['hero'], row['seed'], row['hand_seed']) == (deal, hero, card, hand_seed),
                      'Card, seat or action seed differs')
                check(len(row['utilities']) == SEATS and sum(row['utilities']) == 0
                      and row['chips'] == row['utilities'][hero], 'Invalid terminal utility')
                chips[i, deal, hero] = row['chips']
            divergences.add(*rows)
            count += 1
    check(count == DEALS*SEATS, 'Missing card/seat pairs')
    means = chips.mean(axis=2)
    for block, result in zip(means, (left, right), strict=True):
        np.testing.assert_allclose(block, result['deal_blocks'], rtol=0, atol=1e-8)
    return means


def audit():
    summary, plan = (read(STUDY/name) for name in ('summary.json', 'evaluation-plan.json'))
    design, stage = (read(STUDY/name) for name in ('plan.json', 'stage-ready.json'))
    check(summary['completed'] and summary['completed_cells'] == 18 and summary['verified_hands'] == 2160000
          and summary['independent_card_blocks'] == DEALS and summary['new_selfplay_iterations_per_seed'] == 12
          and summary['warm_start_weight_chain_verified'] and not summary['model_promoted'], 'Incomplete experiment')
    check(digest(STUDY/'evaluation-plan.json') == summary['evaluation_plan_sha256']
          and (plan['card_start'], plan['deals']) == (START, DEALS)
          and all(end <= START or start >= START+DEALS for start, end in read(STUDY/'config.json')['prior_card_ranges']),
          'Different evaluation plan or overlapping cards')
    for s in map(str, SEEDS):
        check(plan['policies'][POLICIES[0]][s] == design['parent_policies'][s]
              and plan['policies'][POLICIES[1]][s] == stage['arms'][POLICIES[1]][s]['policy'],
              'Candidate differs from training or the paired baseline')
    fallback, simultaneous, groups, moves = {}, {}, {}, {}
    z = NormalDist().inv_cdf(1-0.05/(2*len(FIELDS)))
    with closing(Bridge()) as engine:
        for field in FIELDS:
            traces = FirstActionDifferences(engine)
            card_means = []
            behavior = {label: Counter() for label in POLICIES}
            for seed in SEEDS:
                cells = [verified_cell(plan, summary, seed, field, label) for label in POLICIES]
                card_means.append(paired_cards(cells, traces))
                for label, (_, result, _) in zip(POLICIES, cells, strict=True):
                    behavior[label].update(result['behavior'])
            scores = np.asarray(card_means)  # seed, policy, card
            for i, label in enumerate(POLICIES):
                check_estimate(scores[:,i,:].mean(axis=0), summary['absolute_scores'][label][field])
                counts = behavior[label]
                hits, total = (sum(n for k, n in counts.items() if k.startswith('opponent/') and k.endswith('/'+suffix))
                               for suffix in ('fallback', 'decisions'))
                if total:
                    fallback[f'{label}/{field}'] = hits/total
            seed_deltas = scores[:,1,:]-scores[:,0,:]
            delta = seed_deltas.mean(axis=0)
            estimate = summary['trained_minus_frozen_parent'][field]
            check_estimate(delta, estimate)
            check([r['seed'] for r in estimate['per_seed']] == list(SEEDS), 'Different training seeds')
            np.testing.assert_allclose(seed_deltas.mean(axis=1), [r['mean'] for r in estimate['per_seed']], rtol=0, atol=1e-8)
            half = z*delta.std(ddof=1)/math.sqrt(DEALS)
            bounds = dict(lower=float(delta.mean()-half), upper=float(delta.mean()+half))
            for k, value in bounds.items():
                np.testing.assert_allclose(value, summary['simultaneous_95_bonferroni'][field][k], rtol=0, atol=1e-8)
            simultaneous[field] = bounds
            groups[field], moves[field] = traces.results(delta)
    pairs = DEALS*SEATS*len(SEEDS)*len(FIELDS)
    return dict(checked_utc=utc(), summary_sha256=digest(STUDY/'summary.json'), new_cells=18, reused_cells=0,
                raw_hand_records_reaggregated=2*pairs, matched_seat_card_pairs=pairs,
                independent_card_blocks=DEALS, fixed_training_seeds=list(SEEDS), seats_per_card=SEATS,
                common_opponents_engine_dependencies_and_action_rng_protocol=True,
                both_policy_routing_and_weights_sha256_verified=True, raw_records_and_provenance_sha256_verified=18,
                each_source_result_has_terminal_action_replay=True,
                absolute_scores_paired_scores_per_seed_and_intervals_recomputed=True,
                interval='Paired card-block intervals after averaging the six seats and three fixed training seeds',
                secondary_simultaneous_95_bonferroni_492_minus_480=simultaneous,
                first_action_divergence=groups, nonopening_preflop_first_action_change=moves,
                first_divergence_scope='Exploratory terminal differences per 100 of ALL hands, grouped by the first '
                    'differing action. Opponent-first pairs are included. All later decisions contribute to the payoff.',
                action_lookup_fallback_rate=fallback)


if __name__ == '__main__':
    atomic_json(STUDY/'final-audit.json', audit())
