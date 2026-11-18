"""Attribute the audited 420→480 public-field delta to disjoint paired hands."""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
from itertools import zip_longest
import json
import math
from pathlib import Path

import numpy as np

from evaluate_average_baseline import hand_seed, interval
from fullhouse_runtime import BRIDGE, Bridge, ROOT
from fullhouse_v3_common import atomic_json, digest


RUN = '20261117T063900Z'
STUDY = ROOT / 'artifacts/average-extension' / RUN
SEEDS = (342, 343, 344)
DEALS = 20000
START = 2037400000
SUMMARY_SHA = 'e4e316230cc8b508fffa47beb26c470c5ad382240810f466e3c3a1171f368374'


def verified_cell(study, summary, seed, field, label):
    cell = f'{field}-seed-{seed}-{label}'
    folder = study / 'evaluation' / cell
    if digest(folder / 'results.json') != summary['result_files'][cell]:
        raise ValueError(f'Cell digest mismatch: {cell}')
    result = json.loads((folder / 'results.json').read_text())
    if not (result['complete'] and result['raw_hands_verified'] and result['terminal_payoffs_replayed']
            and (result['deal_seed'], result['deals'], result['hands'], result['big_blind_chips'])
            == (START, DEALS, DEALS * 6, 100)):
        raise ValueError(f'Incomplete or incompatible cell: {cell}')
    return folder, result


def first_difference(left, right, hero):
    for index, (a, b) in enumerate(zip_longest(left, right)):
        if a != b:
            if not a or not b or a['seat'] != hero or b['seat'] != hero:
                raise ValueError('Traces diverged before a paired hero decision')
            prefix = tuple((v['seat'], v['kind'], v.get('to')) for v in left[:index])
            return prefix, a, b
    return None


def audit(study):
    if digest(study / 'evaluation-summary.json') != SUMMARY_SHA:
        raise ValueError('Expected audited 480 evaluation')
    summary = json.loads((study / 'evaluation-summary.json').read_text())
    by_street = defaultdict(lambda: np.zeros((len(SEEDS), DEALS), dtype=np.int64))
    preflop_nodes = defaultdict(lambda: np.zeros((len(SEEDS), DEALS), dtype=np.int64))
    node_counts = Counter()
    counts, actions, action_chips = Counter(), Counter(), Counter()
    fallback = {}
    engine = Bridge()

    @lru_cache(maxsize=32768)
    def public_position(prefix):
        # With fixed stacks and button, the street and acting seat depend only
        # on the public betting prefix. Hole cards never enter this lookup.
        state = engine.call(dict(op='reset', seed=START, button=0))
        for seat, kind, amount in prefix:
            if state['terminal'] or state['seat_to_act'] != seat:
                raise ValueError('Illegal common action prefix')
            action = dict(seat=seat, kind=kind)
            if amount is not None:
                action['to'] = amount
            state = engine.call(dict(op='step', action=action))
        return state['street'], state['seat_to_act']

    try:
        for si, seed in enumerate(SEEDS):
            cells = [verified_cell(study, summary, seed, 'public', label) for label in ('parent420', 'current480')]
            streams = []
            for folder, result in cells:
                path = folder / 'hands.jsonl.gz'
                if digest(path) != result['raw_hands_sha256'] or digest(BRIDGE) != result['engine_sha256']:
                    raise ValueError('Raw evidence or rules engine mismatch')
                streams.append(path)
            if cells[0][1]['opponents'] != cells[1][1]['opponents']:
                raise ValueError('Different opponent in paired comparison')
            paired = 0
            with gzip.open(streams[0], 'rt') as old, gzip.open(streams[1], 'rt') as new:
                for index, (left, right) in enumerate(zip_longest(old, new)):
                    deal, hero = divmod(index, 6)
                    if not left or not right or deal >= DEALS:
                        raise ValueError('Invalid paired trace length')
                    a, b = json.loads(left), json.loads(right)
                    for row in (a, b):
                        if (row['deal'], row['hero'], row['seed'], row['hand_seed']) != (
                                deal, hero, START + deal, hand_seed(START + deal, hero)):
                            raise ValueError('Mismatched card or seat order')
                        if sum(row['utilities']) or row['chips'] != row['utilities'][hero]:
                            raise ValueError('Invalid terminal utility')
                    diff = first_difference(a['actions'], b['actions'], hero)
                    delta = b['chips'] - a['chips']
                    if diff is None:
                        if a['utilities'] != b['utilities']:
                            raise ValueError('Identical trace has different payoff')
                        street = 'same_trace'
                    else:
                        prefix, before, after = diff
                        street, actor = public_position(prefix)
                        if actor != hero:
                            raise ValueError('Common prefix points to another player')
                        if street == 'preflop':
                            node = 'first_actor_unopened' if not prefix else 'other_preflop'
                            node_counts[node] += 1
                            preflop_nodes[node][si, deal] += delta
                        change = f"{street}/{before['kind']}->{after['kind']}"
                        if before['kind'] == after['kind'] == 'raise':
                            change += '_larger' if after['to'] > before['to'] else '_smaller'
                        actions[change] += 1
                        action_chips[change] += delta
                    counts[street] += 1
                    by_street[street][si, deal] += delta
                    paired += 1
            if paired != DEALS * 6:
                raise ValueError('Missing paired hands')
            expected = np.subtract(cells[1][1]['deal_blocks'], cells[0][1]['deal_blocks'])
            if not np.allclose(sum(v[si] for v in by_street.values()) / 6, expected, rtol=0, atol=1e-9):
                raise ValueError('Attribution differs from archived card-block results')
    finally:
        engine.close()
    for field in ('mixed', 'noregrets'):
        for label in ('parent420', 'current480'):
            behavior = Counter()
            for seed in SEEDS:
                _, result = verified_cell(study, summary, seed, field, label)
                behavior.update(result['behavior'])
            hits = sum(v for k, v in behavior.items() if k.endswith('/fallback'))
            total = sum(v for k, v in behavior.items() if k.endswith('/decisions'))
            fallback[f'{field}/{label}'] = dict(fallback=hits, blueprint_decisions=total, rate=hits/total)
    def contributions(groups, sizes):
        return {name:dict(pairs=sizes[name], per_seed_bb100=(data.mean(axis=1)/6).tolist(),
                          **interval((data.mean(axis=0)/6).tolist())) for name, data in sorted(groups.items())}
    streets = contributions(by_street, counts)
    if not np.array_equal(sum(preflop_nodes.values()), by_street['preflop']):
        raise ValueError('Opening and later preflop attribution do not sum to street delta')
    denom = DEALS * len(SEEDS) * 6
    total = sum(v['mean'] for v in streets.values())
    expected = summary['diagnostic_comparisons']['public/current480-minus-parent420']['mean']
    if not math.isclose(total, expected, abs_tol=1e-9) or not math.isclose(sum(action_chips.values())/denom, total, abs_tol=1e-9):
        raise ValueError('Overall paired attribution mismatch')
    return dict(run=RUN, evaluation_summary_sha256=SUMMARY_SHA, card_start=START, independent_blocks=DEALS,
                seeds=list(SEEDS), paired_hands=denom, raw_hand_rows=denom*2, total_public_delta_bb100=total,
                first_difference_street=streets, first_difference_preflop_node=contributions(preflop_nodes, node_counts),
                first_difference_action={k:dict(pairs=actions[k],
                    contribution_bb100=action_chips[k]/denom) for k in sorted(actions)},
                blueprint_fallback=fallback, notes='Exploratory additive contributions per 100 of ALL hands. '
                'Intervals use the original card blocks and fixed seeds, without subgroup multiplicity correction. '
                'All actions after first divergence contribute to the terminal payoff. No causal attribution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    atomic_json(args.output, audit(STUDY))
