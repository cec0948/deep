"""Score frozen policies on paired original cards, conditioning on the named opponent panel."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from fullhouse_v3_common import atomic_json, digest


LABELS = ('steps0', 'steps256')
CAPS = (50, 100, 200)
ROLES = ('primary', 'diagnostic', 'holdout')


def _validate_design(groups, seed_ids):
    if not seed_ids or len(set(seed_ids)) != len(seed_ids) or any(type(s) is not int for s in seed_ids):
        raise ValueError('Expected distinct integer training seeds')
    if not any(group['role'] == 'primary' for group in groups.values()):
        raise ValueError('A primary group is required')
    for field, group in groups.items():
        if group['role'] not in ROLES or not group['family']:
            raise ValueError(f'Invalid family or role for {field}')
        if type(group['deal_seed']) is not int or type(group['deals']) is not int or group['deals'] < 2:
            raise ValueError(f'Invalid card range for {field}')
        if group['role'] == 'holdout':
            low, high = group['deal_seed'], group['deal_seed'] + group['deals']
            for other in groups.values():
                start = other['deal_seed']
                if other['role'] != 'holdout' and low < start + other['deals'] and start < high:
                    raise ValueError(f'Holdout {field} overlaps the design cards')


def _estimate(values, z):
    mean = float(np.mean(values))
    radius = float(z * np.std(values, ddof=1) / math.sqrt(len(values)))
    return {'mean': mean, 'lower': mean - radius, 'upper': mean + radius}


def _breadth(fields, field_scores, weights, label):
    results = {field: field_scores[field][label] for field in fields}
    positive = [field for field, stats in results.items() if stats['mean'] > 0]
    confident = [field for field, stats in results.items() if stats['lower'] > 0]
    worst = min(fields, key=lambda field: results[field]['mean'])
    return {
        'positive_fields': positive,
        'positive_fields_simultaneous_lower_above_zero': confident,
        'family_balanced_positive_share': sum(weights[field] for field in positive),
        'family_balanced_confirmed_positive_share': sum(weights[field] for field in confident),
        'weakest_field_by_point_estimate': worst,
        'minimum_field_mean_bb100': {
            key: min(stats[key] for stats in results.values()) for key in ('mean', 'lower', 'upper')
        },
    }


def _capped_delta(row, labels, cap):
    """Propagate simultaneous bounds using monotonicity and the unit slope bound."""
    old, new = (row[label] for label in labels)
    diff = row['paired_change']
    lower = min(cap, new['lower']) - min(cap, old['upper'])
    upper = min(cap, new['upper']) - min(cap, old['lower'])
    return {
        'mean': min(cap, new['mean']) - min(cap, old['mean']),
        'lower': max(lower, min(0.0, diff['lower'])),
        'upper': min(upper, max(0.0, diff['upper'])),
    }


def _role_summary(fields, groups, field_scores, labels, caps):
    families = Counter(groups[field]['family'] for field in fields)
    weights = {field: 1 / (len(families) * families[groups[field]['family']]) for field in fields}
    capped = {}
    for cap in caps:
        deltas = {field: _capped_delta(field_scores[field], labels, cap) for field in fields}
        entry = {
            label: sum(weights[f] * min(cap, field_scores[f][label]['mean']) for f in fields)
            for label in labels
        }
        entry['paired_change'] = {
            key: sum(weights[f] * deltas[f][key] for f in fields) for key in ('mean', 'lower', 'upper')
        }
        capped[str(cap)] = entry
    return {
        'field_weights': weights,
        'breadth': {label: _breadth(fields, field_scores, weights, label) for label in labels},
        'uncapped_family_balanced_change_bb100': {
            key: sum(weights[f] * field_scores[f]['paired_change'][key] for f in fields)
            for key in ('mean', 'lower', 'upper')
        },
        'family_balanced_capped_bb100': capped,
    }


def score_panel(panel, groups, seed_ids, *, baseline=LABELS[0], candidate=LABELS[1], caps=CAPS):
    """panel[field][label][seed] holds one six-seat mean in bb/100 per ordered card."""
    _validate_design(groups, seed_ids)
    if set(panel) != set(groups) or baseline == candidate:
        raise ValueError('Panel fields or policy labels do not match')
    if not caps or len(set(caps)) != len(caps) or any(not math.isfinite(c) or c <= 0 for c in caps):
        raise ValueError('Expected distinct finite positive caps')
    labels = (baseline, candidate)
    comparison_count = 3 * len(groups)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * comparison_count))
    scores = {}
    for field, group in groups.items():
        policies = panel[field]
        if set(policies) != set(labels):
            raise ValueError(f'Missing or unexpected policy in {field}')
        values = {}
        for label, rows in policies.items():
            if set(rows) != set(seed_ids):
                raise ValueError(f'Training seed coverage differs for {field}/{label}')
            matrix = np.asarray([rows[seed] for seed in seed_ids], dtype=float)
            if matrix.shape != (len(seed_ids), group['deals']) or not np.isfinite(matrix).all():
                raise ValueError(f'Incomplete or nonfinite card blocks for {field}/{label}')
            values[label] = matrix.mean(axis=0)
        scores[field] = {
            'family': group['family'], 'role': group['role'],
            'deal_seed': group['deal_seed'], 'original_card_blocks': group['deals'],
            **{label: _estimate(values[label], z) for label in labels},
            'paired_change': _estimate(values[candidate] - values[baseline], z),
        }
    roles = {}
    for role in ROLES:
        fields = [f for f in groups if groups[f]['role'] == role]
        if fields:
            roles[role] = _role_summary(fields, groups, scores, labels, caps)
    return {
        'baseline': baseline, 'candidate': candidate, 'training_seeds': list(seed_ids),
        'units': 'bb/100', 'fields': scores, 'roles': roles,
        'uncertainty': {
            'confidence': 0.95, 'simultaneous_estimates': comparison_count, 'critical_z': z,
            'method': 'Large-sample Bonferroni intervals for both absolute means and the paired change per '
                      'field. Six seats and all fixed training seeds are averaged inside the original card. '
                      'Group bounds propagate these simultaneous intervals and allow arbitrary field correlation.',
        },
        'scope': 'Conditional on the fixed policies, training seeds, and declared opponent lineups. '
                 'Weights are equal across families and equal within family. Saturation applies to each '
                 'lineup mean before weighting. The holdout has its own cards and separate summary.',
    }


def score_study(study):
    plan_path = study / 'evaluation-plan.json'
    plan = json.loads(plan_path.read_text())
    groups, seed_ids = plan['fields'], plan['seeds']
    _validate_design(groups, seed_ids)
    panel = {}
    hashes = {}
    for field, group in groups.items():
        panel[field] = {label: {} for label in LABELS}
        for label in LABELS:
            for seed in seed_ids:
                name = f'{field}-seed-{seed}-{label}'
                path = study / 'evaluation' / name / 'results.json'
                result = json.loads(path.read_text())
                ref = plan['policies'][label][str(seed)]
                counts = result['verification']
                if (result['field'] != field or result['deals'] != group['deals']
                        or result['deal_seed'] != group['deal_seed'] or result['hands'] != 6 * group['deals']
                        or result.get('label', label) != label or result.get('training_seed', seed) != seed
                        or result['candidate_sha256'] != ref['sha256'] or result['big_blind_chips'] != 100
                        or not all(result.get(k) is True for k in
                                   ('complete', 'raw_hands_verified', 'terminal_payoffs_replayed'))
                        or counts.get('terminal_payoffs_replayed') is not True
                        or counts.get('aggregate_recomputed') is not True
                        or counts.get('chip_conservation_verified') is not True
                        or counts['hands'] != result['hands'] or counts['card_blocks'] != result['deals']):
                    raise ValueError(f'Wrong policy, cards or verification metadata in {name}')
                blocks = result['deal_blocks']
                # At 100 chips/big blind, a six-seat mean hero payoff in chips equals bb/100.
                if not math.isclose(float(np.mean(blocks)), result['bb_per_100'], rel_tol=0, abs_tol=1e-8):
                    raise ValueError(f'Card blocks and bb/100 disagree in {name}')
                panel[field][label][seed] = blocks
                hashes[name] = digest(path)
    summary = score_panel(panel, groups, seed_ids, caps=plan.get('caps_bb100', CAPS))
    summary.update(evaluation_plan_sha256=digest(plan_path), cell_result_hashes=hashes)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = score_study(args.study)
    if args.output is None:
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        atomic_json(args.output, report)
