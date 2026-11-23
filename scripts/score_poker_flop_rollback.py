"""Verify and score the paired, in-sample single-street routing comparison."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import verify_results
from fullhouse_v3_common import atomic_json, digest
from score_poker_population import score_panel


BASE = 'specialist480'
DIVERSE = 'diverse504'
HYBRID = 'flop480late504'
SLOTS = (486, 492, 498, 504)
Z95 = NormalDist().inv_cdf(.975)


def gap_recovery(gap, gain):
    """Return the recovery ratio and its paired Fieller interval, without clipping it."""
    gap, gain = (np.asarray(values, dtype=float) for values in (gap, gain))
    if (gap.ndim != 1 or gap.shape != gain.shape or gap.size < 2
            or not np.isfinite(gap).all() or not np.isfinite(gain).all()):
        raise ValueError('Paired finite public card blocks are required')
    d, g = float(gap.mean()), float(gain.mean())
    variance = np.cov(gap, gain, ddof=1) / len(gap)
    a = d*d - Z95**2 * variance[0, 0]
    b = -2 * (d*g - Z95**2 * variance[0, 1])
    c = g*g - Z95**2 * variance[1, 1]
    discriminant = b*b - 4*a*c
    bounds = None
    if d > 0 and a > 0:
        # Perfectly proportional block returns can yield a tiny negative from roundoff.
        tolerance = 32 * np.finfo(float).eps * (abs(b*b) + abs(4*a*c))
        if discriminant < -tolerance:
            raise ValueError('Inconsistent variance for public recovery ratio')
        radius = math.sqrt(max(0, discriminant))
        bounds = [float((-b-radius)/(2*a)), float((-b+radius)/(2*a))]
    width = Z95 * math.sqrt(variance[0, 0])
    return dict(numerator_bb100=g, denominator_bb100=d, denominator_ci95=[d-width, d+width],
                fraction=g/d if d > 0 else None, paired_fieller_ci95=bounds,
                interval_status='bounded' if bounds is not None else 'positive gap unresolved',
                method='Diagnostic ratio of paired public means, (hybrid minus diverse504) / '
                       '(specialist480 minus diverse504). Nominal Fieller interval with uncertainty '
                       'in both means and their covariance, conditional on the selected cards.')


def _jobs(plan, labels):
    jobs = [row for row in plan['jobs'] if row['label'] in labels]
    expected = {f'{f}-seed-{s}-{label}' for f in plan['fields'] for s in plan['seeds']
                for label in labels}
    for row in jobs:
        if row['id'] != f"{row['field']}-seed-{row['seed']}-{row['label']}":
            raise ValueError('Invalid field, seed or policy in the cell identity')
    if len(jobs) != len(expected) or {row['id'] for row in jobs} != expected:
        raise ValueError('Repeated or missing evaluation cell')
    return jobs


def _validate_sources(study, previous_study):
    path = study/'evaluation-plan.json'
    plan_sha = digest(path)
    verify_files({path: read(study/'bound.json')['evaluation_plan_sha256']})
    plan = read(path)
    previous_path = previous_study/'evaluation-plan.json'
    reference_path = previous_study/'reference-plan.json'
    summary_path = previous_study/'summary.json'
    verify_files({previous_path: plan['previous_plan_sha256'],
                  reference_path: plan['previous_reference_plan_sha256'],
                  summary_path: plan['previous_summary_sha256']})
    previous, reference, summary = (read(p) for p in (previous_path, reference_path, summary_path))
    if (not summary['complete'] or summary['plan_sha256'] != plan['previous_plan_sha256']
            or summary['reference_plan_sha256'] != previous['reference_plan_sha256']
            or previous['reference_plan_sha256'] != plan['previous_reference_plan_sha256']
            or read(previous_study/'evaluation-bound.json')['plan_sha256'] != summary['plan_sha256']
            or read(previous_study/'reference-bound.json')['plan_sha256'] != summary['reference_plan_sha256']):
        raise ValueError('Inherited evaluation bindings differ')
    for context in (plan, previous, reference):
        verify_files(context['frozen_files'])
        if context['fields'] != previous['fields'] or context['seeds'] != [342, 343, 344]:
            raise ValueError('Field, role, card schedule or fixed seeds changed')
    if set(plan['policies']) != {HYBRID} or len(plan['jobs']) != 27:
        raise ValueError('Unexpected new policy or number of new cells')
    new_jobs = _jobs(plan, {HYBRID})
    old_jobs = _jobs(previous, {BASE, DIVERSE})
    inherited = {j['id']: summary['cell_result_sha256'][j['id']] for j in old_jobs}
    if inherited != plan['previous_cells']:
        raise ValueError('Previous result bindings differ from the archived summary')
    return plan, plan_sha, new_jobs, previous, reference, summary, old_jobs


def _snapshot_trace(path, result):
    """Independently check every recorded hand selection and compare order using its digest."""
    if result['label'] == BASE:
        if result.get('snapshot_hand_counts'):
            raise ValueError('Unexpected snapshot mixture in specialist480')
        return None
    trace = bytearray()
    counts = Counter()
    with gzip.open(path/'hands.jsonl.gz', 'rt') as stream:
        for line in stream:
            slot = json.loads(line)['snapshot_iteration']
            if type(slot) is not int or slot not in SLOTS:
                raise ValueError(f'Unknown per-hand snapshot at {path.name}')
            counts[str(slot)] += 1
            trace.append(SLOTS.index(slot))
    if (set(counts) != {str(slot) for slot in SLOTS}
            or counts != result['snapshot_hand_counts'] or len(trace) != result['hands']):
        raise ValueError(f'Raw snapshot totals differ at {path.name}')
    return dict(counts=dict(counts), hand_choice_sha256=hashlib.sha256(trace).hexdigest())


def _verify_cell(directory, job, context, plan_sha, inherited_hash=None):
    field, seed, label, key = (job[k] for k in ('field','seed','label','id'))
    path = directory/'evaluation'/key
    ref = context['policies'][label][str(seed)]
    spec = context['fields'][field]
    cell_sha = digest(path/'results.json')
    status = read(path/'status.json')
    if (status.get('phase') != 'complete' or status['results_sha256'] != cell_sha
            or status['verified_hands'] != 6*spec['deals']
            or (inherited_hash is not None and inherited_hash != cell_sha)):
        raise ValueError(f'Changed or incomplete cell {key}')
    expected = dict(label=label, training_seed=seed, field=field, deal_seed=spec['deal_seed'],
                    deals=spec['deals'], hands=6*spec['deals'], big_blind_chips=100,
                    candidate_sha256=ref['sha256'], terminal_payoffs_replayed=True,
                    evaluation_plan_sha256=plan_sha)
    result = verify_results(path, expected=expected)
    provenance = read(path/'provenance.json')
    if (any(result.get(k) != v for k, v in provenance.items())
            or Path(result['candidate']).resolve() != Path(ref['path']).resolve()
            or result['dependency_hashes'].get(str(Path(ref['path']).resolve())) != ref['sha256']):
        raise ValueError(f'Policy or provenance mismatch in {key}')
    trace = _snapshot_trace(path, result)
    matched = {k: result[k] for k in ('opponents','first_deal_clockwise_lineup','varies_by_card',
                                       'deal_seed','deals','big_blind_chips')}
    return result['deal_blocks'], cell_sha, trace, matched


def _public_comparison(values, seeds, base, candidate):
    old, new = (np.array([values[label][s] for s in seeds],dtype=float) for label in (base, candidate))
    diff = (new-old).mean(axis=0)
    mean = float(diff.mean())
    radius = float(Z95 * diff.std(ddof=1) / math.sqrt(len(diff)))
    return dict(baseline=base, candidate=candidate, baseline_bb100=float(old.mean()),
                candidate_bb100=float(new.mean()), by_seed=dict(zip(map(str,seeds),(new-old).mean(axis=1).tolist())),
                mean=mean, paired_ci95=[mean-radius,mean+radius])


def summarize_study(study, previous_study):
    """Verify all 81 cells and write the source summary for the reused diagnostic cards."""
    study, previous_study = Path(study), Path(previous_study)
    plan, plan_sha, new_jobs, previous, reference, prior_summary, old_jobs = _validate_sources(study, previous_study)
    fields, seeds = plan['fields'], plan['seeds']
    early = {j['id'] for j in reference['jobs']}
    rows = {field: {label: {} for label in (BASE, DIVERSE, HYBRID)} for field in fields}
    traces, matched_inputs, hashes = {}, {}, {}
    for directory, jobs, context in ((previous_study, old_jobs, previous), (study, new_jobs, plan)):
        for job in jobs:
            field, seed, label, key = (job[k] for k in ('field','seed','label','id'))
            old = label != HYBRID
            ident = plan_sha
            if old:
                ident = plan['previous_reference_plan_sha256'] if key in early else plan['previous_plan_sha256']
            blocks, hashes[key], trace, inputs = _verify_cell(
                directory, job, context, ident, plan['previous_cells'].get(key) if old else None)
            rows[field][label][seed] = blocks
            pair = f'{field}/{seed}'
            if (matched_inputs.setdefault(pair, inputs) != inputs
                    or plan['opponents'][f'{field}-seed-{seed}'] != inputs['opponents']):
                raise ValueError(f'Unmatched opponents, card schedule or lineup for {pair}')
            if trace is not None:
                if (traces.setdefault(pair, trace) != trace
                        or prior_summary['snapshot_hand_counts'][pair] != trace['counts']):
                    raise ValueError(f'Unmatched private snapshot choices for {pair}')
    def count_hands(jobs):
        return sum(6*fields[j['field']]['deals'] for j in jobs)
    unique = set()
    for spec in fields.values():
        unique.update(range(spec['deal_seed'],spec['deal_seed']+spec['deals']))
    new_hands, reference_hands = count_hands(new_jobs), count_hands(old_jobs)
    if (new_hands, reference_hands, len(unique)) != (648000, 1296000, 26000) or (
            new_hands != plan['expected_new_hands'] or reference_hands != plan['expected_reference_hands']
            or len(unique) != plan['unique_original_card_blocks']):
        raise ValueError('Card or hand totals differ from registration')
    def panel(base):
        paired = {field: {label: rows[field][label] for label in (base,HYBRID)} for field in fields}
        return score_panel(paired, fields, seeds, baseline=base, candidate=HYBRID)
    comparisons = {f'{new}_minus_{base}': _public_comparison(rows['public'],seeds,base,new)
                   for base,new in ((DIVERSE,HYBRID),(BASE,HYBRID),(BASE,DIVERSE))}
    means = {label: np.mean([rows['public'][label][s] for s in seeds],axis=0)
             for label in (BASE,DIVERSE,HYBRID)}
    result = dict(complete=True,verified_utc=utc(),model_promoted=False,
                  new_cells=len(new_jobs),reference_cells=len(old_jobs),completed_cells=len(hashes),
                  new_hands=new_hands,reference_hands=reference_hands,completed_hands=new_hands+reference_hands,
                  unique_original_card_blocks=len(unique),primary_public_comparison=f'{HYBRID}_minus_{DIVERSE}',
                  public_comparisons=comparisons,
                  public_gap_recovery=gap_recovery(means[BASE]-means[DIVERSE],means[HYBRID]-means[DIVERSE]),
                  treatment_vs_diverse504=panel(DIVERSE),treatment_vs_specialist480=panel(BASE),
                  scope='Street rollback diagnostic on the previously observed opponent-diversity cards. '
                        'These data motivated selection; all interval estimates are descriptive. '
                        'Six seats and three fixed seeds repeat each original card. The nine field '
                        'labels and original field roles are retained. No fitting or new self-play.',
                  plan_sha256=plan_sha,previous_plan_sha256=plan['previous_plan_sha256'],
                  previous_reference_plan_sha256=plan['previous_reference_plan_sha256'],
                  previous_summary_sha256=plan['previous_summary_sha256'],
                  cell_result_sha256=hashes,snapshot_hand_assignments=traces)
    verify_files({study/'evaluation-plan.json':plan_sha,previous_study/'summary.json':plan['previous_summary_sha256']})
    atomic_json(study/'summary.json', result)
    return result
