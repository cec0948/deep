"""Score matched forced flop actions with original card deals as independent units."""
import argparse
from collections import Counter
import gzip
import json
import math
from os.path import commonpath
from pathlib import Path
from statistics import NormalDist

import numpy as np

from fullhouse_v3_common import atomic_json, digest


LINEUPS = ('self480', 'public', 'tight-value')
SNAPSHOTS = ('480', '486', '492', '498', '504')
CANDIDATES = ('mixture', '486', '504')


def clustered_means(values, seeds, deals, fixed_seeds):
    """Average within seed/deal, then within seed, giving each fixed seed equal weight.

    Return column means and their standard errors. A repeated original deal uses a
    single influence contribution even when played by several distinct policies.
    """
    values = np.asarray(values, dtype=float)
    seeds = np.asarray(seeds)
    deals = np.asarray(deals)
    if (values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all()
            or seeds.shape != (len(values),) or deals.shape != (len(values),)):
        raise ValueError('Expected one finite vector, policy seed and original deal ID per observation')
    if not fixed_seeds or len(set(fixed_seeds)) != len(fixed_seeds) or set(seeds) != set(fixed_seeds):
        raise ValueError('Fixed seed coverage is incomplete or ambiguous')
    unique_deals, cluster = np.unique(deals, return_inverse=True)
    influence = np.zeros((len(unique_deals), values.shape[1]))
    seed_means = {}
    for seed in fixed_seeds:
        present = seeds == seed
        counts = np.bincount(cluster[present], minlength=len(unique_deals))
        observed = counts > 0
        n = np.count_nonzero(observed)
        if n < 2:
            raise ValueError(f'Seed {seed} requires at least two independent original deals')
        totals = np.zeros_like(influence)
        np.add.at(totals, cluster[present], values[present])
        deal_means = totals[observed] / counts[observed, None]
        seed_mean = deal_means.mean(axis=0)
        seed_means[seed] = seed_mean
        influence[observed] += (deal_means - seed_mean) / math.sqrt(n * (n - 1))
    mean = np.mean(list(seed_means.values()), axis=0)
    se = np.sqrt((influence ** 2).sum(axis=0)) / len(fixed_seeds)
    return mean, se, seed_means


def clustered_intervals(values, seeds, deals, fixed_seeds, names):
    """Nominal and simultaneous normal intervals for a preregistered set of contrasts."""
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected unique contrast names')
    means, ses, by_seed = clustered_means(values, seeds, deals, fixed_seeds)
    if len(means) != len(names):
        raise ValueError('Number of contrast names must match number of columns')
    z = NormalDist().inv_cdf(.975)
    family_z = NormalDist().inv_cdf(1 - .025 / len(names))
    summaries = {}
    for idx, name in enumerate(names):
        mean = float(means[idx])
        se = float(ses[idx])
        summaries[name] = dict(mean_bb=mean, cluster_se_bb=se,
                               paired_95_bb=[mean-z*se, mean+z*se],
                               simultaneous_95_bb=[mean-family_z*se, mean+family_z*se],
                               mean_bb_by_seed={str(seed): float(v[idx]) for seed, v in by_seed.items()})
    return summaries


def empirical_direction_counts(selfplay, external):
    """Observed state-level signs are descriptive because rollout values have sampling noise."""
    def direction(value):
        if value > 1e-10:
            return 'positive'
        if value < -1e-10:
            return 'negative'
        return 'tie'

    counts = Counter((direction(a), direction(b)) for a, b in zip(selfplay, external, strict=True))
    signs = ('positive', 'negative', 'tie')
    return {first: {second: counts[first, second] for second in signs} for first in signs}


def state_key(state):
    seed, deal = state['seed'], state['deal_seed']
    if type(seed) is not int or type(deal) is not int:
        raise ValueError('Training seed and original deal ID must be integers')
    return seed, deal


def probabilities(state):
    actions = state['legal_actions']
    if (not actions or len({json.dumps(a, sort_keys=True) for a in actions}) != len(actions)
            or set(state['policies']) != set(SNAPSHOTS)):
        raise ValueError('Missing policies or ambiguous concrete root actions')
    rows = np.array([state['policies'][label]['probabilities'] for label in SNAPSHOTS], dtype=float)
    if (rows.shape != (len(SNAPSHOTS), len(actions)) or not np.isfinite(rows).all()
            or np.any(rows < 0) or not np.allclose(rows.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Root action probabilities must form finite normalized distributions')
    # Inference may export float32 probabilities slightly off 1 when summed in float64.
    return rows / rows.sum(axis=1, keepdims=True)


def action_values(row, required, repeats, big_blind):
    payoffs = row['action_payoffs_chips']
    if row['repeats'] != repeats or len(payoffs) != len(required):
        raise ValueError('Incomplete repeat or root action coverage')
    q = np.zeros(len(required))
    for index, (needed, outcomes) in enumerate(zip(required, payoffs)):
        if outcomes is None:
            if needed:
                raise ValueError('Missing rollout payoff for positive-probability root action')
            continue
        if not needed or len(outcomes) != repeats or any(type(v) is not int for v in outcomes):
            raise ValueError('Unexpected action coverage, repeats or terminal chip amounts')
        q[index] = sum(outcomes) / (repeats * big_blind)
    return q


def score_flop_diagnostic(states, rollouts, *, seeds=(342, 343, 344), states_per_seed=1024, repeats=8):
    """Integrate each policy over the same concrete-action return estimates, in big blinds."""
    if not seeds or len(set(seeds)) != len(seeds) or states_per_seed < 2 or repeats <= 0:
        raise ValueError('Invalid fixed sampling design')
    sample = {}
    deltas = []
    top_action_pairs = []
    for state in states:
        key = state_key(state)
        if key in sample:
            raise ValueError('Repeated state in the frozen flop sample')
        probs = probabilities(state)
        candidate_probs = np.stack((probs[1:].mean(axis=0), probs[1], probs[-1]))
        deltas.append(candidate_probs - probs[0])
        tops = [np.flatnonzero(p >= p.max() - 1e-10) for p in (probs[0], candidate_probs[0])]
        top_action_pairs.append(tuple(int(top[0]) for top in tops) if all(len(top) == 1 for top in tops) else None)
        sample[key] = (len(sample), state, np.any(probs > 0, axis=0))
    coverage = Counter(seed for seed, _ in sample)
    if coverage != Counter({s: states_per_seed for s in seeds}):
        raise ValueError('Incomplete or unexpected fixed-seed flop sample')
    original_deals = [deal for _, deal in sample]
    if len(set(original_deals)) != len(original_deals):
        raise ValueError('This registered sample requires disjoint card ranges across policy seeds')
    gains = np.full((len(sample), len(CANDIDATES), len(LINEUPS)), np.nan)
    absolute_baseline = np.full((len(sample), len(LINEUPS)), np.nan)
    top_changes = np.full((len(sample), len(LINEUPS)), np.nan)
    terminal_branches = 0
    for row in rollouts:
        key = state_key(row)
        if key not in sample or row['lineup'] not in LINEUPS:
            raise ValueError('Rollout is outside the frozen state or lineup design')
        pos, state, supported = sample[key]
        field = LINEUPS.index(row['lineup'])
        if (row['hero'] != state['hero'] or row['legal_actions'] != state['legal_actions']
                or row.get('state_id') != state.get('state_id')):
            raise ValueError('Rollout identity or legal action order differs from frozen sample')
        if not math.isnan(absolute_baseline[pos, field]):
            raise ValueError('Duplicate rollout for a sampled state and lineup')
        q = action_values(row, supported, repeats, big_blind=100)
        baseline = np.asarray(state['policies']['480']['probabilities'], dtype=float)
        absolute_baseline[pos, field] = baseline @ q / baseline.sum()
        # Adding a constant to every eligible action value cannot affect a policy contrast.
        centered = np.where(supported, q - q[supported][0], 0)
        gains[pos, :, field] = deltas[pos] @ centered
        pair = top_action_pairs[pos]
        if pair is not None and pair[0] != pair[1]:
            top_changes[pos, field] = q[pair[1]] - q[pair[0]]
        terminal_branches += int(np.count_nonzero(supported)) * repeats
    if not np.isfinite(gains).all():
        raise ValueError('At least one matched lineup is missing')
    policy_seeds = [seed for seed, _ in sample]
    names = list(LINEUPS) + [f'{lineup}_minus_self480' for lineup in LINEUPS[1:]]
    comparisons = {}
    for j, label in enumerate(CANDIDATES):
        field_gains = gains[:, j, :]
        changes = np.column_stack((field_gains, field_gains[:, 1:] - field_gains[:, [0]]))
        estimates = clustered_intervals(changes, policy_seeds, original_deals, seeds, names)
        changed = np.array([np.abs(d[j]).sum() > 1e-10 for d in deltas])
        signs = {lineup: empirical_direction_counts(field_gains[changed, 0], field_gains[changed, k])
                 for k, lineup in enumerate(LINEUPS[1:], start=1)}
        comparisons[label] = dict(
            contrasts=estimates, states_with_probability_change=int(changed.sum()),
            observed_signs_on_changed_states_self480_by_external=signs)
    changed_top = np.isfinite(top_changes[:, 0])
    seed_ids = np.asarray(policy_seeds)
    changed_counts = {str(s): int((changed_top & (seed_ids == s)).sum()) for s in seeds}
    unique = sum(pair is not None for pair in top_action_pairs)
    selected = top_changes[changed_top]
    switches = dict(
        scope='Only states where 480 and the mixture have different, unique most-probable root actions. '
              'Force each of those two actions and compare their Q. Descriptive subset; a win over a '
              'specific continuation does not establish optimal play.',
        most_likely_tie_tolerance=1e-10, eligible_unique_tops=unique, ties_excluded=len(sample)-unique,
        different_top=int(changed_top.sum()), same_unique_top=unique-int(changed_top.sum()),
        share_of_all_sampled_states=float(changed_top.mean()), changed_states_by_seed=changed_counts,
        by_lineup={})
    for f, lineup in enumerate(LINEUPS):
        per_seed = {str(s): float(top_changes[changed_top & (seed_ids == s), f].mean())
                    for s in seeds if changed_counts[str(s)] > 0}
        switches['by_lineup'][lineup] = dict(
            new_top_minus_old_top_bb=(sum(per_seed.values()) / len(seeds) if len(per_seed) == len(seeds) else None),
            mean_bb_by_seed=per_seed, positive=int((selected[:, f] > 0).sum()),
            negative=int((selected[:, f] < 0).sum()), tied=int((selected[:, f] == 0).sum()))
    return dict(
        units='big blinds per sampled first hero flop decision', big_blind_chips=100,
        states=len(sample), distinct_original_deals=len(set(original_deals)),
        states_per_training_seed={str(s): coverage[s] for s in seeds}, repeats_per_action_per_lineup=repeats,
        evaluated_terminal_branches=terminal_branches, registered_lineups=list(LINEUPS),
        absolute_480_bb={name: float(absolute_baseline[:, i].mean()) for i, name in enumerate(LINEUPS)},
        primary_mixture_minus_480=comparisons.pop('mixture'), secondary_minus_480=comparisons,
        different_most_likely_root_actions=switches,
        estimand='Conditional mean utility difference from replacing only the root action distribution '
                 'at the first reached hero flop decision; hero follows 480 after the forced root. '
                 'The mixture equally weights the four frozen root policies.',
        uncertainty='Action branches and repeated continuations share their original deal. Their payoffs '
                    'are averaged inside that deal. Fixed policy seeds receive equal '
                    'weight. Paired normal intervals cluster by original deal. Five primary contrasts have '
                    'simultaneous 95% Bonferroni intervals. Each secondary policy has its own five-contrast '
                    'family; those intervals do not jointly cover primary and secondary results.',
        descriptive_signs='Signs use a 1e-10 BB numerical tie tolerance. A per-state sign is a noisy '
                          'estimate from finite continuations; sign counts are descriptive.')


def json_lines(paths):
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt') as stream:
            for line in stream:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--states', nargs='+', type=Path, required=True)
    parser.add_argument('--rollouts', nargs='+', type=Path, required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = score_flop_diagnostic(json_lines(args.states), json_lines(args.rollouts))
    inputs = [p.resolve() for p in (*args.states, *args.rollouts)]
    parent = Path(commonpath([p.parent for p in inputs]))
    result['input_sha256'] = {str(p.relative_to(parent)): digest(p) for p in inputs}
    if len(result['input_sha256']) != len(inputs):
        raise ValueError('Input path supplied more than once')
    atomic_json(args.output, result)


if __name__ == '__main__':
    main()
