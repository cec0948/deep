"""Describe the four preassigned postflop snapshots on already evaluated public hands."""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import random
from statistics import NormalDist

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import hand_seed
from fullhouse_v3_common import ROOT, atomic_json, digest


STUDY = ROOT / 'artifacts/preflop-followup/480-v1/postflop-averaging-20261120'
REGISTRATION = '47d64d029bbbad2ee115dd64f235237145d24e5ebfed0522d12b59d80b8aa370'
POPULATION = STUDY / 'population'
FOLDER = STUDY / 'snapshot-path'
REPORT = ROOT / 'reports/preflop-followup/480-v1/postflop-averaging-20261120/snapshot-path'
POLICIES = ('specialist480', 'mixture504', 'slow504')


def ratio_by_card(gains, counts):
    """Ratio of totals and its card-clustered, linearized standard error."""
    sample = float(counts.sum())
    if sample <= 0 or len(gains) < 2 or gains.shape != counts.shape:
        raise ValueError('Invalid snapshot coverage')
    mean = float(gains.sum() / sample)
    errors = gains - mean * counts
    se = float(errors.std(ddof=1) / (np.sqrt(len(gains)) * counts.mean()))
    return mean, se


def summarize():
    design = read(FOLDER / 'design.json')
    verify_files({FOLDER/'design.json': REGISTRATION, STUDY/'plan.json': design['training_plan_sha256'],
                  POPULATION/'design.json': design['population_design_sha256']})
    plan_file = POPULATION/'evaluation-plan.json'
    plan_sha = read(POPULATION/'bound.json')['evaluation_plan_sha256']
    verify_files({plan_file: plan_sha})
    plan = read(plan_file)
    n, start, captures = design['deals'], design['deal_seed'], design['captures']
    seeds = design['seeds']
    if (plan['fields']['public']['deal_seed'], plan['fields']['public']['deals'], plan['seeds']) != (start, n, seeds):
        raise ValueError('Design and evaluation plan disagree')
    counts = np.zeros((len(captures), n), dtype=np.int64)
    base = np.zeros_like(counts)
    mixture = np.zeros_like(counts)
    result_hashes = {}
    matched_endpoint = 0
    for seed in seeds:
        mixture_path = Path(plan['policies']['mixture504'][str(seed)]['path'])
        endpoint_path = Path(plan['policies']['slow504'][str(seed)]['path'])
        verify_files({mixture_path: plan['policies']['mixture504'][str(seed)]['sha256'],
                      endpoint_path: plan['policies']['slow504'][str(seed)]['sha256']})
        mix_spec, end_spec = read(mixture_path), read(endpoint_path)
        last = mix_spec['postflop'][-1]
        if last['iteration'] != 504 or last['sha256'] != end_spec['postflop']['sha256']:
            raise ValueError('Final snapshot and endpoint have different model identities')
        verify_files({mixture_path.parent/last['file']: last['sha256'],
                      end_spec['postflop']['path']: last['sha256']})
        with ExitStack() as stack:
            streams, results = {}, {}
            for label in POLICIES:
                key = f'public-seed-{seed}-{label}'
                folder = POPULATION/'evaluation'/key
                raw = folder/'hands.jsonl.gz'
                result = read(folder/'results.json')
                if (result['evaluation_plan_sha256'] != plan_sha or result['field'] != 'public'
                    or result['label'] != label or result['training_seed'] != seed
                    or result['deal_seed'] != start or result['deals'] != n or result['hands'] != n*6
                    or result['candidate_sha256'] != plan['policies'][label][str(seed)]['sha256']
                    or not result['complete'] or not result['terminal_payoffs_replayed']):
                    raise ValueError('Unverified or unmatched public cell')
                verify_files({raw: result['raw_hands_sha256'], folder/'provenance.json': result['provenance_sha256']})
                streams[label] = stack.enter_context(gzip.open(raw, 'rt'))
                results[label] = result
                result_hashes[key] = digest(folder/'results.json')
            for deal in range(n):
                sums = {label: 0 for label in POLICIES}
                for hero in range(6):
                    hseed = hand_seed(start+deal, hero)
                    idx = random.Random(b'scheduled-postflop-choice-v1\0' + hseed.to_bytes(8, 'big')).randrange(4)
                    old, mix, endpoint = [json.loads(next(streams[label])) for label in POLICIES]
                    for label, row in zip(POLICIES, (old, mix, endpoint)):
                        if ((row['deal'], row['hero'], row['seed'], row['hand_seed']) !=
                            (deal, hero, start+deal, hseed) or row['chips'] != row['utilities'][hero]
                            or sum(row['utilities']) != 0):
                            raise ValueError('Invalid terminal hand matching')
                        sums[label] += row['chips']
                    if mix.pop('snapshot_iteration') != captures[idx]:
                        raise ValueError('Frozen private draw disagrees with the ledger')
                    if idx == 3:
                        if mix != endpoint:
                            raise ValueError('Final snapshot actions or payoffs differ from the endpoint')
                        matched_endpoint += 1
                    counts[idx, deal] += 1
                    base[idx, deal] += old['chips']
                    mixture[idx, deal] += mix['chips']
                for label, total in sums.items():
                    if not np.isclose(total / 6, results[label]['deal_blocks'][deal], rtol=0, atol=1e-8):
                        raise ValueError('Raw payoffs disagree with verified card-block result')
            if any(stream.readline() for stream in streams.values()):
                raise ValueError('Unscheduled extra hands')
    if not np.all(counts.sum(axis=0) == 6 * len(seeds)) or np.any(counts % len(seeds)):
        raise ValueError('Shared card/seat assignments differ across training seeds')
    z = NormalDist().inv_cdf(1 - .05/(2*len(captures)))
    rows = {}
    for idx, capture in enumerate(captures):
        weight = counts[idx]
        hands = int(weight.sum())
        mean, se = ratio_by_card(mixture[idx] - base[idx], weight)
        rows[str(capture)] = dict(assigned_hands_across_seeds=hands, assigned_card_seats=hands//len(seeds),
            card_blocks_with_assignment=int(np.count_nonzero(weight)), matched_base_bb100=float(base[idx].sum()/hands),
            snapshot_bb100=float(mixture[idx].sum()/hands), paired_change_bb100=mean,
            card_cluster_se=se, simultaneous_95=[mean-z*se, mean+z*se])
    pooled = float((mixture-base).sum() / counts.sum())
    reconstruction = sum(row['assigned_hands_across_seeds']*row['paired_change_bb100'] for row in rows.values()) / counts.sum()
    if not np.isclose(pooled, reconstruction, rtol=0, atol=1e-10):
        raise ValueError('Snapshot contributions do not reconstruct the full paired result')
    return dict(verified_utc=utc(), design_sha256=REGISTRATION, population_plan_sha256=plan_sha, script_sha256=digest(__file__),
                public_only=True, seeds=seeds, original_card_blocks=n, extra_card_blocks=0, extra_hands=0,
                endpoint_traces_and_payoffs_identical=matched_endpoint, primary_mixture_change_reconstructed=pooled,
                results=rows, cell_results_sha256=result_hashes,
                scope='Secondary matched subset description on the same public cards; four snapshot intervals jointly 95% by normal approximation; no policy selection.')


if __name__ == '__main__':
    report = summarize()
    atomic_json(FOLDER/'summary.json', report)
    atomic_json(REPORT/'summary.json', report)
    print(json.dumps(report['results'], indent=2))
