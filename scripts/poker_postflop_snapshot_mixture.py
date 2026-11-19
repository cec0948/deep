"""A frozen two-snapshot postflop blueprint, with a single private choice per hand."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from math import sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from statistics import NormalDist

import numpy as np

from controlled_poker_refit import read, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import FORMAT as STREET_FORMAT, evaluator_policy


FORMAT = 'fixed-preflop-two-postflop-snapshots-v1'
STUDY = ROOT / 'artifacts/preflop-followup/480-v1/fixed-preflop-504'
PARENT_HASH = '6cc8d7759a091ce14ed417088e82917a9fc05792e20d51d2456ca8305df08d9c'
ENDPOINTS = ('specialist480', 'fixed504')
SEEDS = (342, 343, 344)
FIELDS = ('public', 'mixed', 'noregrets')


def source_records():
    verify_files({STUDY/'summary.json': PARENT_HASH})
    summary = read(STUDY/'summary.json')
    plan = STUDY/'evaluation-plan.json'
    verify_files({plan: summary['evaluation_plan_sha256']})
    return summary, read(plan)


def write_manifest(path, seed):
    if seed not in SEEDS:
        raise ValueError('Unknown frozen training seed')
    summary, plan = source_records()
    refs = [plan['policies'][label][str(seed)] for label in ENDPOINTS]
    verify_files({ref['path']: ref['sha256'] for ref in refs})
    policies = [read(ref['path']) for ref in refs]
    if (not summary['entire_preflop_frozen'] or any(p['format'] != STREET_FORMAT for p in policies)
            or policies[0]['preflop']['sha256'] != policies[1]['preflop']['sha256']
            or policies[0]['postflop']['sha256'] == policies[1]['postflop']['sha256']):
        raise ValueError('Expected identical preflop weights and distinct postflop weights')
    atomic_json(path, dict(format=FORMAT, training_seed=seed, source_summary_sha256=PARENT_HASH,
                           preflop=policies[0]['preflop'], postflop=[p['postflop'] for p in policies]))
    return path


def snapshot_index(hand_seed):
    if type(hand_seed) is not int or not 0 <= hand_seed < 2**64:
        raise ValueError('Expected a separate 64-bit random hand seed')
    # Only this separate stream selects a snapshot. Later cards and actions
    # cannot change the selection or reveal the private draw to the opponents.
    key = b'postflop-endpoint-choice-v1\0' + hand_seed.to_bytes(8, 'big')
    return hashlib.sha256(key).digest()[0] & 1


class PostflopSnapshotMixture:
    def __init__(self, model, source):
        self.model = Path(model).resolve()
        spec = read(self.model)
        self.model_type = FORMAT
        if (spec.get('format') != FORMAT or spec.get('training_seed') not in SEEDS
                or spec.get('source_summary_sha256') != PARENT_HASH or len(spec['postflop']) != 2):
            raise ValueError('Invalid endpoint mixture manifest')
        # A manifest claim alone cannot establish which two snapshots it contains.
        _, plan = source_records()
        sources = [plan['policies'][label][str(spec['training_seed'])] for label in ENDPOINTS]
        verify_files({p['path']: p['sha256'] for p in sources})
        endpoints = [read(p['path']) for p in sources]
        if (spec['preflop'] != endpoints[0]['preflop']
                or spec['preflop']['sha256'] != endpoints[1]['preflop']['sha256']
                or spec['postflop'] != [p['postflop'] for p in endpoints]):
            raise ValueError('Snapshot identities differ from the frozen campaign')
        refs = [spec['preflop'], *spec['postflop']]
        self.dependency_hashes = {p['path']: p['sha256'] for p in [*sources, *refs]}
        self.dependency_hashes[str(Path(__file__).resolve())] = digest(__file__)
        verify_files(self.dependency_hashes)
        self.preflop, *self.postflop = [evaluator_policy(ref['path'], source) for ref in refs]
        if any(p.model_type != 'compact_advantage' for p in [self.preflop, *self.postflop]):
            raise ValueError('Unexpected frozen policy format')
        for p in [self.preflop, *self.postflop]:
            self.dependency_hashes.update(p.dependency_hashes)
        self.active = None

    def begin_hand(self, hand_seed):
        self.active = snapshot_index(hand_seed)

    def decide(self, state, seed):
        if self.active is None:
            raise ValueError('Call begin_hand before making any decisions')
        street = state['street']
        if street not in ('preflop', 'flop', 'turn', 'river'):
            raise ValueError('Invalid betting street')
        policy = self.preflop if street == 'preflop' else self.postflop[self.active]
        return policy.decide(state, seed)


def retrospective():
    """Integrate the private hand coin over already evaluated endpoint trajectories."""
    summary, plan = source_records()
    n = plan['deals']
    simultaneous_z = NormalDist().inv_cdf(1 - .05 / (2*len(FIELDS)))
    fields = {}
    for field in FIELDS:
        endpoints = []
        for label in ENDPOINTS:
            seed_blocks = []
            for seed in SEEDS:
                cell = f'{field}-seed-{seed}-{label}'
                path = STUDY/'evaluation'/cell/'results.json'
                verify_files({path: summary['result_sha256'][cell]})
                row = read(path)
                if (row['field'] != field or row['deal_seed'] != plan['card_start'] or row['deals'] != n
                        or row['hands'] != 6*n or not row['terminal_payoffs_replayed']
                        or row['candidate_sha256'] != plan['policies'][label][str(seed)]['sha256']):
                    raise ValueError('Unmatched or unaudited evaluation block')
                blocks = np.asarray(row['deal_blocks'], dtype=float)
                if blocks.shape != (n,) or not np.isfinite(blocks).all():
                    raise ValueError('Missing or invalid card-block payoffs')
                seed_blocks.append(blocks)
            endpoints.append(np.mean(seed_blocks, axis=0))
        mixture = (endpoints[0] + endpoints[1])/2
        delta = mixture - endpoints[0]
        mean = float(delta.mean())
        half = float(simultaneous_z * delta.std(ddof=1)/sqrt(n))
        fields[field] = dict(mixture_bb100=float(mixture.mean()), change_vs_480_bb100=mean,
                             change_simultaneous_95=[mean-half, mean+half])
    return dict(endpoint_weights={'480': .5, '504': .5}, independent_card_blocks=n,
                first_card=plan['card_start'], last_card=plan['card_start']+n-1,
                reused_terminal_hands=n*6*len(SEEDS)*len(FIELDS)*len(ENDPOINTS), new_hands=0,
                new_fitting_steps=0, new_selfplay_rounds=0, source_summary_sha256=PARENT_HASH,
                scope='Retrospective on the fixed 504 study opponents, marginal over the private hand coin',
                fields=fields)


def verify_runtime():
    """Replay a balanced prefix of the original paired deals through the mixture runtime."""
    import evaluate_average_baseline as evaluator

    summary, plan = source_records()
    per_cell = 8
    selected = {label: 0 for label in ENDPOINTS}
    with TemporaryDirectory(prefix='poker-snapshot-mixture-') as work:
        work = Path(work)
        previous = evaluator.policy_for
        try:
            evaluator.policy_for = PostflopSnapshotMixture
            for seed in SEEDS:
                spec = write_manifest(work/f'seed-{seed}.json', seed)
                for field in FIELDS:
                    output = work/f'{field}-{seed}'
                    result = evaluator.run(spec, plan['source'], field, per_cell, plan['card_start'], 120, output)
                    if not result['terminal_payoffs_replayed']:
                        raise ValueError('Runtime evaluation did not replay all concrete terminal payoffs')
                    with ExitStack() as stack:
                        streams = []
                        for label in ENDPOINTS:
                            name = f'{field}-seed-{seed}-{label}'
                            cell = STUDY/'evaluation'/name
                            verify_files({cell/'results.json': summary['result_sha256'][name]})
                            raw = cell/'hands.jsonl.gz'
                            verify_files({raw: read(cell/'results.json')['raw_hands_sha256']})
                            streams.append(stack.enter_context(gzip.open(raw, 'rt')))
                        check = stack.enter_context(gzip.open(output/'hands.jsonl.gz', 'rt'))
                        observed = 0
                        for line in check:
                            observed += 1
                            row = json.loads(line)
                            options = [json.loads(next(stream)) for stream in streams]
                            # The replayed episode fixes cards, all action draws and seat;
                            # only the independent private coin chooses its endpoint.
                            choice = snapshot_index(row['hand_seed'])
                            if row != options[choice]:
                                raise ValueError('Mixed-policy trace differs from the frozen endpoint replay')
                            selected[ENDPOINTS[choice]] += 1
                        if observed != per_cell*6:
                            raise ValueError('Incomplete mixture trace verification')
        finally:
            evaluator.policy_for = previous
    if min(selected.values()) == 0:
        raise ValueError('Trace check did not exercise both snapshots')
    return dict(card_blocks=per_cell, seeds=SEEDS, fields=FIELDS, reverified_hands=sum(selected.values()),
                selected_endpoint_counts=selected, full_action_traces_and_terminal_payoffs_matched=True,
                independent_new_strength_samples=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    make = modes.add_parser('manifest')
    make.add_argument('seed', choices=SEEDS, type=int)
    make.add_argument('output', type=Path)
    modes.add_parser('retrospective')
    modes.add_parser('verify')
    args = parser.parse_args()
    if args.mode == 'manifest':
        write_manifest(args.output, args.seed)
    else:
        result = verify_runtime() if args.mode == 'verify' else retrospective()
        print(json.dumps(result, indent=2, allow_nan=False))
