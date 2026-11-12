"""Read-only information-loss and frozen-replay coverage audit.

Synthetic witnesses are accepted by the trusted game engine. A feature collision
proves lost observable information, not that the optimal actions differ or that
the collision caused an observed evaluation loss.
"""
import argparse
import datetime
import gc
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

from fullhouse_v3_common import ROOT, atomic_json, digest
from regret_partition import base_strata, is_opening


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def suit_isomorphic(hole_a, board_a, hole_b, board_b):
    """Whether one global suit permutation maps both public observations."""
    for target in itertools.permutations('cdhs'):
        suits = dict(zip('cdhs', target))
        mapped = lambda cards: sorted(card[0] + suits[card[1]] for card in cards)
        if mapped(hole_a) == sorted(hole_b) and mapped(board_a) == sorted(board_b):
            return True
    return False


def witness_requests():
    first_raise = [dict(seat=3, kind='raise', to=200), dict(seat=4, kind='call'),
                   dict(seat=5, kind='fold'), dict(seat=0, kind='fold'),
                   dict(seat=1, kind='call'), dict(seat=2, kind='call')]
    limp_raise = [dict(seat=3, kind='call'), dict(seat=4, kind='raise', to=200),
                  dict(seat=5, kind='fold'), dict(seat=0, kind='fold'),
                  dict(seat=1, kind='call'), dict(seat=2, kind='call'),
                  dict(seat=3, kind='call')]
    def request(actions, board):
        return dict(op='replay', hero=1, button=0, hole=['As', 'Kd'], board=board,
                    actions=actions, seed=202611120917)
    return [dict(name='preflop_aggressor_and_action_sequence',
                 interpretation='Seat3 raises and seat4 calls versus seat3 limps, seat4 raises and seat3 calls; same flop material state.',
                 requests=[request(first_raise, ['2c', '7d', 'Th']),
                           request(limp_raise, ['2c', '7d', 'Th'])]),
            dict(name='board_rank_suit_association',
                 interpretation='Board suit placement changes while hero cards stay fixed; these observations are not related by one global suit relabeling.',
                 requests=[request(first_raise, ['2s', '7d', 'Th']),
                           request(first_raise, ['2h', '7d', 'Ts'])])]


def prove_witnesses(parent):
    from fullhouse_runtime import Bridge
    from fullhouse_v4_runtime import FullhouseV4
    cfg = json.loads((parent / 'config.json').read_text())
    policies = {seed: FullhouseV4(parent / f'seed-{seed}/current_00120.npz', parent / 'source')
                for seed in cfg['seeds']}
    encoder = policies[cfg['seeds'][0]]
    engine = Bridge(); results = []
    try:
        for case in witness_requests():
            states = [engine.call(request) for request in case['requests']]
            if any(s['terminal'] or s['seat_to_act'] != 1 or s['street'] != 'flop' for s in states):
                raise AssertionError('Engine did not reach the declared legal observation')
            encoded = [encoder.features.encode_state_dict(s, encoder.lookup._equity_tables) for s in states]
            x, y = [item[0] for item in encoded]
            masks_equal = np.array_equal(encoded[0][1], encoded[1][1])
            if not np.array_equal(x, y) or not masks_equal or json_digest(states[0]) == json_digest(states[1]):
                raise AssertionError('Declared distinct-state feature collision did not reproduce')
            policy_outputs = {}
            for seed, policy in policies.items():
                left, right = [policy._strategy(s) for s in states]
                if not np.array_equal(left, right) or not np.isclose(left.sum(), 1.) or (left < 0).any():
                    raise AssertionError('Same-input deterministic deployment parity failed')
                policy_outputs[str(seed)] = dict(probabilities=left.tolist(), byte_identical=True,
                    model_sha256=policy.digest)
            iso = suit_isomorphic(states[0]['your_cards'], states[0]['community_cards'],
                                  states[1]['your_cards'], states[1]['community_cards'])
            if case['name'] == 'board_rank_suit_association' and iso:
                raise AssertionError('Card witness is merely an irrelevant global suit permutation')
            results.append(dict(name=case['name'], interpretation=case['interpretation'],
                engine_accepted_both=True, terminal=False, hero=1, street='flop',
                requests=case['requests'], states=states,
                state_sha256=[json_digest(s) for s in states],
                feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                feature_dtype=str(x.dtype), feature_dimension=len(x),
                byte_identical_features=True, legal_masks_equal=masks_equal,
                legal_mask=encoded[0][1].tolist(), features=x.tolist(),
                same_card_observation=states[0]['your_cards'] == states[1]['your_cards'] and states[0]['community_cards'] == states[1]['community_cards'],
                card_observations_globally_suit_isomorphic=iso,
                policy_outputs=policy_outputs,
                demonstrated='Distinct engine-legal public observations map to one policy input.',
                not_demonstrated='Different optimal actions, frequency in real play, exploitability cost, or causality for the previous regression.'))
    finally:
        engine.close()
    return results


def validate_replay(x, y, weights):
    if x.ndim != 2 or x.shape[1] != 51 or y.shape != (len(x), 5) or weights.shape != (len(x),):
        raise ValueError('Unexpected replay shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Nonfinite replay or nonpositive iteration weight')
    if not np.isin(x[:, 17:27], [0., 1.]).all() or not np.all(x[:, 17:21].sum(1) == 1) or not np.all(x[:, 21:27].sum(1) == 1):
        raise ValueError('Malformed street or position one-hot')
    legal = x[:, 46:51]
    if not np.isin(legal, [0., 1.]).all() or not (legal.sum(1) >= 1).all():
        raise ValueError('Malformed legal action mask')


def summarize_replay(x, y, weights):
    """Summarize actual main replay, not unseen game states or visitation odds."""
    validate_replay(x, y, weights)
    mask = ~is_opening(x)
    x, y, weights = x[mask], y[mask], weights[mask].astype(float)
    groups = base_strata(x); streets = np.argmax(x[:, 17:21], axis=1)
    total_weight = float(weights.sum()); present = np.unique(groups); strata = {}
    for group in present:
        ids = groups == group; gx, gy, gw = x[ids], y[ids], weights[ids]
        keys, inverse, counts = np.unique(gx, axis=0, return_inverse=True, return_counts=True)
        mass = np.bincount(inverse, weights=gw)
        means = np.stack([np.bincount(inverse, weights=gw * gy[:, a]) / mass for a in range(5)], axis=1)
        residual = gy.astype(float) - means[inverse]
        target_second = float(np.average(np.square(gy.astype(float)).mean(1), weights=gw))
        within = float(np.average(np.square(residual).mean(1), weights=gw))
        share = float(gw.sum() / total_weight)
        strata[str(int(group))] = dict(rows=int(ids.sum()), unique_feature_vectors=len(keys),
            rows_in_repeated_feature_groups=int(counts[counts > 1].sum()),
            raw_iteration_weight_share=share, fitter_stratum_probability=1. / len(present),
            fitter_to_raw_weight_share_ratio=float((1. / len(present)) / share),
            target_rms=float(target_second ** .5), weighted_within_feature_target_variance=within,
            within_feature_variance_fraction_of_second_moment=within / target_second if target_second else None,
            iteration_min=float(gw.min()), iteration_max=float(gw.max()))
    return dict(main_rows=len(x), opening_rows_excluded=int((~mask).sum()),
        by_street={name: dict(rows=int((streets == s).sum()),
            row_share=float((streets == s).mean()),
            iteration_weight_share=float(weights[streets == s].sum() / total_weight))
            for s, name in enumerate(('preflop', 'flop', 'turn', 'river'))},
        by_stratum=strata, malformed_rows=0,
        validation_scope=['51-feature/5-target shapes', 'finite features and targets',
                          'positive finite iteration weights', 'street and position one-hot',
                          'binary nonempty legal-action mask'],
        illegal_action_target_nonzero_entries=int(np.count_nonzero(y * (1 - x[:, 46:51]))),
        limitations=['Replay is reservoir-retained training data, not an unbiased count of all game situations.',
                     'Repeated feature inputs can be identical information sets, sampling repetition, or lost information; this audit cannot distinguish them without full observation logs.',
                     'Within-feature regret-target variance also includes Monte Carlo chance and changing opponent policies; it is not proof of incorrect targets.',
                     'Equal stratum fitting mass changes the finite-network projection objective; these ratios do not establish that reweighting is harmful.'])


def audit(parent, output):
    import torch
    parent = Path(parent).resolve(); output = Path(output).resolve()
    if output == parent or parent in output.parents:
        raise ValueError('Audit output must be outside immutable training run')
    cfg = json.loads((parent / 'config.json').read_text())
    manifest = json.loads((parent / 'source-manifest.json').read_text())
    if cfg['iterations'] != 120 or cfg['seeds'] != [342, 343, 344]:
        raise ValueError('Expected declared three-seed 120-round parent')
    frozen = {str(parent / name): sha for name, sha in manifest['files'].items()}
    if digest(ROOT / 'scripts/regret_partition.py') != digest(parent / 'tools/regret_partition.py'):
        raise ValueError('Live partition helper differs from frozen parent')
    for name in ('fullhouse_runtime.py', 'fullhouse_v4_runtime.py', 'regret_partition.py'):
        path = ROOT / 'scripts' / name; frozen[str(path)] = digest(path)
    for seed in cfg['seeds']:
        for name in ('latest.pt', 'current_00120.npz'):
            path = parent / f'seed-{seed}' / name; frozen[str(path)] = digest(path)
    for name, sha in frozen.items():
        if digest(name) != sha:
            raise ValueError('Frozen artifact hash mismatch: ' + name)
    began = time.monotonic()
    report = dict(format='poker-information-audit-v1', complete=False,
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_run=parent.name, parent_iteration=120, source_manifest_sha256=digest(parent/'source-manifest.json'),
        script_sha256=digest(Path(__file__)),
        engine_sha256=digest(ROOT/'.cache/noregrets/target/release/poker-bridge'),
        features_sha256=digest(parent/'source/bot/features.py'),
        runtime_sha256=digest(ROOT/'scripts/fullhouse_v4_runtime.py'),
        partition_sha256=digest(ROOT/'scripts/regret_partition.py'),
        witnesses=prove_witnesses(parent), replay={},
        new_selfplay_samples=0, trained_models=False, parents_modified=False)
    for seed in cfg['seeds']:
        path = parent / f'seed-{seed}/latest.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck['config'] != cfg or ck['source_manifest'] != manifest or ck['iteration'] != 120 or ck['seed'] != seed:
            raise ValueError('Checkpoint provenance mismatch')
        records = ck['buffers']
        x = np.concatenate([b['features'] for b in records])
        y = np.concatenate([b['advantages'] for b in records])
        w = np.concatenate([b['iterations'] for b in records])
        if len(x) > 600000:
            raise ValueError('Audit replay budget exceeded')
        coverage = summarize_replay(x, y, w)
        coverage['buffers'] = [dict(seat=i, retained_rows=b['count'], total_seen=b['total_seen'],
            retention_fraction=b['count'] / b['total_seen'] if b['total_seen'] else None)
            for i, b in enumerate(records)]
        coverage['checkpoint_sha256'] = frozen[str(path)]
        report['replay'][str(seed)] = coverage
        del ck, records, x, y, w
        gc.collect()
        if time.monotonic() - began > 240:
            raise TimeoutError('Information audit exceeded four minute CPU budget')
    for name, sha in frozen.items():
        if digest(name) != sha:
            raise ValueError('Immutable training artifact changed during audit: ' + name)
    report.update(complete=True, seconds=time.monotonic()-began,
        frozen_artifacts_verified=len(frozen),
        findings=['Two reproducible engine-legal public-information collisions were demonstrated.',
                  'All three frozen current120 policies necessarily produce the same action probabilities within each collision pair.',
                  'Existing hand-level evaluation logs have no per-decision public state, feature or action distribution; specific loss attribution requires a separate diagnostic evaluation.'],
        limitations=['Synthetic witnesses establish a representational limit, not its incidence or performance cost.',
                     'No optimal-response solver or six-player exploitability calculation was performed.',
                     'No feature replacement, model promotion, retraining or continuation was performed by this audit.'])
    logs = sorted((parent / 'evaluation').rglob('hands.jsonl.gz'))
    if logs:
        with gzip.open(logs[0], 'rt') as stream:
            fields = sorted(json.loads(next(stream)))
        report['evaluation_log_inspection'] = dict(
            representative_log=str(logs[0].relative_to(ROOT)),
            sha256=digest(logs[0]), first_record_fields=fields,
            evaluator_sha256=digest(ROOT/'scripts/evaluate_compact.py'),
            per_decision_observations_recorded=bool(set(fields) & {'observations', 'decisions', 'features', 'states'}),
            scope='Representative original120 terminal-hand log and current evaluator row schema; no decision-level loss attribution attempted.')
    atomic_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.parent, args.output)
    print(json.dumps(dict(complete=result['complete'], witnesses=len(result['witnesses']),
        seeds=list(result['replay']), seconds=result['seconds'], output=str(args.output))))
