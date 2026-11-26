"""Independent state, update-chain and auxiliary-policy checks for the 504→528 run."""

import argparse
import hashlib
import json
import sys

import numpy as np
import torch
from audit_poker_opponent_diversity_training import reference_assignments
from audit_poker_population import check
from audit_postflop_averaging import checked_ref, net_fingerprint
from compact_lightweight_backup import equal_state
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from run_poker_preflop_online import StreetNetworks, install_street_training_policy
from run_poker_preflop_pilot import CAMPAIGN, SOURCE
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import collect_game_reference, load_checkpoint
from train_poker_fixed_preflop_window import restore_replay
from train_poker_warm_preflop import weights

BASE = ROOT / 'artifacts/preflop-followup/480-v1'
PARENT = BASE / 'opponent-diversity-20261123'
STUDY = BASE / 'pool-extension-528-20261126'
REPORT = ROOT / 'reports' / STUDY.relative_to(ROOT / 'artifacts')
PARENT_STAGE = '8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9'
SEEDS = (342, 343, 344)
CAPTURES = (510, 516, 522, 528)
HISTORY = (300, 360, 420)


def source_hashes():
    """Bind both preserved arms and every source array or policy used by the successor."""
    stage = read(PARENT / 'stage-ready.json')
    plan = read(PARENT / 'training-plan.json')
    verify_files({PARENT / 'stage-ready.json': PARENT_STAGE})
    paths = {
        PARENT / 'stage-ready.json',
        PARENT / 'training-plan.json',
        PARENT / 'training-audit.json',
        CAMPAIGN.parent / 'stage-ready.json',
    }
    for seed in SEEDS:
        sid = str(seed)
        for root in (PARENT / 'pool504', CAMPAIGN):
            folder = root / f'seed-{seed}'
            ptr = read(folder / 'current.json')
            checkpoint = folder / 'checkpoints' / ptr['directory']
            refs = {checkpoint / name: sha for name, sha in ptr['files'].items()}
            verify_files(refs)
            paths.update([folder / 'current.json', *refs])
        for name, entries in stage['policies'].items():
            ref = checked_ref(entries[sid])
            paths.add(ref)
            spec = read(ref)
            if name == 'poolMixture504':
                paths.add(ref.parent / 'plan.json')
                refs = [spec['preflop'], *spec['postflop']]
                for row in refs:
                    p = ref.parent / row['file']
                    verify_files({p: row['sha256']})
                    paths.add(p)
            else:
                paths.update(checked_ref(spec[street]) for street in ('preflop', 'postflop'))
        paths.update(checked_ref(ref) for ref in plan['historical_policies'][sid].values())
        paths.add(PARENT / 'pool504' / f'seed-{seed}' / 'diagnostics.jsonl')
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def snapshot_source():
    hashes = source_hashes()
    path = STUDY / 'source-state-audit.json'
    if path.is_file():
        check(read(path)['files'] == hashes, 'Source files changed since their pre-run audit')
        return
    atomic_json(path, {'utc': utc(), 'source_stage_sha256': PARENT_STAGE, 'files': hashes})


def first_update(seed):
    """Run the original November 23 update kernel once on the retained full state in memory."""
    sys.path[:0] = [str(SOURCE), str(CAMPAIGN / 'tools')]
    from bot.features import load_preflop_equity
    from compact_regret import ARRAY_MAP, CompactRegretNet, install_policy
    from poker_opponent_population import load_historical, pool_update
    from training import r1_support as r1

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    state, checkpoint = load_checkpoint(PARENT / 'pool504' / f'seed-{seed}')
    cfg = state['config']
    spec = read(checked_ref(read(PARENT / 'stage-ready.json')['policies']['pool504'][str(seed)]))
    general, preflop = CompactRegretNet().eval(), CompactRegretNet().eval()
    for net, key, model in (
        (general, 'net', checkpoint / 'current.npz'),
        (preflop, 'specialist_net', checked_ref(spec['preflop'])),
    ):
        net.load_state_dict(state[key])
        verify_export(model, net.state_dict(), ARRAY_MAP)
    net = StreetNetworks(general, preflop)
    install_policy()
    install_street_training_policy()
    hist = load_historical(net, cfg['historical_policies'])
    equity = load_preflop_equity()
    buffers, root, strategy, rng, ref_rng = restore_replay(state)
    check(
        equal_state(state['rng'], r1.pack_rng(rng))
        and equal_state(state['reference_rng'], ref_rng.bit_generator.state),
        'Could not restore original RNG immediately before the 505 update',
    )
    check(
        set(state['rng']) == {'python', 'numpy', 'generator', 'torch'}
        and equal_state(strategy.state_dict(), state['strategy_replay'])
        and equal_state(root.state_dict(), state['root_replay']),
        'Incomplete state restore',
    )
    r1.WEIGHT_HORIZON, r1.AVERAGE_SAMPLER = cfg['weight_horizon'], 'importance'
    connection = weights(net)
    ref = collect_game_reference(net, strategy, ref_rng, 505, cfg['reference_hands'], equity)
    check(equal_state(state['rng'], r1.pack_rng(rng)), 'Reference sampling consumed training RNG')
    row = pool_update(net, buffers, root, rng, 505, cfg, equity, seed, hist)
    row.update(reference_collection=ref, input_weights=connection, output_weights=weights(net))
    return row


def state_counters(state):
    return {
        'regret_rows_by_seat': [int(buf['count']) for buf in state['buffers']],
        'regret_seen_by_seat': [int(buf['total_seen']) for buf in state['buffers']],
        'root_seen': int(state['root_replay']['count'].sum()),
        'strategy_rows': int(state['strategy_replay']['size']),
        'strategy_seen': int(state['strategy_replay']['total_seen']),
    }


def average_recheck(state, checkpoint):
    """Recompute TV/KL from independently loaded export and exact retained weighted labels."""
    from poker_strategy_average import AverageNet

    replay = state['strategy_replay']
    n = replay['size']
    x, inv = np.unique(replay['x'][:n], axis=0, return_inverse=True)
    mass = np.bincount(inv, weights=replay['mass'][:n])
    y = np.column_stack(
        [np.bincount(inv, weights=replay['y'][:n, a] * replay['mass'][:n]) / mass for a in range(5)]
    )
    with np.load(checkpoint / 'average.npz', allow_pickle=False) as data:
        params = {k: torch.from_numpy(data[k].copy()) for k in state['average_state']}
        check(equal_state(params, state['average_state']), 'Auxiliary inference export differs from live network')
        meta = json.loads(str(data['metadata_json']))
    check(
        meta['averaging_window'] == [361, 528]
        and not meta['historical_mixture_available']
        and state['averaging_window'] == [361, 528],
        'Auxiliary average has wrong history',
    )
    model = AverageNet(params['body.0.weight'].shape[0])
    model.load_state_dict(params)
    with torch.inference_mode():
        probs = np.vstack(
            [model(torch.from_numpy(chunk)).exp().numpy() for chunk in np.array_split(x, 16)]
        ).astype(float)
    tv = float(np.average(np.sum(np.abs(probs - y), axis=1) / 2, weights=mass))
    kl = float(
        np.average(np.sum(y * np.log(np.maximum(y, 1e-30) / np.maximum(probs, 1e-30)), axis=1), weights=mass)
    )
    fit = state['average_fit']
    check(
        fit['steps'] == 8192
        and fit['batch_size'] == 512
        and fit['rows'] == n
        and fit['distinct_inputs'] == len(x)
        and fit['total_seen'] == replay['total_seen']
        and abs(fit['weighted_replay_tv'] - tv) < 1e-6
        and abs(fit['weighted_replay_kl'] - kl) < 1e-6
        and fit['within_original_quality_reference'] == (tv <= 0.15),
        'Auxiliary weighted approximation metrics disagree',
    )
    return {
        'weighted_replay_tv': tv,
        'weighted_replay_kl': kl,
        'original_reference_tv': 0.15,
        'meets_original_tv_reference': tv <= 0.15,
        'window': [361, 528],
        'distinct_inputs': len(x),
        'retained_rows': n,
        'auxiliary_training_steps': 8192,
    }


def still_retained(old, new):
    """An occupied old reservoir row survives at its original index or is replaced by a new row."""
    kept = []
    for left, right in zip(old['buffers'], new['buffers'], strict=True):
        keep = right['iterations'] <= 504
        check(
            left['count'] == right['count'] == 100000
            and keep.any()
            and (~keep).any()
            and all(
                np.array_equal(left[k][keep], right[k][keep]) for k in ('features', 'advantages', 'iterations')
            ),
            'Old regret samples changed without a later collection iteration',
        )
        check(
            np.all((right['iterations'][~keep] >= 505) & (right['iterations'][~keep] <= 528)),
            'A replaced sample has an out-of-window round',
        )
        kept.append(int(keep.sum()))
    old_strategy, new_strategy = (s['strategy_replay'] for s in (old, new))
    equal = np.ones(old_strategy['size'], dtype=bool)
    for field in ('x', 'y', 'mass'):
        same = old_strategy[field] == new_strategy[field]
        equal &= same if same.ndim == 1 else same.all(axis=1)
    check(
        0 < equal.sum() < old_strategy['size'] == new_strategy['size'] == 100000
        and not equal_state(old_strategy['rng'], new_strategy['rng']),
        'Strategy reservoir did not evolve',
    )
    return {
        'parent_regret_rows_retained_byte_identical_by_seat': kept,
        'strategy_rows_still_byte_identical_at_same_index': int(equal.sum()),
    }


def verify_seed(seed, stage):
    sys.path[:0] = [str(SOURCE), str(CAMPAIGN / 'tools')]
    from compact_regret import ARRAY_MAP

    key = str(seed)
    worker = STUDY / 'pool528' / f'seed-{seed}'
    captured = STUDY / 'snapshots' / f'seed-{seed}'
    old, parent_ckpt = load_checkpoint(PARENT / 'pool504' / f'seed-{seed}')
    new, new_ckpt = load_checkpoint(worker)
    source = read(PARENT / 'stage-ready.json')
    start, ready, pointer = (
        read(worker / name) for name in ('start-verification.json', 'ready.json', 'current.json')
    )
    rows = [json.loads(line) for line in (worker / 'diagnostics.jsonl').read_text().splitlines()]
    check(
        old['seed'] == new['seed'] == seed
        and old['iteration'] == 504
        and new['iteration'] == 528
        and read(worker / 'status.json')['status'] == 'complete'
        and len(rows) == 24
        and [r['iteration'] for r in rows] == list(range(505, 529))
        and ready == stage['arms']['pool528'][key]
        and ready['files'] == pointer['files']
        and pointer['iteration'] == 528,
        'Seed, range, saved files or completion state mismatch',
    )
    verify_files(
        {
            **{new_ckpt / n: h for n, h in ready['files'].items()},
            worker / 'diagnostics.jsonl': ready['diagnostics_sha256'],
            worker / 'start-verification.json': ready['start_sha256'],
        }
    )
    eq_config = dict(
        old['config'],
        initial_iteration=504,
        iterations=528,
        stages=[528],
        training_arm='pool528',
        resume_iterations={str(s): 504 for s in SEEDS},
    )
    check(
        new['config'] == eq_config
        and old['source_manifest'] == new['source_manifest']
        and old['averaging_window'] == [361, 504]
        and new['averaging_window'] == [361, 528]
        and old['historical_mixture_available'] is False
        and new['historical_mixture_available'] is False,
        'Unexpected configuration drift or history claim',
    )
    check(
        start['checkpoint_files'] == source['arms']['pool504'][key]['files']
        and start['weights'] == source['seed_verification'][key]['final_weights']
        and start['exact_loaded_state']
        and start['restored_root_regret_strategy_rng_byte_verified']
        and set(start['restored_rng'])
        == {'python', 'numpy', 'generator', 'torch', 'reference_generator', 'strategy_reservoir'}
        and start['regret_seen'] == [b['total_seen'] for b in old['buffers']]
        and start['reference_seen'] == old['strategy_replay']['total_seen']
        and start['root_class_counts'] == old['root_replay']['count'].tolist(),
        'Worker connection record disagrees with the archived state',
    )
    source_policy = read(checked_ref(source['policies']['pool504'][key]))
    fixed_parent = read(checked_ref(source['policies']['specialist480'][key]))
    stage_policies = stage['policies']
    check(
        stage_policies['specialist480'][key] == source['policies']['specialist480'][key]
        and fixed_parent['preflop']['sha256'] == source_policy['preflop']['sha256'],
        'Canonical baseline changed',
    )
    single = read(checked_ref(stage_policies['current528'][key]))
    mixture_file = checked_ref(stage_policies['mixture528'][key])
    mixture = read(mixture_file)
    check(
        single['preflop']['sha256'] == source_policy['preflop']['sha256'] == mixture['preflop']['sha256']
        and checked_ref(single['postflop']) == new_ckpt / 'current.npz'
        and checked_ref(stage_policies['average528'][key]) == new_ckpt / 'average.npz'
        and digest(mixture_file.parent / mixture['preflop']['file']) == source_policy['preflop']['sha256']
        and [r['iteration'] for r in mixture['postflop']] == list(CAPTURES)
        and mixture['arm'] == 'pool528'
        and mixture['seed'] == seed
        and mixture['selection'] == 'Equal probability, one private draw per hand',
        'Wrong endpoint/mixed policy',
    )
    verify_export(checked_ref(single['preflop']), new['specialist_net'], ARRAY_MAP)
    verify_export(new_ckpt / 'current.npz', new['net'], ARRAY_MAP)
    check(
        equal_state(old['specialist_net'], new['specialist_net'])
        and all(equal_state(value, new['net'][k]) for k, value in old['net'].items() if k.startswith('opening')),
        'Frozen preflop or opening weights changed',
    )
    parent_weights = {
        'preflop': net_fingerprint(checked_ref(source_policy['preflop']), ARRAY_MAP),
        'general': net_fingerprint(parent_ckpt / 'current.npz', ARRAY_MAP),
    }
    check(start['weights'] == parent_weights, 'Starting network fingerprints disagree')
    expected_seen = [b['total_seen'] for b in old['buffers']]
    previous = parent_weights
    last_strategy_seen = old['strategy_replay']['total_seen']
    branch_visits = {str(code): 0 for code in (0, *HISTORY)}
    hashes = []
    for r in rows:
        t = r['iteration']
        traverser = (t - 1) % 6
        slots = reference_assignments(seed, t)
        pool = r['opponent_population']
        hist_counts = {k: int(np.count_nonzero(slots == int(k))) for k in branch_visits}
        ref = r['reference_collection']
        check(
            r['seed'] == seed
            and r['traverser'] == traverser
            and r['traversals'] == 256
            and r['input_weights'] == previous
            and r['output_weights']['general'] != previous['general']
            and r['output_weights']['preflop'] == parent_weights['preflop']
            and np.isfinite(r['main_loss'])
            and r['general_steps'] == 768
            and r['learning_rate'] == 0.00025
            and r['fit_draws'] == 768 * 512
            and sum(r['fit_draws_by_street']) == r['fit_draws']
            and min(r['fit_draws_by_street']) > 0
            and r['preflop_steps'] == r['opening_steps'] == 0
            and r['frozen_preflop_max_probability_error'] == 0.0
            and r['frozen_preflop_probes'] == start['frozen_preflop_probes']
            and r['buffer_rows'] == [100000] * 6
            and r['root_rows'] == 676
            and r['new_regret_rows'] > 0
            and ref['hands'] == 512
            and ref['rows'] == 100000
            and 0 < ref['total_seen'] - last_strategy_seen <= ref['decision_nodes']
            and hist_counts == pool['opponent_seat_assignments']
            and hashlib.sha256(slots.tobytes()).hexdigest() == pool['assignment_sha256']
            and min(pool['postflop_node_visits'].values()) > 0
            and (r['average_fit'] is None) == (t != 528),
            f'Update differs from frozen protocol at {seed}/{t}',
        )
        for k, n in hist_counts.items():
            branch_visits[k] += n
        expected_seen[traverser] += r['new_regret_rows']
        last_strategy_seen = ref['total_seen']
        previous = r['output_weights']
        if t in CAPTURES:
            cap = next(c for c in mixture['postflop'] if c['iteration'] == t)
            policy = captured / cap['file']
            receipt = read(policy.parent / 'receipt.json')
            check(
                policy.is_file()
                and digest(policy) == cap['sha256'] == receipt['export_sha256']
                and receipt['iteration'] == t
                and receipt['start_sha256'] == ready['start_sha256']
                and digest(captured / 'plan.json') == receipt['plan_sha256'] == mixture['plan_sha256']
                and net_fingerprint(policy, ARRAY_MAP) == previous['general'],
                'Snapshot weights differ from trained round',
            )
            hashes.append(cap['sha256'])
    check(
        len(set(hashes)) == 4
        and hashes[-1] == pointer['files']['current.npz']
        and expected_seen == [b['total_seen'] for b in new['buffers']]
        and last_strategy_seen == new['strategy_replay']['total_seen']
        and np.array_equal(old['root_replay']['count'] + 24 * 4, new['root_replay']['count'])
        and new['average_fit'] == rows[-1]['average_fit']
        and not equal_state(old['reference_rng'], new['reference_rng'])
        and not equal_state(old['rng']['generator'], new['rng']['generator'])
        and previous == stage['seed_verification'][key]['final_weights'],
        'Final state mismatches actual updates',
    )
    reproduced = read(STUDY / f'replay-505-{seed}.json')
    check(
        all(rows[0][k] == v for k, v in reproduced['expected'].items())
        and reproduced['parent_stage_sha256'] == PARENT_STAGE,
        'First update did not reproduce',
    )
    return dict(
        seed=seed,
        parent_checkpoint_sha256=source['arms']['pool504'][key]['files'],
        checkpoint_sha256=pointer['files'],
        diagnostics_sha256=ready['diagnostics_sha256'],
        first_update_bit_reproduced=True,
        first_update_sha256=digest(STUDY / f'replay-505-{seed}.json'),
        confirmed_rounds=[505, 528],
        rounds=len(rows),
        traversals=256 * len(rows),
        fit_updates=768 * len(rows),
        fit_draws=768 * 512 * len(rows),
        new_regret_rows=sum(r['new_regret_rows'] for r in rows),
        preflop_unchanged=True,
        new_snapshots=dict(zip(CAPTURES, hashes, strict=True)),
        history_seat_assignments=branch_visits,
        **still_retained(old, new),
        source_counts=state_counters(old),
        final_counts=state_counters(new),
        auxiliary_average=average_recheck(new, new_ckpt),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('source', 'first', 'final'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == 'source':
        snapshot_source()
        return
    torch.set_num_threads(1)
    if args.action == 'first':
        check(args.seed is not None, 'Specify one training seed')
        check(
            read(STUDY / 'stage-ready.json')['training_complete'], 'Replay audit runs after training completion'
        )
        worker = STUDY / 'pool528' / f'seed-{args.seed}'
        rows = (worker / 'diagnostics.jsonl').read_text().splitlines()
        first = json.loads(rows[0])
        expected = first_update(args.seed)
        differences = [field for field, value in expected.items() if first[field] != value]
        check(
            first['iteration'] == 505 and not differences,
            f'Replayed first update differs at {args.seed}/505: {", ".join(differences)}',
        )
        atomic_json(
            STUDY / f'replay-505-{args.seed}.json',
            {
                'seed': args.seed,
                'round': 505,
                'expected': expected,
                'parent_stage_sha256': PARENT_STAGE,
                'no_retained_model': True,
            },
        )
        return
    check((STUDY / 'source-state-audit.json').is_file(), 'Pre-run source receipt missing')
    snapshot_source()
    stage = read(STUDY / 'stage-ready.json')
    plan = read(STUDY / 'training-plan.json')
    sources = read(STUDY / 'source-state-audit.json')
    verify_files(plan['frozen_files'])
    check(
        stage['iteration'] == 528
        and stage['training_complete']
        and stage['parent_stage_sha256'] == PARENT_STAGE
        and dict(plan['budget'])
        == {
            'learning_rate': 0.00025,
            'traversals': 256,
            'general_steps': 768,
            'main_steps': 768,
            'batch_size': 512,
            'root_repetitions': 4,
            'reference_hands': 512,
            'buffer_capacity': 100000,
            'strategy_capacity': 100000,
            'average_fit_steps': 8192,
            'average_batch': 512,
            'average_width': 128,
            'opening_steps': 0,
            'preflop_steps': 0,
        },
        'Wrong parent or planned training budget',
    )
    observed = read(STUDY / 'training-process-audit.json')
    superv, *workers = observed['matches']
    launch = read(STUDY / 'training-launch.json')
    check(
        launch['utc'] < observed['observed_utc'] < stage['completed_utc']
        and launch['pid'] == superv['pid']
        and str(launch['process_start']) == superv['start_token']
        and superv['argv'][1].endswith('run_poker_pool528.py')
        and superv['argv'][2:] == ['run']
        and [worker['seed'] for worker in workers] == list(SEEDS),
        'Observed training process identities or times do not match the launch',
    )
    for worker in workers:
        seed = worker['seed']
        status = read(STUDY / 'pool528' / f'seed-{seed}' / 'status.json')
        check(
            worker['pid'] == status['pid']
            and worker['ppid'] == superv['pid']
            and worker['start_token'] == str(status['process_start'])
            and worker['proc_state'] not in ('Z', 'X')
            and worker['argv'][1].endswith('run_poker_pool528.py')
            and worker['argv'][2:] == ['worker', str(seed)],
            'Observed worker command or process token does not match its final record',
        )
    seeds = {str(s): verify_seed(s, stage) for s in SEEDS}
    result = {
        'verified_utc': utc(),
        'training_complete': True,
        'stage_sha256': digest(STUDY / 'stage-ready.json'),
        'training_plan_sha256': digest(STUDY / 'training-plan.json'),
        'immutable_source_files': len(sources['files']),
        'source_audit_sha256': digest(STUDY / 'source-state-audit.json'),
        'observed_training_processes_sha256': digest(STUDY / 'training-process-audit.json'),
        'source_manifest_precedes_training': sources['utc'] < read(STUDY / 'training-launch.json')['utc'],
        'registered_rounds_by_seed': 24,
        'actual_round_updates': sum(v['rounds'] for v in seeds.values()),
        'online_traversals': sum(v['traversals'] for v in seeds.values()),
        'gradient_updates': sum(v['fit_updates'] for v in seeds.values()),
        'fitting_row_draws': sum(v['fit_draws'] for v in seeds.values()),
        'new_regret_rows': sum(v['new_regret_rows'] for v in seeds.values()),
        'seeds': seeds,
    }
    check(result['source_manifest_precedes_training'], 'Source snapshot not taken before training launch')
    atomic_json(STUDY / 'training-audit.json', result)
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / 'training-audit.json', result)


if __name__ == '__main__':
    main()
