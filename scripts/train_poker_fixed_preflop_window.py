"""Play 24 new six-player rounds with the deployed 480 preflop policy fixed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import run_poker_preflop_confirmation as confirmation
from compact_lightweight_backup import equal_state
from controlled_poker_refit import assert_finite_arrays, read, utc, verify_files
from fullhouse_v3_common import atomic_json, digest, process_start
from poker_preflop_specialist import file_ref, write_policy
from poker_strategy_average import StrategyReplay, fit_average
from run_poker_preflop_pilot import CAMPAIGN, SEEDS
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import (
    CHECKPOINT_FORMAT, collect_game_reference, commit_checkpoint, load_checkpoint, record_average_quality,
)
from train_poker_warm_preflop import connect, weights


OUTPUT = confirmation.FOLLOWUP / 'fixed-preflop-504'
ARM, PARENT = 'fixed504', 'specialist480'
INITIAL, FINAL = 480, 504
GENERAL_STEPS, PREFLOP_STEPS = 768, 0


def restore_replay(parent):
    from root_regret_replay import RootReplay
    from training import r1_support as r1
    from training.buffer import ReservoirBuffer

    cfg = parent['config']
    buffers = [ReservoirBuffer(cfg['buffer_capacity'], 51) for _ in range(6)]
    for buf, state in zip(buffers, parent['buffers'], strict=True):
        buf.load_state_dict(state)
        if not equal_state(buf.state_dict(), state):
            raise ValueError('Restored regret reservoir differs')
    root = RootReplay()
    root.load_state_dict(parent['root_replay'])
    strategy = StrategyReplay.from_state(parent['strategy_replay'])
    rng, reference_rng = np.random.default_rng(), np.random.default_rng()
    reference_rng.bit_generator.state = parent['reference_rng']
    r1.restore_rng(parent['rng'], rng)
    if (not equal_state(root.state_dict(), parent['root_replay'])
            or not equal_state(strategy.state_dict(), parent['strategy_replay'])
            or not equal_state(r1.pack_rng(rng), parent['rng'])
            or not equal_state(reference_rng.bit_generator.state, parent['reference_rng'])):
        raise ValueError('Opening or strategy replay or RNG restore differs')
    return buffers, root, strategy, rng, reference_rng


def preflop_probes(buffers, root):
    """Include all conditional opening classes and retained later decisions at each seat."""
    from regret_partition import is_opening

    rows = [root.means()[0].copy()]
    if len(rows[0]) != 169 or not is_opening(rows[0]).all():
        raise ValueError('Unexpected parent opening features')
    for buf in buffers:
        x = buf.features[:len(buf)]
        available = np.flatnonzero((x[:, 17] == 1) & ~is_opening(x))
        if len(available) < 16:
            raise ValueError('Insufficient nonopening preflop probes')
        rows.append(x[available[np.linspace(0, len(available)-1, 16, dtype=int)]].copy())
    return np.concatenate(rows)


def played_probabilities(networks, features):
    from training.traverse_fast import NumpyPolicy

    policy = NumpyPolicy(networks)
    probabilities = np.stack([policy.strategy(x, x[46:51]) for x in features])
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-6, rtol=0)
            or (probabilities[features[:, 46:51] == 0] != 0).any()):
        raise ValueError('Invalid preflop probabilities')
    return probabilities


def postflop_sampler(buffers):
    from compact_regret import ReplayView
    from regret_partition import base_strata
    from scaled_regret_fitting import ScaledRegretSampler

    views = []
    counts = np.zeros(3, dtype=np.int64)
    for buf in buffers:
        x = buf.features[:len(buf)]
        streets = x[:, 17:21]
        if not np.isin(streets, (0., 1.)).all() or not (streets.sum(axis=1) == 1).all():
            raise ValueError('Invalid replay street encoding')
        mask = streets[:, 0] == 0
        counts += streets[mask, 1:].sum(axis=0, dtype=np.int64)
        views.append(ReplayView(x[mask], buf.advantages[:len(buf)][mask], buf.iterations[:len(buf)][mask]))
    sampler = ScaledRegretSampler(views, base_strata, 'stratum-scaled-feature-conditional')
    if (min(counts) < 1 or len(sampler.features) != counts.sum() or np.any(sampler.groups < 10)
            or np.any(sampler.groups >= 22) or np.any(sampler.features[:, 17] != 0)):
        raise ValueError('Postflop-only replay or feature stratum verification failed')
    return sampler, counts.tolist()


def update_postflop(networks, buffers, root, rng, iteration, cfg, equity):
    from regret_fitting import train_conditional
    from root_regret_replay import collect_root_replay
    from training.traverse_fast import run_traversals_fast

    seat = (iteration-1) % 6
    buf = buffers[seat]
    before = buf.total_seen
    run_traversals_fast(cfg['traversals'], 6, seat, networks, buf, iteration, rng, equity,
                        opponent_exploration=0., strategy_buffer=None, randomize_stacks=False)
    root_rows = collect_root_replay(networks, rng, root, iteration, cfg['root_repetitions'])
    if buf.total_seen <= before or root_rows != 169*cfg['root_repetitions']:
        raise ValueError('Missing new traversal or root observations')
    sampler, counts = postflop_sampler(buffers)
    model = networks.general
    optimizer = torch.optim.Adam(model.main.parameters(), lr=cfg['learning_rate'])
    loss = train_conditional(model.main, sampler, optimizer, GENERAL_STEPS, cfg['batch_size'])
    groups = np.unique(sampler.groups)
    model.main_scales[groups] = torch.from_numpy(sampler.scales[groups])
    model.eval()
    assert_finite_arrays(model)
    draws = [int(sampler.draws[sampler.features[:, i] == 1].sum()) for i in (18, 19, 20)]
    if not np.isfinite(loss) or sum(draws) != GENERAL_STEPS*cfg['batch_size'] or min(draws) <= 0:
        raise ValueError('Postflop fit budget or numeric checks failed')
    return dict(traverser=seat, traversals=cfg['traversals'], new_regret_rows=buf.total_seen-before,
                root_rows=root_rows, general_steps=GENERAL_STEPS, opening_steps=0, preflop_steps=0,
                fit_rows_by_street=counts, fit_draws=sum(draws), fit_draws_by_street=draws, main_loss=loss)


def train(seed, preflight=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if preflight:
        ref = read(confirmation.OUTPUT/'evaluation-plan.json')['policies']['specialist'][str(seed)]
    else:
        plan = read(OUTPUT/'plan.json')
        verify_files(plan['frozen_files'])
        ref = plan['parent_policies'][str(seed)]
    networks, parent, proof = connect(seed, ref)
    from bot.features import load_preflop_equity
    from compact_regret import ARRAY_MAP, export_compact
    from training import r1_support as r1

    cfg = parent['config']
    expected = dict(traversals=256, batch_size=512, learning_rate=.001, root_repetitions=4,
                    buffer_capacity=100000, strategy_capacity=100000, reference_hands=512)
    if any(cfg[k] != v for k, v in expected.items()):
        raise ValueError('Parent sampling or fitting configuration differs')
    equity = load_preflop_equity()
    buffers, root, strategy, rng, reference_rng = restore_replay(parent)
    probe_features = preflop_probes(buffers, root)
    expected_probabilities = played_probabilities(networks, probe_features)
    initial_opening = {k: v.clone() for k, v in networks.general.opening.state_dict().items()}
    initial_scale = networks.general.opening_scale.clone()
    proof.update(replay_root_reference_and_rng_verified=True, frozen_preflop_probes=len(probe_features))
    if preflight:
        _, counts = postflop_sampler(buffers)
        return dict(**proof, postflop_rows_by_street=counts)

    folder = OUTPUT/ARM/f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    iteration = saved = INITIAL
    token = process_start(os.getpid())

    def mark(status, **fields):
        atomic_json(folder/'status.json', dict(status=status, observed_utc=utc(), pid=os.getpid(),
                    process_start=token, seed=seed, arm=ARM, iteration=iteration,
                    last_committed_iteration=saved, **fields))

    try:
        mark('connected')
        atomic_json(folder/'start-verification.json', proof)
        manifest = parent['source_manifest']
        del parent
        worker_cfg = dict(cfg, initial_iteration=INITIAL, iterations=FINAL, stages=[FINAL],
                          resume_iterations={str(s): INITIAL for s in SEEDS}, training_arm=ARM,
                          main_steps=GENERAL_STEPS, general_steps=GENERAL_STEPS, preflop_steps=0,
                          opening_steps=0, fit_streets=['flop', 'turn', 'river'])
        r1.WEIGHT_HORIZON = cfg['weight_horizon']
        r1.AVERAGE_SAMPLER = 'importance'
        previous = proof['weights']
        began = time.monotonic()
        for iteration in range(INITIAL+1, FINAL+1):
            if time.monotonic()-began > cfg['training_seconds_per_worker']:
                raise TimeoutError('Fixed training time budget reached')
            tick = time.monotonic()
            mark('training')
            verify_files(plan['frozen_files'])
            entering = weights(networks)
            if entering != previous:
                raise ValueError('Input model differs from preceding round')
            learner_rng = r1.pack_rng(rng)
            observation = collect_game_reference(networks, strategy, reference_rng, iteration,
                                                 cfg['reference_hands'], equity)
            if not equal_state(learner_rng, r1.pack_rng(rng)):
                raise ValueError('Independent reference collection changed the learning RNG')
            row = update_postflop(networks, buffers, root, rng, iteration, cfg, equity)
            previous = weights(networks)
            probabilities = played_probabilities(networks, probe_features)
            if (previous['general'] == entering['general'] or previous['preflop'] != proof['weights']['preflop']
                    or not equal_state(initial_opening, networks.general.opening.state_dict())
                    or not torch.equal(networks.general.opening_scale, initial_scale)
                    or not np.array_equal(probabilities, expected_probabilities)):
                raise ValueError('Updated postflop or frozen preflop failed identity checks')

            average, avg_report = None, None
            if iteration == FINAL:
                mark('fitting_window_average')
                average, report = fit_average(strategy, seed=seed*100000+iteration,
                                              steps=cfg['average_fit_steps'], batch_size=cfg['average_batch'],
                                              width=cfg['average_width'])
                avg_report = record_average_quality(report, cfg['average_max_replay_tv'])
            if iteration % 6 == 0:
                state = dict(checkpoint_format=CHECKPOINT_FORMAT, seed=seed, iteration=iteration,
                             config=worker_cfg, source_manifest=manifest, net=networks.general.state_dict(),
                             specialist_net=networks.specialist.state_dict(), buffers=[b.state_dict() for b in buffers],
                             root_replay=root.state_dict(), rng=r1.pack_rng(rng), strategy_replay=strategy.state_dict(),
                             reference_rng=reference_rng.bit_generator.state, average_fit=avg_report,
                             average_state=None if average is None else average.state_dict(),
                             averaging_window=[361, iteration], historical_mixture_available=False)
                commit_checkpoint(folder, state, lambda p: export_compact(networks.general, p), average)
                saved = iteration
                del state
            row.update(seed=seed, iteration=iteration, input_weights=entering, output_weights=previous,
                       buffer_rows=[len(b) for b in buffers], reference_collection=observation,
                       checkpoint_iteration=saved, frozen_preflop_probes=len(probe_features),
                       frozen_preflop_max_probability_error=float(np.max(np.abs(probabilities-expected_probabilities))),
                       average_fit=avg_report, seconds=time.monotonic()-tick)
            with (folder/'diagnostics.jsonl').open('a') as log:
                log.write(json.dumps(row, allow_nan=False)+'\n')

        state, checkpoint = load_checkpoint(folder)
        target = folder/'preflop.npz'
        base = read(ref['path'])['preflop']
        shutil.copyfile(base['path'], target)
        verify_files({target: base['sha256']})
        verify_export(target, state['specialist_net'], ARRAY_MAP)
        verify_export(checkpoint/'current.npz', state['net'], ARRAY_MAP)
        policy = write_policy(folder/'policy.json', target, checkpoint/'current.npz')
        atomic_json(folder/'ready.json', dict(policy=file_ref(policy), checkpoint=str(checkpoint),
                    preflop_export=file_ref(target), files=read(folder/'current.json')['files'],
                    diagnostics_sha256=digest(folder/'diagnostics.jsonl'),
                    start_sha256=digest(folder/'start-verification.json')))
        mark('complete')
    except BaseException as error:
        mark('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'worker'])
    parser.add_argument('seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode == 'worker':
        owner = read(OUTPUT/'training-status.json')
        if (owner['pid'] != os.getppid() or not owner['process_start']
                or process_start(owner['pid']) != owner['process_start']):
            raise RuntimeError('Worker does not belong to the active supervisor')
    result = train(args.seed, preflight=args.mode == 'preflight')
    if result:
        print(json.dumps(result, allow_nan=False))
