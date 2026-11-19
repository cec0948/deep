"""Continue both street networks from the evaluated 480-round specialist."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

import run_poker_preflop_confirmation as confirmation
from compact_lightweight_backup import equal_state
from controlled_poker_refit import assert_finite_arrays, read, utc, verify_files
from fullhouse_runtime import Bridge
from fullhouse_v3_common import atomic_json, digest, process_start
from poker_preflop_specialist import evaluator_policy, file_ref, write_policy
from poker_strategy_average import StrategyReplay, fit_average
from run_poker_preflop_online import StreetNetworks, fit_preflop, install_street_training_policy
from run_poker_preflop_pilot import CAMPAIGN, SEEDS, SOURCE
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import (CHECKPOINT_FORMAT, collect_game_reference,
    commit_checkpoint, load_checkpoint, record_average_quality)


OUTPUT = confirmation.FOLLOWUP / 'warm-start'
ARM = 'warm492'
PARENT = 'specialist480'
INITIAL, FINAL = 480, 492
GENERAL_STEPS, PREFLOP_STEPS = 768, 256


def fingerprint(network):
    h = hashlib.sha256()
    for name, value in network.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def weights(networks):
    return dict(general=fingerprint(networks.general), preflop=fingerprint(networks.specialist))


def connect(seed, ref):
    """Restore the precise baseline and test the real traversal factory against deployment."""
    sys.path[:0] = [str(SOURCE), str(CAMPAIGN / 'tools')]
    from compact_regret import ARRAY_MAP, CompactRegretNet, install_policy
    from regret_partition import is_opening
    from training import traverse_fast

    state, checkpoint = load_checkpoint(CAMPAIGN / f'seed-{seed}')
    if state['seed'] != seed or state['iteration'] != INITIAL or state['config'] != read(CAMPAIGN/'config.json'):
        raise ValueError('Unexpected source checkpoint, seed or configuration')
    verify_files({ref['path']: ref['sha256']})
    spec = read(ref['path'])
    if spec['postflop'] != file_ref(checkpoint/'current.npz'):
        raise ValueError('The frozen policy uses a different postflop network')
    general = CompactRegretNet().eval()
    general.load_state_dict(state['net'])
    specialist = deepcopy(general)
    with np.load(spec['preflop']['path'], allow_pickle=False) as data:
        specialist.main.load_state_dict({key: torch.from_numpy(data[name].copy())
                                         for key, name in ARRAY_MAP.items()})
        specialist.main_scales.copy_(torch.from_numpy(data['main_scales'].copy()))
    for net, street in ((general, 'postflop'), (specialist, 'preflop')):
        asset = spec[street]
        verify_files({asset['path']: asset['sha256']})
        verify_export(asset['path'], net.state_dict(), ARRAY_MAP)
        assert_finite_arrays(net)
    networks = StreetNetworks(general, specialist)
    install_policy()
    factory = install_street_training_policy()
    if traverse_fast.NumpyPolicy is not factory:
        raise ValueError('Traversal factory did not bind the selected street policy')
    played = traverse_fast.NumpyPolicy(networks)
    base = factory(general)
    old_path = sys.path.copy()
    old_data = os.environ.get('BOT_DATA_DIR')
    engine = Bridge()
    counts = dict(opening=0, preflop=0, flop=0, turn=0, river=0)
    different = 0
    max_error = 0.
    try:
        frozen = evaluator_policy(ref['path'], SOURCE)
        for deal in range(2037560100, 2037560116):
            public = engine.call(dict(op='reset', seed=deal, button=0))
            while not public['terminal']:
                street = public['street']
                deployed = frozen.policies['preflop' if street == 'preflop' else 'postflop']
                features, legal = deployed.features.encode_state_dict(public, deployed.lookup._equity_tables)
                legal = legal.astype(np.float32)
                actual = played.strategy(features, legal)
                expected = deployed.lookup.get_strategy(public)
                error = float(np.max(np.abs(expected - actual)))
                if not np.isfinite(error) or error > 1e-6:
                    raise ValueError(f'Policy mismatch before first deal: {seed}/{street}/{error}')
                max_error = max(max_error, error)
                group = 'opening' if is_opening(features) else street
                counts[group] += 1
                if group == 'preflop' and not np.allclose(actual, base.strategy(features, legal), rtol=0, atol=1e-6):
                    different += 1
                action = dict(seat=public['seat_to_act'], kind='check' if public['can_check'] else 'call')
                public = engine.call(dict(op='step', action=action))
    finally:
        engine.close()
        sys.path[:] = old_path
        if old_data is None:
            os.environ.pop('BOT_DATA_DIR', None)
        else:
            os.environ['BOT_DATA_DIR'] = old_data
    if min(counts.values()) == 0 or different == 0:
        raise ValueError('Initial policy probes did not exercise the selected specialist')
    proof = dict(seed=seed, iteration=INITIAL, weights=weights(networks),
                 deployed_policy_sha256=ref['sha256'], street_weight_sha256={k:spec[k]['sha256'] for k in ('preflop','postflop')},
                 action_probability_probes=counts, changed_nonopening_preflop_vs_general=different,
                 max_probability_error=max_error, checkpoint_files=read(CAMPAIGN/f'seed-{seed}'/'current.json')['files'])
    return networks, state, proof


def update(networks, buffers, root, rng, iteration, cfg, equity):
    from compact_regret import fit_compact, fit_opening
    from root_regret_replay import collect_root_replay
    from training.traverse_fast import run_traversals_fast

    traverser = (iteration-1) % 6
    replay = buffers[traverser]
    before = replay.total_seen
    run_traversals_fast(cfg['traversals'], 6, traverser, networks, replay, iteration, rng, equity,
                        opponent_exploration=0., strategy_buffer=None, randomize_stacks=False)
    root_rows = collect_root_replay(networks, rng, root, iteration, cfg['root_repetitions'])
    if replay.total_seen <= before or root_rows != 169 * cfg['root_repetitions']:
        raise ValueError('Expected new traversal and conditional opening samples')
    general = networks.general
    fit = fit_compact(general, buffers, GENERAL_STEPS, cfg['batch_size'], root_replay=root,
                      opening_steps=cfg['opening_steps'], lr=cfg['learning_rate'])
    x, y, mass = root.means()
    legal = x[:,46:51]
    positive = np.maximum(y,0) * legal
    totals = positive.sum(1,keepdims=True)
    expected = np.divide(positive, totals, out=legal.astype(float)/legal.sum(1,keepdims=True), where=totals>0)
    steps = cfg['opening_steps']
    while True:
        pred = general.get_strategy(torch.from_numpy(x), torch.from_numpy(legal)).numpy()
        tv = float(np.average(np.abs(pred-expected).sum(1)/2, weights=mass))
        if not np.isfinite(tv) or not np.isfinite(fit['main_loss']):
            raise FloatingPointError('Nonfinite main fit or opening probabilities')
        if tv <= cfg['max_opening_tv']:
            break
        extra = min(1024, cfg['opening_max_steps']-steps)
        if extra <= 0:
            raise ValueError('Opening fit exceeded the inherited quality bound')
        fit_opening(general, (x,y,mass), extra, cfg['learning_rate'])
        steps += extra
    # Start the non-opening preflop update from its own previous weights.
    specialist, pre_loss = fit_preflop(networks.specialist, buffers, PREFLOP_STEPS, cfg)
    specialist.opening.load_state_dict(general.opening.state_dict())
    specialist.opening_scale.copy_(general.opening_scale)
    assert_finite_arrays(general)
    assert_finite_arrays(specialist)
    return StreetNetworks(general, specialist), dict(traverser=traverser, traversals=cfg['traversals'],
        new_regret_rows=replay.total_seen-before, root_rows=root_rows, opening_tv=tv, opening_steps=steps,
        general_steps=GENERAL_STEPS, preflop_steps=PREFLOP_STEPS, main_loss=fit['main_loss'], preflop_loss=pre_loss)


def train(seed, preflight=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if preflight:
        specialist = read(confirmation.OUTPUT/'evaluation-plan.json')['policies']['specialist'][str(seed)]
        return connect(seed, specialist)[2]
    plan = read(OUTPUT/'plan.json')
    verify_files(plan['frozen_files'])
    folder = OUTPUT/ARM/f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    def mark(status, **values):
        atomic_json(folder/'status.json', dict(status=status, observed_utc=utc(), pid=os.getpid(),
                    process_start=process_start(os.getpid()), seed=seed, arm=ARM, **values))
    mark('checking_start', iteration=INITIAL)
    try:
        networks, parent, proof = connect(seed, plan['parent_policies'][str(seed)])
        from bot.features import load_preflop_equity
        from compact_regret import ARRAY_MAP, export_compact
        from root_regret_replay import RootReplay
        from training import r1_support as r1
        from training.buffer import ReservoirBuffer
        cfg = parent['config']
        if (cfg['main_steps'] != GENERAL_STEPS+PREFLOP_STEPS or cfg['batch_size'] != 512
                or cfg['buffer_capacity'] != 100000 or cfg['strategy_capacity'] != 100000):
            raise ValueError('Parent budgets differ')
        worker_cfg = dict(cfg, initial_iteration=INITIAL, iterations=FINAL, stages=[FINAL],
            resume_iterations={str(s):INITIAL for s in SEEDS}, training_arm=ARM,
            general_steps=GENERAL_STEPS, preflop_steps=PREFLOP_STEPS)
        buffers = [ReservoirBuffer(cfg['buffer_capacity'],51) for _ in range(6)]
        for buf, saved in zip(buffers,parent['buffers'],strict=True):
            buf.load_state_dict(saved)
        root = RootReplay()
        root.load_state_dict(parent['root_replay'])
        strategy = StrategyReplay.from_state(parent['strategy_replay'])
        rng, reference_rng = np.random.default_rng(), np.random.default_rng()
        equity = load_preflop_equity()
        reference_rng.bit_generator.state = parent['reference_rng']
        r1.restore_rng(parent['rng'],rng)
        if (not equal_state(r1.pack_rng(rng),parent['rng']) or not equal_state(reference_rng.bit_generator.state,parent['reference_rng'])
                or not equal_state(strategy.state_dict(),parent['strategy_replay'])
                or not equal_state(root.state_dict(),parent['root_replay'])
                or any(not equal_state(b.state_dict(),s) for b,s in zip(buffers,parent['buffers'],strict=True))):
            raise ValueError('Replay, root statistics or RNG restore differs before sampling')
        proof.update(replay_root_reference_and_rng_verified=True)
        atomic_json(folder/'start-verification.json', proof)
        manifest = parent['source_manifest']
        del parent
        r1.WEIGHT_HORIZON = cfg['weight_horizon']
        r1.AVERAGE_SAMPLER = 'importance'
        previous = proof['weights']
        start = time.monotonic()
        for iteration in range(INITIAL+1, FINAL+1):
            if time.monotonic()-start > cfg['training_seconds_per_worker']:
                raise TimeoutError('Training time budget reached')
            mark('training', iteration=iteration)
            tick = time.monotonic()
            verify_files(plan['frozen_files'])
            entering = weights(networks)
            if entering != previous:
                raise ValueError('Weights changed between consecutive selfplay iterations')
            learner_rng = r1.pack_rng(rng)
            observation = collect_game_reference(networks,strategy,reference_rng,iteration,cfg['reference_hands'],equity)
            if not equal_state(learner_rng,r1.pack_rng(rng)):
                raise ValueError('Reference collection consumed learner RNG')
            networks, row = update(networks,buffers,root,rng,iteration,cfg,equity)
            previous = weights(networks)
            if any(previous[k] == entering[k] for k in entering):
                raise ValueError('An assigned network did not update')
            average, avg_report = None, None
            if iteration == FINAL:
                mark('fitting_window_average', iteration=iteration)
                average, av = fit_average(strategy,seed=seed*100000+iteration,steps=cfg['average_fit_steps'],
                    batch_size=cfg['average_batch'],width=cfg['average_width'])
                avg_report = record_average_quality(av,cfg['average_max_replay_tv'])
            if iteration % 6 == 0:
                state = dict(checkpoint_format=CHECKPOINT_FORMAT, seed=seed, iteration=iteration, config=worker_cfg,
                    source_manifest=manifest, net=networks.general.state_dict(), specialist_net=networks.specialist.state_dict(),
                    buffers=[b.state_dict() for b in buffers], root_replay=root.state_dict(), rng=r1.pack_rng(rng),
                    strategy_replay=strategy.state_dict(), reference_rng=reference_rng.bit_generator.state,
                    average_fit=avg_report, average_state=None if average is None else average.state_dict(),
                    averaging_window=[361,iteration], historical_mixture_available=False)
                commit_checkpoint(folder,state,lambda path:export_compact(networks.general,path),average)
                del state
            row.update(seed=seed, iteration=iteration, input_weights=entering, output_weights=previous,
                       buffer_rows=[len(b) for b in buffers], reference_collection=observation,
                       average_fit=avg_report, seconds=time.monotonic()-tick)
            with (folder/'diagnostics.jsonl').open('a') as log:
                log.write(json.dumps(row,allow_nan=False)+'\n')
        final, checkpoint = load_checkpoint(folder)
        target = folder/'preflop.npz'
        export_compact(networks.specialist,target)
        verify_export(target,final['specialist_net'],ARRAY_MAP)
        verify_export(checkpoint/'current.npz',final['net'],ARRAY_MAP)
        policy = write_policy(folder/'policy.json',target,checkpoint/'current.npz')
        atomic_json(folder/'ready.json',dict(policy=file_ref(policy), checkpoint=str(checkpoint),
            preflop_export=file_ref(target), files=read(folder/'current.json')['files'],
            diagnostics_sha256=digest(folder/'diagnostics.jsonl'), start_sha256=digest(folder/'start-verification.json')))
        mark('complete', iteration=FINAL)
    except BaseException as error:
        mark('failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight','worker'])
    parser.add_argument('seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.mode == 'worker':
        owner = read(OUTPUT/'training-status.json')
        if owner['pid'] != os.getppid() or process_start(owner['pid']) != owner['process_start']:
            raise RuntimeError('Worker does not belong to the running supervisor')
    result = train(args.seed, args.mode == 'preflight')
    if result:
        print(json.dumps(result, allow_nan=False))
