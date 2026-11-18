"""Run the fixed 480-to-492 paired selfplay comparison after confirmation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

import run_poker_preflop_confirmation as confirmation
from compact_lightweight_backup import equal_state
from controlled_poker_refit import assert_finite_arrays, read, utc, verify_files
from evaluate_compact_learning_curve import adoption_conditions
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import file_ref, preflop_rows, write_policy
from poker_strategy_average import StrategyReplay, fit_average
from run_poker_preflop_pilot import CAMPAIGN, PARENT, SEEDS, paired, verify_parent
from train_poker_average_continuation import (CHECKPOINT_FORMAT, collect_game_reference,
    commit_checkpoint, legacy_update, load_checkpoint, record_average_quality)


SCRIPT = Path(__file__).resolve()
OUTPUT = confirmation.FOLLOWUP/'online'
CFG = confirmation.CFG
ARMS = ('control492', 'specialist492')


@dataclass
class StreetNetworks:
    general: object
    specialist: object


def install_street_training_policy():
    import compact_regret
    import training.traverse_fast as traverse
    from regret_partition import is_opening

    legacy_policy = compact_regret.CompactNumpyPolicy

    class RoutedPolicy:
        def __init__(self, networks):
            self.general = legacy_policy(networks.general)
            self.specialist = legacy_policy(networks.specialist)

        def strategy(self, features, legal):
            if features[17] == 1 and not is_opening(features):
                return self.specialist.strategy(features, legal)
            return self.general.strategy(features, legal)

    def policy_for(net):
        return RoutedPolicy(net) if isinstance(net, StreetNetworks) else legacy_policy(net)

    # Both legacy external-sampling traversals and UTG continuation rollouts
    # load this factory. The isolated average collector uses the same routing.
    compact_regret.CompactNumpyPolicy = policy_for
    traverse.NumpyPolicy = policy_for
    return policy_for


def fit_preflop(net, buffers, steps, cfg):
    from regret_fitting import train_conditional
    from regret_partition import base_strata
    from scaled_regret_fitting import ScaledRegretSampler

    replay = [preflop_rows(b.state_dict()) for b in buffers]
    fitted = deepcopy(net)
    sampler = ScaledRegretSampler(replay, base_strata, 'stratum-scaled-feature-conditional')
    optimizer = torch.optim.Adam(fitted.main.parameters(), lr=cfg['learning_rate'])
    loss = train_conditional(fitted.main, sampler, optimizer, steps, cfg['batch_size'])
    groups = np.unique(sampler.groups)
    fitted.main_scales[groups] = torch.from_numpy(sampler.scales[groups])
    fitted.eval()
    assert_finite_arrays(fitted)
    if (not np.isfinite(loss) or sampler.draws.sum() != steps*cfg['batch_size']
            or not equal_state(fitted.opening.state_dict(), net.opening.state_dict())
            or not torch.equal(fitted.opening_scale, net.opening_scale)):
        raise ValueError('Invalid preflop fit or changed opening expert')
    return fitted, loss


def first_collection_digest(buffers, root, strategy, rng, reference_rng):
    """Each seed in both arms must see identical samples before its first update."""
    checksum = hashlib.sha256()
    for replay in [*(b.state_dict() for b in buffers), root.state_dict(), strategy.state_dict()]:
        for name, value in sorted(replay.items()):
            checksum.update(name.encode())
            if isinstance(value, np.ndarray):
                checksum.update(str((value.shape, value.dtype.str)).encode())
                checksum.update(value.tobytes())
            else:
                checksum.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    for generator in (rng, reference_rng):
        checksum.update(json.dumps(generator.bit_generator.state, sort_keys=True).encode())
    return checksum.hexdigest()


def train(study, arm, seed):
    plan = read(study/'plan.json')
    verify_files(plan['frozen_files'])
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unexpected training arm or seed')
    folder = study/arm/f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    initial, final = plan['initial_iteration'], plan['final_iteration']
    general_steps = plan['step_budget'][arm]['general']
    preflop_steps = plan['step_budget'][arm]['preflop']
    source = Path(plan['campaign'])
    sys.path[:0] = [str(source/'source'), str(source/'tools')]
    from bot.features import load_preflop_equity
    from compact_regret import CompactRegretNet, export_compact, install_policy
    from root_regret_replay import RootReplay
    from training import r1_support as r1
    from training.buffer import ReservoirBuffer

    cfg = read(source/'config.json')
    if general_steps + preflop_steps != cfg['main_steps']:
        raise ValueError('Main update budgets differ')
    worker_cfg = dict(cfg, initial_iteration=initial, iterations=final, stages=[final],
        resume_iterations={str(s): initial for s in SEEDS}, training_arm=arm,
        general_steps=general_steps, preflop_steps=preflop_steps)
    update_cfg = dict(cfg, main_steps=general_steps)
    install_policy()
    if preflop_steps:
        install_street_training_policy()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    state, checkpoint = load_checkpoint(source/f'seed-{seed}')
    if state['seed'] != seed or state['iteration'] != initial or state['config'] != cfg:
        raise ValueError('Training must begin with the verified 480-round state')
    buffers = [ReservoirBuffer(cfg['buffer_capacity'], 51) for _ in range(6)]
    for buffer, saved in zip(buffers, state['buffers']):
        buffer.load_state_dict(saved)
    root = RootReplay()
    root.load_state_dict(state['root_replay'])
    general = CompactRegretNet()
    general.load_state_dict(state['net'])
    general.eval()
    rng = np.random.default_rng()
    r1.restore_rng(state['rng'], rng)
    strategy = StrategyReplay.from_state(state['strategy_replay'])
    reference_rng = np.random.default_rng()
    reference_rng.bit_generator.state = state['reference_rng']
    manifest = state['source_manifest']
    del state
    r1.WEIGHT_HORIZON = cfg['weight_horizon']
    r1.AVERAGE_SAMPLER = 'importance'
    equity = load_preflop_equity()
    policy = general
    started = time.monotonic()
    saved_round = initial
    iteration = initial

    def mark(status, **values):
        atomic_json(folder/'status.json', dict(status=status, observed_utc=utc(), pid=os.getpid(),
            process_start=process_start(os.getpid()), seed=seed, arm=arm, iteration=iteration,
            last_committed_iteration=saved_round, **values))

    mark('training', parent_checkpoint=str(checkpoint))
    try:
        for iteration in range(initial + 1, final + 1):
            if time.monotonic() - started > cfg['training_seconds_per_worker']:
                raise TimeoutError('Fixed online training time budget exhausted')
            mark('training')
            verify_files(plan['frozen_files'])
            tick = time.monotonic()
            global_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
            collected = collect_game_reference(policy, strategy, reference_rng, iteration,
                                               cfg['reference_hands'], equity)
            if not equal_state(global_rng, (random.getstate(), np.random.get_state(), torch.get_rng_state())):
                raise ValueError('Reference collection consumed learner RNG')
            general, fit, tv, opening_steps = legacy_update(policy, buffers, root, rng, iteration, update_cfg, equity)
            assert_finite_arrays(general)
            first_sha = (first_collection_digest(buffers, root, strategy, rng, reference_rng)
                         if iteration == initial + 1 else None)
            specialist = None
            preflop_loss = None
            if preflop_steps:
                specialist, preflop_loss = fit_preflop(general, buffers, preflop_steps, cfg)
            policy = StreetNetworks(general, specialist) if specialist is not None else general
            average, average_fit = None, None
            if iteration == final:
                mark('fitting_window_average')
                average, average_fit = fit_average(strategy, seed=seed*100000+iteration,
                    steps=cfg['average_fit_steps'], batch_size=cfg['average_batch'], width=cfg['average_width'])
                average_fit = record_average_quality(average_fit, cfg['average_max_replay_tv'])
                assert_finite_arrays(average)
            if iteration % 6 == 0:
                state = dict(checkpoint_format=CHECKPOINT_FORMAT, seed=seed, iteration=iteration, config=worker_cfg,
                    source_manifest=manifest, net=general.state_dict(), buffers=[b.state_dict() for b in buffers],
                    root_replay=root.state_dict(), rng=r1.pack_rng(rng), strategy_replay=strategy.state_dict(),
                    reference_rng=reference_rng.bit_generator.state, average_fit=average_fit,
                    average_state=None if average is None else average.state_dict(),
                    averaging_window=[361, iteration], historical_mixture_available=False)
                if specialist is not None:
                    state['specialist_net'] = specialist.state_dict()
                commit_checkpoint(folder, state, lambda path: export_compact(general, path), average)
                del state
                saved_round = iteration
            row = dict(iteration=iteration, seed=seed, arm=arm, seconds=time.monotonic()-tick,
                first_collection_sha256=first_sha, traverser=(iteration-1)%6, traversals=cfg['traversals'],
                general_steps=general_steps, preflop_steps=preflop_steps, buffer_rows=[len(b) for b in buffers],
                opening_steps=opening_steps, opening_tv=tv, main_loss=fit['main_loss'], preflop_loss=preflop_loss,
                reference_collection=collected, average_fit=average_fit)
            with (folder/'diagnostics.jsonl').open('a') as log:
                log.write(json.dumps(row, allow_nan=False)+'\n')
        state, checkpoint = load_checkpoint(folder)
        if saved_round != final or state['iteration'] != final:
            raise ValueError('Final saved iteration differs')
        if preflop_steps:
            if not equal_state(state['specialist_net'], specialist.state_dict()):
                raise ValueError('Restored specialist weights differ from the final network')
            target = folder/'preflop.npz'
            export_compact(specialist, target)
            write_policy(folder/'policy.json', target, checkpoint/'current.npz')
            model = folder/'policy.json'
        else:
            model = checkpoint/'current.npz'
        atomic_json(folder/'ready.json', dict(policy=file_ref(model), checkpoint=str(checkpoint),
            files=read(folder/'current.json')['files'], diagnostics_sha256=digest(folder/'diagnostics.jsonl'),
            preflop_export=None if preflop_steps == 0 else file_ref(folder/'preflop.npz')))
        mark('complete')
    except BaseException as error:
        mark('failed', error=f'{type(error).__name__}: {error}')
        raise


def check_confirmation():
    stage = confirmation.OUTPUT
    summary = read(stage/'summary.json')
    if (summary['smoke'] or not summary['independent_confirmation_pass']
            or summary['evaluation_deal_seed'] != CFG['confirmation_deal_seed']
            or summary['independent_card_blocks'] != CFG['evaluation_deals']
            or summary['completed_cells'] != 27 or summary['new_selfplay_iterations'] != 0):
        raise ValueError('Only a complete, passing independent confirmation authorizes this study')
    verify_files({stage/'evaluation-plan.json': summary['evaluation_plan_sha256'],
                  **{str(stage/'evaluation'/name/'results.json'): sha for name, sha in summary['result_sha256'].items()}})
    verify_files(read(stage/'evaluation-plan.json')['frozen_files'])
    return stage/'summary.json'


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmation.check_no_jobs()
        confirmed = check_confirmation()
        verify_parent()
        start, deals = CFG['online_deal_seed'], CFG['evaluation_deals']
        previous = confirmation.check_fresh_cards(OUTPUT, start, deals)
        if shutil.disk_usage(OUTPUT.parent).free < 2*1024**3:
            raise RuntimeError('Insufficient space for both training arms and verified hand records')
        parent_cfg = read(CAMPAIGN/'config.json')
        frozen = {path: sha for path, sha in parent_cfg['frozen_files'].items()
                  if Path(path).is_relative_to(CAMPAIGN/'source') or Path(path).is_relative_to(CAMPAIGN/'tools')}
        paths = [SCRIPT, confirmation.CONFIG, confirmation.SCRIPT, confirmed, CAMPAIGN/'config.json',
                 PARENT/'stage-ready.json', ROOT/'scripts/train_poker_average_continuation.py',
                 ROOT/'scripts/poker_preflop_specialist.py', ROOT/'scripts/poker_strategy_average.py',
                 ROOT/'scripts/compact_lightweight_backup.py']
        frozen.update({str(path): digest(path) for path in paths})
        verify_files(frozen)
        OUTPUT.mkdir(exist_ok=False)
        initial, final = CFG['initial_iteration'], CFG['final_iteration']
        plan = dict(created_utc=utc(), initial_iteration=initial, final_iteration=final, arms=ARMS,
            seeds=SEEDS, campaign=str(CAMPAIGN), frozen_files=frozen,
            parent_archive_sha256=read(PARENT/'backup-ready.json')['sha256'],
            step_budget=dict(control492=dict(general=parent_cfg['main_steps'], preflop=0),
                             specialist492=dict(general=CFG['online_general_steps'], preflop=CFG['online_preflop_steps'])),
            new_selfplay_per_arm_seed=final-initial, evaluation_deal_seed=start, evaluation_deals=deals,
            comparison='Compare both 492-round policies on the same new cards against each other and original120/current480',
            automatic_retry=False, model_promotion=False)
        atomic_json(OUTPUT/'plan.json', plan)
        atomic_json(OUTPUT/'config.json', dict(evaluation_deal_seed=start, evaluation_deals=deals,
            prior_card_ranges=previous, initial_iteration=initial, final_iteration=final))
        try:
            commands = {f'{arm}-seed-{s}': [str(PYTHON), str(SCRIPT), '_train', arm, str(s)]
                        for s in SEEDS for arm in ARMS}
            confirmation.run_jobs(OUTPUT, commands, CFG['evaluation_workers'], 'training')
            verify_files(frozen)
            models, opponents, inherited = confirmation.frozen_pilot()
            del models['specialist']
            models.update({arm: {} for arm in ARMS})
            records, first_digests, opening_totals = {}, {}, {}
            for arm in ARMS:
                records[arm], first_digests[arm], opening_totals[arm] = {}, {}, {}
                for seed in SEEDS:
                    key = str(seed)
                    folder = OUTPUT/arm/f'seed-{seed}'
                    ready = read(folder/'ready.json')
                    pointer = read(folder/'current.json')
                    if pointer['iteration'] != final or pointer['files'] != ready['files']:
                        raise ValueError('Final checkpoint no longer matches its ready record')
                    verify_files({Path(ready['checkpoint'])/name: sha for name, sha in ready['files'].items()})
                    verify_files({ready['policy']['path']: ready['policy']['sha256'],
                                  str(folder/'diagnostics.jsonl'): ready['diagnostics_sha256']})
                    if ready['preflop_export'] is not None:
                        ref = ready['preflop_export']
                        verify_files({ref['path']: ref['sha256']})
                    rows = [json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines()]
                    iterations = range(initial + 1, final + 1)
                    if ([r['iteration'] for r in rows] != list(iterations)
                            or [r['traverser'] for r in rows] != [(i-1) % 6 for i in iterations]
                            or any(r['traversals'] != parent_cfg['traversals']
                                   or r['general_steps']+r['preflop_steps'] != parent_cfg['main_steps']
                                   for r in rows)):
                        raise ValueError('Actual training steps or traverser schedule differ from plan')
                    first_digests[arm][key] = rows[0]['first_collection_sha256']
                    opening_totals[arm][key] = sum(r['opening_steps'] for r in rows)
                    records[arm][key] = ready
                    models[arm][key] = ready['policy']
            if first_digests[ARMS[0]] != first_digests[ARMS[1]] or not all(first_digests[ARMS[0]].values()):
                raise ValueError('Paired first-iteration sample/RNG parity failed')
            atomic_json(OUTPUT/'stage-ready.json', dict(training_complete=True, iteration=final, arms=records,
                first_collection_parity=True, opening_steps=opening_totals, completed_utc=utc()))
            rows, outcome = confirmation.evaluate(OUTPUT, models, opponents, {**inherited, **frozen}, start, deals)
            changes = {f'{after}_minus_{before}': paired(rows, before, after, deals)
                       for after, before in [(a, b) for a in ARMS for b in ('original120', 'current480')]
                       + [('specialist492', 'control492')]}
            conditions = {name: adoption_conditions(change)['conditions'] for name, change in changes.items()}
            outcome.update(completed_utc=utc(), completed=True, comparisons=changes, conditions=conditions,
                training_iterations_per_arm_seed=final-initial, first_collection_parity=True,
                opening_steps=opening_totals, model_promoted=False,
                specialist_extension_supported=all(all(conditions[f'specialist492_minus_{b}'].values())
                    for b in ('original120', 'current480', 'control492')))
            atomic_json(OUTPUT/'summary.json', outcome)
            return outcome
        except BaseException as error:
            atomic_json(OUTPUT/'failure.json', dict(observed_utc=utc(), error=f'{type(error).__name__}: {error}'))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', '_train'))
    parser.add_argument('internal', nargs='*', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.action == '_train':
        arm, seed = args.internal
        owner = read(OUTPUT/'training-status.json')
        if owner['pid'] != os.getppid() or process_start(owner['pid']) != owner['process_start']:
            raise ValueError('Unowned online training worker')
        train(OUTPUT, arm, int(seed))
    elif not args.internal:
        result = run()
        print(json.dumps({k: v for k, v in result.items() if k not in ('result_sha256', 'noregrets_fallback_and_decisions')},
                         indent=2, allow_nan=False))
    else:
        parser.error('No overrides for this fixed study')


if __name__ == '__main__':
    main()
