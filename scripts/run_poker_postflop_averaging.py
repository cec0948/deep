"""Train a seed-matched online postflop window at a smaller learning rate and capture four policies."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import poker_capture_postflop_snapshots as snapshots
import run_poker_fixed_preflop_window as comparison
import run_poker_preflop_confirmation as confirmation
import train_poker_fixed_preflop_window as original
from compact_lightweight_backup import equal_state
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import file_ref, write_policy
from poker_strategy_average import fit_average
from run_poker_preflop_pilot import SEEDS, SOURCE
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import (CHECKPOINT_FORMAT, collect_game_reference, commit_checkpoint,
                                             load_checkpoint, record_average_quality)
from train_poker_warm_preflop import connect, weights


OUTPUT = confirmation.FOLLOWUP / 'postflop-averaging-20261120'
SCRIPT = Path(__file__).resolve()
ARM = 'slow504'
INITIAL, FINAL = 480, 504
CAPTURES = [486, 492, 498, 504]
RATE = .00025
OLD_SUMMARY_SHA = '6cc8d7759a091ce14ed417088e82917a9fc05792e20d51d2456ca8305df08d9c'
POP_SUMMARY_SHA = read(ROOT/'reports/preflop-followup/480-v1/population/status.json')['panel_summary_sha256']


def matched_first_round(row, control):
    keys = ('traverser', 'traversals', 'new_regret_rows', 'root_rows', 'fit_rows_by_street',
            'fit_draws_by_street', 'fit_draws', 'reference_collection', 'input_weights')
    if any(row[k] != control[k] for k in keys) or row['output_weights']['general'] == control['output_weights']['general']:
        raise ValueError('First round did not match the original incoming policy and sampled counts with a different fitted policy')


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def train(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    owner = read(OUTPUT/'training-status.json')
    if owner['pid'] != os.getppid() or process_start(owner['pid']) != owner['process_start']:
        raise RuntimeError('Worker must belong to the registered training supervisor')
    plan = read(OUTPUT/'plan.json')
    verify_files({**plan['frozen_files'], OUTPUT/'plan.json': read(OUTPUT/'registration.json')['plan_sha256']})
    ref = plan['parent_policies'][str(seed)]
    folder = OUTPUT/ARM/f'seed-{seed}'
    capture_folder = OUTPUT/'snapshots'/f'seed-{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    token = process_start(os.getpid())
    iteration = saved = INITIAL

    def mark(status, **fields):
        atomic_json(folder/'status.json', dict(status=status, observed_utc=utc(), pid=os.getpid(),
            process_start=token, seed=seed, arm=ARM, iteration=iteration, last_committed_iteration=saved, **fields))

    try:
        mark('connecting')
        networks, parent, proof = connect(seed, ref)
        from bot.features import load_preflop_equity
        from compact_regret import ARRAY_MAP, export_compact
        from training import r1_support as r1

        cfg = parent['config']
        expected = dict(traversals=256, batch_size=512, learning_rate=.001, root_repetitions=4,
                        buffer_capacity=100000, strategy_capacity=100000, reference_hands=512)
        if any(cfg[k] != value for k, value in expected.items()):
            raise ValueError('Unexpected parent budget')
        run_cfg = dict(cfg, learning_rate=RATE, initial_iteration=INITIAL, iterations=FINAL, stages=[FINAL],
                       resume_iterations={str(s): INITIAL for s in SEEDS}, training_arm=ARM,
                       main_steps=768, general_steps=768, preflop_steps=0, opening_steps=0,
                       fit_streets=['flop', 'turn', 'river'])
        buffers, root, strategy, rng, reference_rng = original.restore_replay(parent)
        features = original.preflop_probes(buffers, root)
        expected_prob = original.played_probabilities(networks, features)
        opening = {k:v.clone() for k,v in networks.general.opening.state_dict().items()}
        opening_scale = networks.general.opening_scale.clone()
        proof.update(replay_root_reference_and_rng_verified=True, frozen_preflop_probes=len(features))
        atomic_json(folder/'start-verification.json', proof)
        source_manifest = parent['source_manifest']
        del parent
        equity = load_preflop_equity()
        r1.WEIGHT_HORIZON = cfg['weight_horizon']
        r1.AVERAGE_SAMPLER = 'importance'
        first_control = read_rows(Path(plan['matched_diagnostics'][str(seed)]))[0]
        previous = proof['weights']
        began = time.monotonic()
        for iteration in range(INITIAL+1, FINAL+1):
            if time.monotonic()-began > cfg['training_seconds_per_worker']:
                raise TimeoutError('Registered worker time budget reached')
            tick = time.monotonic()
            mark('training')
            entering = weights(networks)
            if entering != previous:
                raise ValueError('Incoming model no longer matches previous update')
            rng_before = r1.pack_rng(rng)
            observation = collect_game_reference(networks, strategy, reference_rng, iteration, cfg['reference_hands'], equity)
            if not equal_state(rng_before, r1.pack_rng(rng)):
                raise ValueError('Reference collection changed the learning RNG')
            row = original.update_postflop(networks, buffers, root, rng, iteration, run_cfg, equity)
            previous = weights(networks)
            probs = original.played_probabilities(networks, features)
            if (previous['general'] == entering['general'] or previous['preflop'] != proof['weights']['preflop']
                    or not equal_state(opening, networks.general.opening.state_dict())
                    or not torch.equal(opening_scale, networks.general.opening_scale)
                    or not np.array_equal(probs, expected_prob)):
                raise ValueError('Postflop updates or frozen preflop failed identity checks')
            row.update(seed=seed, iteration=iteration, input_weights=entering, output_weights=previous,
                       learning_rate=RATE, buffer_rows=[len(b) for b in buffers], reference_collection=observation,
                       frozen_preflop_probes=len(features), frozen_preflop_max_probability_error=0.)
            if iteration == INITIAL+1:
                matched_first_round(row, first_control)
            average = report = None
            if iteration == FINAL:
                mark('fitting_window_average')
                average, report = fit_average(strategy, seed=seed*100000+iteration, steps=cfg['average_fit_steps'],
                                              batch_size=cfg['average_batch'], width=cfg['average_width'])
                report = record_average_quality(report, cfg['average_max_replay_tv'])
            if iteration in CAPTURES:
                state = dict(checkpoint_format=CHECKPOINT_FORMAT, seed=seed, iteration=iteration, config=run_cfg,
                             source_manifest=source_manifest, net=networks.general.state_dict(),
                             specialist_net=networks.specialist.state_dict(), buffers=[b.state_dict() for b in buffers],
                             root_replay=root.state_dict(), rng=r1.pack_rng(rng), strategy_replay=strategy.state_dict(),
                             reference_rng=reference_rng.bit_generator.state, average_fit=report,
                             average_state=None if average is None else average.state_dict(),
                             averaging_window=[361,iteration], historical_mixture_available=False)
                commit_checkpoint(folder, state, lambda p: export_compact(networks.general, p), average)
                saved = iteration
                del state
            row.update(checkpoint_iteration=saved, average_fit=report, seconds=time.monotonic()-tick)
            with (folder/'diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                if iteration in CAPTURES:
                    stream.flush()
                    os.fsync(stream.fileno())
            if iteration in CAPTURES:
                snapshots.capture(capture_folder, iteration)

        final_state, checkpoint = load_checkpoint(folder)
        base = read(ref['path'])['preflop']
        target = folder/'preflop.npz'
        shutil.copyfile(base['path'], target)
        verify_files({target: base['sha256']})
        verify_export(target, final_state['specialist_net'], ARRAY_MAP)
        verify_export(checkpoint/'current.npz', final_state['net'], ARRAY_MAP)
        if final_state['config'] != run_cfg:
            raise ValueError('Committed recovery differs from the training configuration')
        snapshots.finalize(capture_folder)
        policy = write_policy(folder/'policy.json', target, checkpoint/'current.npz')
        atomic_json(folder/'ready.json', dict(policy=file_ref(policy), checkpoint=str(checkpoint),
            preflop_export=file_ref(target), files=read(folder/'current.json')['files'],
            diagnostics_sha256=digest(folder/'diagnostics.jsonl'), start_sha256=digest(folder/'start-verification.json'),
            mixture=file_ref(capture_folder/'mixture.json')))
        mark('complete')
    except BaseException as exc:
        mark('failed', error=f'{type(exc).__name__}: {exc}')
        raise


def verify_stage(plan):
    assets, summaries, endpoint, mixtures = {}, {}, {}, {}
    for seed in SEEDS:
        key = str(seed)
        folder = OUTPUT/ARM/f'seed-{seed}'
        cap = OUTPUT/'snapshots'/f'seed-{seed}'
        ready = read(folder/'ready.json')
        rows = read_rows(folder/'diagnostics.jsonl')
        start = read(folder/'start-verification.json')
        ptr = read(folder/'current.json')
        verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256'],
                      folder/'start-verification.json': ready['start_sha256'],
                      **{r['path']:r['sha256'] for r in (ready['policy'], ready['preflop_export'], ready['mixture'])},
                      **{Path(ready['checkpoint'])/k:v for k,v in ready['files'].items()}})
        previous = start['weights']
        first_control = read_rows(Path(plan['matched_diagnostics'][key]))[0]
        if ([r['iteration'] for r in rows] != list(range(INITIAL+1, FINAL+1)) or ptr['iteration'] != FINAL
            or ptr['files'] != ready['files'] or read(folder/'status.json')['status'] != 'complete'
            or start['deployed_policy_sha256'] != plan['parent_policies'][key]['sha256']
            or start['street_weight_sha256']['preflop'] != ready['preflop_export']['sha256']):
            raise ValueError('Unexpected worker, parent or checkpoint identity')
        matched_first_round(rows[0], first_control)
        for r in rows:
            if (r['input_weights'] != previous or r['learning_rate'] != RATE or r['general_steps'] != 768
                or r['traversals'] != 256 or r['traverser'] != (r['iteration']-1)%6
                or r['fit_draws'] != 768*512 or sum(r['fit_draws_by_street']) != r['fit_draws']
                or min(r['fit_draws_by_street']) <= 0 or not np.isfinite(r['main_loss'])
                or r['new_regret_rows'] <= 0 or r['root_rows'] != 169*4 or r['buffer_rows'] != [100000]*6
                or r['reference_collection']['rows'] != 100000 or r['reference_collection']['hands'] != 512
                or r['preflop_steps'] or r['opening_steps'] or r['frozen_preflop_max_probability_error'] != 0.
                or r['frozen_preflop_probes'] != start['frozen_preflop_probes']
                or r['output_weights']['general'] == previous['general']
                or r['output_weights']['preflop'] != start['weights']['preflop']):
                raise ValueError(f'Unexpected training evidence: {seed}/{r["iteration"]}')
            previous = r['output_weights']
        mix = snapshots.manifest(cap)
        if (mix != read(cap/'mixture.json') or [r['iteration'] for r in mix['postflop']] != CAPTURES
            or len({r['sha256'] for r in mix['postflop']}) != len(CAPTURES)
            or mix['postflop'][-1]['sha256'] != ready['files']['current.npz']):
            raise ValueError('Mixture or endpoint differs from captured checkpoints')
        assets[key] = ready
        summaries[key] = dict(start=start, iteration=FINAL, updates=len(rows), final_weights=previous,
            general_steps=sum(r['general_steps'] for r in rows), traversals=sum(r['traversals'] for r in rows),
            new_regret_rows=sum(r['new_regret_rows'] for r in rows), reference_hands=sum(r['reference_collection']['hands'] for r in rows),
            fit_draws_by_street=[sum(r['fit_draws_by_street'][i] for r in rows) for i in range(3)],
            checkpoints=CAPTURES, first_control_sampling_counts_match=True, all_preflop_unchanged=True,
            final_average=rows[-1]['average_fit'])
        endpoint[key], mixtures[key] = ready['policy'], ready['mixture']
    return dict(training_complete=True, iteration=FINAL, completed_utc=utc(), arms={ARM:assets},
                policies={'specialist480':plan['parent_policies'], ARM:endpoint, 'mixture504':mixtures},
                warm_start_weight_chain_verified=True, seed_verification=summaries)


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        comparison.check_workers()
        if OUTPUT.exists():
            raise FileExistsError('Registered window already exists; inspect its original output')
        parents, _, frozen = comparison.frozen_inputs()
        old = original.OUTPUT
        pop = confirmation.FOLLOWUP/'population'
        frozen.update({str(old/'summary.json'):OLD_SUMMARY_SHA, str(pop/'summary.json'):POP_SUMMARY_SHA,
                       str(SCRIPT):digest(SCRIPT), str(Path(snapshots.__file__)):digest(snapshots.__file__)})
        matched = {}
        for seed in SEEDS:
            old_worker = old/original.ARM/f'seed-{seed}'
            ready = read(old/'stage-ready.json')['arms'][original.ARM][str(seed)]
            matched[str(seed)] = str(old_worker/'diagnostics.jsonl')
            frozen[matched[str(seed)]] = ready['diagnostics_sha256']
        verify_files(frozen)
        OUTPUT.mkdir(exist_ok=False)
        try:
            plan = dict(created_utc=utc(), arm=ARM, arms=[ARM], primary_policy='mixture504', secondary_policy=ARM,
                        comparison_policy='specialist480', learning_rate=RATE, prior_learning_rate=.001,
                        general_steps=768, preflop_steps=0, opening_steps=0, main_batch=512,
                        initial_iteration=INITIAL, final_iteration=FINAL, checkpoint_every=6, capture_iterations=CAPTURES,
                        training='6-player self-play; unchanged 480 specialist preflop; postflop Adam only',
                        time_average='four equally likely postflop policies, one private choice per hand',
                        seeds=SEEDS, campaign=str(comparison.CAMPAIGN), parent_policies=parents,
                        parent_archive_sha256=confirmation.pilot.CFG['restore_dependency_480_sha256'],
                        parent_policy_bundle_sha256=comparison.PARENT_BUNDLE,
                        frozen_files=frozen, matched_diagnostics=matched, automatic_retry=False,
                        automatic_extension=False, model_promotion=False)
            atomic_json(OUTPUT/'plan.json', plan)
            sha = digest(OUTPUT/'plan.json')
            for seed in SEEDS:
                snapshots.register(OUTPUT/'snapshots'/f'seed-{seed}', OUTPUT/ARM/f'seed-{seed}',
                                   seed, ARM, parents[str(seed)]['path'], CAPTURES, SOURCE)
            atomic_json(OUTPUT/'registration.json', dict(utc=utc(), plan_sha256=sha))
            commands = {f'{ARM}-seed-{s}':[str(PYTHON), str(SCRIPT), 'worker', str(s)] for s in SEEDS}
            confirmation.run_jobs(OUTPUT, commands, len(SEEDS), 'training')
            verify_files({OUTPUT/'plan.json':sha, **frozen})
            atomic_json(OUTPUT/'stage-ready.json', verify_stage(plan))
        except BaseException as exc:
            atomic_json(OUTPUT/'failure.json', dict(utc=utc(), error=f'{type(exc).__name__}: {exc}', automatic_retry=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run','worker'))
    parser.add_argument('seed', choices=SEEDS, nargs='?', type=int)
    args = parser.parse_args()
    if args.mode == 'worker' and args.seed is not None:
        train(args.seed)
    elif args.mode == 'run' and args.seed is None:
        run()
    else:
        parser.error('Specify a seed only for an internal worker')
