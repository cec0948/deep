"""Run the matched 480→504 treatment with seat-fixed historical self-play opponents."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

import poker_capture_postflop_snapshots as snapshots
import poker_opponent_population as population
import run_poker_fixed_preflop_window as prior
import run_poker_postflop_averaging as control
import run_poker_preflop_confirmation as confirmation
import train_poker_fixed_preflop_window as original
from compact_lightweight_backup import equal_state
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import file_ref, write_policy
from poker_strategy_average import fit_average
from run_poker_preflop_pilot import CAMPAIGN, SEEDS, SOURCE
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import (CHECKPOINT_FORMAT, collect_game_reference, commit_checkpoint,
                                             load_checkpoint, record_average_quality)
from train_poker_warm_preflop import connect, weights

STUDY = confirmation.FOLLOWUP/'opponent-diversity-20261123'
CONFIG = ROOT/'configs/poker-opponent-diversity-20261123.json'
SCRIPT = Path(__file__).resolve()
ARM = 'pool504'


def bind_training():
    cfg, reg = read(CONFIG), read(STUDY/'registration.json')
    if (cfg['start'],cfg['end'],cfg['seeds'],cfg['snapshots'],cfg['learning_rate'],
            cfg['traversals_per_iteration'],cfg['postflop_adam_updates_per_iteration'],
            cfg['batch_size'],cfg['replay_capacity_per_seat']) != (480,504,SEEDS,control.CAPTURES,.00025,256,768,512,100000):
        raise ValueError('Registered training settings differ')
    mix = cfg['treatment_opponents']
    if (mix['current_probability'],mix['historical_probability'],mix['historical_iterations'],
            mix['training_external_opponents']) != (.5,.5,list(population.STAGES),[]):
        raise ValueError('Changed opponent assignment design')
    if reg['config_sha256'] != digest(CONFIG):
        raise ValueError('Config differs from public registration')
    parents, _, frozen = prior.frozen_inputs()
    hist, sources = population.historical_inputs(SEEDS)
    frozen.update(sources)
    old_stage = control.OUTPUT/'stage-ready.json'
    old = read(old_stage)
    if parents != old['policies']['specialist480'] or not old['training_complete']:
        raise ValueError('Parent policies do not match the completed control')
    paths = [CONFIG, STUDY/'registration.json', SCRIPT, Path(population.__file__), old_stage,
             Path(control.__file__), Path(snapshots.__file__), SOURCE/'training/traverse_fast.py',
             SOURCE/'bot/features.py', CAMPAIGN/'tools/compact_regret.py']
    frozen.update({str(p):digest(p) for p in paths})
    control_logs = {}
    for seed in SEEDS:
        key = str(seed)
        folder = control.OUTPUT/control.ARM/f'seed-{seed}'
        diag = folder/'diagnostics.jsonl'
        frozen[str(diag)] = old['arms'][control.ARM][key]['diagnostics_sha256']
        frozen[str(folder/'start-verification.json')] = old['arms'][control.ARM][key]['start_sha256']
        control_logs[key] = str(diag)
    verify_files(frozen)
    return dict(utc=utc(), arm=ARM, seeds=SEEDS, initial_iteration=480, final_iteration=504,
                capture_iterations=control.CAPTURES, historical_policies=hist, parent_policies=parents,
                campaign=str(CAMPAIGN), parent_archive_sha256=confirmation.pilot.CFG['restore_dependency_480_sha256'],
                parent_policy_bundle_sha256=prior.PARENT_BUNDLE, control_logs=control_logs, frozen_files=frozen)


def train(seed):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read_bound()
    owner = read(STUDY/'training-status.json')
    if owner['pid'] != os.getppid() or process_start(owner['pid']) != owner['process_start']:
        raise RuntimeError('Worker does not belong to the registered supervisor')
    key = str(seed)
    ref = plan['parent_policies'][key]
    folder = STUDY/ARM/f'seed-{seed}'
    cap = STUDY/'snapshots'/f'seed-{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    token = process_start(os.getpid())
    iteration = saved = 480

    def mark(status, **fields):
        atomic_json(folder/'status.json',dict(status=status,observed_utc=utc(),pid=os.getpid(),
                    process_start=token,seed=seed,arm=ARM,iteration=iteration,last_committed_iteration=saved,**fields))
    try:
        mark('connecting')
        networks, parent, proof = connect(seed, ref)
        from bot.features import load_preflop_equity
        from compact_regret import ARRAY_MAP, export_compact
        from training import r1_support as r1

        source_manifest = parent['source_manifest']
        cfg = dict(parent['config'], learning_rate=.00025, initial_iteration=480, iterations=504, stages=[504],
                   resume_iterations={str(s):480 for s in SEEDS}, training_arm=ARM, main_steps=768, general_steps=768,
                   preflop_steps=0, opening_steps=0,fit_streets=['flop','turn','river'],
                   opponent_assignment='independent-seat-per-traversal-v1', historical_policies=plan['historical_policies'][key])
        buffers, root, strategy, rng, reference_rng = original.restore_replay(parent)
        del parent
        probes = original.preflop_probes(buffers,root)
        expected = original.played_probabilities(networks,probes)
        opening = {k:v.clone() for k,v in networks.general.opening.state_dict().items()}
        opening_scale = networks.general.opening_scale.clone()
        proof.update(replay_root_reference_and_rng_verified=True,frozen_preflop_probes=len(probes))
        control_start = read(Path(plan['control_logs'][key]).parent/'start-verification.json')
        if proof != control_start:
            raise ValueError('Starting policies, checkpoint files or probes differ from matched control')
        atomic_json(folder/'start-verification.json',proof)
        rng_before = r1.pack_rng(rng)
        pool = population.load_historical(networks, plan['historical_policies'][key])
        if not equal_state(rng_before, r1.pack_rng(rng)):
            raise ValueError('Connecting the opponent pool changed the learning RNG')
        equity = load_preflop_equity()
        r1.WEIGHT_HORIZON, r1.AVERAGE_SAMPLER = cfg['weight_horizon'], 'importance'
        previous = proof['weights']
        began = time.monotonic()
        for iteration in range(481,505):
            if time.monotonic()-began > cfg['training_seconds_per_worker']:
                raise TimeoutError('Registered worker time budget reached')
            tick = time.monotonic()
            mark('training')
            entering = weights(networks)
            if entering != previous:
                raise ValueError('Broken model update chain')
            rng_before = r1.pack_rng(rng)
            reference = collect_game_reference(networks,strategy,reference_rng,iteration,cfg['reference_hands'],equity)
            if not equal_state(rng_before,r1.pack_rng(rng)):
                raise ValueError('Reference collection changed the learning RNG')
            row = population.pool_update(networks,buffers,root,rng,iteration,cfg,equity,seed,pool)
            previous = weights(networks)
            if (previous['general']==entering['general'] or previous['preflop']!=proof['weights']['preflop']
                    or not equal_state(opening,networks.general.opening.state_dict())
                    or not torch.equal(opening_scale,networks.general.opening_scale)
                    or not np.array_equal(expected,original.played_probabilities(networks,probes))):
                raise ValueError('Unexpected unchanged training weights or changed frozen preflop')
            row.update(seed=seed,iteration=iteration,input_weights=entering,output_weights=previous,
                       learning_rate=cfg['learning_rate'],buffer_rows=[len(b) for b in buffers],
                       reference_collection=reference,frozen_preflop_probes=len(probes),
                       frozen_preflop_max_probability_error=0.)
            model = report = None
            if iteration==504:
                mark('fitting_window_average')
                model, report = fit_average(strategy,seed=seed*100000+iteration,steps=cfg['average_fit_steps'],
                                             batch_size=cfg['average_batch'],width=cfg['average_width'])
                report = record_average_quality(report,cfg['average_max_replay_tv'])
            if iteration in control.CAPTURES:
                state = dict(checkpoint_format=CHECKPOINT_FORMAT,seed=seed,iteration=iteration,config=cfg,
                             source_manifest=source_manifest,net=networks.general.state_dict(),
                             specialist_net=networks.specialist.state_dict(),buffers=[b.state_dict() for b in buffers],
                             root_replay=root.state_dict(),rng=r1.pack_rng(rng),strategy_replay=strategy.state_dict(),
                             reference_rng=reference_rng.bit_generator.state,average_fit=report,
                             average_state=None if model is None else model.state_dict(),averaging_window=[361,iteration],
                             historical_mixture_available=False)
                commit_checkpoint(folder,state,lambda p:export_compact(networks.general,p),model)
                saved = iteration
                del state
            row.update(checkpoint_iteration=saved,average_fit=report,seconds=time.monotonic()-tick)
            with (folder/'diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if iteration in control.CAPTURES:
                    stream.flush()
                    os.fsync(stream.fileno())
            if iteration in control.CAPTURES:
                snapshots.capture(cap,iteration)

        final, checkpoint = load_checkpoint(folder)
        pre = read(ref['path'])['preflop']
        target = folder/'preflop.npz'
        shutil.copyfile(pre['path'],target)
        verify_files({target:pre['sha256']})
        verify_export(target,final['specialist_net'],ARRAY_MAP)
        verify_export(checkpoint/'current.npz',final['net'],ARRAY_MAP)
        if cfg != final['config']:
            raise ValueError('Committed recovery config differs')
        snapshots.finalize(cap)
        policy = write_policy(folder/'policy.json',target,checkpoint/'current.npz')
        ready = dict(policy=file_ref(policy),checkpoint=str(checkpoint),preflop_export=file_ref(target),
                     files=read(folder/'current.json')['files'],diagnostics_sha256=digest(folder/'diagnostics.jsonl'),
                     start_sha256=digest(folder/'start-verification.json'),mixture=file_ref(cap/'mixture.json'))
        atomic_json(folder/'ready.json',ready)
        mark('complete')
    except BaseException as error:
        mark('failed',error=f'{type(error).__name__}: {error}')
        raise


def verify_stage(plan):
    assets, summaries, endpoints, mixtures = {}, {}, {}, {}
    for seed in SEEDS:
        key = str(seed)
        folder = STUDY/ARM/f'seed-{seed}'
        cap = STUDY/'snapshots'/f'seed-{seed}'
        ready = read(folder/'ready.json')
        start = read(folder/'start-verification.json')
        ptr = read(folder/'current.json')
        verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256'],
                      folder/'start-verification.json':ready['start_sha256'],
                      **{ref['path']:ref['sha256'] for ref in (ready['policy'],ready['preflop_export'],ready['mixture'])},
                      **{Path(ready['checkpoint'])/filename:sha for filename,sha in ready['files'].items()}})
        rows = control.read_rows(folder/'diagnostics.jsonl')
        previous = start['weights']
        control_file = Path(plan['control_logs'][key])
        first_control = control.read_rows(control_file)[0]
        if (start!=read(control_file.parent/'start-verification.json') or ptr['iteration']!=504
                or read(folder/'status.json')['status']!='complete' or ptr['files']!=ready['files']
                or [r['iteration'] for r in rows]!=list(range(481,505))
                or rows[0]['reference_collection']!=first_control['reference_collection']):
            raise ValueError('Incomplete stage or different initialization/reference from control')
        hist_visits = {k:0 for k in (0,*population.STAGES)}
        for r in rows:
            t = r['iteration']
            seat = (t-1)%6
            slots = population.assignments(seed,t,256,seat)
            d = r['opponent_population']
            counts = {str(k):int(np.count_nonzero(slots==k)) for k in hist_visits}
            if (r['traverser']!=seat or not np.all(slots[:,seat]==-1)
                    or d['assignment_sha256']!=sha256(slots.tobytes()).hexdigest()
                    or d['opponent_seat_assignments']!=counts or sum(counts.values())!=5*256
                    or min(d['postflop_node_visits'].values())<=0 or r['traversals']!=256 or r['general_steps']!=768
                    or r['fit_draws']!=768*512 or min(r['fit_draws_by_street'])<=0 or sum(r['fit_draws_by_street'])!=768*512
                    or r['preflop_steps']!=0 or r['opening_steps']!=0 or r['buffer_rows']!=[100000]*6
                    or r['root_rows']!=676 or r['reference_collection']['hands']!=512 or r['reference_collection']['rows']!=100000
                    or not np.isfinite(r['main_loss']) or r['new_regret_rows']<=0 or r['learning_rate']!=.00025
                    or r['input_weights']!=previous or r['output_weights']['general']==previous['general']
                    or r['output_weights']['preflop']!=start['weights']['preflop'] or r['frozen_preflop_max_probability_error']!=0.):
                raise ValueError(f'Unverified update {seed}/{t}')
            previous = r['output_weights']
            for slot in hist_visits:
                hist_visits[slot] += d['postflop_node_visits'][str(slot)]
        mix = snapshots.manifest(cap)
        if (mix!=read(cap/'mixture.json') or [r['iteration'] for r in mix['postflop']]!=control.CAPTURES
                or len({r['sha256'] for r in mix['postflop']})!=len(control.CAPTURES)
                or mix['postflop'][-1]['sha256']!=ptr['files']['current.npz']):
            raise ValueError('Incomplete or mismatched snapshot policies')
        summaries[key] = dict(start=start,iteration=504,updates=len(rows),final_weights=previous,checkpoints=control.CAPTURES,
                              traversals=sum(r['traversals'] for r in rows),general_steps=sum(r['general_steps'] for r in rows),
                              new_regret_rows=sum(r['new_regret_rows'] for r in rows),all_preflop_unchanged=True,
                              postflop_opponent_node_visits=hist_visits,final_average=rows[-1]['average_fit'])
        endpoints[key],mixtures[key],assets[key] = ready['policy'],ready['mixture'],ready
    return dict(training_complete=True,completed_utc=utc(),iteration=504,warm_start_weight_chain_verified=True,
                policies={'specialist480':plan['parent_policies'],ARM:endpoints,'poolMixture504':mixtures},
                arms={ARM:assets},seed_verification=summaries)


def read_bound():
    path = STUDY/'training-plan.json'
    verify_files({path:read(STUDY/'training-bound.json')['plan_sha256']})
    plan = read(path)
    verify_files(plan['frozen_files'])
    return plan


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior.check_workers()
        path = STUDY/'training-plan.json'
        if path.exists():
            raise FileExistsError(path)
        plan = bind_training()
        atomic_json(path,plan)
        atomic_json(STUDY/'training-bound.json',dict(utc=utc(),plan_sha256=digest(path)))
        try:
            for s in SEEDS:
                snapshots.register(STUDY/'snapshots'/f'seed-{s}',STUDY/ARM/f'seed-{s}',s,ARM,
                                   plan['parent_policies'][str(s)]['path'],control.CAPTURES,SOURCE)
            commands = {f'{ARM}-seed-{s}':[str(PYTHON),str(SCRIPT),'worker',str(s)] for s in SEEDS}
            confirmation.run_jobs(STUDY,commands,len(SEEDS),'training')
            atomic_json(STUDY/'stage-ready.json',verify_stage(read_bound()))
        except BaseException as exc:
            atomic_json(STUDY/'training-failure.json',dict(utc=utc(),error=f'{type(exc).__name__}: {exc}',automatic_retry=False))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['preflight','run','worker'])
    parser.add_argument('seed',nargs='?',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=='worker' and args.seed:
        train(args.seed)
    elif args.action=='preflight' and args.seed is None:
        print(list(bind_training()['historical_policies']))
    elif args.action=='run' and args.seed is None:
        run()
    else:
        parser.error('Expected a seed only for worker')
