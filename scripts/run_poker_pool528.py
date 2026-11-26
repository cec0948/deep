"""Continue the archived diverse 504 state to 528, without changing the sample or fit budget."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

import poker_capture_postflop_snapshots as snapshots
import poker_opponent_population as population
import run_poker_opponent_diversity_train as parent_run
import run_poker_preflop_confirmation as confirmation
import train_poker_fixed_preflop_window as fixed
from compact_lightweight_backup import equal_state
from controlled_poker_refit import assert_finite_arrays, read, utc, verify_files
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest, process_start
from poker_preflop_specialist import file_ref, write_policy
from poker_strategy_average import fit_average
from run_poker_preflop_online import StreetNetworks, install_street_training_policy
from run_poker_preflop_pilot import CAMPAIGN, SEEDS, SOURCE
from snapshot_poker_preflop_online import verify_export
from train_poker_average_continuation import (CHECKPOINT_FORMAT, collect_game_reference, commit_checkpoint,
                                             load_checkpoint, record_average_quality)
from train_poker_warm_preflop import weights


PARENT = parent_run.STUDY
STUDY = PARENT.parent / 'pool-extension-528-20261126'
CONFIG = ROOT / 'configs/poker-pool-extension-528-20261126.json'
SCRIPT = Path(__file__).resolve()
PARENT_STAGE = '8a59fdc2480bde76d882f134e76501df7e2feb0b6eefad797c87a7754adc4cf9'
PARENT_ARCHIVE = '129ca3bc40648d81f6ae59f60ff9eec73fd7e29906cdf92ef162aaaaec7c7352'
ARM = 'pool528'
INITIAL, FINAL = 504, 528
CAPTURES = [510, 516, 522, 528]
BUDGET = dict(learning_rate=.00025, traversals=256, general_steps=768, main_steps=768, batch_size=512,
              root_repetitions=4, reference_hands=512, buffer_capacity=100000, strategy_capacity=100000,
              average_fit_steps=8192, average_batch=512, average_width=128, opening_steps=0, preflop_steps=0)


def bind_plan():
    registration = read(STUDY/'registration.json')
    verify_files({CONFIG: registration['config_sha256'], PARENT/'stage-ready.json': PARENT_STAGE})
    if registration['source_stage_sha256'] != PARENT_STAGE:
        raise ValueError('Registration differs from the source stage')
    prior_plan = parent_run.read_bound()
    stage = read(PARENT/'stage-ready.json')
    frozen = dict(prior_plan['frozen_files'])
    frozen.update(registration['frozen_files'])
    hist, sources = population.historical_inputs(SEEDS)
    if hist != prior_plan['historical_policies'] or not stage['training_complete'] or stage['iteration'] != INITIAL:
        raise ValueError('Source stage or historical opponent pool changed')
    frozen.update(sources)
    assets = stage['arms']['pool504']
    for seed in SEEDS:
        key = str(seed)
        folder = PARENT/'pool504'/f'seed-{seed}'
        checkpoint = folder/'checkpoints'/'iter-00504'
        ptr = read(folder/'current.json')
        ready = assets[key]
        if (read(folder/'ready.json') != ready or ptr != dict(iteration=INITIAL,directory=checkpoint.name,files=ready['files'])
                or Path(ready['checkpoint']) != checkpoint or ready['policy'] != stage['policies']['pool504'][key]):
            raise ValueError('Archived worker pointer differs from the completed stage')
        frozen.update({str(checkpoint/name):h for name,h in ready['files'].items()})
        for ref in (ready['policy'], ready['preflop_export'], ready['mixture'], stage['policies']['specialist480'][key]):
            frozen[ref['path']] = ref['sha256']
        frozen[str(folder/'diagnostics.jsonl')] = ready['diagnostics_sha256']
        frozen[str(folder/'current.json')] = digest(folder/'current.json')
        frozen[str(folder/'ready.json')] = digest(folder/'ready.json')
    for path in (SCRIPT, CONFIG, STUDY/'registration.json', PARENT/'training-plan.json', PARENT/'stage-ready.json',
                 Path(fixed.__file__), Path(population.__file__), Path(snapshots.__file__)):
        frozen[str(path)] = digest(path)
    verify_files(frozen)
    return dict(registered_utc=utc(), arm=ARM, initial_iteration=INITIAL, final_iteration=FINAL, seeds=SEEDS,
                capture_iterations=CAPTURES, budget=BUDGET, parent_archive_sha256=PARENT_ARCHIVE,
                historical_policies=hist, parent_policies=stage['policies']['pool504'],
                baseline_policies=stage['policies']['specialist480'], parent_stage_sha256=PARENT_STAGE,
                source_checkpoint=assets, frozen_files=frozen)


def read_bound():
    path = STUDY/'training-plan.json'
    verify_files({path:read(STUDY/'training-bound.json')['plan_sha256']})
    plan = read(path)
    verify_files(plan['frozen_files'])
    return plan


def connect(seed, plan):
    sys.path[:0] = [str(SOURCE), str(CAMPAIGN/'tools')]
    from compact_regret import ARRAY_MAP, CompactRegretNet, install_policy
    from training import r1_support as r1

    key = str(seed)
    state, checkpoint = load_checkpoint(PARENT/'pool504'/f'seed-{seed}')
    cfg = state['config']
    old = plan['source_checkpoint'][key]
    if (state['iteration'] != INITIAL or state['seed'] != seed or state['averaging_window'] != [361, INITIAL]
            or cfg['training_arm'] != 'pool504' or cfg['historical_policies'] != plan['historical_policies'][key]
            or any(cfg[k] != value for k,value in BUDGET.items()) or state['average_state'] is None):
        raise ValueError('Incorrect full-state source or changed training settings')
    policy = read(old['policy']['path'])
    if policy['postflop'] != file_ref(checkpoint/'current.npz') or policy['preflop'] != old['preflop_export']:
        raise ValueError('Starting weights differ from the played 504 policy')
    networks = StreetNetworks(CompactRegretNet().eval(), CompactRegretNet().eval())
    for model, source, asset in ((networks.general, state['net'], policy['postflop']),
                                 (networks.specialist, state['specialist_net'], policy['preflop'])):
        model.load_state_dict(source, strict=True)
        if not equal_state(source,model.state_dict()):
            raise ValueError('Model does not match serialized full training state')
        verify_export(asset['path'],model.state_dict(),ARRAY_MAP)
        assert_finite_arrays(model)
    install_policy()
    install_street_training_policy()
    buffers, root, strategy, rng, reference_rng = fixed.restore_replay(state)
    if (set(state['rng']) != {'python','numpy','generator','torch'} or len(buffers) != 6
            or any(len(b) != 100000 for b in buffers) or strategy.size != 100000):
        raise ValueError('Expected six full regret buffers, full strategy memory and complete RNG state')
    historical = population.load_historical(networks, plan['historical_policies'][key])
    probes = fixed.preflop_probes(buffers, root)
    expected = fixed.played_probabilities(networks,probes)
    if not equal_state(r1.pack_rng(rng),state['rng']) or reference_rng.bit_generator.state != state['reference_rng']:
        raise ValueError('Connection consumed the parent RNG stream')
    last = read(PARENT/'stage-ready.json')['seed_verification'][key]['final_weights']
    if weights(networks) != last:
        raise ValueError('Final parent network fingerprints do not match the continuation')
    proof = dict(seed=seed, iteration=INITIAL, weights=last, deployed_policy_sha256=old['policy']['sha256'],
                 street_weight_sha256={k:policy[k]['sha256'] for k in ('preflop','postflop')},
                 checkpoint_files=old['files'], exact_loaded_state=True,
                 regret_replay_rows=[len(b) for b in buffers], regret_seen=[b.total_seen for b in buffers],
                 root_class_counts=[int(n) for n in root.state_dict()['count']],
                 reference_rows=strategy.size, reference_seen=strategy.total_seen,
                 restored_rng=['python','numpy','generator','torch','reference_generator','strategy_reservoir'],
                 prior_average_window=state['averaging_window'], historical_policies=plan['historical_policies'][key],
                 restored_root_regret_strategy_rng_byte_verified=True, frozen_preflop_probes=len(probes))
    return state, networks, historical, buffers, root, strategy, rng, reference_rng, probes, expected, proof


def train(seed):
    owner = read(STUDY/'training-status.json')
    if owner['pid'] != os.getppid() or process_start(owner['pid']) != owner['process_start']:
        raise RuntimeError('Worker not owned by live registered supervisor')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read_bound()
    key = str(seed)
    folder = STUDY/ARM/f'seed-{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    cap = STUDY/'snapshots'/f'seed-{seed}'
    token = process_start(os.getpid())
    iteration = saved = INITIAL

    def mark(status, **fields):
        atomic_json(folder/'status.json',dict(status=status,observed_utc=utc(),pid=os.getpid(),process_start=token,
                    arm=ARM,seed=seed,iteration=iteration,last_committed_iteration=saved,**fields))

    try:
        mark('restoring')
        state, networks, hist, buffers, root, strategy, rng, ref_rng, probes, expected, proof = connect(seed,plan)
        from bot.features import load_preflop_equity
        from compact_regret import ARRAY_MAP, export_compact
        from training import r1_support as r1

        cfg = dict(state['config'], training_arm=ARM,initial_iteration=INITIAL,iterations=FINAL,stages=[FINAL],
                   resume_iterations={str(s):INITIAL for s in SEEDS})
        manifest = state['source_manifest']
        del state
        atomic_json(folder/'start-verification.json',proof)
        equity = load_preflop_equity()
        r1.WEIGHT_HORIZON, r1.AVERAGE_SAMPLER = cfg['weight_horizon'],'importance'
        opening = {k:v.clone() for k,v in networks.general.opening.state_dict().items()}
        opening_scale = networks.general.opening_scale.clone()
        previous = proof['weights']
        began = time.monotonic()
        for iteration in range(INITIAL+1, FINAL+1):
            tick = time.monotonic()
            if tick-began > cfg['training_seconds_per_worker']:
                raise TimeoutError('Registered training time budget exhausted')
            mark('training')
            entering = weights(networks)
            if previous != entering:
                raise ValueError('Previous-round weight chain changed')
            learner_rng = r1.pack_rng(rng)
            observation = collect_game_reference(networks,strategy,ref_rng,iteration,cfg['reference_hands'],equity)
            if not equal_state(learner_rng,r1.pack_rng(rng)):
                raise ValueError('Reference collection consumed the training RNG')
            row = population.pool_update(networks,buffers,root,rng,iteration,cfg,equity,seed,hist)
            previous = weights(networks)
            if (previous['general'] == entering['general'] or previous['preflop'] != proof['weights']['preflop']
                    or not equal_state(opening,networks.general.opening.state_dict())
                    or not torch.equal(opening_scale,networks.general.opening_scale)
                    or not np.array_equal(expected,fixed.played_probabilities(networks,probes))):
                raise ValueError('Postflop update or frozen preflop verification failed')
            row.update(seed=seed,iteration=iteration,input_weights=entering,output_weights=previous,
                       learning_rate=cfg['learning_rate'],buffer_rows=[len(b) for b in buffers],
                       reference_collection=observation,frozen_preflop_probes=len(probes),
                       frozen_preflop_max_probability_error=0.)
            average = report = None
            if iteration == FINAL:
                mark('fitting_window_average')
                average, report = fit_average(strategy,seed=seed*100000+iteration,steps=cfg['average_fit_steps'],
                                              batch_size=cfg['average_batch'],width=cfg['average_width'])
                report = record_average_quality(report,cfg['average_max_replay_tv'])
            if iteration in CAPTURES:
                state = dict(checkpoint_format=CHECKPOINT_FORMAT,seed=seed,iteration=iteration,config=cfg,
                             source_manifest=manifest,net=networks.general.state_dict(),specialist_net=networks.specialist.state_dict(),
                             buffers=[b.state_dict() for b in buffers],root_replay=root.state_dict(),rng=r1.pack_rng(rng),
                             strategy_replay=strategy.state_dict(),reference_rng=ref_rng.bit_generator.state,
                             average_fit=report,average_state=None if average is None else average.state_dict(),
                             averaging_window=[361,iteration],historical_mixture_available=False)
                commit_checkpoint(folder,state,lambda p:export_compact(networks.general,p),average)
                saved = iteration
                del state
            row.update(checkpoint_iteration=saved,average_fit=report,seconds=time.monotonic()-tick)
            with (folder/'diagnostics.jsonl').open('a') as log:
                log.write(json.dumps(row,allow_nan=False)+'\n')
                if iteration in CAPTURES:
                    log.flush()
                    os.fsync(log.fileno())
            if iteration in CAPTURES:
                snapshots.capture(cap,iteration)

        state, ckpt = load_checkpoint(folder)
        pre = plan['source_checkpoint'][key]['preflop_export']
        target = folder/'preflop.npz'
        shutil.copyfile(pre['path'],target)
        verify_files({target:pre['sha256']})
        verify_export(target,state['specialist_net'],ARRAY_MAP)
        verify_export(ckpt/'current.npz',state['net'],ARRAY_MAP)
        if (state['config'] != cfg or state['averaging_window'] != [361,FINAL] or state['average_state'] is None
                or not equal_state(state['rng'],r1.pack_rng(rng))):
            raise ValueError('Invalid final checkpoint or RNG divergence')
        snapshots.finalize(cap)
        policy = write_policy(folder/'policy.json',target,ckpt/'current.npz')
        atomic_json(folder/'ready.json',dict(policy=file_ref(policy),checkpoint=str(ckpt),preflop_export=file_ref(target),
                    files=read(folder/'current.json')['files'],diagnostics_sha256=digest(folder/'diagnostics.jsonl'),
                    start_sha256=digest(folder/'start-verification.json'),mixture=file_ref(cap/'mixture.json'),
                    average=file_ref(ckpt/'average.npz')))
        mark('complete')
    except BaseException as exc:
        mark('failed',error=f'{type(exc).__name__}: {exc}')
        raise


def verify_stage(plan):
    arms, summaries = {}, {}
    policies = {'specialist480':plan['baseline_policies'],'current528':{},'mixture528':{},'average528':{}}
    for seed in SEEDS:
        key = str(seed)
        folder = STUDY/ARM/f'seed-{seed}'
        cap = STUDY/'snapshots'/f'seed-{seed}'
        ready, start = read(folder/'ready.json'), read(folder/'start-verification.json')
        rows = parent_run.control.read_rows(folder/'diagnostics.jsonl')
        ptr = read(folder/'current.json')
        verify_files({folder/'diagnostics.jsonl':ready['diagnostics_sha256'],folder/'start-verification.json':ready['start_sha256'],
                      **{ref['path']:ref['sha256'] for ref in (ready['policy'],ready['preflop_export'],ready['mixture'],ready['average'])},
                      **{Path(ready['checkpoint'])/k:h for k,h in ready['files'].items()}})
        if (read(folder/'status.json')['status']!='complete' or ptr['files']!=ready['files'] or ptr['iteration']!=FINAL
                or [r['iteration'] for r in rows]!=list(range(INITIAL+1,FINAL+1))
                or start['checkpoint_files']!=plan['source_checkpoint'][key]['files']):
            raise ValueError(f'Missing trained round or mismatched worker identity: {seed}')
        previous = start['weights']
        assigned = {str(n):0 for n in (0,*population.STAGES)}
        final_seen = list(start['regret_seen'])
        for r in rows:
            t = r['iteration']
            seat = (t-1)%6
            slots = population.assignments(seed,t,BUDGET['traversals'],seat)
            pool = r['opponent_population']
            counts = {k:int(np.count_nonzero(slots==int(k))) for k in assigned}
            if (r['input_weights']!=previous or r['output_weights']['general']==previous['general']
                    or r['output_weights']['preflop']!=start['weights']['preflop'] or r['frozen_preflop_max_probability_error']!=0
                    or r['traverser']!=seat or r['traversals']!=256 or r['general_steps']!=768 or r['fit_draws']!=768*512
                    or sum(r['fit_draws_by_street'])!=r['fit_draws'] or min(r['fit_draws_by_street'])<=0
                    or r['preflop_steps']!=0 or r['opening_steps']!=0 or r['root_rows']!=676
                    or r['reference_collection']['hands']!=512 or r['reference_collection']['rows']!=100000
                    or r['buffer_rows']!=[100000]*6 or not np.isfinite(r['main_loss']) or r['new_regret_rows']<=0
                    or r['learning_rate']!=.00025 or counts!=pool['opponent_seat_assignments']
                    or sha256(slots.tobytes()).hexdigest()!=pool['assignment_sha256']
                    or min(pool['postflop_node_visits'].values())<=0):
                raise ValueError(f'Failed training checks: {seed}/{t}')
            for k in assigned:
                assigned[k] += counts[k]
            final_seen[seat] += r['new_regret_rows']
            previous = r['output_weights']
        final, checkpoint = load_checkpoint(folder)
        if ([b['total_seen'] for b in final['buffers']] != final_seen
                or final['strategy_replay']['total_seen']!=rows[-1]['reference_collection']['total_seen']
                or final['strategy_replay']['total_seen']<=start['reference_seen']
                or final['average_fit']!=rows[-1]['average_fit']
                or not np.array_equal(final['root_replay']['count'], np.array(start['root_class_counts'])+4*len(rows))):
            raise ValueError('Saved sufficient statistics/replay differ from actual sample counts')
        del final
        mixture = snapshots.manifest(cap)
        if ([slot['iteration'] for slot in mixture['postflop']]!=CAPTURES
                or mixture['postflop'][-1]['sha256']!=ptr['files']['current.npz']
                or len({slot['sha256'] for slot in mixture['postflop']}) != len(CAPTURES)):
            raise ValueError('Missing or mismatched scheduled snapshots')
        for label, field in [('current528','policy'),('mixture528','mixture'),('average528','average')]:
            policies[label][key] = ready[field]
        summaries[key] = dict(start=start,iteration=FINAL,updates=len(rows),final_weights=previous,checkpoints=CAPTURES,
                              traversals=sum(r['traversals'] for r in rows),general_steps=sum(r['general_steps'] for r in rows),
                              new_regret_rows=sum(r['new_regret_rows'] for r in rows),regret_seen=final_seen,
                              fit_draws=sum(r['fit_draws'] for r in rows),all_preflop_unchanged=True,
                              historical_and_current_seat_assignments=assigned,final_average=rows[-1]['average_fit'])
        arms[key] = ready
    return dict(training_complete=True,completed_utc=utc(),iteration=FINAL,warm_start_weight_chain_verified=True,
                parent_stage_sha256=PARENT_STAGE,policies=policies,arms={ARM:arms},seed_verification=summaries)


def run():
    with (confirmation.FOLLOWUP/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        parent_run.prior.check_workers()
        with (STUDY/'training-launch.json').open('x') as log:
            json.dump(dict(utc=utc(),pid=os.getpid(),process_start=process_start(os.getpid())),log)
        try:
            plan = bind_plan()
            path = STUDY/'training-plan.json'
            atomic_json(path,plan)
            atomic_json(STUDY/'training-bound.json',dict(utc=utc(),plan_sha256=digest(path)))
            for seed in SEEDS:
                snapshots.register(STUDY/'snapshots'/f'seed-{seed}',STUDY/ARM/f'seed-{seed}',seed,ARM,
                                   plan['parent_policies'][str(seed)]['path'],CAPTURES,SOURCE)
            commands = {f'{ARM}-seed-{s}':[str(PYTHON),str(SCRIPT),'worker',str(s)] for s in SEEDS}
            confirmation.run_jobs(STUDY,commands,len(SEEDS),'training')
            atomic_json(STUDY/'stage-ready.json',verify_stage(read_bound()))
        except BaseException as exc:
            atomic_json(STUDY/'training-failure.json',dict(utc=utc(),error=f'{type(exc).__name__}: {exc}',automatic_retry=False))
            raise


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode',choices=['preflight','run','worker'])
    cli.add_argument('seed',type=int,choices=SEEDS,nargs='?')
    args = cli.parse_args()
    if args.mode=='preflight':
        # A new process per seed installs the same one-off routing factory as production.
        if args.seed is None:
            cli.error('Specify a seed to exercise full-state restoration')
        torch.set_num_threads(1)
        plan = bind_plan()
        _, *_, proof = connect(args.seed,plan)
        print(json.dumps(proof,allow_nan=False))
    elif args.mode=='worker' and args.seed is not None:
        train(args.seed)
    elif args.mode=='run' and args.seed is None:
        run()
    else:
        cli.error('Unexpected seed')
