"""Continue a restored latest policy/replay without retaining policy history."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

from compact_lightweight_backup import load_state, save_state, validate_state
from fullhouse_v3_common import ROOT, atomic_json, digest


def load_current(campaign, seed):
    folder = Path(campaign)/f'seed-{seed}'; pointer = json.loads((folder/'current.json').read_text())
    name = pointer['directory']
    if not re.fullmatch(r'iter-\d{5,}', name):
        raise ValueError('Unsafe checkpoint directory')
    path = folder/'checkpoints'/name
    if path.is_symlink():
        raise ValueError('Checkpoint cannot be a symlink')
    for filename, sha in pointer['files'].items():
        if filename not in ('state.json','arrays.npz','current.npz') or digest(path/filename) != sha:
            raise ValueError('Committed checkpoint bytes changed')
    if set(pointer['files']) != {'state.json','arrays.npz','current.npz'}:
        raise ValueError('Incomplete committed checkpoint')
    state = load_state(path); validate_state(state, seed, pointer['iteration'])
    return state, path


def prune_checkpoints(folder, keep):
    """Keep the parent, declared stages and latest committed recovery only."""
    for path in (Path(folder)/'checkpoints').iterdir():
        if re.fullmatch(r'iter-\d{5,}', path.name) and int(path.name[5:]) not in keep:
            if path.is_symlink():
                raise ValueError('Unexpected checkpoint symlink')
            shutil.rmtree(path)


def commit_checkpoint(campaign, seed, state, exporter, keep):
    folder = Path(campaign)/f'seed-{seed}'; iteration = state['iteration']
    checkpoints = folder/'checkpoints'; checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints/f'iter-{iteration:05d}'
    if target.exists():
        raise FileExistsError('A committed checkpoint is immutable')
    validate_state(state, seed, iteration)
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=checkpoints) as temp:
        payload = Path(temp)/'state'; save_state(state, payload)
        exporter(payload/'current.npz')
        files = {name:digest(payload/name) for name in ('state.json','arrays.npz','current.npz')}
        payload.rename(target)
    export = folder/f'current_{iteration:05d}.npz'
    if export.exists():
        raise FileExistsError('An immutable policy already exists')
    shutil.copyfile(target/'current.npz', export)
    atomic_json(folder/'current.json', dict(iteration=iteration, directory=target.name, files=files))
    # Intermediate exported policies are not needed by this current-only plan.
    for path in folder.glob('current_*.npz'):
        if int(path.stem.rsplit('_',1)[1]) not in keep | {iteration}:
            path.unlink()
    prune_checkpoints(folder, keep | {iteration})
    return files


def run(campaign, seed, through):
    import numpy as np
    import torch
    campaign=Path(campaign).resolve(); cfg=json.loads((campaign/'config.json').read_text())
    manifest=json.loads((campaign/'source-manifest.json').read_text())
    fixture=cfg.get('verification_fixture',False)
    if seed not in cfg['seeds'] or through not in cfg['stages']:
        raise ValueError('Undeclared training seed or stopping stage')
    if fixture:
        for key,maximum in [('iterations',12),('traversals',8),('main_steps',8),('opening_steps',8),
                            ('opening_max_steps',8),('buffer_capacity',5000),('root_repetitions',1),('root_repetitions_first',1)]:
            if cfg[key]>maximum:raise ValueError('Verification fixture exceeds bound')
    else:
        if (cfg['initial_iteration']!=240 or cfg['stages']!=[300,360] or cfg['iterations']!=360
                or cfg['buffer_capacity']!=100000 or cfg['seeds']!=[342,343,344]):
            raise ValueError('The authorized bounded240-to360 plan changed')
        gate=json.loads((ROOT/cfg['gate_run']/'gates.json').read_text())
        deployment=json.loads((ROOT/cfg['gate_run']/'deployment-verification.json').read_text())
        if cfg['method'] not in gate['eligible_methods'] or not deployment['passed'] or deployment['runtime_sha256']!=digest(ROOT/'scripts/fullhouse_v4_runtime.py'):
            raise ValueError('Frozen learning/deployment gates changed')
    for name,sha in manifest['files'].items():
        if digest(campaign/name)!=sha:raise ValueError('Frozen campaign source changed')
    for name,sha in manifest.get('root_files',{}).items():
        if digest(name)!=sha:raise ValueError('Frozen bounded-training helper changed')
    if digest(Path(__file__))!=manifest['driver_sha256']:
        raise ValueError('Bounded worker source changed')
    folder=campaign/f'seed-{seed}'; lock=(folder/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path[:0]=[str(campaign/'source'),str(campaign/'tools')]
    from training import r1_support as r1
    from training.buffer import ReservoirBuffer
    from training.traverse_fast import run_traversals_fast
    from bot.features import load_preflop_equity
    from compact_regret import CompactRegretNet,install_policy,fit_compact,fit_opening,export_compact
    from root_regret_replay import RootReplay,collect_root_replay
    install_policy();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    state,_=load_current(campaign,seed);start=state['iteration']
    if state['config']!=cfg or state['source_manifest']!=manifest or not start<through<=cfg['iterations']:
        raise ValueError('Checkpoint configuration/source/stopping point mismatch')
    buffers=[ReservoirBuffer(cfg['buffer_capacity'],51) for _ in range(6)]
    for b,s in zip(buffers,state['buffers']):b.load_state_dict(s)
    root=RootReplay();root.load_state_dict(state['root_replay'])
    net=CompactRegretNet();net.load_state_dict(state['net']);net.eval()
    rng=np.random.default_rng();r1.restore_rng(state['rng'],rng)
    r1.WEIGHT_HORIZON=cfg['iterations'];r1.AVERAGE_SAMPLER='importance'
    del state
    equity=load_preflop_equity();began=time.monotonic()
    keep=set(cfg.get('evaluation_iterations',[]))|{cfg['initial_iteration']}
    try:
        for iteration in range(start+1,through+1):
            if time.monotonic()-began>cfg['seconds_per_training_worker']:
                raise TimeoutError('Bounded training stage exhausted its time budget')
            started=time.monotonic();traverser=(iteration-1)%6
            run_traversals_fast(cfg['traversals'],6,traverser,net,buffers[traverser],iteration,rng,equity,
                opponent_exploration=0.,strategy_buffer=None,randomize_stacks=False)
            repeats=cfg['root_repetitions_first'] if iteration==1 else cfg['root_repetitions']
            collect_root_replay(net,rng,root,iteration,repeats)
            net=CompactRegretNet()
            fit=fit_compact(net,buffers,cfg['main_steps'],cfg['batch_size'],root_replay=root,
                            opening_steps=cfg['opening_steps'],lr=cfg['learning_rate'])
            x,y,mass=root.means();opening_steps=cfg['opening_steps']
            expected=np.maximum(y,0)*x[:,46:51];total=expected.sum(1,keepdims=True);legal=x[:,46:51].astype(float)
            expected=np.divide(expected,total,out=legal/legal.sum(1,keepdims=True),where=total>0)
            def measure():
                pred=net.get_strategy(torch.from_numpy(x),torch.from_numpy(x[:,46:51])).numpy()
                tv=np.abs(pred-expected).sum(1)/2;aa=int(np.flatnonzero((x[:,10]==1)&(x[:,7]>.99))[0])
                return dict(weighted_tv=float(np.average(tv,weights=mass)),aa_fold=float(pred[aa,0]),
                    aa_target_fold=float(expected[aa,0]),aa_regrets=y[aa].tolist())
            quality=measure()
            while quality['weighted_tv']>cfg['max_opening_tv'] and opening_steps<cfg['opening_max_steps']:
                extra=min(1024,cfg['opening_max_steps']-opening_steps)
                fit_opening(net,(x,y,mass),extra,cfg['learning_rate']);opening_steps+=extra;quality=measure()
            from run_compact_learning_curve import assert_finite
            row=dict(iteration=iteration,seed=seed,seconds=time.monotonic()-started,
                additional_selfplay_iterations=iteration-cfg['initial_iteration'],buffer_rows=[len(b) for b in buffers],
                root=root.diagnostics(),opening_steps=opening_steps,quality=quality,main_loss=fit['main_loss'])
            assert_finite(row)
            if quality['weighted_tv']>cfg['max_opening_tv'] and not fixture:
                raise RuntimeError('Opening fitting quality failed; no automatic retry')
            with (folder/'diagnostics.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row),flush=True)
            if iteration%cfg['checkpoint_every']==0 or iteration==through:
                state=dict(iteration=iteration,seed=seed,config=cfg,source_manifest=manifest,net=net.state_dict(),
                    buffers=[b.state_dict() for b in buffers],root_replay=root.state_dict(),rng=r1.pack_rng(rng))
                commit_checkpoint(campaign,seed,state,lambda p:export_compact(net,p),keep)
                del state
                atomic_json(folder/'status.json',dict(iteration=iteration,through=through,
                    status='complete' if iteration==through else 'running',quality=quality,root=root.diagnostics()))
    except BaseException as error:
        pointer=json.loads((folder/'current.json').read_text())
        atomic_json(folder/'status.json',dict(status='failed',iteration=pointer['iteration'],through=through,error=str(error)))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--through',type=int,required=True)
    a=p.parse_args();run(a.run_dir,a.seed,a.through)
