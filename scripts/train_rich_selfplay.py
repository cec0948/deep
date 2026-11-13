"""Bounded new-family rich-input residual self-play from a frozen360 teacher.

No old registry is overwritten. Fresh rounds are numbered1..96 separately from
the parent's360 rounds. Latest recovery plus declared stages are retained.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import sys
import tempfile
import time

import numpy as np
import torch

from compact_lightweight_backup import equal_state, load_state, save_state
from fullhouse_v3_common import ROOT, atomic_json, digest
from rich_regret_model import FORMAT, RichRegretModel


CHECKPOINT_FORMAT = 'rich-residual-recovery-v1'


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_frozen(config):
    if not config.get('frozen_files'):
        raise ValueError('Frozen source and teacher manifest required')
    for name, sha in config['frozen_files'].items():
        if digest(name) != sha:
            raise ValueError('Frozen new-family dependency changed: '+name)


def validate_config(config, seed, through):
    fixture = config.get('verification_fixture') is True
    if seed not in config['seeds'] or through not in config['stages']:
        raise ValueError('Undeclared seed or stopping stage')
    if fixture:
        limits = dict(iterations=12,traversals=8,buffer_capacity=5000,main_steps=16,batch_size=128)
        if any(type(config.get(k)) is not int or not 1 <= config[k] <= v for k,v in limits.items()):
            raise ValueError('Fixture exceeds its bounded verification budget')
    elif any(config.get(k) != v for k,v in dict(seeds=[342,343,344],stages=[12,48,96],
            iterations=96,traversals=64,buffer_capacity=5000,main_steps=256,batch_size=128,
            checkpoint_every=6).items()):
        raise ValueError('The predeclared rich-input study changed')
    if not 0 < config.get('learning_rate',0) <= .001:
        raise ValueError('Invalid learning rate')
    if not 0 < config.get('seconds_per_training_worker',0) <= 21600:
        raise ValueError('Worker time budget must be at most6hours')
    if not fixture and config.get('prerequisites_passed') is not True:
        raise ValueError('New neural-information prerequisites must pass before6player learning')
    if config.get('parent_iteration',360) != 360:
        raise ValueError('Declared existing360 teacher required')
    if config.get('fitting_max_prior_mse_ratio',1.05) != 1.05:
        raise ValueError('Predeclared fitting consistency criterion changed')
    check_frozen(config)


def validate_fitting_report(report, config):
    """A retained-replay consistency check, never a strength/holdout gate."""
    if report.get('eligible_rows') == 0 and report.get('steps') == 0:
        return
    after = report.get('replay_mse_after_bb2')
    prior = report.get('teacher_prior_mse_bb2')
    if (type(after) not in (int,float) or type(prior) not in (int,float)
            or not math.isfinite(after) or not math.isfinite(prior)
            or after < 0 or prior < 0
            or after > config.get('fitting_max_prior_mse_ratio',1.05)*prior+1e-6):
        raise ValueError('Retained-replay fitting consistency failed against the zero-residual teacher prior; no automatic retry')


def teacher_spec(campaign, config, seed):
    row = config['teachers'][str(seed)]
    path = Path(row['path'])
    if not path.is_absolute():
        path = Path(campaign)/path
    if not path.is_file() or digest(path) != row['sha256']:
        raise ValueError('Existing teacher file does not match its verified parent')
    return path


def checkpoint_state(model, buffers, config, seed, iteration, rng):
    return dict(format=CHECKPOINT_FORMAT, seed=seed, iteration=iteration,
                parent_iteration=360, family='rich-regret-residual', config=config,
                model=model.state_dict(), buffers=[b.state_dict() for b in buffers],
                rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                         generator=rng.bit_generator.state,torch=torch.get_rng_state()),
                teacher_knowledge_inherited=True, parent_rng_inherited=False,
                legacy_exact_resume=False, average_policy_available=False)


def validate_checkpoint(state, config, seed, dimension):
    if (state.get('format') != CHECKPOINT_FORMAT or state.get('seed') != seed
            or state.get('config') != config or not 0 <= state.get('iteration',-1) <= config['iterations']
            or state.get('parent_iteration') != 360 or len(state.get('buffers',[])) != 6):
        raise ValueError('Rich checkpoint family/configuration mismatch')
    for b in state['buffers']:
        count = b['count']
        if (type(count) is not int or not 0 <= count <= config['buffer_capacity']
                or b['features'].shape != (count,dimension) or b['advantages'].shape != (count,5)
                or b['iterations'].shape != (count,) or b['total_seen'] < count
                or any(not np.isfinite(b[k]).all() for k in ['features','advantages','iterations'])
                or (count and (np.any(b['iterations'] < 1) or np.any(b['iterations'] > state['iteration'])
                               or np.any(b['iterations'] != np.floor(b['iterations']))))):
            raise ValueError('Malformed bounded rich replay checkpoint')
        if count and (not np.isin(b['features'][:,46:51],[0,1]).all()
                      or np.any(b['features'][:,46:51].sum(1)<1)
                      or np.any(b['advantages'][b['features'][:,46:51]==0] != 0)):
            raise ValueError('Malformed legal rich replay')
    if set(state['rng']) != {'python','numpy','generator','torch'}:
        raise ValueError('Complete independent and global RNG state required')


def commit_checkpoint(folder, state, model, config):
    folder = Path(folder); checkpoints = folder/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    iteration = state['iteration']; target = checkpoints/f'iter-{iteration:05d}'
    if target.exists():
        raise FileExistsError('A rich checkpoint is immutable')
    with tempfile.TemporaryDirectory(prefix='.pending-',dir=checkpoints) as temporary:
        payload = Path(temporary)/'state'; save_state(state,payload)
        model.export(payload/'current.npz')
        restored = load_state(payload)
        if not equal_state(state,restored):
            raise ValueError('Lossless rich checkpoint parity failed')
        files = {name:digest(payload/name) for name in ['state.json','arrays.npz','current.npz']}
        payload.rename(target)
    atomic_json(folder/'current.json',dict(format=CHECKPOINT_FORMAT,iteration=iteration,
                directory=target.name,files=files,teacher_sha256=model.teacher_sha256,
                parent_iteration=360,average_policy_available=False))
    descriptor=dict(format='rich-policy-export-v1',model=str(target/'current.npz'),
                    model_sha256=files['current.npz'],teacher=str(model.teacher_path),
                    teacher_sha256=model.teacher_sha256,source=config['source'],
                    iteration=iteration,seed=state['seed'],
                    feature_schema='rich-public-poker-observation-v1',
                    parent_iteration=360,average_policy_available=False)
    atomic_json(folder/f'policy-{iteration:05d}.json',descriptor)
    atomic_json(folder/'policy-current.json',descriptor)
    keep = {0,iteration}|set(config['stages'])
    for p in checkpoints.iterdir():
        if re.fullmatch(r'iter-\d{5,}',p.name) and int(p.name[5:]) not in keep:
            if p.is_symlink():
                raise ValueError('Unexpected checkpoint symlink')
            shutil.rmtree(p)
    for p in folder.glob('policy-*.json'):
        match=re.fullmatch(r'policy-(\d{5,})\.json',p.name)
        if match and int(match.group(1)) not in keep:
            p.unlink()
    return sum((target/name).stat().st_size for name in files)


def load_current(folder, config, seed, dimension):
    pointer = json.loads((Path(folder)/'current.json').read_text())
    if not re.fullmatch(r'iter-\d{5,}',pointer['directory']):
        raise ValueError('Invalid rich checkpoint path')
    path = Path(folder)/'checkpoints'/pointer['directory']
    if path.is_symlink() or set(pointer['files']) != {'state.json','arrays.npz','current.npz'}:
        raise ValueError('Incomplete rich checkpoint pointer')
    for name,sha in pointer['files'].items():
        if digest(path/name) != sha:
            raise ValueError('Committed rich checkpoint bytes changed')
    state = load_state(path);validate_checkpoint(state,config,seed,dimension)
    if state['iteration'] != pointer['iteration']:
        raise ValueError('Checkpoint pointer/state disagree')
    return state,path


def warm_start_verification(model, seed, dimension):
    from training.game import new_hand,legal_actions,step,S_TERMINAL,S_TOACT
    from rich_poker_observation import encode_game,history_start,append_action
    probe_rng = np.random.default_rng(seed+17000);samples=opening=0
    from regret_partition import is_opening
    for _ in range(12):
        state,deck,holes = new_hand(6,int(probe_rng.integers(0,6)),probe_rng)
        history=history_start(state)
        for _ in range(512):
            if state[S_TERMINAL]:break
            actor=int(state[S_TOACT]);legal=legal_actions(state,actor).astype(np.float32)
            features=encode_game(state,deck,holes,actor,legal,history)
            if features.shape != (dimension,):raise ValueError('Warm-start input dimension changed')
            baseline=model.teacher_strategy(features[:51],legal);candidate=model.strategy(features,legal)
            if baseline.dtype != candidate.dtype or not np.array_equal(baseline,candidate):
                raise ValueError('Zero-residual initial policy differs from existing teacher')
            samples+=1;opening+=int(bool(is_opening(features[:51])))
            probabilities=baseline.astype(float);probabilities/=probabilities.sum()
            action=int(probe_rng.choice(5,p=probabilities))
            after=state.copy();step(after,deck,holes,action)
            history=append_action(history,state,after,action);state=after
            if samples>=96:break
        if samples>=96:break
    if samples<32 or opening<1:raise ValueError('Insufficient genuine warm-start observation probes')
    return dict(passed=True,teacher_probability_byte_identical=True,samples=samples,
                opening_samples=opening,zero_residual=model.zero_residual,
                teacher_knowledge_inherited=True,parent_rng_inherited=False,
                parent_iteration=360,legacy_exact_resume=False)


def run(campaign,seed,through):
    campaign=Path(campaign).resolve();config=json.loads((campaign/'config.json').read_text())
    validate_config(config,seed,through)
    source=Path(config['source']);source=source if source.is_absolute() else campaign/source
    sys.path.insert(0,str(source.resolve()))
    from training.buffer import ReservoirBuffer
    from training import r1_support as r1
    from rich_poker_observation import RICH_DIM
    from rich_poker_traversal import run_traversals
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder=campaign/f'seed-{seed}';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    teacher=teacher_spec(campaign,config,seed)
    model=RichRegretModel(teacher,RICH_DIM,source,width=config.get('width',128))
    buffers=[ReservoirBuffer(config['buffer_capacity'],RICH_DIM) for _ in range(6)]
    if (folder/'current.json').exists():
        state,_=load_current(folder,config,seed,RICH_DIM);start=state['iteration'];model.load_state_dict(state['model'])
        for b,saved in zip(buffers,state['buffers']):b.load_state_dict(saved)
        random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
        rng.bit_generator.state=state['rng']['generator'];torch.set_rng_state(state['rng']['torch'])
        del state
    else:
        proof=warm_start_verification(model,seed,RICH_DIM);proof.update(verified_utc=utc(),teacher_sha256=model.teacher_sha256)
        atomic_json(folder/'warm-start-verification.json',proof);start=0
        initial=checkpoint_state(model,buffers,config,seed,0,rng);validate_checkpoint(initial,config,seed,RICH_DIM)
        commit_checkpoint(folder,initial,model,config);del initial
    if not start<through<=config['iterations']:
        raise ValueError('Rich stage is already complete or out of order; no automatic retry')
    began=time.monotonic();pid=os.getpid();raw=Path(f'/proc/{pid}/stat').read_text();token=raw[raw.rfind(')')+2:].split()[19]
    def status(phase,iteration,**fields):
        atomic_json(folder/'status.json',dict(status=phase,phase='training' if phase=='running' else phase,
            observed_utc=utc(),pid=pid,process_start=token,seed=seed,through=through,iteration=iteration,
            parent_iteration=360,new_family_selfplay_iterations=iteration,feature_dim=RICH_DIM,
            replay_rows=[len(b) for b in buffers],average_policy_available=False,
            teacher_knowledge_inherited=True,parent_rng_inherited=False,**fields))
    def interrupt(signum,frame):raise InterruptedError('Worker interrupted; no automatic retry')
    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGTERM,signal.SIGINT)}
    iteration=start
    try:
        status('running',start)
        for iteration in range(start+1,through+1):
            if time.monotonic()-began>config['seconds_per_training_worker']:
                raise TimeoutError('Rich worker exhausted its fixed stage time budget')
            check_frozen(config);tick=time.monotonic();traverser=(iteration-1)%6
            collection=run_traversals(config['traversals'],traverser,model,buffers,iteration,rng,
                                      max_nodes=config.get('max_nodes_per_iteration',200000))
            fitting=model.fit(buffers,steps=config['main_steps'],batch_size=config['batch_size'],
                              learning_rate=config['learning_rate'])
            validate_fitting_report(fitting,config)
            row=dict(iteration=iteration,seconds=time.monotonic()-tick,collection=collection,fitting=fitting,
                     replay_rows=[len(b) for b in buffers])
            json.dumps(row,allow_nan=False)
            with (folder/'diagnostics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            committed=None
            if iteration%config['checkpoint_every']==0 or iteration==through:
                state=checkpoint_state(model,buffers,config,seed,iteration,rng)
                validate_checkpoint(state,config,seed,RICH_DIM);committed=commit_checkpoint(folder,state,model,config);del state
            status('complete' if iteration==through else 'running',iteration,
                   checkpoint_bytes=committed,last_iteration_seconds=row['seconds'],fitting=fitting)
        return dict(complete=True,seed=seed,iteration=through,parent_iteration=360,
                    new_family_selfplay_iterations=through,average_policy_available=False)
    except BaseException as error:
        status('interrupted' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',
               iteration,error_type=type(error).__name__,error=str(error),automatic_retry=False)
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--through',type=int,required=True);args=parser.parse_args()
    print(json.dumps(run(args.campaign,args.seed,args.through),allow_nan=False))
