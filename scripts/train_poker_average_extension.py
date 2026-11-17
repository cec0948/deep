"""Authorized420 to480 continuation with unchanged learning and bounded replay.

The ONLY learning addition is a separate bounded strategy collector/network.
Its random generators and fitting are isolated from the legacy update stream.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

from compact_lightweight_backup import save_state,load_state,equal_state,validate_state
from fullhouse_v3_common import ROOT,atomic_json,digest,process_start
from poker_strategy_average import StrategyReplay,collect_reference,export_average
from poker_strategy_average import fit_average
from controlled_poker_refit import utc

CHECKPOINT_FORMAT='compact-window-average-recovery-v1'


from train_poker_average_continuation import (record_average_quality, verify_frozen, load_checkpoint,
    commit_checkpoint, collect_game_reference, legacy_update)


def run(campaign,seed):
    campaign=Path(campaign).resolve();cfg=json.loads((campaign/'config.json').read_text());verify_frozen(cfg)
    config_sha256=digest(campaign/'config.json')
    folder=campaign/f'seed-{seed}';lock=(folder/'worker.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if seed not in cfg['seeds'] or cfg['initial_iteration']!=420 or cfg['iterations']!=480:
        raise ValueError('Unauthorized baseline seed/window')
    if any(cfg[k]!=v for k,v in dict(traversals=256,buffer_capacity=100000,main_steps=1024,
            batch_size=512,opening_steps=2048,opening_max_steps=8192,max_opening_tv=.1,
            root_repetitions=4,learning_rate=.001,strategy_capacity=100000,
            reference_hands=512,average_fit_steps=8192,average_batch=512,
            weight_horizon=420,average_width=128,average_normalize=False,stages=[480]).items()):
        raise ValueError('Predeclared baseline sample/compute budget changed')
    sys.path[:0]=[str(campaign/'source'),str(campaign/'tools')]
    from training import r1_support as r1
    from training.buffer import ReservoirBuffer
    from training.traverse_fast import run_traversals_fast
    from bot.features import load_preflop_equity
    from compact_regret import CompactRegretNet,install_policy,fit_compact,fit_opening,export_compact
    from root_regret_replay import RootReplay,collect_root_replay
    install_policy();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    state,_=load_checkpoint(folder)
    if state['iteration']!=cfg['resume_iterations'][str(seed)] or state['config']!=cfg:
        raise ValueError('Only the explicitly verified recovery launch is allowed')
    initial_iteration=state['iteration']
    buffers=[ReservoirBuffer(cfg['buffer_capacity'],51) for _ in range(6)]
    for b,saved in zip(buffers,state['buffers']):b.load_state_dict(saved)
    root=RootReplay();root.load_state_dict(state['root_replay'])
    net=CompactRegretNet();net.load_state_dict(state['net']);net.eval()
    rng=np.random.default_rng();r1.restore_rng(state['rng'],rng)
    strategy=StrategyReplay.from_state(state['strategy_replay'])
    reference_rng=np.random.default_rng();reference_rng.bit_generator.state=state['reference_rng']
    manifest=state['source_manifest'];del state
    r1.WEIGHT_HORIZON=cfg['weight_horizon'];r1.AVERAGE_SAMPLER='importance';equity=load_preflop_equity()
    began=time.monotonic();pid=os.getpid();token=process_start(pid)
    def status(phase,iteration,**fields):
        atomic_json(folder/'status.json',dict(status='complete' if phase=='complete' else 'running',
            phase=phase,observed_utc=utc(),pid=pid,process_start=token,seed=seed,iteration=iteration,
            new_selfplay_iterations=iteration-initial_iteration,
            completed_since_parent360=iteration-360,
            planned_new_selfplay_iterations=480-initial_iteration,
            elapsed_seconds=time.monotonic()-began,**fields))
    status('preparing',initial_iteration)
    try:
        for iteration in range(initial_iteration+1,481):
            if time.monotonic()-began>cfg['training_seconds_per_worker']:
                raise TimeoutError('Baseline training fixed budget exhausted; no retry')
            verify_frozen(cfg)
            if digest(campaign/'config.json')!=config_sha256:raise ValueError('Campaign configuration changed')
            tick=time.monotonic()
            # The collector owns its RNG and reservoir. Verify the legacy global
            # streams really remain byte-identical around each collection.
            old_np=np.random.get_state();old_torch=torch.get_rng_state();old_py=random.getstate()
            collected=collect_game_reference(net,strategy,reference_rng,iteration,cfg['reference_hands'],equity)
            if (not equal_state(old_np,np.random.get_state()) or not torch.equal(old_torch,torch.get_rng_state())
                    or old_py!=random.getstate()):
                raise ValueError('Strategy collection changed legacy learning RNG')
            net,fit,tv,opening_steps=legacy_update(net,buffers,root,rng,iteration,cfg,equity)
            average=None;average_fit=None
            if iteration in cfg['stages']:
                status('fitting_window_average',iteration)
                average,average_fit=fit_average(strategy,seed=seed*100000+iteration,
                    steps=cfg['average_fit_steps'],batch_size=cfg['average_batch'],
                    width=cfg['average_width'])
                # The old0.15reference remains reported. A finite auxiliary
                # projection error does not terminate this fixed main-policy study.
                average_fit=record_average_quality(average_fit,cfg['average_max_replay_tv'])
                atomic_json(folder/f'average-fit-{iteration}.json',average_fit)
            row=dict(iteration=iteration,new_selfplay_iterations=iteration-initial_iteration,
                completed_since_parent360=iteration-360,seed=seed,
                seconds=time.monotonic()-tick,buffer_rows=[len(b) for b in buffers],root=root.diagnostics(),
                opening_tv=tv,opening_steps=opening_steps,main_loss=fit['main_loss'],
                reference_collection=collected,average_fit=average_fit,legacy_rng_isolation_verified=True)
            with (folder/'diagnostics.jsonl').open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(row,allow_nan=False),flush=True)
            if iteration%6==0 or average is not None:
                state=dict(checkpoint_format=CHECKPOINT_FORMAT,seed=seed,iteration=iteration,config=cfg,
                    source_manifest=manifest,net=net.state_dict(),buffers=[b.state_dict() for b in buffers],
                    root_replay=root.state_dict(),rng=r1.pack_rng(rng),strategy_replay=strategy.state_dict(),
                    reference_rng=reference_rng.bit_generator.state,average_fit=average_fit,
                    average_state=None if average is None else average.state_dict(),
                    averaging_window=[361,iteration],historical_mixture_available=False)
                commit_checkpoint(folder,state,lambda p:export_compact(net,p),average);del state
            status('complete' if iteration==480 else 'training',iteration,opening_tv=tv,
                strategy_rows=strategy.size,average_fit=average_fit)
    except BaseException as error:
        atomic_json(folder/'status.json',dict(status='failed',phase='stopped',observed_utc=utc(),pid=pid,
            process_start=token,seed=seed,error_type=type(error).__name__,error=str(error),
            last_committed_iteration=json.loads((folder/'current.json').read_text())['iteration'],automatic_retry=False))
        raise
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.campaign,a.seed)
