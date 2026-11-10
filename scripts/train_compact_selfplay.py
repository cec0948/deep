"""Deterministic CPU self-play, dense root replay, exact per-hand policy mixture."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

from fullhouse_v3_common import ROOT,atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--through',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    folder=a.run_dir.resolve();cfg=json.loads((folder/'config.json').read_text());manifest=json.loads((folder/'source-manifest.json').read_text())
    fixture=cfg.get('verification_fixture',False)
    if fixture:
        for name,maximum in [('iterations',12),('traversals',8),('main_steps',8),('opening_steps',8),('opening_max_steps',8),('buffer_capacity',5000),('root_repetitions',1),('root_repetitions_first',1)]:
            if cfg[name]>maximum:raise ValueError('Smoke fixture exceeds bound: '+name)
    else:
        gate_path=ROOT/cfg['gate_run']/'gates.json';gate=json.loads(gate_path.read_text())
        if cfg['method'] not in gate['eligible_methods']:raise ValueError('Both learning gates must pass')
        deployment=json.loads((ROOT/cfg['gate_run']/'deployment-verification.json').read_text())
        if not deployment['passed']:raise ValueError('Deployment verification required')
        if deployment['runtime_sha256']!=digest(ROOT/'scripts/fullhouse_v4_runtime.py'):raise ValueError('Runtime changed')
    if a.seed not in cfg['seeds']:raise ValueError('Undeclared training seed')
    for name,expected in manifest['files'].items():
        if digest(folder/name)!=expected:raise ValueError('Frozen source changed: '+name)
    if digest(Path(__file__))!=manifest['driver_sha256']:raise ValueError('Worker code changed')
    sys.path[:0]=[str(folder/'source'),str(folder/'tools')]
    import numpy as np
    import torch
    from training import r1_support as r1
    from training.buffer import ReservoirBuffer
    from training.traverse_fast import run_traversals_fast
    from bot.features import load_preflop_equity
    from compact_regret import (CompactRegretNet,install_policy,fit_compact,fit_opening,export_compact)
    from root_regret_replay import RootReplay,collect_root_replay
    install_policy();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    r1.WEIGHT_HORIZON=cfg['iterations'];r1.AVERAGE_SAMPLER='importance'
    seed_dir=folder/f'seed-{a.seed}';seed_dir.mkdir(exist_ok=True)
    import fcntl
    lock=(seed_dir/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    buffers=[ReservoirBuffer(cfg['buffer_capacity'],51) for _ in range(6)];root=RootReplay()
    net=CompactRegretNet();net.eval()
    with torch.no_grad():
        for parameter in net.parameters():parameter.zero_()
    start=0;snapshots=[];checkpoint=seed_dir/'latest.pt'
    if a.resume:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if ck['config']!=cfg or ck['source_manifest']!=manifest or ck['seed']!=a.seed:raise ValueError('Checkpoint config/source mismatch')
        net.load_state_dict(ck['net']);root.load_state_dict(ck['root_replay'])
        for b,state in zip(buffers,ck['buffers']):b.load_state_dict(state)
        snapshots=ck['snapshots'];start=ck['iteration'];r1.restore_rng(ck['rng'],rng)
    elif checkpoint.exists():raise ValueError('Checkpoint exists; explicit resume required')
    end=min(a.through or cfg['iterations'],cfg['iterations'])
    if end<=start:raise ValueError('No new iterations requested')
    equity=load_preflop_equity();began=time.monotonic()
    def export(net,path):
        temporary=path.with_name(path.stem+'.tmp.npz');export_compact(net,temporary);os.replace(temporary,path)
    for iteration in range(start+1,end+1):
        if time.monotonic()-began>cfg['seconds_per_training_worker']:raise TimeoutError('CPU training limit')
        # sigma^t is the policy BEFORE update t. Save it with weight t.
        snapshot=seed_dir/f'policy_{iteration:05d}.npz';export(net,snapshot)
        snapshots.append(dict(iteration=iteration,weight=iteration,file=snapshot.name,sha256=digest(snapshot)))
        traverser=(iteration-1)%6;started=time.monotonic()
        run_traversals_fast(cfg['traversals'],6,traverser,net,buffers[traverser],iteration,rng,equity,
                            opponent_exploration=0.,strategy_buffer=None,randomize_stacks=False)
        repeats=cfg['root_repetitions_first'] if iteration==1 else cfg['root_repetitions']
        collect_root_replay(net,rng,root,iteration,repeats)
        net=CompactRegretNet()
        fit=fit_compact(net,buffers,cfg['main_steps'],cfg['batch_size'],root_replay=root,
                        opening_steps=cfg['opening_steps'],lr=cfg['learning_rate'])
        x,y,mass=root.means();opening_steps=cfg['opening_steps']
        expected=np.maximum(y,0)*x[:,46:51];total=expected.sum(1,keepdims=True)
        legal=x[:,46:51].astype(float)
        expected=np.divide(expected,total,out=legal/legal.sum(1,keepdims=True),where=total>0)
        def measure():
            pred=net.get_strategy(torch.from_numpy(x),torch.from_numpy(x[:,46:51])).numpy()
            tv=np.abs(pred-expected).sum(1)/2
            aa=int(np.flatnonzero((x[:,10]==1)&(x[:,7]>.99))[0])
            return dict(weighted_tv=float(np.average(tv,weights=mass)),aa_fold=float(pred[aa,0]),
                        aa_target_fold=float(expected[aa,0]),aa_regrets=y[aa].tolist())
        quality=measure()
        while quality['weighted_tv']>cfg['max_opening_tv'] and opening_steps<cfg['opening_max_steps']:
            extra=min(1024,cfg['opening_max_steps']-opening_steps)
            fit_opening(net,(x,y,mass),extra,cfg['learning_rate']);opening_steps+=extra;quality=measure()
        if quality['weighted_tv']>cfg['max_opening_tv'] and not fixture:
            r1.atomic_save(dict(iteration=iteration,seed=a.seed,config=cfg,net=net.state_dict(),root_replay=root.state_dict(),
                                buffers=[b.state_dict() for b in buffers],rng=r1.pack_rng(rng),quality=quality),seed_dir/'failed-fit.pt')
            raise RuntimeError('Opening fit failed; no later self-play was run')
        row=dict(iteration=iteration,seed=a.seed,seconds=time.monotonic()-started,buffer_rows=[len(b) for b in buffers],
                 root=root.diagnostics(),opening_steps=opening_steps,quality=quality,main_loss=fit['main_loss'])
        with (seed_dir/'diagnostics.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if iteration%cfg['checkpoint_every']==0 or iteration==end:
            current=seed_dir/f'current_{iteration:05d}.npz';export(net,current)
            mixture=dict(model_type='compact_snapshot_mixture',training_revision='r1-cpu-v4',iteration=iteration,
                         selection='Once per hand, independent of cards; weight t for the pre-update policy sigma^t',models=snapshots)
            atomic_json(seed_dir/f'mixture_{iteration:05d}.json',mixture)
            state=dict(iteration=iteration,seed=a.seed,config=cfg,source_manifest=manifest,net=net.state_dict(),
                       buffers=[b.state_dict() for b in buffers],root_replay=root.state_dict(),rng=r1.pack_rng(rng),snapshots=snapshots)
            r1.atomic_save(state,checkpoint)
            atomic_json(seed_dir/'status.json',dict(iteration=iteration,through=end,status='complete' if iteration==end else 'running',quality=quality,root=root.diagnostics()))


if __name__=='__main__':main()
