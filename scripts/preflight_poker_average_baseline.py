"""Actual-parent, full-budget one-update parity and trusted-engine runtime checks.

Runs once in a new directory. The shadow update is a disposable comparison,
never committed as campaign progress and never reused as a selected candidate.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fullhouse_v3_common import ROOT,atomic_json,digest
from controlled_poker_refit import read,utc,verify_files
from compact_lightweight_backup import equal_state
from train_poker_average_baseline import collect_game_reference,legacy_update,load_checkpoint
from poker_strategy_average import StrategyReplay,fit_average,export_average
from evaluate_average_baseline import run as evaluate


def run(study):
    study=Path(study).resolve();plan=read(study/'plan.json');campaign=Path(plan['campaign']);cfg=read(campaign/'config.json')
    output=study/'preflight';output.mkdir(exist_ok=False);began=time.monotonic()
    verify_files(plan['frozen_files'])
    sys.path[:0]=[str(campaign/'source'),str(campaign/'tools')]
    from training import r1_support as r1
    from training.buffer import ReservoirBuffer
    from training.traverse_fast import run_traversals_fast
    from root_regret_replay import RootReplay,collect_root_replay
    from compact_regret import CompactRegretNet,install_policy,fit_compact,fit_opening
    from bot.features import load_preflop_equity
    install_policy();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r1.WEIGHT_HORIZON=420;r1.AVERAGE_SAMPLER='importance';equity=load_preflop_equity()
    parent,_=load_checkpoint(campaign/'seed-342')
    def restore():
        buffers=[ReservoirBuffer(100000,51) for _ in range(6)]
        for b,s in zip(buffers,parent['buffers']):b.load_state_dict(s)
        root=RootReplay();root.load_state_dict(parent['root_replay'])
        net=CompactRegretNet();net.load_state_dict(parent['net']);net.eval()
        rng=np.random.default_rng();r1.restore_rng(parent['rng'],rng)
        return net,buffers,root,rng
    def state(net,buffers,root,rng):
        return dict(net=net.state_dict(),buffers=[b.state_dict() for b in buffers],
            root=root.state_dict(),rng=r1.pack_rng(rng))
    # Execute the actual frozen old worker's loop body through its quality gate,
    # not a second hand-written definition of what an update should do.
    old=campaign/'tools/train_compact_bounded.py';tree=ast.parse(old.read_text())
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
    body=[]
    for node in loop.body:
        if isinstance(node,ast.With):break
        body.append(node)
    code=compile(ast.Module(body=body,type_ignores=[]),str(old),'exec')
    net,buffers,root,rng=restore()
    ns=dict(np=np,torch=torch,time=time,cfg=dict(cfg,seconds_per_training_worker=cfg['training_seconds_per_worker']),fixture=False,seed=342,iteration=361,
        began=time.monotonic(),net=net,buffers=buffers,root=root,rng=rng,equity=equity,
        run_traversals_fast=run_traversals_fast,collect_root_replay=collect_root_replay,
        CompactRegretNet=CompactRegretNet,fit_compact=fit_compact,fit_opening=fit_opening)
    exec(code,ns)
    reference=state(ns['net'],buffers,root,rng)
    del ns,net,buffers,root,rng
    net,buffers,root,rng=restore()
    replay=StrategyReplay(100000,seed=342+110000);reference_rng=np.random.default_rng(342+120000)
    collected=collect_game_reference(net,replay,reference_rng,361,cfg['reference_hands'],equity)
    net,fit,tv,steps=legacy_update(net,buffers,root,rng,361,cfg,equity)
    actual=state(net,buffers,root,rng)
    checks={k:equal_state(reference[k],actual[k]) for k in reference}
    if not all(checks.values()):raise ValueError('Full-budget native legacy one-update parity failed: '+str(checks))
    before_rng=r1.pack_rng(rng)
    average,average_fit=fit_average(replay,seed=34200000+361,steps=512,batch_size=512)
    if not equal_state(before_rng,r1.pack_rng(rng)):raise ValueError('Average fitting changed learner RNG')
    model=output/'protocol-only-average.npz';export_average(average,model,
        averaging_window=[361,361],purpose='Small protocol smoke only; not a performance candidate')
    #20 new cards x6seats x3opponentfields; every action replayed by trusted engine.
    cells=[]
    for field in ('public','mixed','noregrets'):
        result=evaluate(model,campaign/'source',field,20,2037180000,180,output/field)
        if not result['terminal_payoffs_replayed']:raise ValueError('Trusted action replay did not complete')
        cells.append(dict(field=field,hands=result['hands'],results_sha256=digest(output/field/'results.json'),
            terminal_payoffs_replayed=True))
    verify_files(plan['frozen_files'])
    report=dict(passed=True,completed_utc=utc(),plan_sha256=digest(study/'plan.json'),
        native_old_worker_sha256=digest(old),full_native_one_update_parity=checks,
        legacy_fitting_steps=1024,legacy_batch=512,legacy_traversals=256,
        strategy_collection=collected,average_fit_rng_isolation=True,protocol_cells=cells,
        trusted_hands=360,elapsed_seconds=time.monotonic()-began,
        candidate_strength_verified=False,campaign_progress_from_preflight=0,
        preflight_candidate_reused=False,original_parent_unchanged=True)
    atomic_json(study/'preflight.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.study),indent=2))
