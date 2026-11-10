"""Frozen-main ablation of a small separately fitted initial-action expert."""
import argparse
import json
from pathlib import Path
import sys
import time

from fullhouse_v3_common import ROOT,atomic_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/fullhouse-compact-validation.json')
    p.add_argument('--replay-seed',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());pairs=dict(cfg['frozen_pairs'])
    if a.replay_seed not in pairs:p.error('Undeclared seed')
    source=ROOT/cfg['source_run']/'source';sys.path.insert(0,str(source))
    import numpy as np
    import torch
    from training.buffer import ReservoirBuffer
    from compact_regret import CompactRegretNet,BBCompactPredictions,split_replay,fit_opening,load_main,export_compact,METHOD
    import validate_frozen_regret as audit
    audit.np=np;audit.torch=torch;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=source.parent/f'seed-{a.replay_seed}/latest.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    parent=ROOT/cfg['control_run']/f'frozen-{a.replay_seed}/results.json';old=json.loads(parent.read_text())
    base_report=ROOT/cfg['base_fit_run']/f'frozen-{a.replay_seed}/results.json';base=json.loads(base_report.read_text())
    if old['checkpoint_sha256']!=digest(checkpoint) or base['checkpoint_sha256']!=digest(checkpoint):raise ValueError('Replay changed')
    control=next(r for r in old['refits'] if r['mode']=='control')
    selected=next(r for r in base['refits'] if r['mode']=='stratum-scaled-feature-conditional' and r['steps']==32768)
    if digest(ROOT/selected['model'])!=selected['model_sha256']:raise ValueError('Main expert changed')
    buffers=[]
    for state in ck['buffers']:
        b=ReservoirBuffer(state['count'],51);b.load_state_dict(state);buffers.append(b)
    root=audit.root_oracle(buffers);_,opening=split_replay(buffers)
    np.random.seed(pairs[a.replay_seed]);torch.manual_seed(pairs[a.replay_seed])
    net=CompactRegretNet();load_main(net,ROOT/selected['model']);opt=torch.optim.Adam(net.opening.parameters(),lr=.001)
    report=dict(complete=False,replay_seed=a.replay_seed,fit_seed=pairs[a.replay_seed],config=cfg,config_sha256=digest(a.config),
                checkpoint_sha256=digest(checkpoint),source_hashes=old['source_hashes'],original=old['original'],refits=[control],
                validation_script=Path(__file__).name,script_sha256=digest(Path(__file__)),fitter_sha256=digest(ROOT/'scripts/regret_fitting.py'),
                compact_fitter_sha256=digest(ROOT/'scripts/compact_regret.py'),partition_sha256=digest(ROOT/'scripts/regret_partition.py'),
                base_report_sha256=digest(base_report),base_model_sha256=selected['model_sha256'])
    previous=0;start=time.monotonic()
    for steps in cfg['candidate_steps']:
        fitted=fit_opening(net,opening,steps-previous,optimizer=opt)
        metrics=audit.measure(BBCompactPredictions(net).eval(),buffers,root)
        model=a.output/f'{METHOD}-{steps}.npz';export_compact(net,model)
        row=dict(mode=METHOD,steps=steps,batch=len(opening[0]),seconds=time.monotonic()-start,metrics=metrics,
                 model=str(model.relative_to(ROOT)),model_sha256=digest(model),model_bytes=model.stat().st_size,opening=fitted)
        row['gate']=audit.gate(row,control,cfg['fixed_buffer_gates']);report['refits'].append(row)
        atomic_json(a.output/'results.json',report)
        print(json.dumps(dict(steps=steps,seconds=row['seconds'],root=metrics['root'],gate=row['gate'],bytes=row['model_bytes'])),flush=True);previous=steps
    report['complete']=True;atomic_json(a.output/'results.json',report)


if __name__=='__main__':main()
