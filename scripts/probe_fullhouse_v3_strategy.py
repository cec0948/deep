"""Validate conditional-average fitting on frozen replay; never resume it."""
import json
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cache/fullhouse-v3'))
from training import r1_support as r1
from training.buffer import ReservoirBuffer
from training.network import AverageStrategyNet
from training.train import train_strategy_network


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r1.WEIGHT_HORIZON=600;r1.STRATEGY_BATCH_SIZE=512
    checkpoint=ROOT/'artifacts/fullhouse-medium/20261106T023201457778Z/training/latest.pt'
    ck=torch.load(checkpoint,weights_only=False,map_location='cpu')
    b=ReservoirBuffer(200000,51,store_sample_weights=True);b.load_state_dict(ck['strategy_buffer'])
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    audit=out/'frozen-strategy-audit.jsonl'
    if audit.exists():raise FileExistsError(audit)
    result={'purpose':'Frozen old replay tests fitting only. These data/models are excluded from the new self-play campaign.',
            'thresholds':{'root_weighted_tv':.08,'aa_fold_absolute_error':.05,'aa_raise_absolute_error':.10},'fits':[]}
    for seed in (7,8,9):
        for steps in (128,512):
            np.random.seed(seed);torch.manual_seed(seed)
            net=AverageStrategyNet();opt=torch.optim.Adam(net.parameters(),lr=.001)
            train_strategy_network(net,b,opt,steps,128,torch.device('cpu'))
            r1.write_strategy_audit(audit,600,net,b)
            row=json.loads(audit.read_text().splitlines()[-1]);row.update(seed=seed,fit_steps=steps)
            result['fits'].append(row)
            print(json.dumps(dict(seed=seed,fit_steps=steps,root=row['root'])),flush=True)
    result['passed']=False
    (out/'strategy-fitting.json').write_text(json.dumps(result,indent=2)+'\n')
    final=[r['root'] for r in result['fits'] if r['fit_steps']==512]
    for r in final:
        assert r['weighted_tv']<=.08,r
        pair=next(p for p in r['premium_pairs'] if p['hand']=='AA')
        assert abs(pair['prediction'][0]-pair['target'][0])<=.05,pair
        assert abs(sum(pair['prediction'][2:])-sum(pair['target'][2:]))<=.10,pair
    result['passed']=True;(out/'strategy-fitting.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
