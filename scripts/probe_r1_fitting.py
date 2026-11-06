"""Frozen replay fitting probe; row holdout is not poker generalization testing."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from fullhouse_runtime import ROOT

sys.path.insert(0,str(ROOT/'.cache/fullhouse-r1'))
from training import r1_support as r1
from training.buffer import ReservoirBuffer
from training.network import DuelingAdvantageNet,AverageStrategyNet
from training.train import train_network,train_strategy_network


def split_buffer(state,strategy,rng):
    n=state['count'];order=rng.permutation(n);cut=max(1,int(.8*n));parts=[]
    for ids in (order[:cut],order[cut:]):
        b=ReservoirBuffer(max(1,len(ids)),51,store_sample_weights=strategy)
        for idx in ids:b.add(state['features'][idx],state['advantages'][idx],state['iterations'][idx],sample_weight=state.get('sample_weights',np.ones(n))[idx])
        parts.append(b)
    return parts


def score(net,buffers,strategy):
    total=0.;sums={};weights=[]
    with torch.no_grad():
        for b in buffers:
            for start in range(0,b.size,512):
                end=min(b.size,start+512)
                x=torch.from_numpy(b.features[start:end]);target=torch.from_numpy(b.advantages[start:end]);mask=x[:,46:51]
                w=b.iterations[start:end].astype(float)/100
                if strategy:w=w*b.sample_weights[start:end]
                weights.extend(w.tolist());total+=w.sum()
                if strategy:
                    target=target*mask;target=target/target.sum(1,keepdim=True).clamp(min=1e-12)
                    logp=torch.log_softmax(net(x,mask),1)
                    ce=-(target*logp).sum(1)
                    entropy=-(target*torch.log(target.clamp(min=1e-12))).sum(1)
                    values={'cross_entropy':ce,'kl_to_target':ce-entropy,'total_variation':(logp.exp()-target).abs().sum(1)/2}
                else:
                    values={'mse':(net(x,mask)-target*mask).square().mean(1),'zero_predictor_mse':(target*mask).square().mean(1)}
                for key,value in values.items():sums[key]=sums.get(key,0.)+float(np.dot(w,value.numpy().astype(float)))
    if total<=0:raise ValueError('Empty scoring mass')
    return {**{k:v/total for k,v in sums.items()},'weight_diagnostics':r1.weight_diagnostics(weights)}


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);args=p.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);r1.WEIGHT_HORIZON=100
    ck=torch.load(args.checkpoint,weights_only=False,map_location='cpu')
    if ck['r1_config']['horizon']!=100:raise ValueError('Probe expects horizon 100')
    out=ROOT/'artifacts/r1-fitting'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    state=dict(complete=False,checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=[],
        limitation='Seeded random row split; trajectories are correlated. Scores are fitting diagnostics, not game strength.')
    data_rng=np.random.default_rng(701)
    advantage=[split_buffer(b,False,data_rng) for b in ck['buffers']]
    strategy=split_buffer(ck['strategy_buffer'],True,data_rng)
    for kind,train,valid in [('advantage',[b[0] for b in advantage],[b[1] for b in advantage]),('strategy',[strategy[0]],[strategy[1]])]:
        for seed in (7,8,9):
            np.random.seed(seed);torch.manual_seed(seed)
            net=AverageStrategyNet() if kind=='strategy' else DuelingAdvantageNet()
            opt=torch.optim.Adam(net.parameters(),lr=1e-3);previous=0;elapsed=0.
            for steps in (0,16,128,512):
                start=time.monotonic()
                if steps>previous:
                    if kind=='strategy':train_strategy_network(net,train[0],opt,steps-previous,128,torch.device('cpu'))
                    else:train_network(net,train,opt,steps-previous,128,torch.device('cpu'))
                elapsed+=time.monotonic()-start
                state['results'].append(dict(kind=kind,initialization_seed=seed,steps=steps,optimizer_seconds=elapsed,
                    train=score(net,train,kind=='strategy'),validation=score(net,valid,kind=='strategy')))
                (out/'results.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
                previous=steps
    state['complete']=True;(out/'results.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');print(out,flush=True)

if __name__=='__main__':main()
