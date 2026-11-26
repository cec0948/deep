"""Fit paired-action residual critics on a fixed ledger of fresh self-play branches."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_flop_repair import (FORMAT, MAX_UPDATE, HIDDEN, INPUT_DIM, MODELS, FlopRepairPolicy,
                               critic_input, first_flop_decision, improved_distribution)
from poker_preflop_specialist import file_ref
from train_poker_flop_repair import DESIGN_HASH, REL, SEEDS, STUDY, sources


STEPS = 256
BATCH = 128


def load_training(folder, seed, parent):
    receipt=read(folder/'collection-complete.json')
    verify_files({folder/'positions.jsonl.gz':receipt['position_ledger_sha256']})
    states, inputs, masks, targets, baselines, all_heads = [], [], [], [], [], []
    with gzip.open(folder/'positions.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            state=row['root_observation']
            if row['seed']!=seed or not first_flop_decision(state) or state['seat_to_act']!=row['hero']:
                raise ValueError('Invalid training first-flop observation')
            actions=row['legal_actions']
            baseline=np.array(row['parent_probabilities'])
            features,heads=critic_input(parent.policies['postflop'],state,actions,baseline)
            payouts=np.array(row['action_payoffs_chips'])
            if payouts.shape!=(len(actions),4) or not np.isfinite(payouts).all():
                raise ValueError('Incomplete or nonfinite matched action labels')
            q=payouts.mean(axis=1)
            target=np.zeros(5)
            target[heads]=(q-q@baseline)/max(500.,state['pot'])
            mask=np.zeros(5,dtype=bool)
            mask[heads]=True
            states.append(state)
            inputs.append(features)
            masks.append(mask)
            targets.append(target)
            baselines.append(baseline)
            all_heads.append(heads)
    if len(states)!=1024 or receipt['positions']!=len(states) or np.sum(masks)*4!=receipt['terminal_rollouts']:
        raise ValueError('Incomplete self-play training budget')
    return states, np.array(inputs), np.array(masks), np.array(targets), baselines, all_heads


def train_heads(inputs,masks,targets,seed):
    center=inputs.mean(axis=0)
    scale=np.maximum(inputs.std(axis=0),.08)
    x=torch.tensor(np.clip((inputs-center)/scale,-6.,6.),dtype=torch.float32)
    y=torch.tensor(targets,dtype=torch.float32)
    valid=torch.tensor(masks)
    exports={name:[] for name in ('w1','b1','w2','b2')}
    fits=[]
    for head in range(MODELS):
        rng=np.random.default_rng([20261126,seed,head])
        torch.manual_seed(seed*100+head)
        model=nn.Sequential(nn.Linear(INPUT_DIM,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,5))
        nn.init.zeros_(model[2].weight)
        nn.init.zeros_(model[2].bias)
        # Resample whole independent root deals, preserving all matched action branches together.
        boot=rng.integers(len(x),size=len(x))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        losses=[]
        for _ in range(STEPS):
            ix=boot[rng.integers(len(boot),size=BATCH)]
            residual=(model(x[ix])-y[ix]).square()*valid[ix]
            loss=(residual.sum(dim=1)/valid[ix].sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        for name, value in zip(exports,[model[0].weight,model[0].bias,model[2].weight,model[2].bias],strict=True):
            exports[name].append(value.detach().numpy().copy())
        with torch.no_grad():
            residual=((model(x)-y).square()*valid).sum(1)/valid.sum(1)
        fits.append(dict(head=head,update_steps=STEPS,position_draws=STEPS*BATCH,
                         bootstrap_distinct_positions=len(np.unique(boot)),mean_training_batch_loss=float(np.mean(losses)),
                         full_training_set_mse=float(residual.mean()),constant_zero_mse=float(((y.square()*valid).sum(1)/valid.sum(1)).mean())))
    return dict(center=center,scale=scale,**{k:np.stack(v) for k,v in exports.items()}),fits


def fit_seed(seed):
    folder=STUDY/f'seed-{seed}'
    _,base,source,parent,_=sources(seed)
    states,inputs,masks,targets,baselines,heads=load_training(folder,seed,parent)
    weights,training=train_heads(inputs,masks,targets,seed)
    filepath=folder/'critic.npz'
    with filepath.open('xb') as output:
        np.savez_compressed(output,**weights)
    manifest=folder/'policy.json'
    atomic_json(manifest,dict(format=FORMAT,seed=seed,max_update_mass=MAX_UPDATE,base480=base,
                              feature_source=dict(path=str(source),sha256=digest(source/'bot/features.py')),
                              critic=dict(path=filepath.name,sha256=digest(filepath))))
    policy=FlopRepairPolicy(manifest,source)
    mass=[]
    changed_distribution=0
    for i,(state,baseline,feature,slots) in enumerate(zip(states,baselines,inputs,heads,strict=True)):
        expected,alpha=improved_distribution(feature,slots,baseline,weights)
        reply=policy.decide(state,i)
        if not np.isclose(reply['flop_repair_mass'],alpha,rtol=0,atol=1e-12):
            raise ValueError('Exported model differs from training weights')
        if alpha==0.:
            incumbent=parent.decide(state,i)
            if incumbent['action']!=reply['action'] or incumbent['distribution']!=reply['distribution']:
                raise ValueError('Incumbent seeded action differs on fallback')
        changed_distribution+=float(np.abs(expected-baseline).sum())>1e-8
        mass.append(alpha)
    if not changed_distribution:
        raise ValueError('Trained correction has no effect on any training position')
    result=dict(complete=True,seed=seed,completed_utc=utc(),training_design_sha256=DESIGN_HASH,
                ledger_sha256=digest(folder/'positions.jsonl.gz'),positions=len(states),training=training,
                fit_position_draws=STEPS*BATCH*MODELS,fit_optimizer_updates=STEPS*MODELS,
                concrete_action_labels=int(masks.sum()),training_terminal_branches=int(masks.sum())*4,
                parent_sha256=base['sha256'],critic_sha256=digest(filepath),policy_sha256=digest(manifest),
                changed_distributions_on_training_positions=int(changed_distribution),
                mean_bounded_mixture_mass=float(np.mean(mass)),max_bounded_mixture_mass=float(max(mass)),
                functional_check_positions=len(mass),performance_matches=0)
    atomic_json(folder/'training-receipt.json',result)
    return result


def fit_all():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if read(STUDY/'collection-status.json')['state']!='complete':
        raise ValueError('Fresh training labels are incomplete')
    with (STUDY/'fitting.lock').open('x'):
        pass
    seeds={str(s):fit_seed(s) for s in SEEDS}
    summary=dict(training_complete=True,completed_utc=utc(),kind='derived-policy-probe',
                 method='First flop only; two bootstrap MLP action critics per seed; fixed incumbent thereafter',
                 training_design_sha256=DESIGN_HASH,seeds=seeds,training_positions=sum(v['positions'] for v in seeds.values()),
                 action_labels=sum(v['concrete_action_labels'] for v in seeds.values()),
                 matched_terminal_rollouts=sum(v['training_terminal_branches'] for v in seeds.values()),
                 optimizer_updates=sum(v['fit_optimizer_updates'] for v in seeds.values()),
                 total_training_position_draws=sum(v['fit_position_draws'] for v in seeds.values()),
                 new_online_selfplay_iterations=0, performance_matches=0, strength_verified=False)
    atomic_json(STUDY/'fit-complete.json',summary)
    atomic_json(ROOT/'reports'/REL/'fit-complete.json',summary)
    return summary


if __name__=='__main__':
    fit_all()
