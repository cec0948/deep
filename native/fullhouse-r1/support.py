"""Corrected CPU reference utilities. No claim of six-player Nash convergence."""
import json
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch

WEIGHT_HORIZON = 1
REVISION = 'r1-cpu-v2'
AVERAGE_SAMPLER = 'importance'

def time_weights(iterations):
    return iterations / float(WEIGHT_HORIZON)

def atomic_save(obj, path):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(obj,f);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        d=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(name):os.unlink(name)

def pack_rng(rng):
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'generator':rng.bit_generator.state,'torch':torch.get_rng_state()}

def restore_rng(state,rng):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    rng.bit_generator.state=state['generator'];torch.set_rng_state(state['torch'])

def strategy_weight(own_reach,sample_reach):
    if sample_reach<=0:raise ValueError('Sampling support lost')
    return own_reach/sample_reach

def collect_importance(trainer,rng,n_traversals,net,buffer,iteration):
    """Independent full-support trajectories; Horvitz-Thompson own-reach mass.

    At each history E_q[1{visit} * pi_i/q_path] = pi_i. Chance is sampled
    independently, giving a fixed chance-weighted information-set average.
    This avoids opponent visitation weighting and covers zero-policy paths.
    High importance weights can cause variance: this is a CPU reference.
    """
    from .traverse_fast import NumpyPolicy,_encode_fast
    from .game import new_hand,legal_actions,step,S_TERMINAL,S_TOACT
    policy=NumpyPolicy(net)
    collected=[]
    for _ in range(n_traversals):
        state,deck,hole=new_hand(6,int(rng.integers(0,6)),rng)
        own=np.ones(6,dtype=np.float64);sample_reach=1.;steps=0
        while not state[S_TERMINAL]:
            seat=int(state[S_TOACT]);legal=legal_actions(state,seat)
            features=_encode_fast(state,hole,deck,seat,legal)
            sigma=policy.strategy(features,legal.astype(np.float32)).astype(float)
            sigma=np.maximum(sigma,0)*legal;sigma/=sigma.sum()
            weight=strategy_weight(own[seat],sample_reach)
            if not np.isfinite(weight) or weight>np.finfo(np.float32).max:raise FloatingPointError('Average importance weight overflow')
            collected.append(weight)
            if weight>0:buffer.add(features,sigma.astype(np.float32),iteration,sample_weight=weight)
            q=.5*sigma+.5*legal/legal.sum()
            a=int(rng.choice(5,p=q/q.sum()))
            own[seat]*=sigma[a];sample_reach*=q[a]
            step(state,deck,hole,a);steps+=1
            if steps>512:raise RuntimeError('Strategy sampling failed to terminate')

    return weight_diagnostics(collected)

def weight_diagnostics(weights):
    """Descriptive ESS only: correlated histories are not independent samples."""
    w=np.asarray(weights,dtype=np.float64)
    if not len(w):return {'count':0,'ess':0.,'ess_fraction':0.}
    if not np.isfinite(w).all() or (w<0).any():raise FloatingPointError('Invalid weights')
    scale=w.max()
    z=w/scale if scale>0 else w
    total=z.sum();squares=np.square(z).sum()
    ess=float(total**2/squares) if squares>0 else 0.
    return dict(count=len(w),positive_count=int((w>0).sum()),mean=float(w.mean()),
        p50=float(np.quantile(w,.5)),p95=float(np.quantile(w,.95)),p99=float(np.quantile(w,.99)),
        maximum=float(scale),ess=ess,ess_fraction=ess/len(w),
        max_mass_fraction=float(z.max()/total) if total else 0.,
        top_one_percent_mass_fraction=float(np.sort(z)[-max(1,int(np.ceil(len(w)*.01))):].sum()/total) if total else 0.)

def log_diagnostics(path,iteration,collection,buffer,adv_loss,strategy_loss):
    effective=buffer.iterations[:buffer.size].astype(float)/WEIGHT_HORIZON*buffer.sample_weights[:buffer.size]
    row=dict(iteration=iteration,collection=collection,reservoir=weight_diagnostics(effective),
        reservoir_total_seen=buffer.total_seen,advantage_loss=float(adv_loss),strategy_loss=float(strategy_loss))
    if not np.isfinite([adv_loss,strategy_loss]).all():raise FloatingPointError('Nonfinite loss')
    with open(path,'a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

def collect_fixed_opponent(trainer,rng,n_traversals,net,buffer,iteration):
    """Focal follows sigma; all opponents use a fixed uniform legal policy.

    For a perfect-recall information set I of focal i, visitation mass is
    C_tau(I)*pi_i^t(I), with C_tau fixed across t. Recording t*sigma with
    unit sample weights thus estimates the own-reach time average at I.
    This changes the state weighting for shared, imperfect-recall features.
    It is an explicit experimental sampler, not equal per-history mass.
    """
    from .traverse_fast import NumpyPolicy,_encode_fast
    from .game import new_hand,legal_actions,step,S_TERMINAL,S_TOACT
    policy=NumpyPolicy(net);collected=[]
    for _ in range(n_traversals):
        focal=int(rng.integers(0,6))
        state,deck,hole=new_hand(6,int(rng.integers(0,6)),rng);steps=0
        while not state[S_TERMINAL]:
            seat=int(state[S_TOACT]);legal=legal_actions(state,seat)
            if seat==focal:
                features=_encode_fast(state,hole,deck,seat,legal)
                sigma=policy.strategy(features,legal.astype(np.float32)).astype(float)
                sigma=np.maximum(sigma,0)*legal;sigma/=sigma.sum()
                buffer.add(features,sigma.astype(np.float32),iteration,sample_weight=1.)
                collected.append(1.);q=sigma
            else:q=legal/legal.sum()
            a=int(rng.choice(5,p=q/q.sum()));step(state,deck,hole,a);steps+=1
            if steps>512:raise RuntimeError('Strategy sampling failed to terminate')
    return weight_diagnostics(collected)

def collect_average(trainer,rng,n_traversals,net,buffer,iteration):
    if AVERAGE_SAMPLER=='importance':return collect_importance(trainer,rng,n_traversals,net,buffer,iteration)
    if AVERAGE_SAMPLER=='fixed-opponent':return collect_fixed_opponent(trainer,rng,n_traversals,net,buffer,iteration)
    raise ValueError('Unknown average sampler: '+AVERAGE_SAMPLER)
