"""Predeclared exact multiplayer estimator and sampled-neural Leduc checks.

The unchanged legacy regret learner is NOT certified for six-player convergence.
This gate isolates the NEW strategy collector, bounded replay, fitter and export.
The Leduc teacher trajectory uses the existing full-tree shared-neural learner;
all strategy-network targets come from the new sampled reference collector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from fullhouse_v3_common import atomic_json,digest,process_start
from controlled_poker_refit import utc
from poker_strategy_average import (StrategyReplay,collect_reference,fit_average,
    export_average,load_average,average_strategy)
from train_neural_information_gate import NeuralBudget,neural_iteration
from validate_strength_training import (Replay,adapter_strata,exact_iteration,exploitability_of,
    game_and_infosets,independent_average_contributions,kuhn_features,regret_match)
from scaled_regret_fitting import ScaledRegretSampler

GATE = dict(seeds=[71,72,73],iterations=32,reference_hands_per_iteration=1024,
    strategy_capacity=100000,average_fit_steps=2048,average_batch=512,
    exact_estimator_max_error=1e-12,max_weighted_full_infoset_tv=.15,
    max_average_exploitability_fraction_uniform=.5,max_exploitability_over_exact_average=.35,
    max_seconds=1200,automatic_retry=False)


def math_check(players):
    """Enumerate the exact expectation of the collector, independently of sampling.

    Compare against a separate own-reach-per-infoset definition. Deliberately
    compute the WRONG on-policy-frequency average as a negative control.
    """
    import pyspiel
    game=pyspiel.load_game('kuhn_poker',{'players':players})
    num={};den={};wrongnum={};wrongden={};own_num={};own_den={};q_mass={}
    for iteration in range(1,5):
        seen={}
        def sigma(state):
            key=state.information_state_string();legal=state.legal_actions()
            raw=np.array([1+(int.from_bytes(hashlib.sha256(f'{iteration}:{key}:{a}'.encode()).digest()[:4],'big')%19) for a in legal],dtype=float)
            return raw/raw.sum()
        def walk(state,q,reach,chance_reach,opponent_reach):
            if state.is_terminal():return
            if state.is_chance_node():
                for a,p in state.chance_outcomes():walk(state.child(a),q*p,reach,chance_reach*p,opponent_reach)
                return
            i=state.current_player();key=state.information_state_string();s=sigma(state)
            if key in seen and not np.isclose(seen[key],reach[i],atol=1e-14,rtol=0):
                raise ValueError('Perfect recall oracle disagrees inside an infoset')
            seen[key]=reach[i]
            mass=q*iteration*reach[i]
            num[key]=num.get(key,np.zeros(len(s)))+mass*s;den[key]=den.get(key,0.)+mass
            if iteration==1:q_mass[key]=q_mass.get(key,0.)+q
            bad=chance_reach*np.prod(reach)*iteration
            wrongnum[key]=wrongnum.get(key,np.zeros(len(s)))+bad*s;wrongden[key]=wrongden.get(key,0.)+bad
            for a,p in zip(state.legal_actions(),s):
                updated=reach.copy();updated[i]*=p
                walk(state.child(a),q/len(s),updated,chance_reach,opponent_reach)
        walk(game.new_initial_state(),1.,np.ones(players),1.,None)
        def own(state,reach):
            if state.is_terminal():return
            if state.is_chance_node():
                for a,_ in state.chance_outcomes():own(state.child(a),reach)
                return
            i=state.current_player();key=state.information_state_string();s=sigma(state)
            if key not in own_seen:
                own_seen.add(key);mass=iteration*reach[i]
                own_num[key]=own_num.get(key,np.zeros(len(s)))+mass*s
                own_den[key]=own_den.get(key,0.)+mass
            for a,p in zip(state.legal_actions(),s):
                updated=reach.copy();updated[i]*=p;own(state.child(a),updated)
        own_seen=set();own(game.new_initial_state(),np.ones(players))
    error=max(float(np.max(np.abs(num[k]/den[k]-own_num[k]/own_den[k]))) for k in num)
    wrong=max(float(np.max(np.abs(wrongnum[k]/wrongden[k]-own_num[k]/own_den[k]))) for k in num)
    return dict(players=players,infosets=len(num),max_error=error,wrong_on_policy_frequency_error=wrong,
        passed=error<GATE['exact_estimator_max_error'] and wrong>1e-3)


def train_seed(seed,output,deadline,progress):
    game,keys,infosets=game_and_infosets('leduc_poker')
    features=kuhn_features(keys,infosets);bykey=dict(zip(keys,features))
    current={k:features[j,46:51].astype(float)/features[j,46:51].sum() for j,k in enumerate(keys)}
    initial=exploitability_of(game,current)
    numerator={k:np.zeros(5) for k in keys};mass={k:0. for k in keys}
    replay=StrategyReplay(GATE['strategy_capacity'],seed=seed+50000)
    rng=np.random.default_rng(seed+60000);history=[]
    np.random.seed(seed)
    budget=NeuralBudget()
    for iteration in range(1,GATE['iterations']+1):
        if time.monotonic()>deadline:raise TimeoutError('Average validation fixed budget exhausted')
        own=independent_average_contributions(game,current)
        for k in keys:
            numerator[k]+=iteration*own[k]*current[k];mass[k]+=iteration*own[k]
        def observe(state):
            x=bykey[state.information_state_string()]
            return x,x[46:51]
        for _ in range(GATE['reference_hands_per_iteration']):
            collect_reference(game.new_initial_state(),2,replay,iteration,rng,
                terminal=lambda s:s.is_terminal(),chance=lambda s:s.is_chance_node(),
                outcomes=lambda s:s.chance_outcomes(),actor=lambda s:s.current_player(),observe=observe,
                strategy=lambda s,x,l:current[s.information_state_string()],child=lambda s,a:s.child(a))
        regret,_,_=exact_iteration(game,current,infosets)
        history.append(np.stack([regret[k] for k in keys]).astype(np.float32))
        r=Replay(np.tile(features,(iteration,1)),np.concatenate(history),
            np.repeat(np.arange(1,iteration+1,dtype=np.float32),len(keys)))
        sampler=ScaledRegretSampler([r],adapter_strata,'stratum-scaled-feature-conditional')
        _,scores,_=neural_iteration(features,sampler,seed*100000+iteration,budget)
        current={k:regret_match(scores[j],features[j,46:51]) for j,k in enumerate(keys)}
        progress(seed=seed,iteration=iteration,strategy_rows=replay.size,strategy_seen=replay.total_seen)
    exact={k:numerator[k]/mass[k] if mass[k]>0 else np.full(5,.2) for k in keys}
    # Zero-reach infosets may choose any legal strategy; keep normalization legal.
    for k in keys:
        if mass[k]==0:exact[k]=bykey[k][46:51]/bykey[k][46:51].sum()
    net,fit=fit_average(replay,seed=seed+70000,steps=GATE['average_fit_steps'],batch_size=GATE['average_batch'])
    export=output/f'seed-{seed}-average.npz';export_average(net,export,game='leduc_poker',seed=seed)
    layers,_=load_average(export)
    learned={k:average_strategy(layers,bykey[k],bykey[k][46:51]).astype(float) for k in keys}
    with torch.no_grad():batch=net(torch.from_numpy(features)).exp().numpy()
    parity=max(float(np.max(np.abs(learned[k]-batch[j]))) for j,k in enumerate(keys))
    errors=np.array([np.abs(learned[k]-exact[k]).sum()/2 for k in keys])
    weights=np.array([mass[k] for k in keys]);tv=float(np.average(errors,weights=weights))
    exact_exp=exploitability_of(game,exact);learned_exp=exploitability_of(game,learned)
    passed=bool(tv<=GATE['max_weighted_full_infoset_tv']
        and learned_exp<=initial*GATE['max_average_exploitability_fraction_uniform']
        and learned_exp<=exact_exp+GATE['max_exploitability_over_exact_average'] and parity<2e-5)
    result=dict(seed=seed,passed=passed,initial_exploitability=initial,
        exact_neural_trajectory_average_exploitability=exact_exp,
        learned_average_exploitability=learned_exp,full_infoset_weighted_tv=tv,
        numpy_torch_max_probability_error=parity,fit=fit,model_sha256=digest(export),
        average_targets='Sampled fixed-reference trajectories only',
        exact_policy_table_deployed=False,six_player_strength_verified=False)
    atomic_json(output/f'seed-{seed}.json',result)
    return result


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    began=time.monotonic();deadline=began+GATE['max_seconds'];pid=os.getpid();token=process_start(pid)
    sources={str(Path(__file__).resolve()):digest(__file__)}
    for name in ('poker_strategy_average.py','train_neural_information_gate.py','validate_strength_training.py',
                 'scaled_regret_fitting.py','regret_fitting.py','poker_average_inference.py'):
        path=Path(__file__).with_name(name);sources[str(path.resolve())]=digest(path)
    plan=dict(created_utc=utc(),gate=GATE,source_sha256=sources,
        scope='New strategy estimator and neural approximation only; unchanged legacy regret learner',
        production_information_loss_unresolved=True,full_paper_reproduction=False)
    atomic_json(output/'plan.json',plan)
    def progress(**fields):
        atomic_json(output/'status.json',dict(status='running',observed_utc=utc(),pid=pid,
            process_start=token,elapsed_seconds=time.monotonic()-began,**fields))
    def timeout(*args):raise TimeoutError('Average validation exceeded fixed1200second budget')
    previous=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,GATE['max_seconds'])
    results=[]
    try:
        math=[math_check(p) for p in (2,3)]
        if not all(x['passed'] for x in math):raise ValueError('Exact multiplayer average estimator check failed')
        atomic_json(output/'estimator-check.json',math)
        for seed in GATE['seeds']:
            results.append(train_seed(seed,output,deadline,progress))
        if any(digest(p)!=sha for p,sha in sources.items()):raise ValueError('Frozen validation source changed')
        passed=all(r['passed'] for r in results)
        report=dict(status='complete',passed=passed,completed_utc=utc(),estimator_checks=math,
            seeds=results,source_sha256=sources,plan_sha256=digest(output/'plan.json'),
            elapsed_seconds=time.monotonic()-began,automatic_retry=False,
            six_player_strength_verified=False,full_paper_reproduction=False)
        atomic_json(output/'report.json',report)
        atomic_json(output/'status.json',dict(status='complete',observed_utc=utc(),pid=pid,
            process_start=token,passed=passed,elapsed_seconds=time.monotonic()-began))
        return report
    except BaseException as error:
        report=dict(status='failed',passed=False,observed_utc=utc(),error=str(error),
            error_type=type(error).__name__,seed_results=results,automatic_retry=False)
        atomic_json(output/'report.json',report);atomic_json(output/'status.json',report)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=run(a.output);print(json.dumps(dict(status=report['status'],passed=report['passed'])))
