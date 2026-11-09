"""Differential replay of synthetic training histories in the deployment engine."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from fullhouse_runtime import ROOT, Bridge

RUN = ROOT / 'artifacts/fullhouse-medium/20261106T023201457778Z'
sys.path.insert(0,str(RUN/'source'))
from training import game as g
from bot.features import encode_state, encode_state_dict, load_preflop_equity


def card(c):
    return '23456789TJQKA'[int(c)%13] + 'cdhs'[int(c)//13]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hands',type=int,default=2000)
    parser.add_argument('--policy-iteration',type=int,choices=[120,300,600])
    parser.add_argument('--repair-pending',action='store_true',help='In-process diagnostic ablation only; source stays immutable')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(61971)
    eq=load_preflop_equity()
    bridge=Bridge()
    counts=Counter();examples=[]
    if args.repair_pending:
        def next_actor_with_outstanding_call(state,last):
            n=int(state[g.S_NPLAYERS])
            for offset in range(1,n+1):
                seat=(last+offset)%n
                owes=state[g.S_BETS+seat] < state[g.S_CURBET]
                if (state[g.S_PENDING+seat] or owes) and not state[g.S_FOLDED+seat] and not state[g.S_ALLIN+seat]:
                    return seat
            return -1
        g._next_actor=next_actor_with_outstanding_call
    if args.policy_iteration:
        import torch
        from training.network import DuelingAdvantageNet
        from training.traverse_fast import NumpyPolicy
        torch.set_num_threads(1)
        ck=torch.load(RUN/f'training/iter_{args.policy_iteration:05d}.pt',weights_only=False,map_location='cpu')
        net=DuelingAdvantageNet();net.load_state_dict(ck['net']);policy=NumpyPolicy(net)
    try:
        for hand in range(args.hands):
            s,deck,hole=g.new_hand(6,int(rng.integers(6)),rng)
            history=[]
            while not s[g.S_TERMINAL]:
                counts['training_decisions']+=1
                seat=int(s[g.S_TOACT]);street=int(s[g.S_STREET])
                legal=g.legal_actions(s,seat)
                request=dict(op='replay',hero=seat,button=int(s[g.S_DEALER]),
                             hole=[card(c) for c in hole[seat]],
                             board=[card(c) for c in deck[12:12+(0,3,4,5)[street]]],actions=history)
                issue=None
                try:
                    real=bridge.call(request)
                    x=encode_state(s,hole,deck,seat,legal,eq)
                    y,_=encode_state_dict(real,eq)
                    diffs=np.flatnonzero(~np.isclose(x,y,rtol=1e-5,atol=1e-6))
                    counts['matched_public_states']+=1
                    if len(diffs):
                        counts['feature_mismatch_states']+=1
                        for d in diffs:counts[f'feature_{d}']+=1
                        issue={'kind':'features','indices':diffs.tolist(),'training':x[diffs].tolist(),'deployment':y[diffs].tolist()}
                except (ValueError,TypeError) as exc:
                    counts['unreplayable_states']+=1
                    counts[type(exc).__name__+': '+str(exc)]+=1
                    issue={'kind':'replay_error','error':str(exc)}
                if issue and len(examples)<20:
                    examples.append(dict(hand=hand,request=json.loads(json.dumps(request)),issue=issue,
                                         training_state=s.tolist(),training_deck=deck.tolist()))
                if issue and issue['kind']=='features' and counts['feature_mismatch_states']<=5:
                    examples.append(dict(hand=hand,request=json.loads(json.dumps(request)),issue=issue,
                                         training_state=s.tolist(),training_deck=deck.tolist()))
                if issue and issue['kind']=='replay_error':
                    break
                # Coverage probe, not the training distribution: half call, half uniform legal.
                probs=legal.astype(float)/legal.sum()*.5
                if args.policy_iteration:
                    features=encode_state(s,hole,deck,seat,legal,eq)
                    sigma=policy.strategy(features,legal.astype(np.float32)).astype(float)
                    sigma*=legal;sigma/=sigma.sum()
                    probs+=.5*sigma
                else:
                    probs[1]+=.5
                a=int(rng.choice(5,p=probs))
                owed=max(0,int(s[g.S_CURBET]-s[g.S_BETS+seat]))
                if a==0:action={'kind':'fold'}
                elif a==1:action={'kind':'call' if owed else 'check'}
                else:
                    chips=g._bet_chip_amount(a,int(s[g.S_POT]),owed,int(s[g.S_STACKS+seat]))
                    action={'kind':'raise','to':int(s[g.S_BETS+seat]+chips)}
                history.append(dict(action,seat=seat))
                g.step(s,deck,hole,a)
                if len(history)>512:raise RuntimeError('Nonterminating training hand')
            counts['hands']+=1
    finally:
        bridge.close()
    result=dict(counts=counts,examples=examples,policy_iteration=args.policy_iteration,repair_pending=args.repair_pending,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitation='Synthetic fresh histories; stop each hand at first replay failure. Policy mode uses the actual importance-collector mixture with a fixed checkpoint; random mode uses call/uniform. Neither estimates exact contamination of historical replay or regret trees. Public-state/feature comparison, not exhaustive payout parity.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts),flush=True)


if __name__=='__main__':
    main()
