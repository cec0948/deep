"""Independently verify strict river reach, public inference and paired terminal chip settlement."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime
import gzip
from itertools import combinations
import json
import math
from pathlib import Path
import random
import statistics

import eval7
import numpy as np

from audit_poker_flop_repair import nonce, require
from audit_poker_river_dominance import fixtures
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import semantic_seed, _validate_reply
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_flop_repair import FlopRepairPolicy
from poker_river_call_learning import eligible, features, RiverCallPolicy
from poker_river_dominance import guard_fold
from poker_river_runtime import action_key
from train_poker_flop_repair import sources

REL = Path('preflop-followup/480-v1/river-call-learning-20261126')
STUDY, REPORT = ROOT/'artifacts'/REL, ROOT/'reports'/REL
SELECTED = STUDY.parent/'street-progress-20261126/flop/selected.json'
DESIGN = 'ea1db38fc35ae39a62ba234c8a1117d5ac12f6b836788c4002e9258b4eabbed8'
SEEDS = (342,343,344)
CARDS = list(eval7.Deck().cards)
PANEL = (480,480,480,300,360,420)


def geometry(s):
    if s['terminal'] or s['street']!='river' or s['amount_owed']<=0:
        return False
    hero = s['seat_to_act']
    live = [p for p in s['players'] if not p['is_folded']]
    if len(live)!=2 or len(s['players'])!=6 or hero not in [p['seat'] for p in live]:
        return False
    villain = next(p for p in live if p['seat']!=hero)
    contributions = {p['seat']:10000-p['stack'] for p in s['players']}
    owed = s['amount_owed']
    if not (owed<=s['your_stack'] and villain['bet_this_street']==s['your_bet_this_street']+owed
            and sum(contributions.values())==s['pot'] and max(contributions.values())==contributions[villain['seat']]
            and contributions[hero]+owed==contributions[villain['seat']]):
        return False
    river = []
    for event in reversed(s['action_log']):
        if event['action']=='deal':
            break
        river.append(event)
    wagers = [e['seat'] for e in river if e['action'] in ('raise','bet')]
    return wagers==[villain['seat']] and all(e['action']=='check' for e in river if e.get('seat')==hero)


def five(cards):
    ranks = ['23456789TJQKA'.index(c[0])+2 for c in cards]
    descending = sorted(set(ranks),reverse=True)
    order = sorted(((n,r) for r,n in Counter(ranks).items()),reverse=True)
    straight = descending[0] if len(descending)==5 and descending[0]-descending[-1]==4 else 0
    if descending==[14,5,4,3,2]:
        straight=5
    flush = len({c[1] for c in cards})==1
    if straight and flush:
        return 8,straight
    if order[0][0]==4:
        return 7,order[0][1],order[1][1]
    if order[0][0]==3 and order[1][0]==2:
        return 6,order[0][1],order[1][1]
    if flush:
        return (5,*descending)
    if straight:
        return 4,straight
    if order[0][0]==3:
        return (3,*(r for n,r in order))
    if order[0][0]==2:
        return (2 if order[1][0]==2 else 1,*(r for n,r in order))
    return (0,*descending)


def payouts(row):
    s = row['root_observation']
    hero, opp = row['hero'],row['opponent']
    board, hole, other = s['community_cards'],s['your_cards'],row['opponent_cards']
    require(len(hole)==len(other)==2 and len(board)==5 and len(set(board+hole+other))==9,'Invalid deck')
    a, b = [max(five(c) for c in combinations(h+board,5)) for h in (hole,other)]
    share = .5 if a==b else float(a>b)
    commits = [10000-p['stack'] for p in s['players']]
    commits[hero] += s['amount_owed']
    won, prev = 0, 0
    for level in sorted(set(commits)-{0}):
        layer = sum(x>=level for x in commits)*(level-prev)
        require(commits[hero]>=level and commits[opp]>=level,'Uncovered pot layer')
        if share==1:
            won += layer
        elif share==.5:
            won += layer//2 + ((layer%2) if hero<opp else 0)
        prev=level
    fold = s['your_stack']-10000
    call = won-commits[hero]
    require((share,call,fold,call-fold)==tuple(row[k] for k in ('showdown_share','call_utility_chips','fold_utility_chips','call_minus_fold_chips')),
            'Seven-card rank, odd-chip layer split, or settlement differs')
    return share


def lineup(seed,deal,hero):
    return [480 if i==hero else PANEL[random.Random(nonce('opponent',seed,deal,i)).randrange(6)] for i in range(6)]


def actual_reach(br,seed,start,deal,parent,hero_model,pool):
    """Re-simulate registered original hands, including many without an eligible decision."""
    hero = (deal-start)%6
    stages = lineup(seed,deal,hero)
    s = br.call(dict(op='reset',seed=deal,button=0))
    seen, actions = {}, []
    while not s['terminal'] and not s['players'][hero]['is_folded'] and s['players'][hero]['stack']:
        actor = s['seat_to_act']
        seen[actor] = s['your_cards']
        allowed = geometry(s)
        require(eligible(s)==allowed,'Playing eligibility differs from independent public geometry')
        if actor==hero and allowed:
            opp = next(p['seat'] for p in s['players'] if p['seat']!=hero and not p['is_folded'])
            return actions,s,seen[opp]
        bot = parent if s['street']=='preflop' else hero_model if actor==hero else pool[stages[actor]]
        reply = bot.decide(s,semantic_seed(deal,hero,s))
        if actor==hero:
            reply = guard_fold(s,reply)
        action = dict(reply['action'],seat=actor)
        actions.append(action)
        s = br.call(dict(op='step',action=action))
    return None


def public_features(s):
    board, hole = [eval7.Card(c) for c in s['community_cards']], [eval7.Card(c) for c in s['your_cards']]
    h = eval7.evaluate(board+hole)
    unseen = [c for c in CARDS if c not in board+hole]
    ranks = [eval7.evaluate(board+[a,b]) for a,b in combinations(unseen,2)]
    require(len(ranks)==990,'Visibility range must have 990 legal villain holes')
    outcomes = Counter((h>r)-(h<r) for r in ranks)
    p = max(.0005,min(.9995,(outcomes[1]+.5*outcomes[0])/990))
    hero = s['seat_to_act']
    villain = next(a for a in s['players'] if not a['is_folded'] and a['seat']!=hero)
    boundary=max(i for i,a in enumerate(s['action_log']) if a['action']=='deal')
    prior = Counter((a.get('seat'),a['action']) for a in s['action_log'][:boundary])
    return np.array([math.log(p/(1-p)),s['amount_owed']/(s['pot']+s['amount_owed']),math.log1p(s['pot']/100),
                     villain['is_all_in'],any(a.get('seat')==hero and a['action']=='check' for a in s['action_log'][boundary+1:]),
                     sum(prior[villain['seat'],k] for k in ('raise','bet')),sum(prior[hero,k] for k in ('raise','bet')),
                     prior[hero,'call'],len({c[0] for c in s['community_cards']}),max(Counter(c[1] for c in s['community_cards']).values())]),outcomes


def distribution(reply):
    rows=reply['distribution']
    d={action_key(r['action']):r['probability'] for r in rows}
    require(len(d)==len(rows) and all(0<=v<=1 for v in d.values()) and abs(sum(d.values())-1)<1e-7,'Invalid action distribution')
    return d


def verify_fit(candidate,train_checks):
    """Check the frozen first-order optimum on only the registered public training inputs."""
    weights=read(candidate['model']['path'])
    centers,ranges,coefs=[np.array(weights[k]) for k in ('center','scale','coefficients')]
    rows=[]; shares=[]
    for seed in SEEDS:
        ledger=STUDY/'train'/f'seed-{seed}'/'positions.jsonl.gz'
        recorded=train_checks[str(seed)]['ledger_sha256']
        verify_files({ledger:recorded})
        require(candidate['seeds'][str(seed)]['train_ledger_sha256']==recorded,'Wrong fitted ledger')
        with gzip.open(ledger,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                x,_=public_features(row['root_observation'])
                rows.append(x); shares.append(row['showdown_share'])
    x=np.asarray(rows); y=np.array(shares)
    require(np.allclose(x.mean(axis=0),centers,rtol=0,atol=1e-12) and
            np.allclose(np.maximum(x.std(axis=0),.1),ranges,rtol=0,atol=1e-12) and
            len(y)==candidate['training_positions'],'Wrong training features, centering or batch')
    z=(x-centers)/ranges
    logits=x[:,0]+weights['intercept']+z@coefs
    loss=float(np.mean(np.logaddexp(0.,logits)-y*logits)+.01*coefs@coefs)
    residual=1/(1+np.exp(-logits))-y
    gradient=np.r_[np.mean(residual),np.mean(residual[:,None]*z,axis=0)+.02*coefs]
    require(float(np.abs(gradient).max())<1e-6 and
            abs(candidate['fit_metrics']['training_regularized_objective']-loss)<1e-10,
            'Frozen learner is not the stationary point of the registered train-only objective')
    return dict(training_rows=len(y),public_model_inputs=x.shape[1],regularized_loss=loss,
                maximum_absolute_gradient=float(np.abs(gradient).max()))


def verify_candidate(learner,model,row,x,outcomes):
    s=row['root_observation']; g=row['deal_seed']
    base_reply=learner.parent.decide(s,g)
    before=distribution(base_reply)
    require(before==distribution({'distribution':row['parent_distribution']}),'Wrong protected parent')
    if outcomes[-1]:
        protected=False
    else:
        commits=[10000-p['stack']+(s['amount_owed'] if p['seat']==row['hero'] else 0) for p in s['players']]
        last=0; half=0
        for level in sorted(set(commits)-{0}):
            half+=(level-last)*sum(c>=level for c in commits)//2
            last=level
        protected=(half if outcomes[0] else sum(commits))>s['amount_owed']
    center,scale,coef=[np.array(model[name]) for name in ('center','scale','coefficients')]
    logit=x[0]+model['intercept']+sum(coef*((x-center)/scale))
    owed=s['amount_owed']; pot=s['pot']
    advantage=logit-math.log(owed/pot)
    fold=('fold',None); call=('call',None)
    delta=0. if protected or advantage==0 else .25*before.get(fold,0.) if advantage>0 else -.25*before.get(call,0.)
    after_reply=_validate_reply(s,learner.decide(s,g)); after=distribution(after_reply)
    expected={**before,call:before.get(call,0.)+delta,fold:before.get(fold,0.)-delta}
    require(all(abs(expected.get(k,0)-after.get(k,0))<2e-13 for k in set(expected)|set(after)),
            'Candidate differs from independently reconstructed public-only logistic calculation')
    if base_reply['action']['kind']=='raise' or after_reply['action']['kind']=='raise':
        require(base_reply['action']==after_reply['action'],'Original sampled raise changed')
    return delta*row['call_minus_fold_chips']/100., bool(protected)


def verify_routing(learner,br,seed):
    protected=0
    for spec in fixtures().values():
        state=br.call(dict(op='replay',**spec))
        require(learner.decide(state,seed)==learner.parent.decide(state,seed),'Protected playing fixture changed')
        protected+=1
    # A min-raise and a further opposing min-raise can reach the same hero
    # again on this river. This real legal hand must take the parent route.
    raw=STUDY/'train'/f'seed-{seed}'/'positions.jsonl.gz'
    with gzip.open(raw,'rt') as stream:
        for line in stream:
            row=json.loads(line); state=row['root_observation']
            if state['raise_bounds'] is None:
                continue
            state=br.call(dict(op='reset',seed=row['deal_seed'],button=0))
            for action in row['prefix']:
                state=br.call(dict(op='step',action=action))
            for _ in range(2):
                if state['terminal'] or state['raise_bounds'] is None:
                    break
                state=br.call(dict(op='step',action=dict(seat=state['seat_to_act'],kind='raise',to=state['raise_bounds'][0])))
            else:
                require(not state['terminal'] and state['seat_to_act']==row['hero'] and
                        not geometry(state) and not eligible(state) and
                        learner.decide(state,seed)==learner.parent.decide(state,seed),'Unexpected second eligible river decision')
                return dict(protected_cases=protected,legal_reraise_fallback=1)
    raise ValueError('No live re-raise fixture in the training split')


def check_seed(study,split,seed,candidate=None):
    folder=STUDY/split/f'seed-{seed}'
    receipt=read(folder/'collection-complete.json'); raw=folder/'positions.jsonl.gz'
    cfg=study['collection'][split]; bounds=cfg['by_seed'][str(seed)]; start,limit=bounds['start'],bounds['limit']
    parent_ref=read(SELECTED)['policies'][str(seed)]
    _, ref, _, parent, pool=sources(seed)
    verify_files({raw:receipt['position_ledger_sha256'], parent_ref['path']:receipt['hero_v1_sha256'],
                  ref['path']:receipt['preflop_parent_sha256']})
    require(receipt['complete'] and receipt['design_sha256']==DESIGN and receipt['first_card']==start and
            receipt['attempted_deals']==receipt['last_card']-start+1,'Wrong collector input or scan denominator')
    hmodel=FlopRepairPolicy(parent_ref['path'])
    learner=None
    if candidate:
        learned=candidate['policies'][str(seed)]
        verify_files({learned['path']:learned['sha256'],candidate['model']['path']:candidate['model']['sha256']})
        learner=RiverCallPolicy(learned['path']); model=read(candidate['model']['path'])
    previous=start-1; hit={}; gains=[]; counts=Counter(); n=cfg['positions_per_seed']
    scan_ids=set(random.Random(seed*91).sample(range(start,receipt['last_card']+1),64))
    with closing(Bridge()) as br,gzip.open(raw,'rt') as src:
        for i,line in enumerate(src):
            row=json.loads(line); deal=row['deal_seed']; hero=row['hero']; s=row['root_observation']
            require(previous<deal<limit and hero==(deal-start)%6 and row['seed']==seed and row['split']==split and
                    s['seat_to_act']==hero and geometry(s) and eligible(s),'Ineligible, duplicate, or reused root')
            previous=deal
            require(lineup(seed,deal,hero)==row['opponents_by_seat'],'Historical lineup differs')
            request=dict(op='replay',hero=hero,button=0,hole=s['your_cards'],board=s['community_cards'],actions=row['prefix'])
            require(br.call(request)==s,'Player observation contains concealed information or differs from public reconstruction')
            original=br.call(dict(op='reset',button=0,seed=deal)); opponent_cards=None
            for move in row['prefix']:
                if move['seat']==row['opponent']:
                    opponent_cards=original['your_cards']
                original=br.call(dict(op='step',action=move))
            require(original==s and opponent_cards==row['opponent_cards'],'Privileged label cards differ from the original hand')
            share=payouts(row)
            counts['ties']+=int(share==.5)
            counts['terminal_utilities_reconstructed']+=2
            if i in (0,n//2,n-1) or deal in scan_ids:
                hit[deal]=(row['prefix'],s,row['opponent_cards']); scan_ids.add(deal)
            if learner is not None:
                encoded,outcomes=public_features(s)
                require(np.allclose(encoded,features(s),atol=1e-12,rtol=0),'Engine feature matrix is not reproducible from actor view')
                gain,protected=verify_candidate(learner,model,row,encoded,outcomes)
                gains.append(gain); counts['truly_protected_roots']+=int(protected)
            counts['positions']+=1
        for deal in sorted(scan_ids):
            require(actual_reach(br,seed,start,deal,parent,hmodel,pool)==hit.get(deal),
                    'Missing, extra or nonfirst reached event, or incorrect actual hole cards')
        if learner is not None:
            counts.update(verify_routing(learner,br,seed))
    require(counts['positions']==receipt['positions']==n and previous==receipt['last_card'],'Incorrect scan endpoints or position count')
    out=dict(counts,original_deals_scanned=receipt['attempted_deals'], ledger_sha256=digest(raw),
             sampled_original_hands_completely_replayed=len(scan_ids),sampled_reached_deals=len(hit))
    if gains:
        out.update(mean_bb=statistics.mean(gains),sample_variance=statistics.variance(gains),
                   fraction_eligible=n/receipt['attempted_deals'],
                   observed_contribution_bb_per_100_dealt=100*sum(gains)/receipt['attempted_deals'])
    return out


def run(split):
    verify_files({STUDY/'design.json':DESIGN,REPORT/'design.json':DESIGN})
    design=read(STUDY/'design.json')
    verify_files({SELECTED:design['flop_selection_sha256'],ROOT/'scripts/poker_river_dominance.py':design['guard_sha256']})
    intervals=sorted((v['start'],v['limit']) for r in design['collection'].values() for v in r['by_seed'].values())
    require(len(intervals)==6 and all(a<b for a,b in intervals) and all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),
            'Overlapping train and held-out reservations')
    candidate=None
    if split=='holdout':
        selection=read(STUDY/'selection.json'); candidate=read(STUDY/'candidate.json')
        require(selection['complete'] and candidate['training_complete'] and candidate['design_sha256']==DESIGN and
                selection['candidate_sha256']==digest(STUDY/'candidate.json'),'Wait until independent holdout scoring and fit freeze')
        train_checks=read(STUDY/'train-independent.json')['seeds']
        fit_check=verify_fit(candidate,train_checks)
        frozen=datetime.fromisoformat(candidate['frozen_utc'])
        scored=datetime.fromisoformat(selection['completed_utc'])
        completed={s:datetime.fromisoformat(read(STUDY/'holdout'/f'seed-{s}'/'collection-complete.json')['completed_utc'])
                   for s in SEEDS}
        require(frozen<=scored and max(completed.values())<=scored,'Scoring predates fit freeze or terminal label collection')
    per_seed={str(s):check_seed(design,split,s,candidate) for s in SEEDS}
    result=dict(verified_utc=utc(),design_sha256=DESIGN,split=split,seeds=per_seed,auditor_sha256=digest(__file__))
    if split=='holdout':
        mean=statistics.mean(s['mean_bb'] for s in per_seed.values())
        se=math.sqrt(sum(s['sample_variance']/s['positions'] for s in per_seed.values()))/3
        ci=[mean-1.96*se,mean+1.96*se]
        official=selection['paired_gain']
        for k,v in zip(('mean','standard_error','lower','upper'),(mean,se,*ci)):
            require(abs(official[k]-v)<1.e-9,'Overall held-out estimate or sampling unit differs')
        for seed,s in per_seed.items():
            require(abs(official['by_seed'][seed]['mean']-s['mean_bb'])<1e-9,'Seed estimate differs')
        require(selection['selected']==('candidate' if ci[0]>0 else 'parent'),'Unexpected playing-policy selection')
        result.update(complete=True,selected=selection['selected'],selected_policy_sha256=selection['policy_sha256'],
                      conditional_gain_bb=mean,conditional_se=se,conditional_ci=ci,
                      observed_dealt_contribution_bb_per_100_equal_seed=statistics.mean(s['observed_contribution_bb_per_100_dealt'] for s in per_seed.values()),
                      interpretation='Contribution from first eligible call/fold policy on sampled original deals, counting all other deals as zero. No entire-game win rate is estimated.',
                      freeze_and_scoring=dict(model_frozen_utc=frozen.isoformat(),evaluation_completed_utc=scored.isoformat(),
                                              holdout_labels_completed_before_freeze_seeds=[s for s,t in completed.items() if t<frozen],
                                              model_reconstructed_from_training_only=True),
                      training_check=train_checks, fit_reconstruction=fit_check)
        for path in (STUDY/'independent-audit.json',REPORT/'independent-audit.json'):
            atomic_json(path,result)
    else:
        atomic_json(STUDY/'train-independent.json',result)
    print(json.dumps(result,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=('train','holdout'))
    run(parser.parse_args().split)
