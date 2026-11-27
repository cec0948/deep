"""Reconstruct the diverse river fit and its selection using the original deals."""
from argparse import ArgumentParser
from contextlib import closing
from datetime import datetime
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np
from scipy.special import expit

from audit_poker_flop_repair import require
from audit_poker_river_call_learning import (distribution, geometry, lineup, payouts, public_features)
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_reference_opponents import ReferenceOpponent
from poker_river_call_diverse import DiverseRiverCallPolicy
from poker_river_call_learning import RiverCallPolicy
from poker_solverlike_opponent import SolverLikeOpponent
from run_poker_candidate_confirmation import prior_card_ranges
from train_poker_flop_repair import sources


REL = Path('preflop-followup/480-v1/river-call-continuation-20261127')
STUDY, REPORT = ROOT/'artifacts'/REL, ROOT/'reports'/REL
PRIOR = STUDY.parent/'river-call-learning-20261126'
DESIGN = 'f5a05053ba0f296de162a6b777cca753527210b49fc92b2c6c28c2f761c84969'
SEEDS = (342,343,344)
FOLD, CALL = ('fold',None), ('call',None)


def preflight():
    verify_files({folder/'design.json':DESIGN for folder in (STUDY,REPORT)})
    d=read(STUDY/'design.json')
    previous=read(PRIOR/'candidate.json')
    verify_files({PRIOR/'play-ready.json':d['parent_play_ready_sha256'],
                  previous['model']['path']:d['parent_model_sha256']})
    for sid, sha in d['parent_policy_sha256'].items():
        verify_files({PRIOR/'play'/f'seed-{sid}'/'policy.json':sha,
                      PRIOR/'train'/f'seed-{sid}'/'positions.jsonl.gz':d['parent_train_ledger_sha256'][sid]})
    old={tuple(entry['range']) for entry in prior_card_ranges(STUDY).values()}
    flop=read(STUDY.parent/'flop-repair-20261126/training-design.json')
    old.update((v['train_card_start'],v['train_card_exclusive_limit']) for v in flop['seed_sources'].values())
    streets=read(STUDY.parent/'street-progress-20261126/design.json')
    small=read(STUDY.parent/'river-natural-20261126/design.json')
    old.update((start,start+small['deals_per_seed']) for start in small['start'].values())
    for specs in [s for st in streets['collection'].values() for s in st.values()]+list(read(PRIOR/'design.json')['collection'].values()):
        old.update((v['start'],v['limit']) for v in specs['by_seed'].values())
    new=sorted((v['start'],v['limit']) for split in d['collection'].values()
               for group in split['by_group'].values() for v in group.values())
    require(len(new)==15 and all(a<b for a,b in new) and all(x[1]<=y[0] for x,y in zip(new,new[1:]))
            and all(a>=d1 or b<=c for a,b in new for c,d1 in old),'Overlapping card reservations')
    identities=d['group_policy_sha256']
    unseen={identities[g] for g in d['collection']['holdout']['groups']}
    history={v for s in flop['seed_sources'].values()
             for v in (*s['historical_selfplay_sha256'].values(),s['postflop_sha256'],s['parent_policy_sha256'])}
    training={identities['pressure'],identities['tight-value'],*identities['snapshot-420'].values()}
    require(len(unseen)==2 and unseen.isdisjoint(history|training),'Repeated held-out opponent source')
    return d,len(old)


@lru_cache(maxsize=3)
def opponents(seed):
    _, _, _, opening, past = sources(seed)
    external={'pressure':ReferenceOpponent('pressure'),'tight-value':ReferenceOpponent('tight-value'),
              'snapshot-420':past[420], 'public-fullhouse':Fullhouse(), 'solverlike':SolverLikeOpponent()}
    return opening,past,external


def replay_trajectory(engine, seed, split, group, start, deal, hero_model):
    """Replay an original unfiltered hand, independently assigning every policy."""
    hero=(deal-start)%6
    opening,past,external=opponents(seed)
    by_seat=lineup(seed,deal,hero)
    s=engine.call(dict(op='reset',button=0,seed=deal))
    moves=[]
    seen={}
    while not s['terminal'] and not s['players'][hero]['is_folded'] and s['players'][hero]['stack']:
        actor=s['seat_to_act']
        seen[actor]=s['your_cards']
        if actor==hero and geometry(s):
            villain=next(p['seat'] for p in s['players'] if not p['is_folded'] and p['seat']!=hero)
            return moves,s,seen[villain]
        if actor==hero:
            policy=hero_model
        elif split=='holdout' or s['street']=='river':
            policy=external[group]
        elif s['street']=='preflop':
            policy=opening
        else:
            policy=past[by_seat[actor]]
        move=dict(policy.decide(s,semantic_seed(deal,hero,s))['action'],seat=actor)
        moves.append(move)
        s=engine.call(dict(op='step',action=move))
    return None


def check_cards(engine,row):
    s=row['root_observation']
    hero,villain=row['hero'],row['opponent']
    obs=engine.call(dict(op='replay',hero=hero,button=0,hole=s['your_cards'],
                         board=s['community_cards'],actions=row['prefix']))
    require(s==obs and geometry(s),'State fails actor-only reconstruction or river closure')
    state=engine.call(dict(op='reset',button=0,seed=row['deal_seed']))
    hidden=None
    for action in row['prefix']:
        if state['seat_to_act']==villain:
            hidden=state['your_cards']
        state=engine.call(dict(op='step',action=action))
    require(state==s and hidden==row['opponent_cards'],'Wrong original deal or private label')
    return payouts(row)


def independent_shift(row,learner,parent,model):
    state=row['root_observation']
    x,outcomes=public_features(state)
    amount=state['amount_owed']
    commitments=[10000-p['stack']+(amount if p['seat']==row['hero'] else 0) for p in state['players']]
    prev=payout=0
    for level in sorted(set(commitments)-{0}):
        tier=(level-prev)*sum(v>=level for v in commitments)
        payout+=tier//2 if outcomes[0] else tier
        prev=level
    protected=not outcomes[-1] and payout>amount
    z=(x-np.array(model['center']))/np.array(model['scale'])
    q=float(expit(x[0]+model['intercept']+np.dot(z,model['coefficients'])))
    pot_odds=amount/(state['pot']+amount)
    preferred=None
    if not protected:
        if q>=pot_odds+.05:
            preferred=CALL
        elif q<=pot_odds-.05:
            preferred=FOLD
    # An independent coupling uses the live parent's sampled action (including any
    # raise) and a separate namespace for the newly eligible passive shift.
    key=semantic_seed(row['deal_seed'],row['hero'],state)
    old=parent.decide(state,key)
    before=distribution(old)
    require(before==distribution(dict(distribution=row['parent_distribution'])),'Wrong incumbent distribution')
    shift=.25*before.get(FOLD,0.) if preferred==CALL else -.25*before.get(CALL,0.) if preferred==FOLD else 0.
    expected={**before,FOLD:before.get(FOLD,0.)-shift,CALL:before.get(CALL,0.)+shift}
    reply=learner.decide(state,key)
    actual=distribution(reply)
    require(all(abs(actual.get(a,0.)-expected.get(a,0.))<1.e-12 for a in expected.keys()|actual.keys()),
            'Learned distribution leaked information or changed protected mass')
    action=old['action']
    if (shift and action['kind'] in ('fold','call') and action['kind']!=preferred[0]
            and random.Random(f'river-call-passive:continuation:{key}').random()<.25):
        action={'kind':preferred[0]}
    require(action==reply['action'],'Sampled action coupling disagrees')
    old_share=float(expit(x[0]+parent.intercept+np.dot(z,parent.coef)))
    old_shift=0.
    if not protected and old_share>=pot_odds+.05:
        old_shift=.25*before.get(FOLD,0.)
    if not protected and old_share<=pot_odds-.05:
        old_shift=-.25*before.get(CALL,0.)
    price=row['call_minus_fold_chips']/100.
    return shift*price,old_shift*price,abs(shift-old_shift)>1e-12,protected


def check_cell(d,split,group,seed,learner=None,model=None):
    path=STUDY/split/group/f'seed-{seed}'
    ledger=path/'positions.jsonl.gz'
    rec=read(path/'collection-complete.json')
    verify_files({ledger:rec['position_ledger_sha256']})
    cfg=d['collection'][split]
    bounds=cfg['by_group'][group][str(seed)]
    start,limit=bounds['start'],bounds['limit']
    n=cfg['positions_per_seed_group']
    source=d['group_policy_sha256'][group]
    group_sha=source[str(seed)] if isinstance(source,dict) else source
    require(rec['river_policy_sha256']==group_sha and
            rec['parent_policy_sha256']==d['parent_policy_sha256'][str(seed)],
            'Collector used a different opponent or parent')
    require(rec['first_card']==start and rec['attempted_deals']==rec['last_card']-start+1
            and rec['complete'] and rec['positions']==n and rec['design_sha256']==DESIGN,'Wrong collection window')
    parent=RiverCallPolicy(PRIOR/'play'/f'seed-{seed}'/'policy.json')
    sampled=set(random.Random(start+37).sample(range(start,rec['last_card']+1),32))
    hits={}
    gains=[]
    old_weight_gains=[]
    different_incremental_updates=0
    ties=protected=0
    odds_bands=[0,0,0]
    sum_showdown_share=0.
    last=start-1
    rows=0
    with closing(Bridge()) as engine,gzip.open(ledger,'rt') as stream:
        for i,line in enumerate(stream):
            row=json.loads(line)
            deal=row['deal_seed']
            require(last<deal<limit and (deal-start)%6==row['hero'] and (group,seed,split)==
                    (row['group'],row['seed'],row['split']), 'Duplicate, crossed or mislabeled deal')
            last=deal
            rows+=1
            require(row['opponents_by_seat']==(lineup(seed,deal,row['hero']) if split=='train' else None) and
                    row['river_policy_sha256']==group_sha and row['parent_policy_sha256']==parent.digest,
                    'Inconsistent seated opponent or parent')
            share=check_cards(engine,row)
            ties+=int(share==.5)
            sum_showdown_share+=share
            s=row['root_observation']
            price=s['amount_owed']/(s['pot']+s['amount_owed'])
            odds_bands[0 if price<.2 else 1 if price<1/3 else 2]+=1
            if i in (0,n//4,n//2,3*n//4,n-1) or deal in sampled:
                sampled.add(deal)
                hits[deal]=row['prefix'],row['root_observation'],row['opponent_cards']
            if learner is not None:
                gain,old_gain,different,active=independent_shift(row,learner,parent,model)
                gains.append(gain)
                old_weight_gains.append(old_gain)
                different_incremental_updates+=int(different)
                protected+=int(active)
        for deal in sorted(sampled):
            actual=replay_trajectory(engine,seed,split,group,start,deal,parent)
            require(actual==hits.get(deal), f'Organic parent/opponent reach mismatch {split}/{group}/{seed} {deal}')
    require(rows==n and last==rec['last_card'],'Incomplete cell')
    proof=dict(original_deals_scanned=rec['attempted_deals'],positions=rows,
               raw_sha256=digest(ledger),independent_chip_outcomes=2*rows,ties=ties,
               mean_showdown_share=sum_showdown_share/rows,required_call_equity_bands=odds_bands,
               sampled_full_policy_trajectories=len(sampled),sampled_trajectories_with_root=len(hits))
    if gains:
        refit_gains=[gain-reference for gain,reference in zip(gains,old_weight_gains,strict=True)]
        proof.update(protected_roots=protected,mean=statistics.fmean(gains),variance=statistics.variance(gains),
                     extra_step_with_old_weights=statistics.fmean(old_weight_gains),
                     learned_minus_old_extra_step=statistics.fmean(refit_gains),
                     learned_minus_old_extra_step_variance=statistics.variance(refit_gains),
                     different_incremental_updates=different_incremental_updates,
                     observed_contribution_bb_per_100_deals=100*math.fsum(gains)/rec['attempted_deals'])
    return proof


def fit_check(d,candidate,train):
    """Rebuild the convex training objective without the fitter or its public feature code."""
    parent=read(PRIOR/'models/model.json')
    learned=read(candidate['model']['path'])
    verify_files({candidate['model']['path']:candidate['model_sha256'], PRIOR/'models/model.json':d['parent_model_sha256']})
    require(candidate['source_model_sha256']==learned['source_model_sha256']==d['parent_model_sha256'] and
            parent['center']==learned['center'] and parent['scale']==learned['scale']
            and learned['features']==parent['features'],'Changed normalization, parent weights or feature map')
    groups=d['collection']['train']['groups']
    cells=[(PRIOR/'train'/f'seed-{s}'/'positions.jsonl.gz',d['parent_train_ledger_sha256'][str(s)],.5/3) for s in SEEDS]
    cells.extend((STUDY/'train'/g/f'seed-{s}'/'positions.jsonl.gz',train[g][str(s)]['raw_sha256'],.5/(len(groups)*3))
                 for g in groups for s in SEEDS)
    inputs=[]; labels=[]; weight=[]
    for path,checksum,fraction in cells:
        verify_files({path:checksum})
        with gzip.open(path,'rt') as src:
            shares=[]
            for line in src:
                row=json.loads(line)
                x,_=public_features(row['root_observation'])
                inputs.append(x)
                shares.append(row['showdown_share'])
        labels.extend(shares)
        weight.extend([fraction/len(shares)]*len(shares))
    x=np.array(inputs)
    y=np.array(labels)
    w=np.array(weight)
    z=(x-np.array(parent['center']))/np.array(parent['scale'])
    before=x[:,0]+parent['intercept']+z@parent['coefficients']
    b=learned['intercept']-parent['intercept']
    slopes=np.array(learned['coefficients'])-np.array(parent['coefficients'])
    after=before+b+z@slopes
    initial=float(w@(np.logaddexp(0.,before)-y*before))
    loss=float(w@(np.logaddexp(0.,after)-y*after)+.025*(slopes@slopes)+.005*b*b)
    errors=w*(expit(after)-y)
    grad=np.r_[errors.sum()+.01*b,errors@z+.05*slopes]
    metrics=candidate['fit_metrics']
    require(candidate['training_positions']==len(y) and abs(w.sum()-1.)<1e-12 and
            max(abs(grad))<1e-6 and abs(loss-metrics['fitted_training_objective'])<1e-10 and
            abs(initial-metrics['source_training_objective'])<1e-10 and loss<initial,
            'Reported fit fails the registered optimum or its warm initialization')
    return dict(train_rows=len(y),source_model_sha256=d['parent_model_sha256'],candidate_model_sha256=candidate['model_sha256'],
                new_source_weight=.5,original_source_weight=.5,public_features=x.shape[1],
                source_objective=initial,final_objective=loss,max_abs_gradient=float(max(abs(grad))))


def run(split):
    d,old_ranges=preflight()
    cfg=d['collection'][split]
    candidate=None
    if split=='holdout':
        candidate=read(STUDY/'candidate.json')
        selection=read(STUDY/'selection.json')
        require(candidate['training_complete'] and selection['complete'] and
                candidate['design_sha256']==selection['design_sha256']==DESIGN and
                selection['candidate_sha256']==digest(STUDY/'candidate.json') and
                datetime.fromisoformat(candidate['frozen_utc'])<=datetime.fromisoformat(selection['completed_utc']),
                'Fixed candidate and completed one-shot scoring are required before independent outcome access')
    model=read(candidate['model']['path']) if candidate else None
    rows={}
    for group in cfg['groups']:
        rows[group]={}
        for seed in SEEDS:
            learner=DiverseRiverCallPolicy(candidate['policies'][str(seed)]['path']) if candidate else None
            rows[group][str(seed)]=check_cell(d,split,group,seed,learner,model)
    result=dict(verified_utc=utc(),design_sha256=DESIGN,split=split,auditor_sha256=digest(__file__),
                disjoint_prior_ranges=old_ranges,disjoint_registered_ranges=15,groups=rows,
                price_bins='Required equity owed/(pot+owed), counted in <20%, 20% to <1/3, and >=1/3.')
    if split=='train':
        atomic_json(STUDY/'train-independent.json',result)
        return result
    train=read(STUDY/'train-independent.json')
    require(train['design_sha256']==DESIGN,'Invalid training audit')
    fit=fit_check(d,candidate,train['groups'])
    pooled=[x for seed_rows in rows.values() for x in seed_rows.values()]
    def summarize(cells):
        gain=statistics.fmean(x['mean'] for x in cells)
        se=math.sqrt(math.fsum(x['variance']/x['positions'] for x in cells))/len(cells)
        dealt=statistics.fmean(x['observed_contribution_bb_per_100_deals'] for x in cells)
        return dict(mean=gain,standard_error=se,lower=gain-1.96*se,upper=gain+1.96*se,
                    observed_contribution_bb_per_100=dealt)
    official=selection['paired_gain']
    overall=summarize(pooled)
    for scope,summary in [(official,overall)]+[(official['by_opponent'][g],summarize(list(group.values()))) for g,group in rows.items()]:
        require(all(abs(scope[key]-value)<1e-10 for key,value in summary.items()),'Different stratified estimate')
    name='candidate' if overall['lower']>0. and overall['observed_contribution_bb_per_100']>0. else 'parent'
    expected=candidate['policy_sha256'] if name=='candidate' else d['parent_policy_sha256']
    require(selection['selected']==name and selection['policy_sha256']==expected,'Incorrect deployment selection')
    parts={k:statistics.fmean(x[k] for x in pooled)
           for k in ('extra_step_with_old_weights','learned_minus_old_extra_step')}
    parts['different_incremental_updates']=sum(x['different_incremental_updates'] for x in pooled)
    diff_se=math.sqrt(math.fsum(x['learned_minus_old_extra_step_variance']/x['positions'] for x in pooled))/len(pooled)
    parts['learned_minus_old_extra_step_95_interval']=[
        parts['learned_minus_old_extra_step']-1.96*diff_se,parts['learned_minus_old_extra_step']+1.96*diff_se]
    require(abs(parts['extra_step_with_old_weights']+parts['learned_minus_old_extra_step']-overall['mean'])<1.e-12,
            'Descriptive gain components disagree')
    attribution=selection['nonselecting_attribution']
    separate=attribution['continued_minus_old_coefficients_with_same_extra_update']
    require(abs(attribution['extra_update_with_old_coefficients']['mean']-parts['extra_step_with_old_weights'])<1e-10
            and parts['different_incremental_updates']==attribution['decisions_with_different_incremental_shift']
            and abs(separate['standard_error']-diff_se)<1e-10,'Different continued-weight attribution')
    result.update(complete=True,training=train,fit=fit,holdout_gain=overall,selected=name,
                  descriptive_incremental_contributions=parts,
                  opponent_reach='All five opponents use the same specified held-out policy on every street.',
                  scope='Descriptive per-dealt contribution under the registered tables. Conditional effects concern only first eligible closing river bets, across fixed opponent identities and fixed training seeds.')
    for folder in (STUDY,REPORT):
        atomic_json(folder/'independent-audit.json',result)
    return result


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=('train','holdout'))
    print(json.dumps(run(parser.parse_args().split),allow_nan=False))
