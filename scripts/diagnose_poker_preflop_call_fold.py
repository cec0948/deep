"""Test call versus fold at the first public divergence of the frozen 0/256-step policies."""
from collections import Counter
from contextlib import closing
import gzip
import json
import math
from pathlib import Path
from statistics import NormalDist
import os

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import evaluator_policy
from regret_partition import is_opening
from run_poker_preflop_confirmation import check_fresh_cards
from run_poker_preflop_pilot import CAMPAIGN, FIELDS, SEEDS
from train_poker_average_continuation import load_checkpoint


BASE = ROOT / 'artifacts/preflop-followup/480-v1'
OLD = BASE / 'preflop-dose'
OUT = BASE / 'call-fold-diagnosis'
START, DEALS = 2037900000, 2000
SUMMARY_SHA = '6298d5e5a89f3212bed1f706376e41b298ec06d60f70e0f32d474358946e9610'
AUDIT_SHA = '5f502fe9459f7f985166ba5b138ce35c9460161042e49c2847d4004170aa50c3'
PLAN_SHA = '642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6'
POSITIONS = ('BTN','SB','BB','UTG','HJ','CO')
COSTS = ('<=1bb','(1,3]bb','(3,10]bb','>10bb')
ODDS = ('<=20%','(20,33.333%]','>33.333%')
BB = 100


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def design():
    return dict(evaluation_deal_seed=START, evaluation_deals=DEALS, seeds=SEEDS, fields=FIELDS,
        hero_seats=6, candidate='confirmed480 versus steps256 at the first sampled action disagreement',
        selection='All common-prefix nonopening preflop 480-call/256-fold decisions before any payoff; stop on any earlier concrete action difference.',
        worlds='New independent uniform card blocks played from the start. Reaching the common decision conditions on previous actions under the actual fixed opponents and matched hero policy draws.',
        branches='Force call and fold in the same reached world, use the original 480 hero for all subsequent decisions, same opponent models and semantic action RNG protocol.',
        estimand='Mean call minus fold payoff in big blinds per selected decision, averaged over reached states and sampled future outcomes.',
        uncertainty='Ratio of total paired differences to count. Cluster influence by original card block, including zero-event blocks. Simultaneous 95% Bonferroni across three fields. Other breakdowns descriptive.',
        coverage='Count nonopening 480 replay rows by position, payable call cost and pot odds. Counts are coarse sample support, not hand-value estimates.',
        cost_bounds_bb=[1,3,10], pot_odds_bounds=[.2,1/3], new_selfplay_training_iterations=0,
        source_summary_sha256=SUMMARY_SHA, source_final_audit_sha256=AUDIT_SHA, source_evaluation_plan_sha256=PLAN_SHA)


def bucket(position, cost, pot):
    # Use integer comparisons so exact 20% or one-third pot odds never changes bins.
    price = cost + pot
    check(0 <= position < 6 and cost > 0 and pot > 0, 'Unexpected payable call or position')
    return (POSITIONS[position], COSTS[int(cost>100)+int(cost>300)+int(cost>1000)],
            ODDS[int(5*cost > price)+int(3*cost > price)])


def replay_support(seed, source_plan):
    state, checkpoint = load_checkpoint(CAMPAIGN/f'seed-{seed}')
    ref = read(source_plan['policies']['steps0'][str(seed)]['path'])
    check(state['iteration'] == 480 and state['seed'] == seed and
          ref['postflop']['sha256'] == digest(checkpoint/'current.npz'), 'Wrong replay checkpoint')
    counts = Counter()
    all_nonopening = 0
    for saved in state['buffers']:
        features = saved['features'][:saved['count']]
        rows = features[(features[:,17]==1) & ~is_opening(features)]
        all_nonopening += len(rows)
        stack = np.rint(rows[:,28]*10000).astype(int)
        pot = np.rint(rows[:,27]*10000).astype(int)
        cost = np.minimum(np.rint(rows[:,42]*stack).astype(int), stack)
        pos = np.argmax(rows[:,21:27], axis=1)
        counts.update(bucket(p,c,t) for p,c,t in zip(pos,cost,pot) if c>0)
    check(sum(read(OLD/f'seed-{seed}'/'fit.json')['selected_rows_per_seat'])==all_nonopening,'Support rows differ from fitted replay')
    return dict(selected_training_rows=all_nonopening, facing_call_rows=sum(counts.values()),
                by_position_cost_odds={'|'.join(k):v for k,v in sorted(counts.items())})


def discovery(source_summary, source_audit, plan):
    totals = {}
    for field in FIELDS:
        count = chips = 0
        for seed in SEEDS:
            paths = []
            for arm in ('steps0','steps256'):
                name = f'{field}-seed-{seed}-{arm}'
                folder = OLD/'evaluation'/name
                verify_files({folder/'results.json':source_summary['result_sha256'][name]})
                result = read(folder/'results.json')
                check(result['terminal_payoffs_replayed'] and result['candidate_sha256']==plan['policies'][arm][str(seed)]['sha256'],
                      'Unverified discovery cell')
                raw = folder/'hands.jsonl.gz'
                verify_files({raw:result['raw_hands_sha256']})
                paths.append(raw)
            with gzip.open(paths[0],'rb') as old, gzip.open(paths[1],'rb') as changed:
                size = 0
                for idx, pair in enumerate(zip(old,changed,strict=True)):
                    a,b = map(json.loads,pair)
                    i,hero=divmod(idx,6)
                    check(a['hero']==b['hero']==hero and a['seed']==b['seed']==plan['card_start']+i, 'Unmatched discovery cards')
                    size += 1
                    first = next(((x,y) for x,y in zip(a['actions'],b['actions']) if x!=y), None)
                    if first and first[0]['seat']==hero and first[1]['seat']==hero and (first[0]['kind'],first[1]['kind'])==('call','fold'):
                        count += 1
                        chips += b['chips']-a['chips']
                check(size==plan['deals']*6,'Incomplete discovery ledgers')
        expected=source_audit['nonopening_preflop_first_action_change']['256_minus_0'][field]['call_to_fold']
        impact=chips/(plan['deals']*6*len(SEEDS))
        check(count==expected['paired_hands'] and math.isclose(impact,expected['contribution_bb100'],abs_tol=1e-8),
              'Discovery call-to-fold pairs differ from original full audit')
        totals[field]=dict(call_to_fold_pairs=count, contribution_bb100_all_hands=impact)
    return totals


def fixed_action(engine, public, hero_policy, field, card, hero, state):
    actor=state['seat_to_act']
    seed=semantic_seed(card,hero,state)
    if actor==hero:
        reply=hero_policy.decide(state,seed)
    elif field=='public' or (field=='mixed' and (actor-hero)%6%2==1):
        reply=public.decide(state,seed)
    else:
        reply=engine.call(dict(op='blueprint',seed=seed))
    return dict(reply['action'],seat=actor)


def payoff(engine, public, policy, field, card, hero, forced):
    actions=[dict(kind=forced,seat=hero)]
    state=engine.call(dict(op='step',action=actions[0]))
    for _ in range(512):
        if state['terminal']:
            chips=state['utilities']
            check(len(chips)==6 and all(type(v) is int for v in chips) and sum(chips)==0,'Terminal chip conservation failed')
            return chips[hero], actions
        action=fixed_action(engine,public,policy,field,card,hero,state)
        actions.append(action)
        state=engine.call(dict(op='step',action=action))
    raise RuntimeError('Nonterminal replay')


def decision(engine, public, policies, field, card, hero):
    base, changed = policies
    state=engine.call(dict(op='reset',seed=card,button=0))
    prefix=[]
    while not state['terminal'] and state['street']=='preflop':
        action=fixed_action(engine,public,base,field,card,hero,state)
        if state['seat_to_act']==hero:
            other=dict(changed.decide(state,semantic_seed(card,hero,state))['action'],seat=hero)
            if other!=action:
                if (action['kind'],other['kind'])!=('call','fold'):
                    return None
                check(prefix and not state['can_check'] and state['amount_owed']>0, 'Illegal call versus fold')
                cost=min(state['amount_owed'],state['your_stack'])
                group=bucket((hero-state['dealer'])%6,cost,state['pot'])
                call, call_actions=payoff(engine,public,base,field,card,hero,'call')
                repeated=engine.call(dict(op='reset',seed=card,button=0))
                for prev in prefix:
                    repeated=engine.call(dict(op='step',action=prev))
                check(repeated==state,'Reset did not reproduce the same decision observation')
                fold, fold_actions=payoff(engine,public,base,field,card,hero,'fold')
                check(fold==state['your_stack']-10000,'Folded hero lost chips after folding')
                return dict(card=card, hero=hero, position=group[0], cost_bin=group[1], odds_bin=group[2],
                            cost_chips=cost,pot_chips=state['pot'],call_chips=call,fold_chips=fold,
                            prefix=prefix, call_actions=call_actions, fold_actions=fold_actions)
            if action['kind']=='fold':
                return None
        prefix.append(action)
        check(len(prefix)<=512,'Excessively long common betting prefix')
        state=engine.call(dict(op='step',action=action))
    return None


def estimate(events, totals=False):
    totals_per_card=np.zeros(DEALS)
    counts_per_card=np.zeros(DEALS,dtype=int)
    for e in events:
        i=e['card']-START
        counts_per_card[i]+=1
        totals_per_card[i]+=(e['call_chips']-e['fold_chips'])/BB
    n=len(events)
    if not n:
        return dict(events=0,represented_card_blocks=0,call_minus_fold_bb=None)
    mean=totals_per_card.sum()/n
    result=dict(events=n,represented_card_blocks=int(np.count_nonzero(counts_per_card)),call_minus_fold_bb=float(mean))
    if totals:
        # Delta method for the event-weighted mean, clustering all seats and training seeds by original card.
        residuals=totals_per_card-mean*counts_per_card
        se=math.sqrt(DEALS/(DEALS-1)*float(residuals@residuals))/n
        z=NormalDist().inv_cdf(1-.05/(2*len(FIELDS)))
        result.update(cluster_se=se, simultaneous_95_bb=[float(mean-z*se),float(mean+z*se)])
    return result


def verify_rollouts(events):
    with closing(Bridge()) as engine:
        for e in events:
            for move in ('call','fold'):
                state=engine.call(dict(op='reset',seed=e['card'],button=0))
                for action in e['prefix']:
                    check(not state['terminal'] and action['seat']==state['seat_to_act'],'Invalid common path on replay')
                    state=engine.call(dict(op='step',action=action))
                hero=e['hero']
                check(state['street']=='preflop' and state['seat_to_act']==hero and not state['can_check'] and
                      state['pot']==e['pot_chips'] and min(state['amount_owed'],state['your_stack'])==e['cost_chips'],
                      'Recorded selection state differs in replay')
                check(e[move+'_actions'][0]==dict(kind=move,seat=hero),'Incorrect forced action')
                for action in e[move+'_actions']:
                    check(not state['terminal'] and action['seat']==state['seat_to_act'],'Wrong acting seat on replay')
                    state=engine.call(dict(op='step',action=action))
                check(state['terminal'] and sum(state['utilities'])==0 and state['utilities'][hero]==e[move+'_chips'],
                      'Raw controlled payoff differs from terminal replay')


def run():
    verify_files({OLD/'summary.json':SUMMARY_SHA,OLD/'final-audit.json':AUDIT_SHA,OLD/'evaluation-plan.json':PLAN_SHA})
    plan, summary, audit=(read(OLD/n) for n in ('evaluation-plan.json','summary.json','final-audit.json'))
    verify_files(plan['frozen_files'])
    check(not OUT.exists(),'Diagnostic output already exists')
    prior=check_fresh_cards(OUT,START,DEALS)
    OUT.mkdir()
    atomic_json(OUT/'owner.json',dict(pid=os.getpid(),created_utc=utc()))
    atomic_json(OUT/'config.json',dict(**design(),created_utc=utc(),prior_card_ranges=prior,implementation_sha256=digest(__file__)))
    config_sha=digest(OUT/'config.json')
    baseline_counts=discovery(summary,audit,plan)
    coverage={str(s):replay_support(s,plan) for s in SEEDS}
    atomic_json(OUT/'discovery.json',dict(observed=baseline_counts,training_replay=coverage))
    registry_path=ROOT/'artifacts/selfplay/current.json'
    registry=read(registry_path)
    opponent=ROOT/registry['model']
    public=Fullhouse()
    verify_files({opponent:registry['sha256']})
    events=[]
    planned=DEALS*len(FIELDS)*len(SEEDS)*6
    completed=0
    raw=OUT/'matched-decisions.jsonl.gz'
    with gzip.open(raw,'xt',compresslevel=1) as stream:
        for seed in SEEDS:
            policies=[evaluator_policy(plan['policies'][arm][str(seed)]['path'],plan['source']) for arm in ('steps0','steps256')]
            for field in FIELDS:
                check(plan['opponents'][field]==dict(noregrets=registry['sha256'],public=public.digest if field!='noregrets' else None),
                      'Opponent identities differ from the source evaluation')
                with closing(Bridge(None if field=='public' else opponent)) as engine:
                    for i in range(DEALS):
                        for hero in range(6):
                            row=decision(engine,public,policies,field,START+i,hero)
                            if row is not None:
                                row.update(training_seed=seed,field=field)
                                stream.write(json.dumps(row,allow_nan=False)+'\n')
                                events.append(row)
                            completed+=1
                        if (i+1)%250==0:
                            atomic_json(OUT/'status.json',dict(observed_utc=utc(),completed_traces=completed,planned=planned,
                                       matched_decisions=len(events)))
    group_fields=('position','cost_bin','odds_bin')
    stats={}
    for f in FIELDS:
        rows=[e for e in events if e['field']==f]
        stats[f]=dict(**estimate(rows,True),planned_traces=DEALS*6*len(SEEDS),
            by_training_seed={str(s):estimate([e for e in rows if e['training_seed']==s]) for s in SEEDS},
            descriptive_groups={k:{v:estimate([e for e in rows if e[k]==v]) for v in sorted({e[k] for e in rows})}
                                for k in group_fields})
        support=[coverage[str(e['training_seed'])]['by_position_cost_odds'].get('|'.join(e[k] for k in group_fields),0) for e in rows]
        stats[f]['training_rows_in_same_position_cost_odds_bucket']=dict(minimum=min(support),median=float(np.median(support)),
            events_in_buckets_under_100=sum(v<100 for v in support),events_in_buckets_with_zero_rows=sum(v==0 for v in support))
    with gzip.open(raw,'rt') as stream:
        rebuilt=[json.loads(line) for line in stream]
    check(rebuilt==events and completed==planned, 'Incomplete paired outcome readback')
    verify_rollouts(rebuilt)
    verify_files(plan['frozen_files'])
    verify_files({OUT/'config.json':config_sha,OLD/'summary.json':SUMMARY_SHA,OLD/'final-audit.json':AUDIT_SHA,opponent:registry['sha256']})
    atomic_json(OUT/'summary.json',dict(completed_utc=utc(),completed=True,plan_sha256=config_sha,source_summary_sha256=SUMMARY_SHA,
        source_audit_sha256=AUDIT_SHA,prior_discovery=baseline_counts,training_replay=coverage,first_card=START,
        last_card=START+DEALS-1,independent_card_blocks=DEALS,common_prefix_traces=completed,matched_decisions=len(events),
        paired_terminal_rollouts=2*len(events),all_terminal_actions_independently_replayed=True,raw_decisions_sha256=digest(raw),fields=stats,new_model_fits=0,
        estimand=design()['estimand'],statistical_scope='Fresh card blocks, three fixed training seeds, fixed opponent policies. Exploratory finer strata; no EV claim for individual hands.'))
    return stats


if __name__=='__main__':
    try:
        print(json.dumps({k:dict(events=v['events'],call_minus_fold_bb=v['call_minus_fold_bb'],
                                 simultaneous_95_bb=v['simultaneous_95_bb']) for k,v in run().items()},allow_nan=False))
    except BaseException as exc:
        if ((OUT/'owner.json').is_file() and read(OUT/'owner.json')['pid']==os.getpid()
                and not (OUT/'summary.json').exists() and not (OUT/'failure.json').exists()):
            atomic_json(OUT/'failure.json',dict(observed_utc=utc(),error=f'{type(exc).__name__}: {exc}'))
        raise
