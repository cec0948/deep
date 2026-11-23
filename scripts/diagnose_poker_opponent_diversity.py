"""Check historical-opponent diversity and paired first-flop effects on an existing audit sample."""
from contextlib import closing
from itertools import combinations
from pathlib import Path
import argparse
import gzip
import json

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import Bridge, Fullhouse
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_capture_postflop_snapshots import ScheduledPostflopMixture
from poker_flop_action_values import _play, future_deck, source_cards
from poker_multivalued_turn import Rules
from poker_preflop_specialist import evaluator_policy
from poker_reference_opponents import ReferenceOpponent
from poker_river_runtime import action_key
from register_poker_opponent_diversity import STUDY, REPORT, CONTROL
from score_poker_flop_diagnostic import probabilities, clustered_intervals, LINEUPS, SNAPSHOTS

ORIGINAL = STUDY.parent/'flop-action-diagnostic-20261123'
PLAN = CONTROL/'population/evaluation-plan.json'
LOCAL = STUDY/'flop-diagnostic'
PUBLIC = REPORT/'flop-diagnostic'
SEEDS = (342,343,344)
SLOTS = ('300','360','420','480')
PAIRS = tuple(combinations(SLOTS,2))


def register():
    summary = ORIGINAL/'summary.json'
    check = '11e82ac96c9d05701b40440a296c76c0f6c782aebac1f361076a93ba7f06c056'
    verify_files({summary:check})
    files = {str(ORIGINAL/name):sha for name,sha in read(summary)['input_sha256'].items()}
    files.update({str(summary):check, str(STUDY/'registration.json'):digest(STUDY/'registration.json'),
                  str(PLAN):'87927530b0cfbd8e049769fef29dcdc8a02640869ef189799c1e91e676c6ae07'})
    for name in ['poker_flop_action_values.py','score_poker_flop_diagnostic.py',Path(__file__).name]:
        file = ROOT/'scripts'/name
        files[str(file)] = digest(file)
    verify_files(files)
    design = dict(registered_utc=utc(),source_hashes=files,source_deals=3072,
        pool_check='Mean concrete-action total variation over all pairs of 300,360,420,480 on the same observed flops.',
        effect='Replace only the first hero flop action of the archived control mixture with the new treatment mixture. '
               'Same conditioned 480-reached states and unchanged 480 hero continuation from the prior audit.',
        environments=list(LINEUPS),conditional_repeats=8,missing_outcomes='Replay every newly supported action on the original eight paired decks and action tapes.',
        inference='Three contrast Bonferroni intervals on the 3072 original deals, equal weight by fixed training seed; '
                  'historically known states, explanatory only, independent of the fresh 81-cell full-game panel.')
    LOCAL.mkdir(exist_ok=False)
    PUBLIC.mkdir(exist_ok=True,parents=True)
    for folder in (LOCAL,PUBLIC):
        atomic_json(folder/'design.json',design)


def check_design():
    design=read(LOCAL/'design.json')
    verify_files(design['source_hashes'])
    return read(STUDY/'registration.json'),read(PLAN)


def distribution(model, sample):
    keys = [action_key(a) for a in sample['legal_actions']]
    probs = np.zeros(len(keys))
    for item in model.decide(sample['root_observation'],0)['distribution']:
        probs[keys.index(action_key(item['action']))] += item['probability']
    if not np.isfinite(probs).all() or np.any(probs<0) or not np.isclose(probs.sum(),1,rtol=0,atol=1e-6):
        raise ValueError('Invalid concrete probabilities')
    return probs/probs.sum()


def pool_check():
    registration, plan = check_design()
    data = {}
    for seed in SEEDS:
        refs = registration['historical_policies'][str(seed)]
        verify_files({r['path']:r['sha256'] for r in refs.values()})
        policies = {k:evaluator_policy(refs[k]['path'],plan['source']) for k in SLOTS}
        values = {f'{a}-{b}':[] for a,b in PAIRS}
        with (ORIGINAL/f'seed-{seed}/states.jsonl').open() as file:
            for line in file:
                state=json.loads(line)
                p={k:distribution(model,state) for k,model in policies.items()}
                if not np.allclose(p['480'],probabilities(state)[0],rtol=0,atol=1e-6):
                    raise ValueError('Historical 480 differs from the recorded root policy')
                for a,b in PAIRS:
                    values[f'{a}-{b}'].append(float(abs(p[a]-p[b]).sum()/2))
        if any(len(observations)!=1024 for observations in values.values()):
            raise ValueError('Invalid original deal coverage')
        data[str(seed)]={k:float(np.mean(v)) for k,v in values.items()}
    result=dict(verified_utc=utc(),original_deals=3072,mean_action_total_variation_by_seed=data,
                mean_action_total_variation={k:float(np.mean([x[k] for x in data.values()])) for k in data['342']})
    for folder in (LOCAL,PUBLIC):
        atomic_json(folder/'pool.json',result)


def treatment_effect():
    _,plan=check_design()
    stage=read(STUDY/'stage-ready.json')
    if not stage['training_complete'] or stage['iteration']!=504:
        raise ValueError('Require completed trained policies')
    gains=[]; seeds=[]; cards=[]
    extra_count=0
    extra_file=LOCAL/'missing-actions.jsonl.gz'
    with closing(Bridge()) as bridge,closing(Rules()) as rules,gzip.open(extra_file,'xt') as extra:
        for seed in SEEDS:
            sid=str(seed)
            ref=stage['policies']['poolMixture504'][sid]
            verify_files({ref['path']:ref['sha256']})
            treatment=ScheduledPostflopMixture(ref['path'],plan['source'])
            ref480=plan['policies']['specialist480'][sid]
            parent=evaluator_policy(ref480['path'],plan['source'])
            opponents={'self480':parent,'public':Fullhouse(),'tight-value':ReferenceOpponent('tight-value')}
            folder=ORIGINAL/f'seed-{seed}'
            with (folder/'states.jsonl').open() as states,gzip.open(folder/'values.jsonl.gz','rt') as old_values:
                for line in states:
                    state=json.loads(line); card=state['deal_seed']
                    old=probabilities(state)[1:].mean(axis=0)
                    new=np.mean([distribution(policy,state) for policy in treatment.postflop],axis=0)
                    fields={}
                    fixed=None
                    for _ in LINEUPS:
                        row=json.loads(next(old_values))
                        env=row['lineup']
                        if row['state_id']!=state['state_id'] or row['legal_actions']!=state['legal_actions'] or env in fields:
                            raise ValueError('Original paired rows disagree')
                        q=np.zeros(len(new))
                        for i,values in enumerate(row['action_payoffs_chips']):
                            if old[i]==0 and new[i]==0:
                                continue
                            if values is None:
                                if old[i]>0:
                                    raise ValueError('Missing original supported action')
                                if fixed is None:
                                    fixed=source_cards(bridge,card)
                                values=[]
                                for rep in range(8):
                                    payoff,_=_play(rules,state,future_deck(fixed,card,rep),state['legal_actions'][i],parent,opponents[env],rep,False)
                                    values.append(payoff[state['hero']])
                                extra.write(json.dumps(dict(state_id=state['state_id'],lineup=env,action_index=i,chips=values))+'\n')
                                extra_count+=8
                            if len(values)!=8:
                                raise ValueError('Incomplete action return')
                            q[i]=float(np.mean(values))/100.
                        fields[env]=float((new-old)@q)
                    gains.append([fields[env] for env in LINEUPS])
                    seeds.append(seed)
                    cards.append(card)
                if old_values.readline():
                    raise ValueError('Unexpected original rows')
    if len(set(cards))!=len(cards) or len(cards)!=3072:
        raise ValueError('Unexpected first-flop sample')
    history=read(ORIGINAL/'summary.json')['primary_mixture_minus_480']['contrasts']
    contrasts=clustered_intervals(gains,seeds,cards,SEEDS,LINEUPS)
    result=dict(verified_utc=utc(), original_deals=len(cards), additional_original_deals=0,
          units='BB per first hero flop decision on the previous audit sample',
          added_terminal_branches=extra_count, missing_actions_sha256=digest(extra_file),
          diverse_minus_control=contrasts,diverse_minus_480_point_bb={
              env:contrasts[env]['mean_bb']+history[env]['mean_bb'] for env in LINEUPS},
          stage_sha256=digest(STUDY/'stage-ready.json'),design_sha256=digest(LOCAL/'design.json'))
    for folder in (LOCAL,PUBLIC):
        atomic_json(folder/'effect.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register','pool','effect'))
    args=parser.parse_args()
    {'register':register,'pool':pool_check,'effect':treatment_effect}[args.action]()
