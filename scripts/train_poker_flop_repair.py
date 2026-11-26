"""Generate fresh, matched one-decision self-play labels and fit the bounded flop critic."""
import argparse
from collections import Counter
from contextlib import closing
import gzip
import json
import os
from pathlib import Path
import random

import numpy as np

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply, semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from poker_flop_action_values import _nonce, _play, future_deck, source_cards
from poker_multivalued_turn import Rules
from poker_preflop_specialist import StreetPolicy, evaluator_policy, file_ref
from poker_river_runtime import action_key, legal_menu
from run_poker_preflop_confirmation import run_jobs
from sample_poker_flop_diagnostic import abstract_indices


REL = Path('preflop-followup/480-v1/flop-repair-20261126')
STUDY = ROOT/'artifacts'/REL
SOURCE_PLAN = STUDY.parent/'opponent-diversity-20261123/training-plan.json'
DESIGN_HASH = '09b625d5cf4e13f4af1373fc843f769913dde9dc65ff21a7dd7ac91145e9044b'
PLAN_HASH = 'bc3526a3bd5b96d9f5edccff8414c70fd480db22a962cb8aa09f2423b1e5427d'
SEEDS = (342, 343, 344)
STAGES = (480, 480, 480, 300, 360, 420)
SCRIPT = Path(__file__).resolve()


def sources(seed):
    verify_files({STUDY/'training-design.json': DESIGN_HASH, SOURCE_PLAN: PLAN_HASH})
    design = read(STUDY/'training-design.json')['seed_sources'][str(seed)]
    plan = read(SOURCE_PLAN)
    ref = plan['parent_policies'][str(seed)]
    verify_files({ref['path']: design['parent_policy_sha256']})
    source = Path(plan['campaign'])/'source'
    parent = StreetPolicy(ref['path'], source)
    spec = read(ref['path'])
    verify_files({spec[k]['path']: design[k+'_sha256'] for k in ('preflop','postflop')})
    opponents = {480: parent}
    for stage in (300,360,420):
        old = plan['historical_policies'][str(seed)][str(stage)]
        verify_files({old['path']: design['historical_selfplay_sha256'][str(stage)]})
        opponents[stage] = evaluator_policy(old['path'], source)
    return design, ref, source, parent, opponents


class OpponentsBySeat:
    def __init__(self, policies, lineup):
        self.policies = [policies[t] for t in lineup]

    def decide(self, state, seed):
        return self.policies[state['seat_to_act']].decide(state,seed)


def collect_root(engine, parent, opponents, seed, deal, hero, lineup):
    state = engine.call(dict(op='reset',seed=deal,button=0))
    prefix = []
    while not state['terminal'] and state['street'] in ('preflop','flop'):
        seat = state['seat_to_act']
        actor = parent if state['street']=='preflop' or seat==hero else opponents
        reply = _validate_reply(state,actor.decide(state,semantic_seed(deal,hero,state)))
        if state['street']=='flop' and seat == hero:
            legal = list(legal_menu(state).values())
            if len(legal)<2:
                return None
            probs = {action_key(r['action']):r['probability'] for r in reply['distribution']}
            order = [probs.get(action_key(a),0.) for a in legal]
            if not np.isclose(sum(order),1.):
                raise ValueError('Unsupported parent action')
            return dict(seed=seed,deal_seed=deal,hero=hero,prefix=prefix,root_observation=state,
                        legal_actions=legal,abstract_to_concrete_index=abstract_indices(state,legal),
                        parent_probabilities=order,opponents_by_seat=lineup)
        action = dict(reply['action'],seat=seat)
        state = engine.call(dict(op='step',action=action))
        prefix.append(action)
        if len(prefix)>512:
            raise ValueError('Unbounded hand')
    return None


def collect(seed):
    plan, _, _, parent, policies = sources(seed)
    folder = STUDY/f'seed-{seed}'
    folder.mkdir()
    raw = folder/'positions.jsonl.gz'
    positions = branches = zero_branches = 0
    counts = Counter()
    with closing(Bridge()) as bridge, closing(Rules()) as rules, gzip.open(raw,'xt',compresslevel=1) as out:
        for attempt,deal in enumerate(range(plan['train_card_start'],plan['train_card_exclusive_limit'])):
            hero = attempt % 6
            lineup = [480 if seat == hero else STAGES[random.Random(_nonce('opponent',seed,deal,seat)).randrange(6)]
                      for seat in range(6)]
            opponents = OpponentsBySeat(policies,lineup)
            row = collect_root(bridge,parent,opponents,seed,deal,hero,lineup)
            if row is None:
                continue
            fixed = source_cards(bridge,deal)
            payouts = [[] for _ in row['legal_actions']]
            for rep in range(4):
                deck = future_deck(fixed,deal,rep)
                for k,action in enumerate(row['legal_actions']):
                    utilities,_ = _play(rules,row,deck,action,parent,opponents,rep,False)
                    payouts[k].append(utilities[hero])
                    branches += 1
                    zero_branches += row['parent_probabilities'][k] == 0
            row['action_payoffs_chips'] = payouts
            out.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
            positions += 1
            counts.update(stage for seat,stage in enumerate(lineup) if seat != hero)
            if positions % 128 == 0:
                atomic_json(folder/'progress.json',dict(utc=utc(),root_positions=positions,counterfactual_branches=branches))
            if positions == 1024:
                break
    if positions != 1024:
        raise RuntimeError('Training card scan cap reached')
    receipt = dict(completed_utc=utc(),seed=seed,positions=positions,terminal_rollouts=branches,
                   zero_parent_mass_rollouts=zero_branches,action_labels=branches//4,
                   opponents_by_seat_assignment=counts,first_card=plan['train_card_start'],last_card=deal,
                   attempted_deals=attempt+1,position_ledger_sha256=digest(raw),training_design_sha256=DESIGN_HASH)
    atomic_json(folder/'collection-complete.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('collect','worker','fit'))
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=='collect':
        with (STUDY/'collection.lock').open('x'):
            pass
        run_jobs(STUDY,{f'collect-{s}':[str(PYTHON),str(SCRIPT),'worker','--seed',str(s)]
                        for s in SEEDS},3,'collection')
    elif args.action=='worker' and args.seed in SEEDS:
        collect(args.seed)
    elif args.action=='fit':
        from fit_poker_flop_repair import fit_all
        fit_all()
    else:
        parser.error('Worker requires --seed')
