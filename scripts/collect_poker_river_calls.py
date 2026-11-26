"""Capture isolated, reached river decisions and settle both terminal actions."""
import argparse
from collections import Counter
from contextlib import closing
import gzip
import json
from pathlib import Path
import random

import eval7

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply, semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from poker_flop_action_values import _nonce
from poker_flop_repair import FlopRepairPolicy
from poker_river_call_learning import eligible
from poker_river_dominance import guard_fold
from run_poker_preflop_confirmation import run_jobs
from train_poker_flop_repair import SEEDS, STAGES, OpponentsBySeat, sources


REL = Path('preflop-followup/480-v1/river-call-learning-20261126')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
SELECTED = STUDY.parent/'street-progress-20261126/flop/selected.json'
SCRIPT = Path(__file__).resolve()
GUARD = SCRIPT.with_name('poker_river_dominance.py')


def design(sha):
    verify_files({STUDY/'design.json':sha, REPORT/'design.json':sha})
    return read(STUDY/'design.json')


def settled(state, hero):
    if not state['terminal'] or sum(state['utilities']) or len(state['utilities']) != 6:
        raise ValueError('Forced river action must settle all six seats at zero sum')
    return state['utilities'][hero]


def sample(bridge, incumbent, hero_policy, opponents, deal, hero):
    """Reach the first eligible decision solely by playing the registered parent."""
    state = bridge.call(dict(op='reset',seed=deal,button=0))
    prefix, holes = [], {}
    while not state['terminal']:
        if state['players'][hero]['is_folded'] or state['players'][hero]['stack'] == 0:
            return None
        actor = state['seat_to_act']
        holes[actor] = state['your_cards']
        policy = incumbent if state['street']=='preflop' else hero_policy if actor==hero else opponents
        raw = _validate_reply(state,policy.decide(state,semantic_seed(deal,hero,state)))
        reply = guard_fold(state,raw) if actor==hero else raw
        if actor==hero and state['street']=='river' and eligible(state):
            break
        if actor==hero and reply['action']['kind']=='fold':
            return None
        action = dict(reply['action'],seat=actor)
        prefix.append(action)
        if len(prefix)>512:
            raise ValueError('Exceeded native hand length limit')
        state = bridge.call(dict(op='step',action=action))
    else:
        return None

    other = next(p['seat'] for p in state['players'] if p['seat'] != hero and not p['is_folded'])
    opp_cards, hero_cards, board = holes[other], state['your_cards'], state['community_cards']
    if len(set(opp_cards+hero_cards+board)) != 9:
        raise ValueError('Public and live private cards must be distinct')
    ranks = [eval7.evaluate([eval7.Card(c) for c in cards+board]) for cards in (hero_cards,opp_cards)]
    share = 1 if ranks[0]>ranks[1] else 0 if ranks[0]<ranks[1] else .5
    call = settled(bridge.call(dict(op='step',action=dict(seat=hero,kind='call'))),hero)
    replay = bridge.call(dict(op='reset',seed=deal,button=0))
    for action in prefix:
        replay = bridge.call(dict(op='step',action=action))
    if replay != state:
        raise ValueError('Bound original card or action transcript differs after rewind')
    fold = settled(bridge.call(dict(op='step',action=dict(seat=hero,kind='fold'))),hero)
    if fold != state['your_stack']-10_000:
        raise ValueError('Native fold utility inconsistent with pot geometry')
    total = Counter()
    for r in reply['distribution']:
        total[r['action']['kind']] += r['probability']
    if set(total)-{'fold','call','raise'}:
        raise ValueError('Closing hero unexpectedly has a free action')
    # Hidden hole cards, labels and utilities are siblings of the actor observation.
    return dict(deal_seed=deal,hero=hero,root_observation=state,prefix=prefix,
                v1_distribution=raw['distribution'],parent_distribution=reply['distribution'],
                parent_probabilities={k:total[k] for k in ('fold','call','raise')},opponent=other,
                opponent_cards=opp_cards,showdown_share=share,call_utility_chips=call,
                fold_utility_chips=fold,call_minus_fold_chips=call-fold)


def collect(seed, split, sha):
    spec = design(sha)
    cfg = spec['collection'][split]
    bounds = cfg['by_seed'][str(seed)]
    count = cfg['positions_per_seed']
    reference = read(SELECTED)['policies'][str(seed)]
    verify_files({reference['path']:reference['sha256']})
    _, base_ref, _, parent, pool = sources(seed)
    hero_policy = FlopRepairPolicy(reference['path'])
    folder = STUDY/split/f'seed-{seed}'
    folder.mkdir(parents=True)
    partial = folder/'positions.partial.gz'
    positions = 0
    with closing(Bridge()) as bridge, gzip.open(partial,'xt',compresslevel=1) as stream:
        for offset, deal in enumerate(range(bounds['start'],bounds['limit'])):
            hero = offset%6
            lineup = [480 if s==hero else STAGES[random.Random(_nonce('opponent',seed,deal,s)).randrange(6)]
                      for s in range(6)]
            row = sample(bridge,parent,hero_policy,OpponentsBySeat(pool,lineup),deal,hero)
            if row is not None:
                row.update(seed=seed,split=split,opponents_by_seat=lineup)
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                positions += 1
                if positions == count:
                    break
            if (offset+1)%2000==0:
                atomic_json(folder/'progress.json',dict(utc=utc(),original_deals=offset+1,positions=positions))
        engine_hash = digest(bridge.p.args[0])
    if positions != count:
        raise ValueError(f'Registered range exhausted at {positions}/{count} eligible roots')
    design(sha)
    ledger = folder/'positions.jsonl.gz'
    partial.rename(ledger)
    atomic_json(folder/'collection-complete.json',dict(complete=True,completed_utc=utc(),seed=seed,
                split=split,positions=positions,distinct_original_roots=positions,forced_terminal_actions=2*positions,
                attempted_deals=offset+1,first_card=bounds['start'],last_card=deal,
                design_sha256=sha,training_design_sha256=sha,position_ledger_sha256=digest(ledger),
                hero_v1_sha256=reference['sha256'],preflop_parent_sha256=base_ref['sha256'],
                river_guard_sha256=digest(GUARD),native_engine_sha256=engine_hash,generator_sha256=digest(SCRIPT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split',choices=('train','holdout'))
    parser.add_argument('--design-sha',required=True)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args = parser.parse_args()
    if args.seed is not None:
        collect(args.seed,args.split,args.design_sha)
        return
    design(args.design_sha)
    folder = STUDY/args.split
    folder.mkdir()
    jobs = {str(s):[str(PYTHON),str(SCRIPT),args.split,'--seed',str(s),'--design-sha',args.design_sha] for s in SEEDS}
    run_jobs(folder,jobs,len(jobs),'collection')


if __name__ == '__main__':
    main()
