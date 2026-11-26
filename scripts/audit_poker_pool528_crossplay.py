"""Independently check the directed 528 payoff matrix, its three paired families and cycles."""
from itertools import combinations, permutations
import gzip
import json
import math
from statistics import NormalDist

import numpy as np

from audit_poker_pool528_evaluation import CONFIG, CONFIG_SHA, ROOT, SEEDS, STUDY, band, expected_ref, slot_number
from audit_poker_population import check, close_to
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import hand_seed
from fullhouse_v3_common import atomic_json, digest

FIELDS = ('specialist480','mixture504','mixture528')
FOLDER = STUDY/'crossplay'
START = 2060300000
N = 1000


def matrix_blocks(row, column, seed, hashes, plan_sha):
    name = f'{row}-vs-{column}-seed-{seed}'
    folder = FOLDER/'evaluation'/name
    result = read(folder/'results.json')
    ledger = folder/'hands.jsonl.gz'
    check(read(folder/'status.json')['result_sha256']==hashes[name]==digest(folder/'results.json') and
          result['complete'] and result['raw_sha256']==digest(ledger) and result['plan_sha256']==plan_sha and
          (result['hero'],result['opponents'],result['seed'],result['start'],result['deals'],result['hands'])
          ==(row,column,seed,START,N,6*N),'Different cross-play source/ledger')
    for stage,key in ((row,'hero_policy_sha256'),(column,'opponent_policy_sha256')):
        label = 'current480' if stage=='specialist480' else stage
        check(result[key]==expected_ref(label,seed)['sha256'],'Cross-play policy differs')
    sums = np.zeros(N,dtype=np.int64)
    actions = 0
    hands = 0
    with gzip.open(ledger,'rt') as source:
        for line in source:
            hand = json.loads(line)
            block,hero = divmod(hands,6)
            card = START+block
            check(block<N and [hand['deal'],hand['seed'],hand['hero'],hand['hand_seed']] ==
                  [block,card,hero,hand_seed(card,hero)],'Cross-play card, seat, or stochastic tape differs')
            roles = [row if actor==hero else column for actor in range(6)]
            check(hand['slots']==[None if label=='specialist480' else slot_number(hand_seed(card,actor))
                                  for actor,label in enumerate(roles)],'Incorrect per-seat private draw')
            pay = hand['utilities']
            check(len(pay)==6 and all(type(v) is int for v in pay) and sum(pay)==0 and hand['chips']==pay[hero]
                  and 0<len(hand['actions'])==hand['steps']<=512,'Invalid cross-play hand')
            sums[block] += pay[hero]
            actions += hand['steps']
            hands += 1
    check(hands==6*N and result['terminal_replay']==dict(hands=hands,decisions=actions),'Invalid cross-play settlement coverage')
    values = sums/6
    close_to(result['cards_bb100'],values,'Cross-play per-card chip means differ')
    if row==column:
        check(np.all(sums==0),'Incomplete symmetric self-play rotations')
    return values


def check_interval(result,values,family):
    cards = values.mean(axis=0)
    endpoints = band(cards,NormalDist().inv_cdf(1-.025/family))
    close_to([result['lower'],result['upper']],endpoints,'Cross-play simultaneous interval differs')
    close_to(result['mean'],float(cards.mean()),'Cross-play mean differs')
    close_to([result['by_seed'][str(s)] for s in SEEDS],values.mean(axis=1),'Cross-play fixed-seed means differ')
    return endpoints


def main():
    verify_files({CONFIG:CONFIG_SHA})
    reg = read(CONFIG)['crossplay']
    proof = read(FOLDER/'summary.json')
    plan_sha = read(FOLDER/'bound.json')['plan_sha256']
    check(proof['complete'] and proof['completed_cells']==reg['cells']==27 and proof['terminal_replay_hands']==reg['hands']==162000
          and proof['original_card_blocks']==N and proof['fixed_seeds']==list(SEEDS) and
          proof['plan_sha256']==plan_sha==digest(FOLDER/'evaluation-plan.json') and proof['best_response_computed'] is False
          and proof['six_player_exploitability'] is None,'Changed directed game design')
    hashes = proof['result_sha256']
    check(len(hashes)==27,'Invalid number of cross-play cells')
    values = {(r,c):np.asarray([matrix_blocks(r,c,s,hashes,plan_sha) for s in SEEDS]) for r in FIELDS for c in FIELDS}
    intervals = {(r,c):check_interval(proof['matrix'][r][c],v,6) for (r,c),v in values.items()}
    for c in FIELDS:
        check_interval(proof['new528_minus_parent504_vs_fixed_opponent'][c],values['mixture528',c]-values['mixture504',c],3)
    for a,b in combinations(FIELDS,2):
        check_interval(proof['directed_payoff_minus_reverse'][f'{a}_vs_{b}'],values[a,b]-values[b,a],3)
    edges = {(a,b) for a,b in permutations(FIELDS,2) if intervals[a,b][0]>0 and intervals[b,a][1]<0}
    cycles = [list((FIELDS[0],b,c)) for b,c in permutations(FIELDS[1:]) if {(FIELDS[0],b),(b,c),(c,FIELDS[0])} <= edges]
    check(proof['supported_dominance_edges']==[list(e) for e in sorted(edges)] and proof['supported_cycles']==cycles,
          'Unsupported dominance or cycles')
    summary = dict(complete=True,verified_utc=utc(),cells=27,hands=162000,unique_original_card_blocks=N,
                   self_control_deal_means_all_zero=True,raw_six_seat_utilities_reaggregated=True,
                   private_slot_identity_checked_for_every_seat=True,directed_estimates_checked=9,
                   reciprocal_contrasts_checked=math.comb(len(FIELDS),2),fixed_column_contrasts_checked=len(FIELDS),
                   supported_dominance_edges=sorted(edges),supported_cycles=cycles,summary_sha256=digest(FOLDER/'summary.json'))
    atomic_json(FOLDER/'independent-audit.json',summary)
    atomic_json(ROOT/'reports'/FOLDER.relative_to(ROOT/'artifacts')/'independent-audit.json',summary)


if __name__=='__main__':
    main()
