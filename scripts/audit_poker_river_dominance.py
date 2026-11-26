"""Reproduce three constructed river decisions through both playing backends."""
from contextlib import closing
from itertools import combinations
import json
import random
import subprocess
import sys
import time

import eval7

from controlled_poker_refit import utc
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_multivalued_turn import Rules
from poker_river_dominance import closing_call_floor
from score_poker_street_progress import PUBLIC, SEEDS, STUDY


def fixtures():
    # Each list is the full observed action history before the river decision.
    def actions(items):
        return [dict(seat=seat, kind=kind, **({'to': size[0]} if size else {}))
                for seat, kind, *size in items]

    return {
        'quads': dict(hero=2, button=0, hole=['As','Ah'], board=['Ac','Ad','Js','8d','3s'],
            actions=actions([(3,'fold'),(4,'raise',3500),(5,'fold'),(0,'fold'),(1,'fold'),(2,'call'),
                             (2,'raise',1000),(4,'call'),(2,'check'),(4,'check'),(2,'check'),(4,'raise',5450)])),
        'board_tie': dict(hero=2, button=3, hole=['2s','4s'], board=['Ad','Kd','Qs','Jh','Tc'],
            actions=actions([(0,'fold'),(1,'fold'),(2,'raise',500),(3,'fold'),(4,'fold'),(5,'call'),
                             (5,'check'),(2,'raise',100),(5,'call'),(5,'check'),(2,'raise',1000),
                             (5,'call'),(5,'raise',8400)])),
        'straight_flush': dict(hero=1, button=4, hole=['Td','Jd'], board=['9d','8d','7d','6d','4c'],
            actions=actions([(1,'call'),(2,'fold'),(3,'fold'),(4,'fold'),(5,'fold'),(0,'check'),
                             (0,'check'),(1,'check'),(0,'raise',100),(1,'call'),(0,'raise',100)])),
    }


def terminal_check(rules, request, other, opponent_hole):
    hero = request['hero']
    deck = [None]*52
    deck[2*hero:2*hero+2] = request['hole']
    deck[2*other:2*other+2] = opponent_hole
    deck[12:17] = request['board']
    taken = set(deck) - {None}
    unassigned = iter(str(c) for c in eval7.Deck().cards if str(c) not in taken)
    for index in range(len(deck)):
        if deck[index] is None:
            deck[index] = next(unassigned)
    payoffs = {}
    for action in ('call','fold'):
        rules.call(dict(op='reset', button=request['button'], deck=deck))
        for event in request['actions']:
            state = rules.call(dict(op='step', action=event))
        assert state['your_cards'] == request['hole'] and state['community_cards'] == request['board']
        finished = rules.call(dict(op='step', action=dict(seat=hero,kind=action)))
        assert finished['terminal']
        payoffs[action] = finished['utilities'][hero]
    return payoffs['call']-payoffs['fold']


def protocol(model, backend, cases):
    data = ''.join(json.dumps(request)+'\n' for request in cases)
    response = subprocess.run([sys.executable, str(ROOT/'scripts/run_poker.py'), '--policy', str(model),
                               '--backend', backend], input=data, text=True, capture_output=True,
                              check=True, timeout=60)
    replies = [json.loads(line) for line in response.stdout.splitlines()]
    assert len(replies) == len(cases) and all(reply['ok'] and reply['backend'] == backend for reply in replies)
    return replies


def audit():
    sample = fixtures()
    records = {}
    with closing(Bridge()) as bridge, closing(Rules()) as rules:
        for label, request in sample.items():
            state = bridge.call(dict(request, op='replay'))
            started = time.perf_counter()
            floor = closing_call_floor(state)
            elapsed = time.perf_counter() - started
            assert floor > 0
            distinct = [str(c) for c in eval7.Deck().cards if str(c) not in request['hole']+request['board']]
            remaining = list(combinations(distinct, 2))
            other = next(p['seat'] for p in state['players'] if not p['is_folded'] and p['seat']!=request['hero'])
            deltas = [terminal_check(rules, request, other, holding)
                      for holding in random.Random(96132).sample(remaining,12)]
            assert min(deltas) >= floor
            records[label] = dict(public_request=request, pot_chips=state['pot'], to_call_chips=state['amount_owed'],
                                  exhaustive_opponent_hands=len(remaining), call_over_fold_min_chips=floor,
                                  native_showdown_samples=len(deltas), native_chips_gain_range=[min(deltas),max(deltas)],
                                  certificate_time_ms=1000*elapsed, by_policy_seed={})
    for seed in SEEDS:
        model = STUDY/'play'/f'seed-{seed}'/'policy.json'
        cases = [dict(request, seed=4) for request in sample.values()]
        original = protocol(model,'street-progress',cases)
        revised = protocol(model,'street-progress-river',cases)
        for label, before, after in zip(sample,original,revised,strict=True):
            p = next((item['probability'] for item in before['distribution'] if item['action']['kind']=='fold'),0.)
            assert not any(row['action']['kind']=='fold' and row['probability'] for row in after['distribution'])
            assert (after['action']['kind']=='call' if before['action']['kind']=='fold'
                    else after['action']==before['action'])
            for kind in ('call','raise'):
                old_rows = {r['action'].get('to'):r['probability'] for r in before['distribution'] if r['action']['kind']==kind}
                new_rows = {r['action'].get('to'):r['probability'] for r in after['distribution'] if r['action']['kind']==kind}
                if kind=='call' and p:old_rows[None]=old_rows.get(None,0.)+p
                assert old_rows==new_rows
            records[label]['by_policy_seed'][str(seed)] = dict(original_fold_probability=p,
                corrected_fold_probability=0., policy_sha256=digest(model),
                policy_minimum_gain_bb_per_occurrence=p*records[label]['call_over_fold_min_chips']/100)
    assert all(any(r['original_fold_probability']>0 for r in case['by_policy_seed'].values()) for case in records.values())
    receipt = dict(version='heads-up-river-dominated-fold-v1', completed_utc=utc(), constructed_cases=records,
                   runtime_source_sha256=digest(ROOT/'scripts/poker_river_dominance.py'),
                   audit_source_sha256=digest(__file__),
                   unit='One specific heads-up closing river decision; 100 chips per BB',
                   naturally_observed_frequency_measured_here=False, full_game_winrate_measured=False)
    for dest in (PUBLIC,STUDY):
        atomic_json(dest/'river-dominance.json',receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(audit(),indent=2))
