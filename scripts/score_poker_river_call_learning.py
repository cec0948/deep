"""Score one frozen river-call learner using independent original deals and select a play path."""
import argparse
from contextlib import closing
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import eval7
import numpy as np

from audit_poker_river_dominance import fixtures
from collect_poker_river_calls import GUARD, REPORT, SCRIPT, SELECTED, STUDY
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_river_call_learning import RiverCallPolicy
from poker_river_dominance import RiverDominancePolicy, closing_call_floor
from poker_river_runtime import action_key, legal_menu
from score_poker_street_progress import SEEDS, interval, require
from smoke_poker_street_progress import requests

DESIGN = 'ea1db38fc35ae39a62ba234c8a1117d5ac12f6b836788c4002e9258b4eabbed8'
LEARNED, GUARDED = 'river-call-learning', 'street-progress-river'
CALL, FOLD = ('call', None), ('fold', None)


def distribution(state, reply):
    _validate_reply(state, reply)
    rows = reply['distribution']
    p = {action_key(row['action']):row['probability'] for row in rows}
    require(set(p) <= set(legal_menu(state)) and len(p) == len(rows)
            and p.get(action_key(reply['action']), 0.) > 0., 'Illegal, repeated or unsupported action')
    return p


def compare_distributions(state, before, after):
    for key in set(before) | set(after):
        if key not in {CALL, FOLD}:
            require(before.get(key, 0.) == after.get(key, 0.), 'Nonpassive action probability changed')
    dc = after.get(CALL, 0.) - before.get(CALL, 0.)
    df = after.get(FOLD, 0.) - before.get(FOLD, 0.)
    require(abs(dc+df) <= 2.e-14, 'Transferred call/fold mass differs')
    if closing_call_floor(state) > 0:
        require(dc == df == 0., 'Protected river action changed')
    if dc == 0. or np.isclose(dc, 0., rtol=0, atol=np.finfo(float).eps):
        return 0.
    source = before.get(FOLD if dc > 0 else CALL, 0.)
    require(math.isclose(abs(dc), .25*source, rel_tol=1.e-7, abs_tol=2.e-14), 'Incorrect move budget')
    return dc


def settlement(row, engine):
    """Reconstruct the actual original hidden hand and both closing utilities in the native engine."""
    hero, deal = row['hero'], row['deal_seed']
    pay = {}
    for kind in ('fold','call'):
        state = engine.call(dict(op='reset', seed=deal, button=0))
        for move in row['prefix']:
            require(move['seat'] == state['seat_to_act'], 'Invalid action prefix actor')
            if move['seat'] == row['opponent']:
                require(state['your_cards'] == row['opponent_cards'], 'Actual opponent hole cards differ')
            state = engine.call(dict(op='step', action=move))
        require(state == row['root_observation'] and state['seat_to_act'] == hero
                and state['street'] == 'river' and state['amount_owed'] > 0, 'Actor-visible river root differs')
        final = engine.call(dict(op='step', action=dict(seat=hero,kind=kind)))
        require(final['terminal'] and len(final['utilities']) == 6 and sum(final['utilities']) == 0,
                'No complete zero-sum terminal payoff')
        pay[kind] = final['utilities'][hero]
        require(pay[kind] == row[f'{kind}_utility_chips'], 'Reproduced terminal utility differs')
    players = state['players']
    live = {p['seat']:p for p in players if not p['is_folded']}
    villain = row['opponent']
    require(set(live) == {hero,villain} and hero != villain and len(players) == 6
            and live[hero]['stack']-state['amount_owed'] == live[villain]['stack']
            and live[villain]['stack'] == min(p['stack'] for p in players)
            and sum(10000-p['stack'] for p in players) == state['pot'], 'Ineligible heads-up pot')
    log = state['action_log']
    entry = max(i for i, event in enumerate(log) if event['action'] == 'deal')
    street = log[entry+1:]
    require([e['seat'] for e in street if e['action'] in ('raise','bet')] == [villain]
            and all(e['action']=='check' for e in street if e['seat']==hero), 'Incorrect river bet sequence')
    board = [eval7.Card(card) for card in state['community_cards']]
    a = eval7.evaluate(board + [eval7.Card(card) for card in state['your_cards']])
    b = eval7.evaluate(board + [eval7.Card(card) for card in row['opponent_cards']])
    require((1. if a>b else .5 if a==b else 0.) == row['showdown_share'], 'Incorrect showdown share')
    value = pay['call']-pay['fold']
    require(value == row['call_minus_fold_chips'] and pay['fold'] == state['your_stack']-10000,
            'Counterfactual chip label differs')
    return state,value


def verify_freeze():
    verify_files({STUDY/'design.json':DESIGN,REPORT/'design.json':DESIGN})
    d = read(STUDY/'design.json')
    verify_files({GUARD:d['guard_sha256'], SELECTED:d['flop_selection_sha256'],
                  SELECTED.parents[1]/'play-ready.json':d['parent_play_sha256']})
    require(tuple(d['seeds']) == SEEDS, 'Unexpected deployment seeds')
    intervals = sorted((r['start'],r['limit']) for split in d['collection'].values() for r in split['by_seed'].values())
    require(all(a < b for a,b in intervals) and all(a[1] <= b[0] for a,b in zip(intervals,intervals[1:])),
            'Train and holdout deal ranges overlap')
    candidate = read(STUDY/'candidate.json')
    require(candidate['training_complete'] and candidate['design_sha256'] == DESIGN, 'Training is not frozen')
    verify_files({s['path']:s['sha256'] for s in [*candidate['policies'].values(),candidate['model']]})
    scripts = Path(__file__).resolve().parent
    for name in ('fit_poker_river_call.py','poker_river_call_learning.py','poker_river_dominance.py'):
        verify_files({scripts/name:candidate['code_sha256'][name]})
    parent = read(SELECTED)['policies']
    verify_files({s['path']:s['sha256'] for s in parent.values()})
    return d,candidate,parent


def score():
    d,candidate,parent = verify_freeze()
    frozen = digest(STUDY/'candidate.json')
    cfg = d['collection']['holdout']
    deltas,proofs = {},{}
    with (STUDY/'holdout-score.lock').open('x'), closing(Bridge()) as engine:
        for seed in SEEDS:
            sid = str(seed)
            bounds = cfg['by_seed'][sid]
            path = STUDY/'holdout'/f'seed-{seed}'
            complete = read(path/'collection-complete.json')
            n = cfg['positions_per_seed']
            require(complete['complete'] and complete['positions'] == complete['distinct_original_roots'] == n
                    and complete['forced_terminal_actions'] == 2*n and complete['seed'] == seed
                    and complete['split'] == 'holdout' and complete['design_sha256'] == DESIGN
                    and complete['first_card'] == bounds['start'] and complete['hero_v1_sha256'] == parent[sid]['sha256'],
                    'Inconsistent holdout rollup')
            raw = path/'positions.jsonl.gz'
            verify_files({raw:complete['position_ledger_sha256'], SCRIPT:complete['generator_sha256'],
                          GUARD:complete['river_guard_sha256'], BRIDGE:complete['native_engine_sha256']})
            anchor = RiverDominancePolicy(parent[sid]['path'])
            learner = RiverCallPolicy(candidate['policies'][sid]['path'])
            last = bounds['start']-1
            values, shifts = [], []
            with gzip.open(raw,'rt') as fin:
                for line in fin:
                    row = json.loads(line)
                    deal = row['deal_seed']
                    require(last < deal < bounds['limit'] and row['hero'] == (deal-bounds['start'])%6
                            and row['split']=='holdout' and row['seed']==seed, 'Wrong/reused holdout deal')
                    last = deal
                    state,chips = settlement(row,engine)
                    before = distribution(state,anchor.decide(state,0))
                    require(before == {action_key(x['action']):x['probability'] for x in row['parent_distribution']},
                            'Holdout came from a different protected parent')
                    # The only inference input is the actor view just replayed in the engine.
                    dc = compare_distributions(state,before,distribution(state,learner.decide(state,0)))
                    shifts.append(dc)
                    values.append(dc * chips / 100.)
            require(len(values) == n and last == complete['last_card'], 'Incomplete holdout ledger')
            deltas[sid] = values
            proofs[sid] = dict(original_deals=n,terminal_utilities_reproduced=2*n,
                               increases=sum(v>0 for v in shifts),decreases=sum(v<0 for v in shifts),
                               mean_absolute_call_shift=math.fsum(map(abs,shifts))/n,
                               call_increase_contribution_bb=math.fsum(v for v, shift in zip(values, shifts)
                                                                      if shift > 0) / n,
                               call_decrease_contribution_bb=math.fsum(v for v, shift in zip(values, shifts)
                                                                      if shift < 0) / n,
                               ledger_sha256=complete['position_ledger_sha256'],
                               receipt_sha256=digest(path/'collection-complete.json'))
        verify_freeze()
        verify_files({STUDY/'candidate.json':frozen})
        gain = interval(deltas)
        accepted = gain['lower'] > 0.
        refs = candidate['policies'] if accepted else parent
        result = dict(complete=True,completed_utc=utc(), selected='candidate' if accepted else 'parent',
                      backend=LEARNED if accepted else GUARDED,paired_gain=gain,proofs=proofs,
                      design_sha256=DESIGN,candidate_sha256=frozen,model_sha256=candidate['model']['sha256'],
                      scorer_sha256=digest(__file__),policy_sha256={s:x['sha256'] for s,x in refs.items()},
                      unique_holdout_deals=sum(len(v) for v in deltas.values()),
                      units='BB per reached eligible heads-up river bet decision',whole_game_strength_verified=False)
        atomic_json(REPORT/'selection.json',result)
        atomic_json(STUDY/'selection.json',dict(result,policies=refs))
        return result


def check_protocol(path, backend, seed, engine):
    valid = requests(seed)
    valid['protected_river'] = dict(fixtures()['quads'],seed=seed)
    # A reached training fixture exercises the ordinary river choice model without opening more holdout hands.
    raw = STUDY/'train'/f'seed-{seed}'/'positions.jsonl.gz'
    with gzip.open(raw,'rt') as rows:
        row = json.loads(next(rows))
    state = row['root_observation']
    valid['closing_river'] = dict(hero=row['hero'],button=state['dealer'],hole=state['your_cards'],
                                  board=state['community_cards'],actions=row['prefix'],seed=seed)
    stream = ['{}',*(json.dumps(r) for r in valid.values())]
    result = subprocess.run([sys.executable,str(ROOT/'scripts/run_poker.py'),'--backend',backend,'--policy',str(path)],
                            input='\n'.join(stream)+'\n',capture_output=True,text=True,timeout=60,check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    require(len(replies)==len(stream) and replies[0]['ok'] is False,'Protocol lost framing on an invalid request')
    policy = RiverCallPolicy(path) if backend==LEARNED else RiverDominancePolicy(path)
    for (case,req),reply in zip(valid.items(),replies[1:],strict=True):
        state = engine.call(dict(req,op='replay'))
        p = distribution(state,reply)
        direct = policy.decide(state,seed)
        require(reply['action']==direct['action'] and p==distribution(state,direct) and reply['backend']==backend
                and reply['policy_manifest_sha256']==policy.digest,'Live JSONL alias differs from selected model')
        if case=='protected_river':
            require(closing_call_floor(state)>0 and p.get(FOLD,0.)==0., 'Proved call guard is missing')
    return dict(policy_sha256=policy.digest,valid_requests=list(valid),malformed_request_rejected=True)


def publish():
    d,candidate,parent=verify_freeze()
    selection=read(STUDY/'selection.json')
    verify_files({STUDY/'candidate.json':selection['candidate_sha256']})
    accepted=selection['paired_gain']['lower']>0.
    backend=LEARNED if accepted else GUARDED
    refs=candidate['policies'] if accepted else parent
    require(selection['complete'] and selection['design_sha256']==DESIGN and selection['policies']==refs
            and backend==selection['backend'], 'Invalid published branch')
    dest=STUDY/'play'
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=STUDY) as tmp:
            staged=Path(tmp)/'play'
            for sid,ref in refs.items():
                alias = staged / f'seed-{sid}' / 'policy.json'
                alias.parent.mkdir(parents=True)
                alias.symlink_to(os.path.relpath(ref['path'],dest/f'seed-{sid}'))
            os.rename(staged,dest)
    proofs={}
    with closing(Bridge()) as engine:
        for seed in SEEDS:
            sid = str(seed)
            alias = dest / f'seed-{sid}' / 'policy.json'
            require(alias.resolve(strict=True)==Path(refs[sid]['path']).resolve(strict=True),'Wrong live play alias')
            verify_files({alias:refs[sid]['sha256']})
            proofs[sid]=check_protocol(alias,backend,seed,engine)
    result=dict(complete=True,completed_utc=utc(),backend=backend,selected=selection['selected'],
                design_sha256=DESIGN,selection_sha256=digest(STUDY/'selection.json'),
                public_selection_sha256=digest(REPORT/'selection.json'),policy_sha256=selection['policy_sha256'],
                normal_jsonl_decisions=sum(len(r['valid_requests']) for r in proofs.values()),
                rejected_malformed_requests=len(SEEDS),proofs=proofs,whole_game_strength_verified=False,
                entrypoint=f'python scripts/run_poker.py --backend {backend} --policy '
                           'artifacts/preflop-followup/480-v1/river-call-learning-20261126/play/seed-{seed}/policy.json')
    atomic_json(STUDY/'play-ready.json',result)
    atomic_json(REPORT/'play-ready.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=('score','publish'))
    args=parser.parse_args()
    result=score() if args.step=='score' else publish()
    print(json.dumps(result,allow_nan=False))
