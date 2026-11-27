"""Independently settle and score the frozen diverse-opponent river policy."""

import argparse
from contextlib import closing
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import numpy as np
from scipy.special import expit

from audit_poker_river_call_learning import geometry, payouts, public_features
from audit_poker_river_dominance import fixtures
from controlled_poker_refit import read, utc, verify_files
from fullhouse_runtime import BRIDGE, Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_river_call_learning import RiverCallPolicy
from poker_river_dominance import closing_call_floor
from poker_river_runtime import action_key
from score_poker_river_call_learning import CALL, FOLD, compare_distributions, distribution, settlement
from score_poker_street_progress import SEEDS, require
from smoke_poker_street_progress import requests

REL = Path('preflop-followup/480-v1/river-call-continuation-20261127')
STUDY, REPORT = ROOT/'artifacts'/REL, ROOT/'reports'/REL
PRIOR = STUDY.parent/'river-call-learning-20261126'
LEARNED, PARENT = 'river-call-continuation', 'river-call-learning'


def equal_cell_summary(cells):
    """Independent original deals within fixed opponent/seed strata, with equal stratum weights."""
    require(cells and all(len(v) >= 2 and all(map(math.isfinite, v)) for v in cells),
            'Nonfinite or insufficient paired decisions')
    means = [statistics.fmean(v) for v in cells]
    errors = [statistics.variance(v)/len(v) for v in cells]
    mean = statistics.fmean(means)
    se = math.sqrt(math.fsum(errors))/len(cells)
    return dict(mean=mean, standard_error=se, lower=mean-1.96*se, upper=mean+1.96*se,
                deals=sum(map(len, cells)), strata=len(cells))


def expected_shift(state, model, before, x):
    if x is None:
        return 0.
    z = (x-np.asarray(model['center']))/np.asarray(model['scale'])
    share = expit(x[0] + model['intercept'] + np.dot(model['coefficients'],z))
    gap = share - state['amount_owed']/(state['pot'] + state['amount_owed'])
    if gap >= .05:
        return .25*before.get(FOLD,0.)
    if gap <= -.05:
        return -.25*before.get(CALL,0.)
    return 0.


def verify_candidates(design_sha):
    verify_files({STUDY/'design.json':design_sha, REPORT/'design.json':design_sha})
    d = read(STUDY/'design.json')
    incumbent = read(PRIOR/'selection.json')
    candidate = read(STUDY/'candidate.json')
    require(incumbent['complete'] and incumbent['selected']=='candidate'
            and incumbent['backend']==PARENT and candidate['training_complete']
            and candidate['design_sha256']==design_sha,'Incorrect frozen candidate or parent')
    for candidate_files in (incumbent['policies'], candidate['policies']):
        require(set(candidate_files)==set(map(str,SEEDS)), 'Missing seed manifest')
        verify_files({ref['path']:ref['sha256'] for ref in candidate_files.values()})
    verify_files({candidate['model']['path']:candidate['model_sha256']})
    for name, sha in candidate['code_sha256'].items():
        require(Path(name).name==name,'Invalid source file reference')
        verify_files({Path(__file__).with_name(name):sha})
    return d, incumbent['policies'], candidate


def evaluate_row(row, engine, incumbent, learner, model):
    state, chips = settlement(row,engine)
    require(geometry(state),'Incorrect river eligibility')
    payouts(row)
    key = row['deal_seed']
    original = incumbent.decide(state,key)
    trial = learner.decide(state,key)
    before, after = [distribution(state,r) for r in (original,trial)]
    require(before=={action_key(r['action']):r['probability'] for r in row['parent_distribution']},
            'Stored incumbent differs from deployed policy')
    shift = compare_distributions(state,before,after)
    x = None if closing_call_floor(state)>0 else public_features(state)[0]
    require(abs(shift-expected_shift(state,model,before,x)) < 1.e-12,
            'Actor-visible calibrated probability disagrees with served call/fold shift')
    if 'raise' in (original['action']['kind'], trial['action']['kind']):
        require(original['action']==trial['action'],'A sampled raise changed')
    old_shift=expected_shift(state,incumbent.model,before,x)
    return shift, shift*chips/100., old_shift, old_shift*chips/100.


def smoke(path, seed, backend, policy, train_ledger, engine):
    cases = requests(seed)
    cases['unbeatable_river'] = dict(fixtures()['quads'],seed=seed)
    with gzip.open(train_ledger,'rt') as stream:
        fixture = json.loads(next(stream))
    s = fixture['root_observation']
    cases['ordinary_closing_river'] = dict(seed=seed,hero=fixture['hero'],button=s['dealer'],
            board=s['community_cards'],hole=s['your_cards'],actions=fixture['prefix'])
    lines = ['{}',*(json.dumps(v) for v in cases.values())]
    proc = subprocess.run([sys.executable,str(ROOT/'scripts/run_poker.py'),'--backend',backend,
                           '--policy',str(path)], input='\n'.join(lines)+'\n',capture_output=True,
                          text=True,check=True,timeout=60)
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    require(len(replies)==len(lines) and replies[0]['ok'] is False,'Malformed protocol input was accepted')
    for (case,req),answer in zip(cases.items(),replies[1:],strict=True):
        state=engine.call(dict(op='replay',**req))
        wanted=policy.decide(state,req['seed'])
        probs=distribution(state,answer)
        require(answer['backend']==backend and answer['policy_manifest_sha256']==digest(path)
                and probs==distribution(state,wanted) and answer['action']==wanted['action'],
                'Normal engine output differs from selected model')
        if case=='unbeatable_river':
            require(closing_call_floor(state)>0 and probs.get(FOLD,0.)==0.,'A guaranteed river call was lost')
    return dict(policy_sha256=digest(path), valid_requests=list(cases), malformed_request_rejected=True)


def publish(design_sha):
    from poker_river_call_diverse import DiverseRiverCallPolicy
    _, prior, candidate = verify_candidates(design_sha)
    selected = read(STUDY/'selection.json')
    gain = selected['paired_gain']
    accepted = adoption(gain,gain['observed_contribution_bb_per_100'])
    cls,backend = (DiverseRiverCallPolicy,LEARNED) if accepted else (RiverCallPolicy,PARENT)
    refs = candidate['policies'] if accepted else prior
    require(selected['complete'] and selected['design_sha256']==design_sha and selected['policies']==refs
            and selected['backend']==backend and selected['candidate_sha256']==digest(STUDY/'candidate.json'),
            'Stale selection at publication')
    destination = STUDY/'play'
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=STUDY) as tmp:
            stage = Path(tmp)/'play'
            for sid,ref in refs.items():
                link = stage/f'seed-{sid}'/'policy.json'
                link.parent.mkdir(parents=True)
                link.symlink_to(os.path.relpath(ref['path'],destination/f'seed-{sid}'))
            os.rename(stage,destination)
    proofs={}
    with closing(Bridge()) as engine:
        for seed in SEEDS:
            sid=str(seed)
            path=destination/f'seed-{sid}'/'policy.json'
            require(path.resolve(strict=True)==Path(refs[sid]['path']).resolve(strict=True), 'Incorrect selected alias')
            ledger=next((STUDY/'train').glob(f'*/seed-{sid}/positions.jsonl.gz'))
            proofs[sid]=smoke(path,seed,backend,cls(path),ledger,engine)
    receipt = dict(complete=True,completed_utc=utc(),backend=backend,selected=selected['selected'],
                   selection_sha256=digest(STUDY/'selection.json'),design_sha256=design_sha,
                   public_selection_sha256=digest(REPORT/'selection.json'),policy_sha256=selected['policy_sha256'],
                   normal_jsonl_decisions=sum(len(v['valid_requests']) for v in proofs.values()),
                   rejected_malformed_requests=len(SEEDS),proofs=proofs,whole_game_strength_verified=False,
                   entrypoint=f'python scripts/run_poker.py --backend {backend} --policy '
                              f'artifacts/{REL}/play/seed-{{seed}}/policy.json')
    for directory in (STUDY,REPORT):
        atomic_json(directory/'play-ready.json',receipt)
    return receipt


DESIGN = 'f5a05053ba0f296de162a6b777cca753527210b49fc92b2c6c28c2f761c84969'


def adoption(gain, dealt_contribution):
    return gain['lower'] > 0. and dealt_contribution > 0.


def verify_reach(row, incumbent, opponent, engine):
    """Independently replay all six actors using only their own visible state."""
    from evaluate_average_baseline import semantic_seed
    deal, hero = row['deal_seed'], row['hero']
    state = engine.call(dict(op='reset',seed=deal,button=0))
    for move in row['prefix']:
        actor = state['seat_to_act']
        require(actor != hero or not geometry(state), 'Ledger skipped an earlier eligible hero decision')
        policy = incumbent if actor == hero else opponent
        action = policy.decide(state,semantic_seed(deal,hero,state))['action']
        require(dict(action,seat=actor)==move,'Actual hand trajectory differs from the registered table')
        state = engine.call(dict(op='step',action=move))
    require(state==row['root_observation'],'Dealt public prefix did not reproduce')


def cell_summary(values, shifts, sizes, odds, attempted):
    summary = equal_cell_summary([values])
    sizing = {}
    for label, low, high in (('up_to_0.20',0.,.20),('over_0.20_through_0.35',.20,.35),('over_0.35',.35,1.)):
        indices = [i for i,p in enumerate(odds) if low < p <= high]
        sizing[label] = dict(decisions=len(indices), increased_calls=sum(shifts[i]>0. for i in indices),
                             decreased_calls=sum(shifts[i]<0. for i in indices))
    return dict(**summary, attempted_deals=attempted,
                observed_contribution_bb_per_100=100.*math.fsum(values)/attempted,
                increased_calls=sum(s>0. for s in shifts), decreased_calls=sum(s<0. for s in shifts),
                call_increase_contribution_bb=math.fsum(v for v,s in zip(values,shifts) if s>0.)/len(values),
                call_decrease_contribution_bb=math.fsum(v for v,s in zip(values,shifts) if s<0.)/len(values),
                observed_river_bet_chips=dict(minimum=min(sizes),median=statistics.median(sizes),maximum=max(sizes)),
                pot_odds_exposure=sizing,
                both_terminal_settlements_verified=2*len(values))


def assess_cell(study, group, seed, design, parent_ref, candidate_ref, model, incumbent, learner, opponent, engine):
    cfg = design['collection']['holdout']
    sid = str(seed)
    bounds = cfg['by_group'][group][sid]
    path = study/'holdout'/group/f'seed-{seed}'
    raw, receipt = path/'positions.jsonl.gz', path/'collection-complete.json'
    complete = read(receipt)
    count = cfg['positions_per_seed_group']
    require(complete['complete'] and complete['split']=='holdout' and complete['group']==group
            and complete['seed']==seed and complete['design_sha256']==DESIGN
            and complete['positions']==complete['distinct_original_roots']==count
            and complete['forced_terminal_actions']==2*count and complete['first_card']==bounds['start']
            and design['parent_policy_sha256'][sid]==complete['parent_policy_sha256']==parent_ref['sha256']
            and complete['river_policy_sha256']==design['group_policy_sha256'][group],
            'Inconsistent holdout completion receipt')
    verify_files({raw:complete['position_ledger_sha256'], BRIDGE:complete['native_engine_sha256'],
                  Path(__file__).with_name('collect_poker_river_call_continuation.py'):complete['generator_sha256'],
                  Path(__file__).with_name('collect_poker_river_calls.py'):complete['settlement_source_sha256'],
                  parent_ref['path']:parent_ref['sha256'],candidate_ref['path']:candidate_ref['sha256']})
    values, shifts, sizes, odds, old_update, new_coefficients = [],[],[],[],[],[]
    coefficient_changes=0
    last = bounds['start']-1
    with gzip.open(raw,'rt') as stream:
        for line in stream:
            row=json.loads(line)
            deal=row['deal_seed']
            require(last<deal<bounds['limit'] and row['hero']==(deal-bounds['start'])%6
                    and row['split']=='holdout' and row['group']==group and row['seed']==seed,
                    'Repeated original deal or wrong table/seed/range')
            require(row['opponents_by_seat'] is None and row['river_policy_sha256']==complete['river_policy_sha256']
                    and row['parent_policy_sha256']==complete['parent_policy_sha256']
                    and row['parent_distribution']==row['incumbent_distribution'],'Wrong opponent or parent tag')
            last=deal
            verify_reach(row,incumbent,opponent,engine)
            shift,gain,old_shift,old_gain=evaluate_row(row,engine,incumbent,learner,model)
            values.append(gain)
            shifts.append(shift)
            old_update.append(old_gain)
            new_coefficients.append(gain-old_gain)
            coefficient_changes+=int(abs(shift-old_shift)>1.e-12)
            state=row['root_observation']
            sizes.append(state['amount_owed'])
            odds.append(state['amount_owed']/(state['pot']+state['amount_owed']))
    attempted=complete['attempted_deals']
    require(len(values)==count and last==complete['last_card'] and attempted==last-bounds['start']+1,
            'Incomplete or miscounted original dealt-hand sample')
    result=cell_summary(values,shifts,sizes,odds,attempted)
    result.update(ledger_sha256=digest(raw),receipt_sha256=digest(receipt),
                  decisions_with_different_incremental_shift_from_new_coefficients=coefficient_changes)
    return values,result,old_update,new_coefficients


def summarize_cells(cells, proofs):
    def summarize(keys):
        keys=list(keys)
        result=equal_cell_summary([cells[k] for k in keys])
        result['observed_contribution_bb_per_100']=statistics.fmean(
            proofs[k]['observed_contribution_bb_per_100'] for k in keys)
        return result
    overall=summarize(cells)
    groups=sorted({g for g,s in cells})
    overall['by_opponent']={g:summarize(k for k in cells if k[0]==g) for g in groups}
    overall['by_seed']={str(s):summarize(k for k in cells if k[1]==s) for s in SEEDS}
    return overall


def score():
    import fcntl
    from fullhouse_runtime import Fullhouse
    from poker_river_call_diverse import DiverseRiverCallPolicy
    from poker_solverlike_opponent import SolverLikeOpponent, MODEL_SHA256
    d,prior,candidate=verify_candidates(DESIGN)
    frozen=digest(STUDY/'candidate.json')
    model=read(candidate['model']['path'])
    cfg=d['collection']['holdout']
    expected_groups={'public-fullhouse','solverlike'}
    require(set(cfg['groups'])==expected_groups and set(cfg['groups']).isdisjoint(d['collection']['train']['groups'])
            and tuple(d['seeds'])==SEEDS,'Unexpected or repeated opponent identity')
    bounds=sorted((r['start'],r['limit']) for split in d['collection'].values()
                  for group in split['by_group'].values() for r in group.values())
    require(all(a<b for a,b in bounds) and all(b<=c for (_,b),(c,_) in zip(bounds,bounds[1:])),
            'Overlapping train and holdout deals')
    verify_files({PRIOR/'play-ready.json':d['parent_play_ready_sha256'],
                  Path(__file__).with_name('poker_river_dominance.py'):d['parent_guard_sha256'],
                  Path(__file__).with_name('poker_river_call_learning.py'):d['parent_runtime_sha256']})
    opponents={'public-fullhouse':Fullhouse(),'solverlike':SolverLikeOpponent()}
    require(opponents['public-fullhouse'].digest==d['group_policy_sha256']['public-fullhouse']
            and MODEL_SHA256==d['group_policy_sha256']['solverlike'],'Held-out opposition identity differs')
    cells,proofs,old_update,new_coefficients={},{},{},{}
    with (STUDY/'holdout-score.lock').open('a') as lock, closing(Bridge()) as engine:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (STUDY/'selection.json').exists(),'Selection already published')
        for group in cfg['groups']:
            for seed in SEEDS:
                sid=str(seed)
                original=RiverCallPolicy(prior[sid]['path'])
                learner=DiverseRiverCallPolicy(candidate['policies'][sid]['path'])
                key=group,seed
                cells[key],proofs[key],old_update[key],new_coefficients[key]=assess_cell(
                    STUDY,group,seed,d,prior[sid],candidate['policies'][sid],model,original,learner,opponents[group],engine)
        verify_candidates(DESIGN)
        verify_files({STUDY/'candidate.json':frozen})
        gain=summarize_cells(cells,proofs)
        old_result=equal_cell_summary(list(old_update.values()))
        coefficient_result=equal_cell_summary(list(new_coefficients.values()))
        require(abs(gain['mean']-old_result['mean']-coefficient_result['mean'])<1.e-12,
                'Attribution means do not add to actual served policy gain')
        accepted=adoption(gain,gain['observed_contribution_bb_per_100'])
        refs=candidate['policies'] if accepted else prior
        result=dict(complete=True, completed_utc=utc(), selected='candidate' if accepted else 'parent',
                    backend=LEARNED if accepted else PARENT, design_sha256=DESIGN, candidate_sha256=frozen,
                    model_sha256=candidate['model_sha256'], scorer_sha256=digest(__file__),
                    policy_sha256={sid:r['sha256'] for sid,r in refs.items()},paired_gain=gain,
                    proofs={g:{str(s):proofs[g,s] for s in SEEDS} for g in cfg['groups']},
                    nonselecting_attribution=dict(extra_update_with_old_coefficients=old_result,
                        continued_minus_old_coefficients_with_same_extra_update=coefficient_result,
                        decisions_with_different_incremental_shift=sum(
                            r['decisions_with_different_incremental_shift_from_new_coefficients'] for r in proofs.values())),
                    unique_holdout_deals=sum(map(len,cells.values())), attempted_deals=sum(r['attempted_deals'] for r in proofs.values()),
                    units='BB per reached closing heads-up river bet decision; equal fixed-seed and opponent weights.',
                    all_deal_units='Incremental BB per 100 original dealt hands under this table; other decisions contribute zero.',
                    whole_game_strength_verified=False)
        atomic_json(REPORT/'selection.json',result)
        atomic_json(STUDY/'selection.json',dict(result,policies=refs))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=('score','publish'))
    args=parser.parse_args()
    result=score() if args.step=='score' else publish(DESIGN)
    print(json.dumps(result,allow_nan=False))
