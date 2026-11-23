"""Verify training-policy identities and audit the fixed nine-table evaluation from raw hand ledgers."""
from collections import Counter
from contextlib import ExitStack, closing
from datetime import datetime
import gzip
import importlib.util
import json
from statistics import NormalDist

import numpy as np

from audit_poker_population import RICH, RICH_SHA, check, close_to, expected_identities, matched_role
from audit_postflop_averaging import (
    audit_pair, checked_ref, file_sha, first_difference, hero_policy, paired_stats,
    private_slot, replay, terminal_auditor, verify_terminals,
)
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import hand_seed
from fullhouse_runtime import Bridge, Fullhouse, SOURCE
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_reference_opponents import ReferenceOpponent
from rich_poker_runtime import RichPolicy
from run_poker_candidate_confirmation import prior_card_ranges

STUDY = ROOT/'artifacts/preflop-followup/480-v1/opponent-diversity-20261123'
POP = STUDY/'population'
OLD = STUDY.parent/'postflop-averaging-20261120'
CONFIG = ROOT/'configs/poker-opponent-diversity-evaluation-20261123.json'
TRAIN_CONFIG = ROOT/'configs/poker-opponent-diversity-20261123.json'
OLD_PLAN_SHA = '87927530b0cfbd8e049769fef29dcdc8a02640869ef189799c1e91e676c6ae07'
OLD_STAGE_SHA = '288ec2d5f9068221b5922cfeac29301f226e030734e67b15a4399748224d071e'
SOLVER_SOURCE = SOURCE/'bots/adversarial/solver_like.py'
SOLVER_SHA = '76ef638a9a7e9762316ba34d54eca435bd37086afd9bb30433423346282a5240'
BASE, CONTROL, TREATMENT = 'specialist480', 'control504', 'diverse504'
POLICIES = (BASE, CONTROL, TREATMENT)
SEEDS = (342, 343, 344)
RANGES = {f:(2040200000, 2000) for f in ('noregrets','rich96','calling-station','tight-value','pressure','mixed')}
RANGES.update(public=(2040100000,20000), varied=(2040300000,2000), solverlike=(2040400000,2000))
STRIDE = 311


class AuditedSolverLike:
    """Use the original functions directly and legalize their requested action against the observed state."""
    def __init__(self):
        check(file_sha(SOLVER_SOURCE) == SOLVER_SHA, 'Changed external SolverLike source')
        spec = importlib.util.spec_from_file_location('isolated_solverlike_policy', SOLVER_SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def decide(self, observation, seed):
        bounds = observation['raise_bounds']
        view = dict(observation)
        view['min_raise_to'] = bounds[0] if bounds else view['your_stack'] + view['your_bet_this_street']
        make_move = self.module._decide_preflop if not view['community_cards'] else self.module._decide_postflop
        raw = make_move(view)
        kind = raw['action']
        if kind == 'fold' and not view['can_check']:
            concrete = {'kind':'fold'}
        elif bounds and kind in ('all_in','raise'):
            target = bounds[1] if kind == 'all_in' else raw['amount']
            check(bounds[0] <= target <= bounds[1], 'SolverLike raise out of bounds')
            concrete = {'kind':'raise', 'to':target}
        else:
            check(kind in ('check','call','fold','all_in','raise'), 'Unexpected SolverLike action')
            concrete = {'kind':'check' if view['can_check'] else 'call'}
        return {'action':concrete}


def policy_set(plan):
    old_stage = read(OLD/'stage-ready.json')['policies']
    stage = read(STUDY/'stage-ready.json')
    check(plan['policies'][BASE] == old_stage['specialist480']
          and plan['policies'][CONTROL] == old_stage['mixture504'], 'Parent or control identity differs')
    source = plan['source']
    frozen = read(OLD/'population/evaluation-plan.json')
    check(source == frozen['source'] and set(plan['policies']) == set(POLICIES), 'Different policy source or arms')
    for seed in SEEDS:
        key = str(seed)
        parent = read(checked_ref(plan['policies'][BASE][key]))
        current = stage['policies']['poolMixture504'][key]
        check(current == plan['policies'][TREATMENT][key], 'Wrong treatment policy reference')
        for label in (CONTROL, TREATMENT):
            ref = plan['policies'][label][key]
            manifest = checked_ref(ref)
            actual = read(manifest)
            check(actual['seed'] == seed and actual['preflop']['sha256'] == parent['preflop']['sha256']
                  and actual['format'] == 'fixed-preflop-scheduled-postflop-mixture-v1'
                  and [row['iteration'] for row in actual['postflop']] == [486,492,498,504],
                  'Incorrect policy snapshot schedule, seed or preflop strategy')
            verify_files({manifest.parent/ref['file']:ref['sha256'] for ref in [actual['preflop'],*actual['postflop']]})
        new = read(current['path'])['postflop']
        control = read(plan['policies'][CONTROL][key]['path'])['postflop']
        check(set(r['sha256'] for r in new).isdisjoint(r['sha256'] for r in control), 'Reused control snapshot')


def checked_design():
    registration, plan, report = (read(POP/name) for name in ('registration.json','evaluation-plan.json','summary.json'))
    design = read(POP/'design.json')
    root_reg = read(STUDY/'registration.json')
    hashes = {OLD/'stage-ready.json':OLD_STAGE_SHA, OLD/'population/evaluation-plan.json':OLD_PLAN_SHA,
              CONFIG:registration['config_sha256'], POP/'design.json':registration['design_sha256'],
              POP/'evaluation-plan.json':report['plan_sha256'], TRAIN_CONFIG:root_reg['config_sha256'],
              SOLVER_SOURCE:SOLVER_SHA, RICH:RICH_SHA}
    verify_files(hashes)
    check(digest(CONFIG)=='61cacc2449818c98444e1c9ed5dc140fb4cc88610576df9d2a96441dee6e4dc6',
          'Evaluation protocol differs from the registered design')
    check(datetime.fromisoformat(root_reg['created_utc']) <= datetime.fromisoformat(registration['registered_utc'])
          <= datetime.fromisoformat(plan['created_utc']), 'Unordered registration and binding')
    check(plan['stage_sha256']==digest(STUDY/'stage-ready.json'), 'Wrong final training stage')
    early = read(POP/'reference-plan.json')
    early_sha = digest(POP/'reference-plan.json')
    full_sha = digest(POP/'evaluation-plan.json')
    check(early_sha==read(POP/'reference-bound.json')['plan_sha256']==report['reference_plan_sha256']
          and full_sha==read(POP/'evaluation-bound.json')['plan_sha256']
          and early['policies']=={k:plan['policies'][k] for k in (BASE,CONTROL)}
          and all(early[k]==plan[k] for k in ('design_sha256','source','fields','seeds'))
          and {job['id'] for job in early['jobs']}=={'public-seed-342-specialist480','public-seed-342-control504'},
          'Reference and full plans differ')
    verify_files(early['frozen_files'])
    verify_files(plan['frozen_files'])
    verify_files(root_reg['frozen_files'])
    groups = plan['fields']
    config = read(CONFIG)
    check(config['fields'].keys() == groups.keys() == RANGES.keys()
          and config['previous_evaluation_plan_sha256'] == OLD_PLAN_SHA
          and {f:(v['deal_seed'],v['deals']) for f,v in groups.items()} == RANGES
          and all(groups[f]['role']==r['role'] and groups[f]['family']==r['family'] for f,r in config['fields'].items()),
          'Unexpected table or card coverage')
    ranges = set(RANGES.values())
    past = [r['range'] for r in prior_card_ranges(STUDY).values()]
    check(all(max(start,a)>=min(start+size,b) for a,b in past for start,size in ranges),
          'Card range overlaps a previously reserved or observed range')
    expected = {f'{f}-seed-{s}-{p}':(f,s,p) for f in RANGES for s in SEEDS for p in POLICIES}
    jobs = {job['id']:(job['field'],job['seed'],job['label']) for job in plan['jobs']}
    check(jobs==expected and len(plan['jobs'])==len(expected)==report['completed_cells']==81
          and set(plan['seeds'])==set(SEEDS) and plan['fields']==design['fields']
          and plan['jobs']==design['jobs'] and report['complete'] and not report['model_promoted'],
          'Missing or changed scheduled jobs')
    check(sum(n for _,n in ranges)==report['unique_original_card_blocks']==26000,
          'Distinct original card count mismatch')
    policy_set(plan)
    return plan, report


def bots_for(field, seed, parent, known):
    job = {'field':field,'seed':seed}
    if field == 'solverlike':
        return {'solverlike':known['solverlike']}, None
    ids, roles = expected_identities(job, parent, False)
    bots = {}
    for role in roles:
        if role=='rich96':
            ref = read(RICH)['policies']['rich96'][str(seed)]
            key = (role,seed)
            if key not in known:
                known[key] = RichPolicy(checked_ref(ref),None)
            bots[role] = known[key]
        elif role!='noregrets':
            if role not in known:
                known[role] = ReferenceOpponent(role)
            bots[role] = known[role]
    return bots, ids


def verify_cell(plan, label, field, seed, parent, opponents, engine, traces):
    spec = plan['fields'][field]
    start, deals = spec['deal_seed'], spec['deals']
    job = f'{field}-seed-{seed}-{label}'
    folder = POP/'evaluation'/job
    result, prov, status = (read(folder/n) for n in ('results.json','provenance.json','status.json'))
    result_sha = digest(folder/'results.json')
    verify_files({folder/'hands.jsonl.gz':result['raw_hands_sha256'], folder/'provenance.json':result['provenance_sha256']})
    check(status['phase']=='complete' and status['verified_hands']==deals*6 and status['results_sha256']==result_sha
          and all(result[k] for k in ('complete','terminal_payoffs_replayed','raw_hands_verified'))
          and all(result['verification'][k] for k in ('terminal_payoffs_replayed','chip_conservation_verified','aggregate_recomputed')),
          'Unverified stored hand record')
    ref = plan['policies'][label][str(seed)]
    seen = dict(label=label,field=field,training_seed=seed,deal_seed=start,deals=deals,hands=deals*6,
                big_blind_chips=100,candidate=ref['path'],candidate_sha256=ref['sha256'])
    check(all(prov[k]==result[k]==v for k,v in seen.items()), 'Model identity, cards, or table differ')
    check(prov['dependency_hashes']==result['dependency_hashes'] and all(file_sha(k)==h for k,h in prov['dependency_hashes'].items())
          and file_sha(ref['path'])==ref['sha256'] and prov['opponents']==result['opponents'],
          'Source, opponent or dependency changed')
    early = field=='public' and seed==342 and label in (BASE,CONTROL)
    expected_plan = file_sha(POP/('reference-plan.json' if early else 'evaluation-plan.json'))
    check(prov['evaluation_plan_sha256']==result['evaluation_plan_sha256']==expected_plan
          and prov['design_sha256']==result['design_sha256']==plan['design_sha256'],
          'Wrong assignment to registered reference and full card sets')
    bots, opponent_ids = bots_for(field,seed,parent,opponents)
    if field=='solverlike':
        identities = result['opponents']
        check(set(identities)=={'solverlike'} and (SOLVER_SHA in json.dumps(identities['solverlike']))
              and prov['dependency_hashes'].get(str(SOLVER_SOURCE))==SOLVER_SHA, 'Wrong SolverLike policy')
    else:
        check(result['opponents']==opponent_ids, 'Wrong opponent identity')
    check(list(prov['first_deal_clockwise_lineup'])==[matched_role(field,start,0,i) for i in range(1,6)],
          'Different clockwise lineup')
    bots['hero'] = hero_policy(ref['path'],plan['source'],label!=BASE)
    sums = np.zeros(deals, dtype=np.int64)
    hands = decisions = repeat_hands = repeat_moves = 0
    choices, divergences = Counter(), {}
    raw = folder/'hands.jsonl.gz'
    with gzip.open(raw,'rt') as stream:
        for line in stream:
            row = json.loads(line)
            deal, hero = divmod(hands,6)
            card = start+deal
            rng = hand_seed(card,hero)
            check(deal<deals and (row['deal'],row['hero'],row['seed'],row['hand_seed'])==(deal,hero,card,rng),
                  'Missing or repeated card, seat or private random seed')
            if label!=BASE:
                slot = private_slot(rng)
                check(row['snapshot_iteration']==slot, 'Different snapshot randomization')
                choices[str(slot)] += 1
            else:
                check('snapshot_iteration' not in row, 'Baseline unexpectedly selected a snapshot')
            utilities = row['utilities']
            check(len(utilities)==6 and all(type(u) is int for u in utilities) and sum(utilities)==0
                  and row['chips']==utilities[hero] and 0<len(row['actions'])==row['steps']<=512,
                  'Malformed action or terminal chip ledger')
            sums[deal] += utilities[hero]
            if deal%STRIDE==0 or deal==deals-1:
                trace = replay(row,field,engine,bots)
                key=(deal,hero)
                if label==BASE:
                    traces[key]=trace
                else:
                    street = first_difference(traces[key],trace,hero)
                    check(street!='preflop','Frozen preflop policy changed an action')
                    divergences[street] = divergences.get(street,0)+1
                repeat_hands+=1
                repeat_moves+=row['steps']
            hands+=1
            decisions+=row['steps']
    check(hands==result['verification']['hands']==deals*6 and decisions==result['verification']['concrete_decisions_replayed']
          and verify_terminals(raw)==dict(hands=hands,decisions=decisions), 'Mismatched complete terminal replay')
    check(dict(choices)==result['snapshot_hand_counts'], 'Wrong snapshot totals')
    blocks=sums/6
    close_to(result['deal_blocks'],blocks,'Different original-deal means')
    close_to(result['bb_per_100'],blocks.mean(),'Different table scores')
    proof=dict(hands=hands,decisions=decisions,replayed_model_hands=repeat_hands,replayed_model_actions=repeat_moves,
               sampled_first_different_action=divergences, snapshot_hand_counts=dict(choices),
               result_sha256=result_sha,raw_sha256=result['raw_hands_sha256'])
    return blocks,proof


def audit():
    plan, report = checked_design()
    groups = plan['fields']
    parent_path = STUDY.parent/'preflop-dose/evaluation-plan.json'
    check(file_sha(parent_path)=='642491686308caeaef39f80e90277f8e77d437a6f80b37cae1fa4b47a8a6c4d6',
          'Opponent source registry differs')
    parent = read(parent_path)
    source_registry = read(ROOT/'artifacts/selfplay/current.json')
    blueprint_path=ROOT/source_registry['model']
    public_bot=Fullhouse()
    check(source_registry['sha256']==file_sha(blueprint_path)==parent['opponents']['public']['noregrets']
          and public_bot.digest==parent['opponents']['public']['public'], 'Wrong public or NoRegrets weights')
    opponents={'public':public_bot, 'solverlike':AuditedSolverLike()}
    matrices, proofs, choices = {}, {}, {}
    with ExitStack() as resources:
        ordinary=resources.enter_context(closing(Bridge()))
        blueprint=resources.enter_context(closing(Bridge(blueprint_path)))
        for field in groups:
            roles={matched_role(field,0,0,actor) for actor in range(1,6)}
            engine=blueprint if 'noregrets' in roles else ordinary
            matrices[field]={label:[] for label in POLICIES}
            for seed in SEEDS:
                baseline_traces={}
                for label in POLICIES:
                    vals, proof=verify_cell(plan,label,field,seed,parent,opponents,engine,baseline_traces)
                    matrices[field][label].append(vals)
                    proofs[f'{field}-seed-{seed}-{label}']=proof
                    if label==CONTROL:
                        choices[f'{field}/{seed}']=proof['snapshot_hand_counts']
                    if label==TREATMENT:
                        check(choices[f'{field}/{seed}']==proof['snapshot_hand_counts'],
                              'Mixtures drew different hand-private snapshot sequences')
            matrices[field]={label:np.asarray(rows) for label,rows in matrices[field].items()}
    check({name:row['result_sha256'] for name,row in proofs.items()}==report['cell_result_sha256']
          and choices==report['snapshot_hand_counts'],'Summarized cells or sampled slot totals differ')
    hands=sum(row['hands'] for row in proofs.values())
    actions=sum(row['decisions'] for row in proofs.values())
    check(hands==report['completed_hands']==1944000, 'Incomplete terminal outcomes')
    audit_scores={}
    public={}
    for pair,left,right in [('primary',CONTROL,TREATMENT),('treatment_vs_parent',BASE,TREATMENT),
                            ('control_vs_parent',BASE,CONTROL)]:
        audit_scores[pair]=audit_pair(report[pair],groups,matrices,left,right)
        old,new=matrices['public'][left],matrices['public'][right]
        observed=paired_stats(old,new,NormalDist().inv_cdf(.975))
        key=f'{right}_minus_{left}'
        reported=report['public_comparisons'][key]
        for column, value in [('baseline_bb100',old.mean()),('candidate_bb100',new.mean()),
                              ('mean',observed['mean']),('paired_ci95',observed['interval']),
                              ('by_seed',observed['fixed_seed_changes'])]:
            close_to(reported[column],value,'Different paired public contrast')
        public[key]=observed
    check(set(public)==set(report['public_comparisons']), 'Wrong public comparisons')
    return dict(verified_utc=utc(),audit_source_sha256=digest(__file__),
                summary_sha256=digest(POP/'summary.json'),plan_sha256=digest(POP/'evaluation-plan.json'),
                reference_plan_sha256=digest(POP/'reference-plan.json'),stage_sha256=plan['stage_sha256'],
                original_card_blocks=26000,completed_cells=len(proofs),terminal_replay_hands=hands,
                terminal_replay_actions=actions,terminal_auditor_binary_sha256=digest(terminal_auditor()),
                terminal_auditor_source_sha256=digest(ROOT/'native/noregrets-json/terminal_audit.rs'),
                model_action_replay_hands=sum(row['replayed_model_hands'] for row in proofs.values()),
                model_action_replay_actions=sum(row['replayed_model_actions'] for row in proofs.values()),
                action_sample_card_stride=STRIDE, paired_original_card_scores=audit_scores,
                public_nominal_contrasts=public, matched_private_snapshot_counts=choices, cells=proofs)


if __name__=='__main__':
    atomic_json(STUDY/'final-audit.json', audit())
