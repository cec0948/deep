"""Measure exposure to dominated river folds in ordinary frozen-policy six-seat hands."""
import argparse
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from contextlib import closing
import gzip
import json
from pathlib import Path
import random

from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply, semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, PYTHON, atomic_json, digest
from poker_flop_action_values import _nonce
from poker_flop_repair import FlopRepairPolicy
from run_poker_preflop_confirmation import run_jobs
from train_poker_flop_repair import SEEDS, STAGES, OpponentsBySeat, sources

REL = Path('preflop-followup/480-v1/river-natural-20261126')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
SELECTION = STUDY.parent/'street-progress-20261126/flop/selected.json'
SCRIPT = Path(__file__).resolve()
GUARD = SCRIPT.with_name('poker_river_dominance.py')
DESIGN_SHA = '76c84bd2932d1be976395a2533f9274442cfcf4489c189715c6680e853f74c9d'


SCORE_SHA = 'afe5a34076a8ed94c3f07e66517fd8d70cb9ae6c942f87d55f050dbc631e02c1'
RESTORED = STUDY/'recovered-originals'


def design():
    verify_files({STUDY/'design.json':DESIGN_SHA, REPORT/'design.json':DESIGN_SHA})
    spec = read(STUDY/'design.json')
    verify_files({SELECTION:spec['hero_selection_sha256'], STUDY/'score-plan.json':SCORE_SHA, REPORT/'score-plan.json':SCORE_SHA})
    return spec


def collect(seed):
    spec = design()
    v1 = read(SELECTION)['policies'][str(seed)]
    verify_files({v1['path']:spec['hero_policy'][str(seed)]})
    policy = FlopRepairPolicy(v1['path'])
    _,_,_,incumbent,pool = sources(seed)
    folder = RESTORED/f'seed-{seed}'
    folder.mkdir()
    raw = folder/'hero-river-decisions.jsonl.gz'
    total = 0
    with closing(Bridge()) as bridge, gzip.open(raw, 'xt', compresslevel=1) as output:
        for offset in range(spec['deals_per_seed']):
            deal = spec['start'][str(seed)] + offset
            hero = offset % 6
            stages = [480 if seat == hero else STAGES[random.Random(_nonce('opponent',seed,deal,seat)).randrange(6)]
                      for seat in range(6)]
            opponents = OpponentsBySeat(pool,stages)
            state = bridge.call(dict(op='reset', seed=deal, button=0))
            while not state['terminal']:
                actor = state['seat_to_act']
                engine_policy = incumbent if state['street']=='preflop' else policy if hero==actor else opponents
                reply = _validate_reply(state, engine_policy.decide(state,semantic_seed(deal,hero,state)))
                if hero == actor and state['street'] == 'river':
                    fold = sum(r['probability'] for r in reply['distribution'] if r['action']['kind']=='fold')
                    row = dict(deal=deal, seed=seed, hero=hero, lineup=stages, state=state,
                               fold_probability=fold, chosen=reply['action'])
                    output.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                state = bridge.call(dict(op='step',action=dict(reply['action'],seat=actor)))
            total += 1
        bridge_sha = digest(bridge.p.args[0])
    design()
    atomic_json(folder/'complete.json', dict(complete=True, seed=seed, original_deals=total,
                finished_utc=utc(), ledger_sha256=digest(raw), script_sha256=digest(SCRIPT),
                design_sha256=DESIGN_SHA, native_engine_sha256=bridge_sha))


def score(seed, spec, module, plan):
    if seed == 342:
        raw = STUDY/'seed-342/hero-river-decisions.jsonl.gz'
        verify_files({raw:plan['completed_seed342_ledger_sha256'],
                      STUDY/'logs/frequency-342.log':plan['completed_seed342_log_sha256'],
                      STUDY/'initial_collector.py':plan['original_collector_sha256']})
    else:
        folder = RESTORED/f'seed-{seed}'
        receipt = read(folder/'complete.json')
        if not receipt['complete'] or receipt['original_deals']!=spec['deals_per_seed']:
            raise ValueError('Incomplete original hand ledger')
        raw = folder/'hero-river-decisions.jsonl.gz'
        verify_files({raw:receipt['ledger_sha256']})
    start = spec['start'][str(seed)]
    counts = Counter()
    deals_with_river, positive_deals = set(), set()
    total_exposure = 0.
    events = []
    with gzip.open(raw,'rt') as stream:
        for line in stream:
            row = json.loads(line)
            state = row['state']
            deal,hero = row['deal'], row['hero']
            if not (start<=deal<start+spec['deals_per_seed'] and row['seed']==seed
                    and hero==(deal-start)%6 and state['seat_to_act']==hero and state['street']=='river'):
                raise ValueError('Unexpected original decision identity')
            deals_with_river.add(deal)
            counts['river_decisions'] += 1
            if sum(not p['is_folded'] for p in state['players'])==2 and state['amount_owed']>0:
                counts['heads_up_facing_bet'] += 1
            floor = module.closing_call_floor(state)
            fold = row['fold_probability']
            if floor>0:
                counts['certified_call_over_fold'] += 1
                if fold>0:
                    counts['positive_fold_mass_eligible'] += 1
                    positive_deals.add(deal)
                    total_exposure += floor*fold
                    did_fold = row['chosen']['kind']=='fold'
                    counts['sampled_fold_repaired'] += int(did_fold)
                    events.append(dict(deal=deal,hero=hero,fold_mass=fold,floor_chips=floor,original_sample_folded=did_fold))
    counts['original_deals'] = spec['deals_per_seed']
    counts['deals_with_hero_river_decision'] = len(deals_with_river)
    fields = ('original_deals','deals_with_hero_river_decision','river_decisions','heads_up_facing_bet',
              'certified_call_over_fold','positive_fold_mass_eligible','sampled_fold_repaired')
    return dict(seed=seed,counts={k:counts[k] for k in fields},positive_fold_deals=len(positive_deals),
                recorded_minimum_exposure_chips=total_exposure,positive_events=events,ledger_sha256=digest(raw))


def launch():
    spec = design()
    plan = read(STUDY/'score-plan.json')
    guard_path = STUDY/'frozen_final_guard.py'
    verify_files({guard_path:plan['guard_sha256']})
    module_spec = spec_from_file_location('registered_river_guard',guard_path)
    guard = module_from_spec(module_spec)
    module_spec.loader.exec_module(guard)
    RESTORED.mkdir()
    jobs = {str(s):[str(PYTHON),str(SCRIPT),'--seed',str(s)] for s in SEEDS if s != 342}
    run_jobs(RESTORED,jobs,len(jobs),'frequency')
    per_seed = [score(s,spec,guard,plan) for s in SEEDS]
    counts = {k:sum(r['counts'][k] for r in per_seed) for k in per_seed[0]['counts']}
    chips = sum(r['recorded_minimum_exposure_chips'] for r in per_seed)
    result = dict(complete=True, finished_utc=utc(), design_sha256=DESIGN_SHA, score_plan_sha256=SCORE_SHA,
                  scorer_sha256=digest(SCRIPT), guard_sha256=plan['guard_sha256'], counts=counts,
                  positive_fold_deals=sum(r['positive_fold_deals'] for r in per_seed),
                  observed_action_weighted_minimum_bb=chips/spec['chips_per_bb'],
                  eval_scope='Fresh unselected original hands under the frozen v1 historical pool; descriptive frequencies only. No global win-rate estimate.',
                  per_seed=per_seed)
    atomic_json(STUDY/'summary.json',result)
    atomic_json(REPORT/'summary.json',result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    arg = parser.parse_args()
    launch() if arg.seed is None else collect(arg.seed)
