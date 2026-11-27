"""Score only the changed calibration against the deployed policy on new hands."""
import argparse
from contextlib import closing
import gzip
import json
import os

from audit_poker_river_call_learning import geometry, payouts
from collect_poker_river_call_continuation import field
from collect_poker_river_recalibration import PRIOR, REPORT, STUDY, design
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import semantic_seed
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_river_call_diverse import DiverseRiverCallPolicy
from poker_river_call_recalibrated import RecalibratedRiverCallPolicy
from score_poker_river_call_learning import CALL, compare_distributions, distribution, settlement
from score_poker_river_continuation import adoption, cell_summary, smoke, summarize_cells, verify_reach
from score_poker_street_progress import require


LEARNER = 'river-call-recalibration'
PARENT = 'river-call-continuation'


def assess(raw, receipt, cfg, group, seed, engine, incumbent, learner, opponent):
    start, stop = cfg['ranges'][group][str(seed)]
    last = start-1
    values, shifts, sizes, odds = [], [], [], []
    with gzip.open(raw,'rt') as stream:
        for row in map(json.loads,stream):
            deal = row['deal_seed']
            require(last < deal < stop and row['hero']==(deal-start)%6 and row['seed']==seed
                    and row['group']==group and row['split']=='holdout','Wrong holdout card or group')
            last = deal
            state = row['root_observation']
            require(geometry(state), 'Holdout must be the first eligible closing decision')
            verify_reach(row,incumbent,opponent,engine)
            _, chips = settlement(row,engine)
            payouts(row)
            nonce = semantic_seed(deal,row['hero'],state)
            base, old, new = (p.decide(state,nonce) for p in (incumbent.parent,incumbent,learner))
            first, before, after = (distribution(state,reply) for reply in (base,old,new))
            require(before==distribution(state,dict(ok=True,action=old['action'],
                                                     distribution=row['parent_distribution'])),
                    'Ledger was collected from a different river policy')
            # Each policy must use the same final shift budget relative to
            # the same first-layer distribution. The shifts can oppose.
            compare_distributions(state,first,before)
            compare_distributions(state,first,after)
            require(old['action']==new['action'] or all(r['action']['kind']!='raise' for r in (old,new)),
                    'Changing the calibration also changed a sampled raise')
            shift = after.get(CALL,0.) - before.get(CALL,0.)
            shifts.append(shift)
            values.append(shift * chips / 100.)
            sizes.append(state['amount_owed'])
            odds.append(state['amount_owed']/(state['pot']+state['amount_owed']))
    require(len(values)==cfg['positions_per_cell'] and last==receipt['last_card']
            and receipt['attempted_deals']==last-start+1,'Missing held-out roots')
    proof = cell_summary(values,shifts,sizes,odds,receipt['attempted_deals'])
    proof['changed_decisions'] = sum(abs(value)>1.e-12 for value in shifts)
    return values,proof


def score(sha):
    spec = design(sha)
    frozen = digest(STUDY/'candidate.json')
    candidate = read(STUDY/'candidate.json')
    parent = read(PRIOR/'selection.json')['policies']
    require(candidate['complete'] and candidate['design_sha256']==sha,'Candidate is not frozen')
    verify_files({PRIOR/'selection.json':spec['parent_selection_sha256'],
                  candidate['model']['path']:candidate['model_sha256'],
                  **{ROOT/'scripts'/name:h for name,h in candidate['sources'].items()}})
    cfg = spec['collection']['holdout']
    cells, proofs = {}, {}
    with (STUDY/'score.lock').open('x'), closing(Bridge()) as engine:
        for group, bounds in cfg['ranges'].items():
            for seed in spec['seeds']:
                sid = str(seed)
                folder = STUDY/'holdout'/group/f'seed-{sid}'
                raw = folder/'positions.jsonl.gz'
                receipt = read(folder/'collection-complete.json')
                opponent, opponent_sha = field(group,seed)
                require(receipt['complete'] and receipt['positions']==cfg['positions_per_cell']
                        and receipt['first_card']==bounds[sid][0] and receipt['design_sha256']==sha
                        and receipt['split']=='holdout' and receipt['group']==group and receipt['seed']==seed
                        and receipt['opponent_sha256']==opponent_sha==spec['opponents'][group]
                        and receipt['parent_policy_sha256']==spec['parent_policy_sha256'][sid]==parent[sid]['sha256'],
                        'Invalid held-out data provenance')
                verify_files({raw:receipt['position_ledger_sha256'],parent[sid]['path']:parent[sid]['sha256'],
                              candidate['policies'][sid]['path']:candidate['policy_sha256'][sid]})
                incumbent = DiverseRiverCallPolicy(parent[sid]['path'])
                learner = RecalibratedRiverCallPolicy(candidate['policies'][sid]['path'])
                cells[group,seed], proofs[group,seed] = assess(raw,receipt,cfg,group,seed,engine,incumbent,learner,opponent)
        verify_files({STUDY/'candidate.json':frozen})
        gain = summarize_cells(cells,proofs)
        chosen = adoption(gain,gain['observed_contribution_bb_per_100'])
        refs = candidate['policies'] if chosen else parent
        backend = LEARNER if chosen else PARENT
        receipt = dict(complete=True,completed_utc=utc(),design_sha256=sha,candidate_sha256=frozen,
                       model_sha256=candidate['model_sha256'],selected='candidate' if chosen else 'parent',
                       backend=backend,paired_gain=gain,unique_holdout_deals=sum(map(len,cells.values())),
                       attempted_deals=sum(p['attempted_deals'] for p in proofs.values()),
                       changed_decisions=sum(p['changed_decisions'] for p in proofs.values()),
                       proofs={f'{group}/{seed}':proof for (group,seed),proof in proofs.items()},
                       policy_sha256={s:ref['sha256'] for s,ref in refs.items()},
                       scorer_sha256=digest(__file__),whole_game_win_rate_measured=False)
        atomic_json(REPORT/'selection.json',receipt)
        atomic_json(STUDY/'selection.json',dict(receipt,policies=refs))
    return receipt


def publish(sha):
    spec = design(sha)
    selection = read(STUDY/'selection.json')
    verify_files({STUDY/'candidate.json':selection['candidate_sha256']})
    require(selection['complete'] and selection['design_sha256']==sha,'Incomplete selection')
    backend = selection['backend']
    picked = RecalibratedRiverCallPolicy if backend==LEARNER else DiverseRiverCallPolicy
    tests = {}
    with closing(Bridge()) as engine:
        for seed in spec['seeds']:
            sid = str(seed)
            ref = selection['policies'][sid]
            entry = STUDY/'play'/f'seed-{sid}'/'policy.json'
            entry.parent.mkdir(parents=True,exist_ok=True)
            if not entry.is_symlink():
                entry.symlink_to(os.path.relpath(ref['path'],entry.parent))
            verify_files({entry:selection['policy_sha256'][sid]})
            fixture = next((STUDY/'train').glob(f'*/seed-{sid}/positions.jsonl.gz'))
            tests[sid] = smoke(entry,seed,backend,picked(entry),fixture,engine)
    result = dict(complete=True,completed_utc=utc(),selection_sha256=digest(STUDY/'selection.json'),backend=backend,
                  selected=selection['selected'],policy_sha256=selection['policy_sha256'],
                  normal_jsonl_decisions=sum(len(proof['valid_requests']) for proof in tests.values()),protocol=tests)
    for folder in (STUDY,REPORT):
        atomic_json(folder/'play-ready.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('design_sha')
    parser.add_argument('phase',choices=('score','publish'))
    opts = parser.parse_args()
    print(json.dumps((score if opts.phase=='score' else publish)(opts.design_sha),allow_nan=False))
