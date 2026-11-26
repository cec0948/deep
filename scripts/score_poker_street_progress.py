"""Select one first-street repair on blind root deals using paired rollout action utilities."""
import argparse
from contextlib import closing
import gzip
import json
import math
from pathlib import Path

import numpy as np

from audit_poker_flop_repair import repair_menus, request_at_root
from controlled_poker_refit import read, utc, verify_files
from evaluate_average_baseline import _validate_reply
from fullhouse_runtime import Bridge
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_flop_repair import FlopRepairPolicy
from poker_river_runtime import action_key


REL = Path('preflop-followup/480-v1/street-progress-20261126')
STUDY = ROOT/'artifacts'/REL
PUBLIC = ROOT/'reports'/REL
ORIGINAL = STUDY.parent/'flop-repair-20261126'
DESIGN = '5ce6f2b5ffc7f8a3ea82019845c5923ea3af874ba0de409d8e59a331bdbc5700'
SEEDS = (342, 343, 344)
Z = 1.96


def require(ok, message):
    if not ok:
        raise ValueError(message)


def interval(values):
    rows = [np.asarray(values[str(s)], dtype=np.float64) for s in SEEDS]
    require(all(a.ndim == 1 and len(a) > 1 and np.isfinite(a).all() for a in rows),
            'Nonfinite or incomplete root-level differences')
    means = [float(a.mean()) for a in rows]
    by_seed = {str(seed): {'mean': m, 'deals': len(a), 'standard_error': float(a.std(ddof=1)/np.sqrt(len(a)))}
               for seed, a, m in zip(SEEDS, rows, means, strict=True)}
    se = math.sqrt(sum(r['standard_error']**2 for r in by_seed.values()))/len(SEEDS)
    mean = math.fsum(means)/len(SEEDS)
    return dict(mean=mean, standard_error=se, lower=mean-Z*se, upper=mean+Z*se, by_seed=by_seed)


def probabilities(policy, state, nonce, available):
    reply = _validate_reply(state, policy.decide(state, nonce))
    rows = reply['distribution']
    by_key = {action_key(r['action']): r['probability'] for r in rows}
    require(len(by_key) == len(rows) and by_key.keys() <= set(available)
            and by_key.get(action_key(reply['action']), 0.) > 0., 'Illegal/duplicate action in policy')
    values = np.array([by_key.get(key, 0.) for key in available], dtype=np.float64)
    require(np.isfinite(values).all() and np.min(values) >= 0 and abs(values.sum()-1.) <= 1.e-6,
            'Invalid source policy distribution')
    return values/values.sum()


def references(design, street):
    if street == 'turn':
        source = read(STUDY/'flop/selected.json')
        require(source['complete'] and source['street'] == 'flop', 'Flop path has not been chosen')
        return source['policies'], source['selected_stage']
    refs = {str(seed): dict(path=str(ORIGINAL/f'seed-{seed}/policy.json'),
                            sha256=design['source_policies'][str(seed)]['manifest_sha256']) for seed in SEEDS}
    return refs, 'v1'


def score_seed(street, split, seed, design, candidate_ref, prior_ref, reach_hash, bridge):
    sid = str(seed)
    budget = design['collection'][street][split]
    bounds = budget['by_seed'][sid]
    folder = STUDY/street/split/f'seed-{seed}'
    raw = folder/'positions.jsonl.gz'
    receipt_path = folder/'collection-complete.json'
    receipt = read(receipt_path)
    require(receipt['complete'] and (receipt['street'],receipt['split'],receipt['seed']) == (street,split,seed)
            and receipt['training_design_sha256'] == DESIGN and receipt['first_card'] == bounds['start']
            and (receipt['positions'],receipt['repeats']) == (budget['positions_per_seed'],budget['repeats'])
            and receipt['parent_sha256'] == design['source_policies'][sid]['base480_sha256']
            and receipt['selected_flop_receipt_sha256'] == reach_hash
            and receipt['reach_policy_sha256'] == (prior_ref['sha256'] if street == 'turn' else receipt['parent_sha256']),
            'Incorrect collection source, selected reach policy or root budget')
    verify_files({raw:receipt['position_ledger_sha256'], candidate_ref['path']:candidate_ref['sha256'],
                  prior_ref['path']:prior_ref['sha256']})
    parent = FlopRepairPolicy(prior_ref['path'])
    if street == 'turn':
        from poker_street_progress import StreetProgressPolicy
        candidate = StreetProgressPolicy(candidate_ref['path'])
    else:
        candidate = FlopRepairPolicy(candidate_ref['path'])
    trial, incumbent = [], []
    absolute_shift = modified = labels = 0
    last_deal = bounds['start']-1
    with gzip.open(raw, 'rt') as stream:
        for index,line in enumerate(stream):
            row = json.loads(line)
            deal, state = row['deal_seed'],row['root_observation']
            require((row['street'],row['split'],row['seed']) == (street,split,seed)
                    and last_deal < deal < bounds['limit'] and state['street'] == street
                    and row['hero'] == state['seat_to_act'] == (deal-bounds['start'])%6,
                    'Duplicate card, wrong actor, or inconsistent split')
            last_deal = deal
            available, mapping = repair_menus(state)
            require(available == [action_key(a) for a in row['legal_actions']]
                    and mapping == row['abstract_to_concrete_index'], 'Illegal menu or duplicate raise mapping')
            if index % 191 == 0:
                rebuilt = bridge.call(dict(op='replay', **request_at_root(row)))
                require(rebuilt == state,'Held-out root cannot be reconstructed from actor-visible inputs')
            payouts = np.asarray(row['action_payoffs_chips'])
            require(payouts.shape == (len(available),budget['repeats']) and payouts.dtype.kind in 'iu'
                    and payouts.min() >= -10000 and payouts.max() <= 50000, 'Invalid paired terminal utilities')
            if ('fold',None) in available:
                require(np.all(payouts[available.index(('fold',None))] == state['your_stack']-10000),
                        'Incorrect fold settlement')
            labels += len(available)
            # Center before taking the dot product to cancel the common sunk chip offset exactly.
            q = payouts.mean(axis=1)/100.
            q -= q[0]
            base = probabilities(parent.base,state,deal,available)
            require(np.allclose(base,row['parent_probabilities'],rtol=0,atol=1e-6), 'Wrong 480 anchor')
            old = probabilities(parent,state,deal,available)
            new = probabilities(candidate,state,deal,available)
            if street == 'turn':
                require(np.allclose(old,base,rtol=0,atol=1e-7), 'Selected flop parent should use 480 on turn')
            delta = new-old
            tv = .5*float(np.abs(delta).sum())
            absolute_shift += tv
            modified += int(tv > 1.e-7)
            trial.append(float(delta @ q))
            if street == 'flop' and split == 'dev':
                incumbent.append(float((old-base) @ q))
    n = len(trial)
    require(n == receipt['positions'] and labels == receipt['action_labels'] and last_deal == receipt['last_card']
            and labels*budget['repeats'] == receipt['terminal_rollouts'], 'Incomplete held-out ledger')
    return trial, incumbent, dict(root_deals=n, action_labels=labels, terminal_rollouts=labels*budget['repeats'],
                                  changed_root_distributions=modified, mean_total_variation=absolute_shift/n,
                                  ledger_sha256=digest(raw), collection_sha256=digest(receipt_path))


def public_and_private(street, name, result):
    public = PUBLIC/street
    public.mkdir(parents=True, exist_ok=True)
    atomic_json(public/name,result)
    atomic_json(STUDY/street/name,result)


def public_refs(refs):
    return {sid: ref['sha256'] for sid,ref in refs.items()}


def select(street, positive, current_refs, prior_refs, prior_stage, source_sha, split):
    refs = current_refs if positive else prior_refs
    stage = ('v2' if street == 'flop' else 'v3') if positive else prior_stage
    directory = STUDY/street
    payload = dict(complete=True,selected_utc=utc(),street=street,design_sha256=DESIGN,
                   selected='candidate' if positive else 'parent',selected_stage=stage,
                   candidate_receipt_sha256=source_sha,dev_sha256=digest(directory/'dev-score.json'),
                   confirm_sha256=digest(directory/'confirm-score.json') if split=='confirm' else None,
                   selection_scope='Conditional on the registered reached self-play root distribution.',
                   whole_game_strength_verified=False, global_model_promoted=False)
    atomic_json(PUBLIC/street/'selected.json',dict(payload,policy_sha256=public_refs(refs)))
    # Private last so downstream training can only start after the selection report exists.
    atomic_json(directory/'selected.json',dict(payload,policies=refs))


def run(street,split):
    verify_files({STUDY/'design.json':DESIGN,PUBLIC/'design.json':DESIGN})
    design = read(STUDY/'design.json')
    p = STUDY/street
    candidate_path = p/'candidate.json'
    candidate = read(candidate_path)
    require(candidate['training_complete'], 'Candidate not frozen')
    frozen = digest(candidate_path)
    prior_refs, prior_stage = references(design,street)
    current_refs = candidate['policies']
    reach_hash = digest(STUDY/'flop/selected.json') if street == 'turn' else None
    require(set(prior_refs) == set(current_refs) == set(map(str,SEEDS)), 'Missing seed policy')
    if split == 'confirm':
        dev = read(p/'dev-score.json')
        require(dev['eligible_for_confirmation'] and dev['candidate_receipt_sha256'] == frozen
                and dev['design_sha256'] == DESIGN and dev['prior_policy_sha256'] == public_refs(prior_refs)
                and dev['reach_selection_sha256'] == reach_hash, 'Confirmation not authorized by frozen development')
    name = f'{split}-score.json'
    require(not (p/name).exists() and not (p/'selected.json').exists(), 'Scoring already finished')
    deltas, incumbents, proofs = {}, {}, {}
    with (p/f'{split}-scoring.lock').open('x'), closing(Bridge()) as bridge:
        for seed in SEEDS:
            sid = str(seed)
            deltas[sid],incumbents[sid],proofs[sid] = score_seed(
                street,split,seed,design,current_refs[sid],prior_refs[sid],reach_hash,bridge)
        verify_files({candidate_path:frozen, **({STUDY/'flop/selected.json':reach_hash} if reach_hash else {})})
        for refs in (current_refs, prior_refs):
            verify_files({ref['path']:ref['sha256'] for ref in refs.values()})
        delta = interval(deltas)
        result = dict(complete=True, completed_utc=utc(),street=street,split=split,design_sha256=DESIGN,
                      scorer_sha256=digest(__file__),candidate_receipt_sha256=frozen,
                      candidate_policy_sha256=public_refs(current_refs),prior_policy_sha256=public_refs(prior_refs),
                      reach_selection_sha256=reach_hash,paired_gain=delta,proofs=proofs,
                      unique_original_root_deals=sum(r['root_deals'] for r in proofs.values()),
                      units='bb per reached first decision in the specified self-play pool',
                      error_unit='One distinct reached deal, averaged over action rollouts; equal fixed-seed weights.',
                      one_sided_confirmation_alpha=.025,global_model_promoted=False,match_games=0)
        if split == 'dev':
            result['eligible_for_confirmation'] = delta['mean'] > 0.
            if street == 'flop':
                result['diagnostic_incumbent_v1_minus_480'] = interval(incumbents)
        else:
            result['local_confirmation_passed'] = delta['lower'] > 0.
        public_and_private(street,name,result)
        if split == 'confirm' or not result['eligible_for_confirmation']:
            select(street,split=='confirm' and result['local_confirmation_passed'],
                   current_refs,prior_refs,prior_stage,frozen,split)
        print(json.dumps(dict(street=street,split=split,paired_gain=delta),ensure_ascii=False))


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('street',choices=('flop','turn'))
    cli.add_argument('split',choices=('dev','confirm'))
    args=cli.parse_args()
    run(args.street,args.split)
