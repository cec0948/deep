"""Fit one pot-weighted residual using only frozen training opponents and cards."""
import argparse
import gzip
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from collect_poker_river_recalibration import PRIOR, REPORT, STUDY, design
from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import atomic_json, digest
from poker_preflop_specialist import file_ref
from poker_river_call_diverse import EDGE
from poker_river_call_learning import MASS, eligible, features
from poker_river_call_recalibrated import FEATURE_NAMES, FORMAT, RecalibratedRiverCallPolicy, basis


def train_cell(raw, group, seed, bounds, count, checksum):
    verify_files({raw:checksum})
    states, labels, pots = [], [], []
    previous = bounds[0] - 1
    with gzip.open(raw, 'rt') as source:
        for row in map(json.loads, source):
            state, deal = row['root_observation'], row['deal_seed']
            if (row['split'] != 'train' or row['group'] != group or row['seed'] != seed
                    or not bounds[0] <= deal < bounds[1] or deal <= previous or not eligible(state)
                    or state['seat_to_act'] != row['hero']):
                raise ValueError('Wrong training field, actor or original card')
            previous = deal
            share = row['showdown_share']
            pot = state['pot'] + state['amount_owed']
            value = row['call_utility_chips'] - row['fold_utility_chips']
            if (share not in (0., .5, 1.) or row['call_minus_fold_chips'] != value
                    or abs(value - (share * pot - state['amount_owed'])) > (3 if share == .5 else 0)):
                raise ValueError('Incorrect settlement label')
            states.append(features(state))
            labels.append(share)
            pots.append(pot / 100.)
    if len(labels) != count:
        raise ValueError('Incomplete training cell')
    chip_weights = np.sqrt(np.clip(pots, 5., 120.))
    return np.asarray(states), np.asarray(labels), chip_weights / chip_weights.sum()


def load_panel(spec, sha):
    training = []
    prior_design = read(PRIOR/'design.json')['collection']['train']
    for name, checksum in spec['replay_ledger_sha256'].items():
        _, group, sub, _ = Path(name).parts
        seed = int(sub.removeprefix('seed-'))
        bounds = prior_design['by_group'][group][str(seed)]
        cell = train_cell(PRIOR/name, group, seed, (bounds['start'],bounds['limit']),
                          prior_design['positions_per_seed_group'], checksum)
        training.append((*cell, .5 / len(spec['replay_ledger_sha256'])))
    panel = spec['collection']['train']
    count, records = panel['positions_per_cell'], {}
    for group, by_seed in panel['ranges'].items():
        for sid, bounds in by_seed.items():
            folder = STUDY/'train'/group/f'seed-{sid}'
            receipt = read(folder/'collection-complete.json')
            expected = spec['opponents'][group]
            opponent = expected[sid] if isinstance(expected, dict) else expected
            if (not receipt['complete'] or receipt['positions'] != count
                    or receipt['design_sha256'] != sha or receipt['opponent_sha256'] != opponent
                    or receipt['parent_policy_sha256'] != spec['parent_policy_sha256'][sid]
                    or receipt['native_settlements'] != 2*count or receipt['first_card'] != bounds[0]):
                raise ValueError('Training source has invalid provenance')
            cell = train_cell(folder/'positions.jsonl.gz',group,int(sid),bounds,count,receipt['position_ledger_sha256'])
            training.append((*cell, .5/(len(panel['ranges'])*len(spec['seeds']))))
            records[f'{group}/{sid}'] = receipt['position_ledger_sha256']
    x = np.concatenate([rows for rows,_,_,_ in training])
    y = np.concatenate([labels for _,labels,_,_ in training])
    weight = np.concatenate([shares*mass for _,_,shares,mass in training])
    if not np.isclose(weight.sum(), 1., atol=1.e-12, rtol=0):
        raise ValueError('Expected equally weighted old and fresh training panels')
    return x, y, weight, records


def fit(sha):
    spec = design(sha)
    parent_file = PRIOR/'models/model.json'
    verify_files({parent_file:spec['parent_model_sha256'],PRIOR/'selection.json':spec['parent_selection_sha256']})
    parent = read(parent_file)
    refs = read(PRIOR/'selection.json')['policies']
    x, y, weights, fresh = load_panel(spec,sha)
    center, scale = np.asarray(parent['center']),np.asarray(parent['scale'])
    z = basis(x,center,scale)
    inherited = np.pad(parent['coefficients'],(0,len(FEATURE_NAMES)-len(parent['coefficients'])))
    parent_logits = x[:,0] + parent['intercept'] + z@inherited

    def objective(delta):
        logits = parent_logits + delta[0] + z@delta[1:]
        residual = weights*(expit(logits)-y)
        slopes = delta[1:]
        cost = weights@(np.logaddexp(0.,logits)-y*logits) + .015*(slopes@slopes) + .005*delta[0]**2
        gradient = np.r_[residual.sum()+.01*delta[0], residual@z+.03*slopes]
        return cost, gradient

    initial = np.zeros(len(FEATURE_NAMES)+1)
    baseline = float(objective(initial)[0])
    result = minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1.e-12,'gtol':1.e-9})
    if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun)
            or result.fun >= baseline or result.nit == 0):
        raise ValueError('River residual did not converge to a lower objective')
    model = dict(format=FORMAT,source_model_sha256=spec['parent_model_sha256'],features=list(FEATURE_NAMES),
                 coefficients=(inherited+result.x[1:]).tolist(),intercept=float(parent['intercept']+result.x[0]),
                 passive_shift=MASS,minimum_equity_margin=EDGE)
    with tempfile.TemporaryDirectory(dir=STUDY) as temp:
        staged = Path(temp)/'models'
        staged.mkdir()
        atomic_json(staged/'model.json',model)
        m = digest(staged/'model.json')
        policies = {}
        for seed in spec['seeds']:
            sid = str(seed)
            folder = staged/f'seed-{seed}'
            folder.mkdir()
            source = refs[sid]
            verify_files({source['path']:spec['parent_policy_sha256'][sid]})
            policy = folder/'policy.json'
            atomic_json(policy,dict(format=FORMAT,seed=seed,
                    baseline=dict(path=os.path.relpath(source['path'],STUDY/'models'/folder.name),sha256=source['sha256']),
                    model=dict(path='../model.json',sha256=m)))
            policies[sid] = digest(policy)
        os.rename(staged,STUDY/'models')
    manifests = {str(seed):file_ref(STUDY/'models'/f'seed-{seed}'/'policy.json') for seed in spec['seeds']}
    for sid, source in manifests.items():
        if RecalibratedRiverCallPolicy(source['path']).digest != policies[sid]:
            raise ValueError('Frozen weights failed to restore')
    common = dict(complete=True,frozen_utc=utc(),design_sha256=sha,model_sha256=m,
                  parent_model_sha256=spec['parent_model_sha256'],parent_objective=baseline,
                  fitted_objective=float(result.fun),optimizer_steps=int(result.nit),
                  residual_norm=float(np.linalg.norm(result.x)),gradient_inf_norm=float(np.max(np.abs(result.jac))),
                  training_positions=len(y),fresh_positions=spec['collection']['train']['positions_per_cell']*len(fresh),
                  fresh_ledger_sha256=fresh,policy_sha256=policies,
                  sources={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('poker_river_call_recalibrated.py'))})
    atomic_json(REPORT/'candidate.json',common)
    atomic_json(STUDY/'candidate.json',dict(common,policies=manifests,model=file_ref(STUDY/'models/model.json')))
    return common


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('design_sha')
    args = cli.parse_args()
    with (STUDY/'fit.lock').open('x'):
        print(json.dumps(fit(args.design_sha),allow_nan=False))
