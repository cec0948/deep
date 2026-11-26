"""Fit one registered, shared river showdown calibrator from the training split."""
import gzip
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import file_ref
from poker_river_call_learning import FEATURES, FORMAT, MASS, PENALTY, RiverCallPolicy, eligible, features


REL = Path('preflop-followup/480-v1/river-call-learning-20261126')
STUDY = ROOT/'artifacts'/REL
PUBLIC = ROOT/'reports'/REL
PRIOR = STUDY.parent/'street-progress-20261126'
DESIGN = 'ea1db38fc35ae39a62ba234c8a1117d5ac12f6b836788c4002e9258b4eabbed8'
GUARD = Path(__file__).with_name('poker_river_dominance.py')
RUNTIME = Path(__file__).with_name('poker_river_call_learning.py')


def train_rows(seed, design, parent):
    folder = STUDY/'train'/f'seed-{seed}'
    ledger = folder/'positions.jsonl.gz'
    receipt = read(folder/'collection-complete.json')
    cfg = design['collection']['train']
    count = cfg['positions_per_seed']
    bounds = cfg['by_seed'][str(seed)]
    if (receipt['seed'] != seed or receipt['split'] != 'train' or not receipt['complete']
            or receipt['distinct_original_roots'] != count or receipt['positions'] != count
            or receipt['design_sha256'] != DESIGN or receipt['hero_v1_sha256'] != parent['sha256']
            or receipt['river_guard_sha256'] != design['guard_sha256']):
        raise ValueError('Incomplete or unregistered river call training source')
    verify_files({ledger:receipt['position_ledger_sha256'], parent['path']:parent['sha256']})
    x, shares = [], []
    last = -1
    with gzip.open(ledger, 'rt') as source:
        for line in source:
            row = json.loads(line)
            state = row['root_observation']
            deal = row['deal_seed']
            if (row['split'] != 'train' or row['seed'] != seed or not eligible(state)
                    or not bounds['start'] <= deal < bounds['limit'] or deal <= last
                    or state['seat_to_act'] != row['hero']):
                raise ValueError('Unexpected or repeated river decision')
            last = deal
            x.append(features(state))
            share = row['showdown_share']
            delta = row['call_minus_fold_chips']
            approx = share*(state['pot']+state['amount_owed'])-state['amount_owed']
            if (share not in (0,.5,1) or delta != row['call_utility_chips']-row['fold_utility_chips']
                    or abs(delta-approx) > (3 if share==.5 else 0)):
                raise ValueError('Showdown label conflicts with actual terminal payouts')
            shares.append(share)
    if len(shares) != count:
        raise ValueError('Wrong number of distinct river deals')
    return x, shares, dict(training_positions=count, win_tie_loss=[shares.count(v) for v in (1.,.5,0.)],
                           train_ledger_sha256=receipt['position_ledger_sha256'],
                           parent_sha256=parent['sha256'])


def calibrate(x, y):
    center, scale = x.mean(0), np.maximum(x.std(0), .1)
    z = (x-center)/scale
    offset = x[:,0]
    def objective(param):
        bias, w = param[0], param[1:]
        logits = offset+bias+z@w
        residual = expit(logits)-y
        loss = np.mean(np.logaddexp(0.,logits)-y*logits)+PENALTY*.5*(w@w)
        grad = np.r_[residual.mean(), (residual@z)/len(y)+PENALTY*w]
        return loss, grad
    initial = np.zeros(x.shape[1]+1)
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    if (not result.success or not np.isfinite(result.x).all() or result.fun>objective(initial)[0]
            or not np.isfinite(result.fun) or result.nit == 0):
        raise ValueError('The registered showdown calibration did not converge')
    model = dict(format=FORMAT, features=list(FEATURES), center=center.tolist(), scale=scale.tolist(),
                 coefficients=result.x[1:].tolist(), intercept=float(result.x[0]),
                 penalty=PENALTY, passive_shift=MASS)
    train = dict(optimizer='L-BFGS-B', steps=int(result.nit), training_regularized_objective=float(result.fun),
                 uniform_offset_training_objective=float(objective(initial)[0]),
                 final_gradient_max_abs=float(np.abs(result.jac).max()))
    return model, train


def fit():
    verify_files({STUDY/'design.json':DESIGN, PUBLIC/'design.json':DESIGN})
    design = read(STUDY/'design.json')
    verify_files({PRIOR/'flop/selected.json':design['flop_selection_sha256'],
                  PRIOR/'play-ready.json':design['parent_play_sha256'], GUARD:design['guard_sha256']})
    if read(STUDY/'train/collection-status.json')['state'] != 'complete':
        raise ValueError('River call training split has not completed')
    selected = read(PRIOR/'flop/selected.json')['policies']
    with (STUDY/'fit.lock').open('x'):
        pass
    all_x, all_y, seeds = [], [], {}
    for s in design['seeds']:
        x, y, seeds[str(s)] = train_rows(s, design, selected[str(s)])
        all_x.extend(x)
        all_y.extend(y)
    model, metrics = calibrate(np.asarray(all_x), np.asarray(all_y))
    models = STUDY/'models'
    models.mkdir()
    weights = models/'model.json'
    atomic_json(weights, model)
    model_sha = digest(weights)
    policies = {}
    for s in design['seeds']:
        sid = str(s)
        folder = models/f'seed-{s}'
        folder.mkdir()
        policy = folder/'policy.json'
        parent = selected[sid]
        spec = dict(format=FORMAT, seed=s, baseline=dict(path=os.path.relpath(parent['path'],folder),
                    sha256=parent['sha256']), model=dict(path='../model.json', sha256=model_sha))
        atomic_json(policy, spec)
        restored = RiverCallPolicy(policy)
        if (restored.model != model or restored.parent.digest != parent['sha256']
                or restored.digest != digest(policy)):
            raise ValueError('Restored river call model differs from frozen fitted weights')
        policies[sid] = file_ref(policy)
        seeds[sid]['policy_sha256'] = restored.digest
    common = dict(training_complete=True, frozen_utc=utc(), design_sha256=DESIGN, model_sha256=model_sha,
                  seeds=seeds, training_positions=len(all_y), shared_models=1, fit_metrics=metrics,
                  frozen_before_holdout=True, selected_for_play=False,
                  code_sha256={p.name:digest(p) for p in (Path(__file__),RUNTIME,GUARD)})
    atomic_json(STUDY/'candidate.json', dict(common, policies=policies, model=file_ref(weights)))
    atomic_json(PUBLIC/'candidate.json', common)
    return common


if __name__ == '__main__':
    print(json.dumps(fit()))
