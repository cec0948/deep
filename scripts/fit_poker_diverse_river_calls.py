"""Continue the selected showdown calibration on old and diverse fresh river hands."""
import fcntl
import gzip
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from controlled_poker_refit import read, utc, verify_files
import fit_poker_river_call as original
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_preflop_specialist import file_ref
from poker_river_call_diverse import EDGE, FORMAT, SOURCE_MODEL, DiverseRiverCallPolicy
from poker_river_call_learning import FEATURES, MASS, eligible, features


REL = Path('preflop-followup/480-v1/river-call-continuation-20261127')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
SEEDS = (342,343,344)
RUNTIME = Path(__file__).with_name('poker_river_call_diverse.py')
DESIGN = 'f5a05053ba0f296de162a6b777cca753527210b49fc92b2c6c28c2f761c84969'


def continue_fit(parent, x, y, weights):
    """Fit the residual in the original standardized basis, starting at the exact parent."""
    z = (x-np.asarray(parent['center']))/np.asarray(parent['scale'])
    parent_coef = np.asarray(parent['coefficients'], dtype=np.float64)
    base_logits = x[:,0] + parent['intercept'] + z@parent_coef
    if (x.shape != (len(y),len(FEATURES)) or weights.shape != y.shape
            or not np.isfinite(z).all() or not np.isin(y,(0.,.5,1.)).all()
            or not np.all(weights > 0) or not np.isclose(weights.sum(),1.,atol=1e-12,rtol=0)):
        raise ValueError('Invalid weighted river training examples')

    def objective(delta):
        logits = base_logits + delta[0] + z@delta[1:]
        error = weights*(expit(logits)-y)
        cost = weights @ (np.logaddexp(0.,logits)-y*logits) + .025*(delta[1:]@delta[1:]) + .005*delta[0]**2
        grad = np.r_[error.sum()+.01*delta[0], error@z+.05*delta[1:]]
        return cost, grad

    initial = np.zeros(len(FEATURES)+1)
    parent_cost = float(objective(initial)[0])
    fit = minimize(objective,initial,jac=True,method='L-BFGS-B',
                   options=dict(maxiter=200, ftol=1e-12, gtol=1e-9))
    if not fit.success or not np.isfinite(fit.x).all() or not np.isfinite(fit.fun) or fit.fun > parent_cost+1e-12:
        raise ValueError('Continued calibration failed to converge')
    model = dict(format=FORMAT, source_model_sha256=SOURCE_MODEL, features=list(FEATURES),
                 center=parent['center'], scale=parent['scale'], intercept=float(parent['intercept']+fit.x[0]),
                 coefficients=(parent_coef+fit.x[1:]).tolist(), passive_shift=MASS, minimum_equity_margin=EDGE)
    metrics = dict(optimizer='L-BFGS-B', steps=int(fit.nit), initial_residual_norm=0.,
                   source_training_objective=parent_cost, fitted_training_objective=float(fit.fun),
                   final_residual_norm=float(np.linalg.norm(fit.x)), final_gradient_max_abs=float(np.abs(fit.jac).max()))
    return model, metrics


def old_replay(source_candidate):
    verify_files({original.STUDY/'design.json':original.DESIGN})
    design = read(original.STUDY/'design.json')
    parents = read(original.PRIOR/'flop/selected.json')['policies']
    samples, targets, metadata = [], [], {}
    for seed in SEEDS:
        key = str(seed)
        inputs, shares, proof = original.train_rows(seed,design,parents[key])
        if proof['train_ledger_sha256'] != source_candidate['seeds'][key]['train_ledger_sha256']:
            raise ValueError('Source model and its training replay differ')
        samples.extend(inputs)
        targets.extend(shares)
        metadata[key] = proof
    return samples, targets, metadata


def new_replay(group, seed, bounds, count, design_sha, parent_sha, opponent_sha):
    folder = STUDY/'train'/group/f'seed-{seed}'
    ledger = folder/'positions.jsonl.gz'
    proof = read(folder/'collection-complete.json')
    if (not proof['complete'] or proof['design_sha256'] != design_sha or proof['split'] != 'train'
            or proof['group'] != group or proof['seed'] != seed or proof['positions'] != count
            or proof['distinct_original_roots'] != count or proof['parent_policy_sha256'] != parent_sha
            or proof['river_policy_sha256'] != opponent_sha or proof['first_card'] != bounds['start']
            or proof['forced_terminal_actions'] != 2*count):
        raise ValueError('Invalid diverse river training cell')
    verify_files({ledger:proof['position_ledger_sha256']})
    last = bounds['start']-1
    inputs, shares = [], []
    with gzip.open(ledger,'rt') as source:
        for line in source:
            row = json.loads(line)
            state = row['root_observation']
            card = row['deal_seed']
            if (row['split'] != 'train' or row['group'] != group or row['seed'] != seed or not eligible(state)
                    or not last < card < bounds['limit'] or state['seat_to_act'] != row['hero']
                    or row['river_policy_sha256'] != opponent_sha or row['parent_policy_sha256'] != parent_sha):
                raise ValueError('Wrong river training position or card range')
            last = card
            share = row['showdown_share']
            delta = row['call_utility_chips']-row['fold_utility_chips']
            approx_delta = share*(state['pot']+state['amount_owed'])-state['amount_owed']
            if (share not in (0,.5,1) or row['call_minus_fold_chips'] != delta
                    or abs(delta-approx_delta) > (3 if share==.5 else 0)):
                raise ValueError('Showdown label conflicts with terminal settlement')
            inputs.append(features(state))
            shares.append(share)
    if len(inputs) != count or last != proof['last_card']:
        raise ValueError('Incomplete new training ledger')
    return inputs, shares, dict(positions=count, ledger_sha256=proof['position_ledger_sha256'],
                                collection_sha256=digest(folder/'collection-complete.json'))


def stage_models(design, design_sha, selected):
    source = read(original.STUDY/'candidate.json')
    weights = original.STUDY/'models/model.json'
    verify_files({weights:SOURCE_MODEL, **{ref['path']:ref['sha256'] for ref in selected.values()}})
    if source['model_sha256'] != SOURCE_MODEL:
        raise ValueError('Source is not the previously selected river model')
    inputs, shares, replay_info = old_replay(source)
    original_count = len(inputs)
    row_weights = [.5/original_count]*original_count
    fresh = {}
    train = design['collection']['train']
    groups = train['groups']
    for group in groups:
        fresh[group] = {}
        for seed in SEEDS:
            bounds = train['by_group'][group][str(seed)]
            count = train['positions_per_seed_group']
            source_id = design['group_policy_sha256'][group]
            opponent_sha = source_id[str(seed)] if isinstance(source_id,dict) else source_id
            x,y,info = new_replay(group,seed,bounds,count,design_sha,selected[str(seed)]['sha256'],opponent_sha)
            row_weights.extend([.5/(len(groups)*len(SEEDS)*len(x))]*len(x))
            inputs.extend(x)
            shares.extend(y)
            fresh[group][str(seed)] = info
    model, metrics = continue_fit(read(weights),np.asarray(inputs),np.asarray(shares),np.asarray(row_weights))
    final = STUDY/'models'
    with tempfile.TemporaryDirectory(dir=STUDY) as temp:
        staging = Path(temp)
        model_path = staging/'model.json'
        atomic_json(model_path, model)
        model_hash = digest(model_path)
        policy_sha = {}
        for seed in SEEDS:
            parent_ref = selected[str(seed)]
            folder = staging/f'seed-{seed}'
            folder.mkdir()
            path = folder/'policy.json'
            spec = dict(format=FORMAT, seed=seed,
                        baseline=dict(path=os.path.relpath(parent_ref['path'],folder),sha256=parent_ref['sha256']),
                        model=dict(path='../model.json',sha256=model_hash))
            atomic_json(path,spec)
            restored = DiverseRiverCallPolicy(path)
            if restored.model != model or restored.parent.digest != parent_ref['sha256']:
                raise ValueError('Frozen candidate failed to load with its selected parent')
            policy_sha[str(seed)] = restored.digest
        summary = dict(training_complete=True, frozen_utc=utc(), design_sha256=design_sha,
                       model_sha256=model_hash, source_model_sha256=SOURCE_MODEL,
                       source_candidate_sha256=digest(original.STUDY/'candidate.json'),
                       source_policy_sha256={s:v['sha256'] for s,v in selected.items()},
                       policy_sha256=policy_sha, shared_models=1, training_positions=len(shares),
                       original_replay_positions=original_count, fresh_training_positions=len(shares)-original_count,
                       original_replay=replay_info, fresh_replay=fresh, fit_metrics=metrics,
                       code_sha256={p.name:digest(p) for p in (Path(__file__),RUNTIME)},
                       selected_for_play=False)
        atomic_json(staging/'frozen.json', summary)
        os.rename(staging, final)


def fit(design_sha):
    if design_sha != DESIGN:
        raise ValueError('Unexpected continuation design')
    verify_files({STUDY/'design.json':design_sha, REPORT/'design.json':design_sha})
    design = read(STUDY/'design.json')
    release = read(original.STUDY/'selection.json')
    parent_hashes = {s:v['sha256'] for s,v in release['policies'].items()}
    if (release['selected'] != 'candidate' or release['model_sha256'] != SOURCE_MODEL
            or design['parent_model_sha256'] != SOURCE_MODEL or design['parent_policy_sha256'] != parent_hashes):
        raise ValueError('Parent no longer matches the original selected model')
    verify_files({original.STUDY/'play-ready.json':design['parent_play_ready_sha256'],
                  original.RUNTIME:design['parent_runtime_sha256'], original.GUARD:design['parent_guard_sha256'],
                  **{original.STUDY/'train'/f'seed-{s}'/'positions.jsonl.gz':h
                     for s,h in design['parent_train_ledger_sha256'].items()}})
    with (STUDY/'fit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        models = STUDY/'models'
        if not models.exists():
            stage_models(design,design_sha,release['policies'])
        common = read(models/'frozen.json')
        if common['design_sha256'] != design_sha or common['source_model_sha256'] != SOURCE_MODEL:
            raise ValueError('Existing frozen study belongs to another design')
        policies = {str(s): file_ref(models/f'seed-{s}'/'policy.json') for s in SEEDS}
        weights = file_ref(models/'model.json')
        if common['model_sha256'] != weights['sha256'] or common['policy_sha256'] != {s:v['sha256'] for s,v in policies.items()}:
            raise ValueError('Frozen candidate changed after fitting')
        for folder, data in [(REPORT, common), (STUDY, dict(common, policies=policies, model=weights))]:
            path = folder/'candidate.json'
            if path.exists():
                if read(path) != data:
                    raise ValueError('Candidate receipt differs from the frozen model')
            else:
                atomic_json(path, data)
        return common


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--design-sha',default=DESIGN)
    print(json.dumps(fit(cli.parse_args().design_sha)))
