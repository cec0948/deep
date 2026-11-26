"""Train the next first-flop or first-turn critics using only registered training deals."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from controlled_poker_refit import read, utc, verify_files
from fullhouse_v3_common import ROOT, atomic_json, digest
from poker_flop_repair import FORMAT as FLOP_FORMAT, HIDDEN, INPUT_DIM, MAX_UPDATE, MODELS
from poker_flop_repair import FlopRepairPolicy, critic_input, first_flop_decision, improved_distribution
from poker_preflop_specialist import file_ref
from poker_river_runtime import action_key, legal_menu
from poker_street_progress import FORMAT as TURN_FORMAT, StreetProgressPolicy, first_turn_decision


REL = Path('preflop-followup/480-v1/street-progress-20261126')
STUDY = ROOT/'artifacts'/REL
REPORT = ROOT/'reports'/REL
PREVIOUS = STUDY.parent/'flop-repair-20261126'
DESIGN = '5ce6f2b5ffc7f8a3ea82019845c5923ea3af874ba0de409d8e59a331bdbc5700'
PARAMS = ('w1', 'b1', 'w2', 'b2')


def array_hash(arrays):
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(np.ascontiguousarray(array).tobytes())
    return sha.hexdigest()


def load_rows(folder, street, seed, count, raw_sha, base, bounds=None):
    raw = folder/'positions.jsonl.gz'
    verify_files({raw:raw_sha})
    states, features, targets, masks, baselines, heads = [], [], [], [], [], []
    previous_card = -1
    with gzip.open(raw, 'rt') as inp:
        for line in inp:
            row = json.loads(line)
            state = row['root_observation']
            card = row['deal_seed']
            at_root = first_flop_decision(state) if street == 'flop' else first_turn_decision(state)
            if row['seed'] != seed or row['hero'] != state['seat_to_act'] or not at_root or card <= previous_card:
                raise ValueError('Misrouted or repeated training position')
            if bounds is not None and (row.get('split') != 'train' or row.get('street') != street
                                       or not bounds['start'] <= card < bounds['limit']):
                raise ValueError('Training source is outside registered split')
            previous_card = card
            actions = row['legal_actions']
            baseline = np.array(row['parent_probabilities'])
            feature, slots = critic_input(base.policies['postflop'], state, actions, baseline)
            payout = np.asarray(row['action_payoffs_chips'])
            if (payout.shape != (len(actions), 4) or not np.isfinite(payout).all() or
                    np.any(baseline < 0) or len(set(slots)) != len(slots)):
                raise ValueError('Invalid matched action labels or baseline')
            # A value is the change relative to playing the 480 distribution,
            # measured in pot units with a five-BB floor. Identical branches use
            # identical continuation randomness, so common sunk chips cancel.
            q = payout.mean(axis=1)
            target = np.zeros(5)
            target[slots] = (q - q @ baseline)/max(500., state['pot'])
            mask = np.zeros(5, dtype=bool)
            mask[slots] = True
            states.append(state)
            features.append(feature)
            targets.append(target)
            masks.append(mask)
            baselines.append(baseline)
            heads.append(slots)
    if len(states) != count:
        raise ValueError('Incomplete training ledger')
    return states, features, targets, masks, baselines, heads


def fit_weights(street, seed, inputs, targets, masks, warm, config):
    if warm is None:
        center = inputs.mean(axis=0)
        scale = np.maximum(inputs.std(axis=0), .08)
    else:
        center, scale = warm['center'].copy(), warm['scale'].copy()
    x = torch.tensor(np.clip((inputs-center)/scale, -6., 6.), dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    valid = torch.tensor(masks)
    exported = {k:[] for k in PARAMS}
    stats = []
    steps, batch = config['steps_per_critic'], config['batch_size']
    for h in range(MODELS):
        rng = np.random.default_rng([2026112602, seed, h, int(street == 'turn')])
        torch.manual_seed(seed*100 + 30 + h)
        model = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 5))
        tensors = (model[0].weight, model[0].bias, model[2].weight, model[2].bias)
        with torch.no_grad():
            if warm is not None:
                for key, tensor in zip(PARAMS, tensors, strict=True):
                    tensor.copy_(torch.from_numpy(warm[key][h]))
            else:
                model[2].weight.zero_()
                model[2].bias.zero_()
        initial = array_hash(t.detach().numpy() for t in tensors)
        if warm is not None and initial != array_hash(warm[key][h] for key in PARAMS):
            raise ValueError('Restored tensors differ before the first update')
        boot = rng.integers(len(x), size=len(x))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rates'][street], weight_decay=.01)
        for _ in range(steps):
            ix = boot[rng.integers(len(boot), size=batch)]
            mse = (model(x[ix])-y[ix]).square()*valid[ix]
            loss = (mse.sum(1)/valid[ix].sum(1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        terminal = array_hash(t.detach().numpy() for t in tensors)
        if terminal == initial:
            raise ValueError('Fit produced identical model weights')
        for key, tensor in zip(PARAMS, tensors, strict=True):
            exported[key].append(tensor.detach().numpy().copy())
        with torch.no_grad():
            mse = (((model(x)-y).square()*valid).sum(1)/valid.sum(1)).mean()
        stats.append(dict(head=h, initial_weights_sha256=initial, final_weights_sha256=terminal,
                          steps=steps, position_draws=steps*batch, bootstrapped_distinct_positions=len(np.unique(boot)),
                          mean_normalized_training_squared_error=float(mse)))
    return dict(center=center, scale=scale, **{key:np.stack(v) for key,v in exported.items()}), stats


def fit_seed(street, seed, design, choice):
    sid = str(seed)
    source = design['source_policies'][sid]
    old_folder = PREVIOUS/f'seed-{seed}'
    old_manifest = old_folder/'policy.json'
    verify_files({old_manifest: source['manifest_sha256'], old_folder/'critic.npz': source['weights_sha256']})
    old_policy = FlopRepairPolicy(old_manifest)
    old_spec = read(old_manifest)
    parent_ref = file_ref(old_manifest) if street == 'flop' else choice['policies'][sid]
    verify_files({parent_ref['path']:parent_ref['sha256']})
    parent_spec = read(parent_ref['path'])
    if (parent_spec['format'] != FLOP_FORMAT or parent_spec['seed'] != seed or
            parent_spec['base480']['sha256'] != source['base480_sha256']):
        raise ValueError('Different selected source or seed')
    folder = STUDY/street/'train'/f'seed-{seed}'
    receipt = read(folder/'collection-complete.json')
    budget = design['collection'][street]['train']
    expected_reach = source['base480_sha256'] if street == 'flop' else parent_ref['sha256']
    if (not receipt['complete'] or receipt['street'] != street or receipt['split'] != 'train'
            or receipt['repeats'] != 4 or receipt['seed'] != seed or receipt['training_design_sha256'] != DESIGN
            or receipt['reach_flop_policy_sha256'] != expected_reach
            or receipt['parent_sha256'] != source['base480_sha256']
            or receipt['positions'] != budget['positions_per_seed']):
        raise ValueError('Training ledger provenance differs')
    rows = load_rows(folder, street, seed, budget['positions_per_seed'],receipt['position_ledger_sha256'],
                     old_policy.base, budget['by_seed'][sid])
    if sum(np.sum(mask) for mask in rows[3])*4 != receipt['terminal_rollouts']:
        raise ValueError('Terminal rollout count differs')
    if street == 'flop':
        old_rows = load_rows(old_folder, street, seed, 1024, source['original_flop_ledger_sha256'], old_policy.base)
        rows = tuple(prior + new for prior, new in zip(old_rows, rows, strict=True))
    elif receipt['selected_flop_receipt_sha256'] != digest(STUDY/'flop/selected.json'):
        raise ValueError('Turn reach no longer matches frozen flop choice')
    states, features, targets, masks, baselines, head_ids = rows
    inputs = np.asarray(features)
    cfg = design['training']
    warm = old_policy.weights if street == 'flop' else None
    weights, fits = fit_weights(street, seed, inputs, np.asarray(targets), np.asarray(masks), warm, cfg)
    out = STUDY/'models'/street/f'seed-{seed}'
    out.mkdir(parents=True)
    critic = out/'critic.npz'
    with critic.open('xb') as f:
        np.savez_compressed(f, **weights)
    critic_ref = dict(path=critic.name, sha256=digest(critic))
    manifest = out/'policy.json'
    if street == 'flop':
        atomic_json(manifest, dict(format=FLOP_FORMAT, seed=seed, max_update_mass=MAX_UPDATE,
                                  base480=old_spec['base480'], feature_source=old_spec['feature_source'], critic=critic_ref))
    else:
        parent_path = os.path.relpath(parent_ref['path'], out)
        atomic_json(manifest, dict(format=TURN_FORMAT, seed=seed, max_update_mass=MAX_UPDATE,
                                  flop_policy=dict(path=parent_path, sha256=parent_ref['sha256']), turn_critic=critic_ref))
    policy = StreetProgressPolicy(manifest)
    changed = 0
    key = street + '_repair_mass'
    for i, (state, feat, baseline, ids) in enumerate(zip(states, inputs, baselines, head_ids, strict=True)):
        expected, alpha = improved_distribution(feat, ids, baseline, weights)
        reply = policy.decide(state, i)
        replies = {action_key(v['action']):v['probability'] for v in reply['distribution']}
        menu = list(legal_menu(state).values())
        distribution = [replies.get(action_key(action), 0.) for action in menu]
        if (abs(reply.get(key, -1.)-alpha) > 1e-10 or
                not np.allclose(expected, distribution, atol=1e-9, rtol=0) or alpha > MAX_UPDATE or alpha < 0):
            raise ValueError('Restored inference differs from fit on training root')
        changed += np.abs(expected - baseline).sum() > 1e-8
    info = dict(positions=len(inputs), fresh_train_positions=receipt['positions'],
                train_ledger_sha256=receipt['position_ledger_sha256'], action_labels=int(np.sum(masks)),
                matched_training_rollouts=receipt['terminal_rollouts'], fit_heads=fits,
                initial_flop_sha256=parent_ref['sha256'], normalization_sha256=array_hash([weights['center'], weights['scale']]),
                warm_source_v1_weights_sha256=source['weights_sha256'] if warm is not None else None,
                policy_sha256=digest(manifest), critic_sha256=critic_ref['sha256'], changed_training_distributions=int(changed))
    if warm is not None and info['normalization_sha256'] != array_hash([warm['center'],warm['scale']]):
        raise ValueError('Inherited normalization mismatch')
    return file_ref(manifest), info


def fit(street):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify_files({STUDY/'design.json':DESIGN, REPORT/'design.json':DESIGN})
    design = read(STUDY/'design.json')
    choice = None
    if street == 'turn':
        choice = read(STUDY/'flop/selected.json')
        if not choice['complete'] or choice['selected'] not in ('parent','candidate'):
            raise ValueError('No frozen flop choice')
    if read(STUDY/street/'train/collection-status.json')['state'] != 'complete':
        raise ValueError('Training split incomplete')
    with (STUDY/street/'fit.lock').open('x'):
        pass
    policies, stats = {}, {}
    for seed in (342,343,344):
        policies[str(seed)], stats[str(seed)] = fit_seed(street,seed,design,choice)
    cfg = design['training']
    configs = [Path(__file__), ROOT/'scripts/poker_street_progress.py',
               ROOT/'scripts/poker_flop_repair.py',ROOT/'scripts/serve_poker_flop_repair.py']
    common = dict(training_complete=True, frozen_utc=utc(), street=street, seeds=stats,
                  design_sha256=DESIGN, optimizer_updates=3*MODELS*cfg['steps_per_critic'],
                  fit_position_draws=3*MODELS*cfg['steps_per_critic']*cfg['batch_size'],
                  fresh_train_positions=sum(s['fresh_train_positions'] for s in stats.values()),
                  fit_positions=sum(s['positions'] for s in stats.values()),
                  fresh_matched_training_rollouts=sum(s['matched_training_rollouts'] for s in stats.values()),
                  source_flop_choice_sha256=digest(STUDY/'flop/selected.json') if choice else None,
                  code_sha256={p.name:digest(p) for p in configs}, fit_from_training_only=True,
                  selected_for_play=False, whole_hand_matches=0)
    atomic_json(STUDY/street/'candidate.json', dict(common, policies=policies))
    atomic_json(REPORT/street/'candidate.json', common)
    print(street,common['frozen_utc'],common['fresh_train_positions'],common['optimizer_updates'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('street', choices=['flop','turn'])
    args = parser.parse_args()
    fit(args.street)
