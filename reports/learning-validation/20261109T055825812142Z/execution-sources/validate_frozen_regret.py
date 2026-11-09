"""Predeclared in-sample fitting gates on three immutable V3 replay buffers."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from fullhouse_v3_common import ROOT, atomic_json, digest


def regret_policy(values, legal):
    positive = np.maximum(values, 0)*legal
    total = positive.sum(axis=1, keepdims=True)
    return np.divide(positive, total, out=legal/legal.sum(axis=1, keepdims=True), where=total > 0)


def root_oracle(buffers):
    xs, ys, ws = [], [], []
    for b in buffers:
        x = b.features[:len(b)]
        root = (x[:, 17] == 1) & (x[:, 24] == 1) & np.isclose(x[:, 27], .015)
        xs.append(x[root]); ys.append(b.advantages[:len(b)][root]); ws.append(b.iterations[:len(b)][root])
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws).astype(float)
    keys, inv = np.unique(x, axis=0, return_inverse=True)
    mass = np.bincount(inv, weights=w, minlength=len(keys))
    counts = np.bincount(inv, minlength=len(keys))
    targets = np.stack([np.bincount(inv, weights=w*y[:, a], minlength=len(keys)) for a in range(5)], 1)/mass[:, None]
    return keys, targets, mass, counts


def measure(net, buffers, root):
    sums = np.zeros((5, 3))
    with torch.no_grad():
        for b in buffers:
            for start in range(0, len(b), 4096):
                end = min(start + 4096, len(b))
                x = torch.from_numpy(b.features[start:end]); y = b.advantages[start:end]
                pred = net(x, x[:, 46:51]).numpy()
                if not np.isfinite(pred).all(): raise FloatingPointError('Nonfinite prediction')
                w = b.iterations[start:end].astype(float)
                mse = np.square(pred.astype(float)-y).mean(1)
                zero = np.square(y.astype(float)).mean(1)
                street = np.argmax(x[:, 17:21].numpy(), 1)
                for k, mask in [(0, np.ones(len(x), bool))]+[(s+1, street == s) for s in range(4)]:
                    sums[k] += [w[mask].sum(), (w[mask]*mse[mask]).sum(), (w[mask]*zero[mask]).sum()]
        keys, target, mass, counts = root
        x = torch.from_numpy(keys)
        pred = net(x, x[:, 46:51]).numpy()
    expected = regret_policy(target, keys[:, 46:51].astype(float))
    actual = regret_policy(pred, keys[:, 46:51].astype(float))
    tv = np.abs(actual-expected).sum(1)/2
    aa = np.flatnonzero((keys[:, 10] == 1) & (keys[:, 7] > .99))
    if len(aa) != 1: raise ValueError('Expected exactly one UTG AA input class')
    i = int(aa[0])
    metrics = [dict(mse=float(row[1]/row[0]), zero_mse=float(row[2]/row[0])) for row in sums]
    return dict(global_fit=metrics[0], streets=metrics[1:],
                root=dict(classes=len(keys), weighted_tv=float(np.average(tv, weights=mass)),
                          weighted_mse=float(np.average(np.square(pred-target).mean(1), weights=mass)),
                          aa=dict(rows=int(counts[i]), target=target[i].tolist(), prediction=pred[i].tolist(),
                                  target_policy=expected[i].tolist(), policy=actual[i].tolist())),
                root_table=[dict(features=keys[j].tolist(), rows=int(counts[j]), mass=float(mass[j]),
                                 target=target[j].tolist(), prediction=pred[j].tolist(), tv=float(tv[j]))
                            for j in range(len(keys))])


def gate(result, baseline, limits):
    fit = result['metrics']; control = baseline['metrics']
    checks = dict(root_tv=fit['root']['weighted_tv'] <= limits['max_root_weighted_tv'],
                  aa_target_consistent=fit['root']['aa']['target_policy'][0] == 0,
                  aa_fold=fit['root']['aa']['policy'][0] <= limits['max_aa_fold'],
                  global_mse=fit['global_fit']['mse'] <= control['global_fit']['mse']*limits['max_global_mse_ratio_to_control'],
                  all_streets=all(a['mse'] <= b['mse']*limits['max_street_mse_ratio_to_control']
                                  for a, b in zip(fit['streets'], control['streets'])))
    return dict(passed=all(checks.values()), checks=checks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=ROOT/'configs/fullhouse-learning-validation.json')
    p.add_argument('--replay-seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    pair = [p for p in cfg['frozen_pairs'] if p[0] == args.replay_seed]
    if len(pair) != 1: p.error('Replay seed must occur once in the declared pairs')
    _, fit_seed = pair[0]
    source = ROOT/cfg['source_run']/'source'
    sys.path.insert(0, str(source))
    global np, torch
    import numpy as np
    import torch
    from training.network import DuelingAdvantageNet
    from training.buffer import ReservoirBuffer
    from training.train import train_network, _export_advantage_npz
    from training import r1_support as r1
    from regret_fitting import ConditionalRegretSampler, train_conditional
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT/cfg['source_run']/f'seed-{args.replay_seed}/latest.pt'
    ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
    if ck['iteration'] != 120 or ck['r1_config']['seed'] != args.replay_seed:
        raise ValueError('Wrong replay checkpoint')
    r1.WEIGHT_HORIZON = ck['r1_config']['horizon']
    buffers = []
    for state in ck['buffers']:
        b = ReservoirBuffer(state['count'], 51); b.load_state_dict(state); buffers.append(b)
    root = root_oracle(buffers)
    original = DuelingAdvantageNet(); original.load_state_dict(ck['net']); original.eval()
    report = dict(complete=False, replay_seed=args.replay_seed, fit_seed=fit_seed,
                  checkpoint_sha256=digest(checkpoint), config=cfg, config_sha256=digest(args.config),
                  source_hashes={str(f.relative_to(source)):digest(f) for f in sorted(source.rglob('*.py'))},
                  script_sha256=digest(Path(__file__)), fitter_sha256=digest(ROOT/'scripts/regret_fitting.py'),
                  original=measure(original, buffers, root), refits=[])
    del ck, original
    start_time = time.monotonic()
    def save(): atomic_json(args.output/'results.json', report)
    save()
    for mode in ['control']+cfg['candidate_modes']:
        np.random.seed(fit_seed); torch.manual_seed(fit_seed)
        net = DuelingAdvantageNet(); opt = torch.optim.Adam(net.parameters(), lr=.001)
        sampler = ConditionalRegretSampler(buffers, r1.strategy_strata) if mode == 'stratified-conditional' else None
        milestones = [cfg['baseline_steps']] if mode == 'control' else cfg['candidate_steps']
        batch = cfg['baseline_batch'] if mode == 'control' else cfg['candidate_batch']
        previous = 0; mode_start = time.monotonic()
        for steps in milestones:
            if time.monotonic()-start_time > cfg['limits']['seconds_per_process']:
                raise TimeoutError('Frozen-fit experiment time budget')
            if sampler:
                loss = train_conditional(net, sampler, opt, steps-previous, batch)
            else:
                loss = train_network(net, buffers, opt, steps-previous, batch, torch.device('cpu'))
            metrics = measure(net, buffers, root)
            model = args.output/f'{mode}-{steps}.npz'; _export_advantage_npz(net, model)
            row = dict(mode=mode, steps=steps, batch=batch, seconds=time.monotonic()-mode_start,
                       loss=loss, metrics=metrics, model=str(model.relative_to(ROOT)), model_sha256=digest(model))
            if sampler: row['sampling'] = sampler.diagnostics()
            if mode == 'control': baseline = row
            else: row['gate'] = gate(row, baseline, cfg['fixed_buffer_gates'])
            report['refits'].append(row); save()
            print(json.dumps({k:v for k,v in row.items() if k != 'metrics'} | {'root':metrics['root'], 'global_fit':metrics['global_fit']}), flush=True)
            previous = steps
    report['complete'] = True
    report['elapsed_seconds'] = time.monotonic()-start_time
    save()


if __name__ == '__main__': main()
