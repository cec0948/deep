"""Post-gate diagnostics: replay-label instability and isolated root capacity.

This is NOT an eligible poker candidate. Root-only fitting removes all other
states solely to distinguish representational capacity from shared fitting.
Random half-splits are a sensitivity diagnostic, not independent confidence
intervals: traversal histories and policy targets are correlated.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from fullhouse_v3_common import ROOT, atomic_json, digest
import validate_frozen_regret as frozen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.folder/'config.json').read_text())
    source = ROOT/cfg['source_run']/'source'
    sys.path.insert(0, str(source))
    from training.network import DuelingAdvantageNet
    from training.buffer import ReservoirBuffer
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    frozen.np = np; frozen.torch = torch
    rng = np.random.default_rng(190177)
    report = dict(complete=False, diagnostic_only=True, script_sha256=digest(Path(__file__)),
                  sensitivity_seed=190177, split_repetitions=100,
                  limitation='Correlated finite replay labels; half splits are not confidence intervals. Root-only models cannot be deployed as poker policies.',
                  label_sensitivity=[], root_capacity=[])
    output = args.folder/'target-diagnostics.json'
    for replay_seed, _ in cfg['frozen_pairs']:
        checkpoint = ROOT/cfg['source_run']/f'seed-{replay_seed}/latest.pt'
        ck = torch.load(checkpoint, weights_only=False, map_location='cpu')
        buffers = []
        for state in ck['buffers']:
            b = ReservoirBuffer(state['count'], 51); b.load_state_dict(state); buffers.append(b)
        keys, target, mass, counts = frozen.root_oracle(buffers)
        xs, ys, ws = [], [], []
        for b in buffers:
            x = b.features[:len(b)]
            mask = (x[:, 17] == 1) & (x[:, 24] == 1) & np.isclose(x[:, 27], .015)
            xs.append(x[mask]); ys.append(b.advantages[:len(b)][mask]); ws.append(b.iterations[:len(b)][mask])
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws).astype(float)
        _, inverse = np.unique(x, axis=0, return_inverse=True)
        irreducible = float(np.average(np.square(y-target[inverse]).mean(1), weights=w))
        zero = float(np.average(np.square(y).mean(1), weights=w))
        splits = []
        for _ in range(100):
            differences, masses = [], []
            for j in range(len(keys)):
                ids = rng.permutation(np.flatnonzero(inverse == j)); cut = len(ids)//2
                if not cut: continue
                a, b = ids[:cut], ids[cut:]
                ya = np.average(y[a], axis=0, weights=w[a])[None]
                yb = np.average(y[b], axis=0, weights=w[b])[None]
                legal = keys[j:j+1, 46:51].astype(float)
                differences.append(float(np.abs(frozen.regret_policy(ya, legal)-frozen.regret_policy(yb, legal)).sum()/2))
                masses.append(mass[j])
            splits.append(float(np.average(differences, weights=masses)))
        row = dict(replay_seed=replay_seed, checkpoint_sha256=digest(checkpoint),
                   root_rows=len(x), classes=len(keys), rows_min=int(counts.min()), rows_median=float(np.median(counts)),
                   rows_max=int(counts.max()), root_mass_fraction=float(w.sum()/sum(b.iterations[:len(b)].sum(dtype=float) for b in buffers)),
                   minimum_attainable_row_mse=irreducible, zero_predictor_row_mse=zero,
                   conditional_signal_mse=zero-irreducible,
                   random_half_target_policy_tv_mean=float(np.mean(splits)),
                   random_half_target_policy_tv_min=float(min(splits)), random_half_target_policy_tv_max=float(max(splits)))
        report['label_sensitivity'].append(row); atomic_json(output, report)
        print(json.dumps(row), flush=True)
        if replay_seed == 42: capacity_data = (keys.copy(), target.copy(), mass.copy())
        del ck, buffers, x, y, w
    keys, target, mass = capacity_data
    rms = float(np.sqrt(np.average(np.square(target).mean(1), weights=mass)))
    x = torch.from_numpy(keys); legal = x[:, 46:51]
    weights = torch.from_numpy((mass/mass.mean()).astype(np.float32))
    target_policy = frozen.regret_policy(target, keys[:, 46:51].astype(float))
    for name, scale in [('raw', 1.), ('one-global-scale', rms)]:
        torch.manual_seed(107); net = DuelingAdvantageNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        y = torch.from_numpy((target/scale).astype(np.float32))
        start = time.monotonic()
        for step in range(1, 8193):
            loss = (weights*(net(x, legal)-y).square().mean(1)).mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); optimizer.step()
            if step not in (2048, 8192): continue
            with torch.no_grad(): prediction = net(x, legal).numpy()*scale
            policy = frozen.regret_policy(prediction, keys[:, 46:51].astype(float))
            tv = np.abs(policy-target_policy).sum(1)/2
            row = dict(mode=name, global_scale=scale, steps=step, seconds=time.monotonic()-start,
                       weighted_tv=float(np.average(tv, weights=mass)),
                       conditional_mse=float(np.average(np.square(prediction-target).mean(1), weights=mass)))
            report['root_capacity'].append(row); atomic_json(output, report)
            print(json.dumps(row), flush=True)
    report['complete'] = True; atomic_json(output, report)


if __name__ == '__main__': main()
