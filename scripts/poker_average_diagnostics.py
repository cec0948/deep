"""Frozen-data diagnosis and isolated fitting of the bounded window average.

Input normalization changes only conditioning. It is folded into the exported
first affine layer, preserving the existing NumPy runtime and observation schema.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from poker_strategy_average import AverageNet, export_average
from fullhouse_v3_common import atomic_json, digest
from controlled_poker_refit import utc


def conditional_targets(replay):
    n = replay.size
    if not n:
        raise ValueError('Empty strategy replay')
    x, inverse = np.unique(replay.x[:n], axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=replay.mass[:n], minlength=len(x))
    y = np.stack([np.bincount(inverse,
        weights=replay.mass[:n] * replay.y[:n, a], minlength=len(x))
        for a in range(5)], 1) / mass[:, None]
    if (not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(mass).all() or np.any(mass <= 0)
            or not np.allclose(y.sum(1), 1, atol=1e-6)):
        raise ValueError('Invalid conditional strategy targets')
    return x, y, mass


def metrics(net, x, y, mass):
    errors, divergences = [], []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            p = net(torch.from_numpy(x[start:start+4096])).exp().numpy().astype(np.float64)
            target = y[start:start+4096]
            if not np.isfinite(p).all():
                raise FloatingPointError('Nonfinite average prediction')
            errors.extend(np.abs(p-target).sum(1)/2)
            divergences.extend(np.sum(np.where(target > 0, target * np.log(
                np.maximum(target, 1e-30)/np.maximum(p, 1e-30)), 0), axis=1))
    return dict(distinct_inputs=len(x), mass=float(mass.sum()),
        weighted_tv=float(np.average(errors, weights=mass)),
        weighted_kl=float(np.average(divergences, weights=mass)))


def fold_normalization(net, mean, scale):
    result = copy.deepcopy(net)
    with torch.no_grad():
        weight = result.body[0].weight
        folded = weight / torch.from_numpy(scale)[None, :]
        result.body[0].bias.sub_(folded @ torch.from_numpy(mean))
        weight.copy_(folded)
    result.eval()
    return result


def fit_conditional(x, y, mass, *, seed, steps=16384, batch_size=512,
                    width=128, normalize=False, checkpoints=(), callback=None):
    if (steps not in (8192, 16384) or batch_size != 512 or width not in (128, 256)
            or not len(x) or x.shape[1:] != (51,) or y.shape != (len(x), 5)):
        raise ValueError('Unsupported diagnostic fitting budget')
    mean = np.zeros(51, np.float32)
    scale = np.ones(51, np.float32)
    if normalize:
        mean[:46] = np.average(x[:, :46], axis=0, weights=mass)
        scale[:46] = np.maximum(np.sqrt(np.average(
            (x[:, :46]-mean[:46])**2, axis=0, weights=mass)), .025)
    tx = torch.from_numpy(((x-mean)/scale).astype(np.float32))
    ty = torch.from_numpy(y.astype(np.float32))
    cdf = np.cumsum(mass/mass.sum()); cdf[-1] = 1
    rng = np.random.default_rng(seed)
    began = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = AverageNet(width)
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        for step in range(1, steps+1):
            ids = np.searchsorted(cdf, rng.random(batch_size), side='right')
            loss = -(ty[ids]*net(tx[ids])).sum(1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite average fitting loss')
            optimizer.zero_grad(); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite average gradient')
            optimizer.step()
            if step in checkpoints and callback:
                callback(step, fold_normalization(net, mean, scale))
        folded = fold_normalization(net, mean, scale)
        with torch.no_grad():
            ids = np.linspace(0, len(x)-1, min(2048, len(x)), dtype=int)
            before = net(tx[ids]).exp().numpy()
            after = folded(torch.from_numpy(x[ids])).exp().numpy()
            error = float(np.max(np.abs(before-after)))
        if error > 2e-5:
            raise ValueError('Input normalization export parity failed')
    report = dict(width=width, normalize=normalize, steps=steps, batch_size=batch_size,
        parameters=sum(p.numel() for p in folded.parameters()),
        normalization_fold_max_error=error, elapsed_seconds=time.monotonic()-began,
        input_mean=mean.tolist(), input_scale=scale.tolist(),
        **metrics(folded, x, y, mass))
    return folded, report


def fit_average(replay, *, seed, steps=16384, batch_size=512, width=128, normalize=False):
    x, y, mass = conditional_targets(replay)
    net, report = fit_conditional(x, y, mass, seed=seed, steps=steps,
        batch_size=batch_size, width=width, normalize=normalize)
    report.update(rows=replay.size, total_seen=replay.total_seen,
        weighted_replay_tv=report['weighted_tv'], weighted_replay_kl=report['weighted_kl'],
        objective='Unchanged retained-replay mass weighted soft-label cross entropy',
        scope='Average approximation diagnostic; not poker strength',
        within_original_quality_reference=report['weighted_tv'] <= .15)
    return net, report


def diagnose(input_path, output, seed, width, normalize):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with np.load(input_path, allow_pickle=False) as data:
        x, y, mass, train = (data[k] for k in ('x', 'y', 'mass', 'train'))
    rows = []
    def checkpoint(step, net):
        row = dict(step=step, training=metrics(net, x[train], y[train], mass[train]),
            heldout=metrics(net, x[~train], y[~train], mass[~train]))
        rows.append(row); atomic_json(output/'progress.json', dict(observed_utc=utc(), checkpoints=rows))
    net, report = fit_conditional(x[train], y[train], mass[train], seed=seed,
        width=width, normalize=normalize, checkpoints=(8192, 16384), callback=checkpoint)
    export_average(net, output/'average.npz', scope='Frozen-data approximation diagnosis only')
    result = dict(complete=True, observed_utc=utc(), input_sha256=digest(input_path),
        seed=seed, width=width, normalize=normalize, fit=report, checkpoints=rows,
        new_selfplay_iterations=0, poker_strength_evaluated=False)
    atomic_json(output/'result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--width', type=int, required=True)
    p.add_argument('--normalize', action='store_true'); a = p.parse_args()
    result = diagnose(a.input, a.output, a.seed, a.width, a.normalize)
    print(json.dumps({k: result[k] for k in ('complete', 'width', 'normalize', 'checkpoints')}))
