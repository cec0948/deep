import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from poker_average_diagnostics import conditional_targets, fold_normalization
from poker_strategy_average import AverageNet, StrategyReplay, export_average
from poker_average_inference import load_average, average_strategy


def test_normalization_fold_preserves_probabilities_and_numpy_export(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(41)
    x = rng.normal(size=(100, 51)).astype(np.float32)
    x[:, 46:] = rng.integers(0, 2, size=(100, 5)); x[:, 47] = 1
    mean = rng.normal(size=51).astype(np.float32)
    scale = np.exp(rng.normal(size=51)).astype(np.float32)
    mean[46:] = 0; scale[46:] = 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13); net = AverageNet(256)
    folded = fold_normalization(net, mean, scale)
    with torch.no_grad():
        expected = net(torch.from_numpy((x-mean)/scale)).exp().numpy()
        actual = folded(torch.from_numpy(x)).exp().numpy()
    assert np.max(np.abs(actual-expected)) < 2e-6
    path = tmp_path/'average.npz'; export_average(folded, path)
    layers, _ = load_average(path)
    numpy_result = np.stack([average_strategy(layers, r, r[46:]) for r in x])
    assert np.max(np.abs(numpy_result-expected)) < 2e-6
    assert np.all(numpy_result[x[:, 46:] == 0] == 0)


def test_grouped_targets_preserve_weighted_objective_and_no_duplicate_split():
    replay = StrategyReplay(20)
    x = np.zeros(51, np.float32); x[46:48] = 1
    replay.add(x, [1, 0, 0, 0, 0], 1)
    replay.add(x, [0, 1, 0, 0, 0], 9)
    other = x.copy(); other[0] = .5
    replay.add(other, [.25, .75, 0, 0, 0], 2)
    grouped_x, y, mass = conditional_targets(replay)
    assert len(grouped_x) == 2
    assert np.allclose(y[0], [.1, .9, 0, 0, 0])
    assert np.allclose(mass, [10, 2])
    logp = np.log([.3, .7])
    grouped = -(mass[:, None]*y[:, :2]*logp).sum()/mass.sum()
    original = -(replay.mass[:3, None]*replay.y[:3, :2]*logp).sum()/replay.mass[:3].sum()
    assert abs(grouped-original) < 1e-12
