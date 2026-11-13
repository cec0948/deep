from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_neural_information_gate import (GlobalTargetUnits, NeuralBudget, SharedRegretNet,
    neural_iteration, run)
from validate_strength_training import (Replay, adapter_strata, component_checks,
    exact_iteration, game_and_infosets, kuhn_features, regret_match)
from scaled_regret_fitting import ScaledRegretSampler


def test_network_is_shared_function_and_masks_illegal_actions():
    torch.manual_seed(7)
    net = SharedRegretNet(hidden=16)
    x = torch.randn(4, 51)
    mask = torch.tensor([[1.,1.,0.,0.,0.]]).expand(4,-1)
    first = net(x, mask)
    assert first.shape == (4, 5)
    assert torch.equal(first[:,2:], torch.zeros(4,3))
    assert torch.allclose(net(x[[2,0]], mask[[2,0]]), first[[2,0]], rtol=1e-6, atol=1e-7)
    # No feature-ID table; a previously unseen continuous observation is legal.
    assert torch.isfinite(net(x[:1]+.123, mask[:1])).all()


def test_global_target_units_preserve_sampler_law_and_regret_strategy():
    game, keys, infosets = game_and_infosets()
    x = kuhn_features(keys, infosets)
    _, current = component_checks(game, keys, infosets, x)
    regrets, _, _ = exact_iteration(game, current, infosets)
    sampler = ScaledRegretSampler([Replay(x,np.stack([regrets[k] for k in keys]).astype(np.float32),
        np.ones(len(keys),np.float32))], adapter_strata, 'stratum-scaled-feature-conditional')
    old = sampler.probabilities.copy()
    scaled = GlobalTargetUnits(sampler)
    assert 0 < scaled.scale < 1
    assert np.array_equal(old, sampler.probabilities)
    for row, target in zip(sampler.features, sampler.targets):
        assert regret_match(target,row[46:51]) == pytest.approx(
            regret_match(target/scaled.scale,row[46:51]))


def test_actual_sampler_and_fitter_train_neural_regret_targets():
    torch.set_num_threads(1)
    np.random.seed(71)
    game, keys, infosets = game_and_infosets()
    features = kuhn_features(keys, infosets)
    _, policy = component_checks(game, keys, infosets, features)
    regrets, _, _ = exact_iteration(game, policy, infosets)
    sampler = ScaledRegretSampler([Replay(features,np.stack([regrets[k] for k in keys]).astype(np.float32),
        np.ones(len(keys),np.float32))], adapter_strata, 'stratum-scaled-feature-conditional')
    model, prediction, fit = neural_iteration(features, sampler, 7123,
        NeuralBudget(fit_steps=128,hidden=32))
    assert isinstance(model,SharedRegretNet)
    assert fit['full_replay_normalized_weighted_mse'] < .05
    assert np.isfinite(prediction).all()
    assert np.all(prediction[features[:,46:51]==0] == 0)


@pytest.mark.parametrize('value',[replace(NeuralBudget(),iterations=0),
    replace(NeuralBudget(),fit_steps=513),replace(NeuralBudget(),max_seconds=601),
    replace(NeuralBudget(),learning_rate=1),replace(NeuralBudget(),seeds=(71,))])
def test_fixed_compute_bounds_rejected(value):
    with pytest.raises(ValueError):
        value.validate()


def test_bounded_neural_run_records_training_and_never_claims_six_player_strength(tmp_path):
    output = tmp_path/'neural'
    report = run(output, NeuralBudget(iterations=2,fit_steps=16,hidden=16,max_seconds=30), 'kuhn_poker')
    assert report['status']=='complete'
    assert report['shared_neural_training_executed']
    assert report['six_player_policy_trained'] is False
    assert report['production_six_player_path_validated'] is False
    assert report['automatic_retry'] is False
    assert len(report['seed_results'])==3
    assert all(r['shared_neural_approximator'] and not r['tabular_approximator_used']
               for r in report['seed_results'])
    assert (output/'plan.json').exists() and len(list(output.glob('*-neural.npz')))==3
    before = (output/'report.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(output,NeuralBudget(iterations=2,fit_steps=16,hidden=16,max_seconds=30),'kuhn_poker')
    assert (output/'report.json').read_bytes()==before
