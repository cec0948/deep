"""Independent numerical fixtures for fitting and exact game validation."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from regret_fitting import ConditionalRegretSampler


def test_conditional_sampler_preserves_weighted_targets_and_stratum_mass():
    # One duplicate information set has unequal time weights. Another stratum
    # has 100 times as much original mass. Each stratum gets half the objective.
    class Buffer:
        features = np.zeros((3, 51), np.float32)
        features[2, 0] = 1
        features[:, 46:49] = 1
        advantages = np.array([[2, 1, -.375, 0, 0], [6, -3, 1.625, 0, 0],
                               [-2, 2, 0, 0, 0]], np.float32)
        iterations = np.array([1, 3, 400], np.float32)
        def __len__(self): return 3
    sampler = ConditionalRegretSampler([Buffer()], lambda x: x[:, 0].astype(int))
    np.testing.assert_allclose(sampler.probabilities, [.5, .5])
    np.testing.assert_allclose(sampler.targets, [[5, -2, 1.125, 0, 0], [-2, 2, 0, 0, 0]])
    # The conditional expected squared-loss gradient at a fixed prediction is
    # the same as the independently hand-weighted original three-row objective.
    prediction = np.array([.5, -.5, .25, 0, 0])
    dense_gradient = .5*(prediction-Buffer.advantages[0])/4 + .5*3*(prediction-Buffer.advantages[1])/4 + .5*(prediction-Buffer.advantages[2])
    grouped_gradient = np.sum(sampler.probabilities[:, None]*(prediction-sampler.targets), axis=0)
    np.testing.assert_allclose(grouped_gradient, dense_gradient)
    np.random.seed(719)
    state = np.random.get_state()
    a = sampler.sample(100)
    np.random.set_state(state)
    b = sampler.sample(100)
    for first, second in zip(a, b): np.testing.assert_array_equal(first, second)
    assert sampler.diagnostics()['draws'] == 200


def test_kuhn_best_response_oracle_matches_known_equilibrium():
    from kuhn_validation import exact_metrics, equilibrium, policy_value
    profile = equilibrium()
    metrics = exact_metrics(profile)
    assert abs(metrics['value_player0']+1/18) < 1e-12
    assert abs(metrics['exploitability']) < 1e-12
    assert abs(metrics['best_response_values'][0]+1/18) < 1e-12
    assert abs(metrics['best_response_values'][1]-1/18) < 1e-12
    # Gate comparisons must produce JSON-safe native bools, including after a
    # long run. NumPy bools caused a final-report write failure in the first run.
    import json
    json.dumps({'passed': metrics['exploitability'] < .05}, allow_nan=False)
    # Fold every time when facing a bet: this is exploitable even though all
    # probabilities are finite, legal and normalized.
    profile[6:12] = [1., 0.]
    assert exact_metrics(profile)['exploitability'] > .1
    # The uniform value can be evaluated by enumerating the six card deals.
    assert abs(policy_value(np.full((12, 2), .5))-.125) < 1e-12


def test_tabular_cfr_reduces_exact_exploitability():
    from kuhn_validation import tabular_reference
    result = tabular_reference(1000)
    assert result['exploitability'] < .01
    assert abs(result['value_player0']+1/18) < .005


def test_actual_external_traversal_matches_enumerated_kuhn_regrets():
    import os
    import subprocess
    code = "import sys;sys.path[:0]=['scripts','artifacts/fullhouse-v3/20261109T034714104421Z/source'];from kuhn_validation import traversal_oracle;print(traversal_oracle())"
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'), timeout=60)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr


def test_learning_gate_rejects_missing_or_failed_evidence():
    import copy
    import json
    from check_learning_gates import assess
    cfg = json.loads((ROOT/'configs/fullhouse-learning-validation.json').read_text())
    assert not assess(cfg, {}, {})['six_player_ready']
    # Passing data-fitting evidence alone cannot authorize six-player work.
    good_metrics = dict(root=dict(weighted_tv=.01, aa=dict(policy=[0., 0., 1., 0., 0.], target_policy=[0., 0., 1., 0., 0.])),
                        global_fit=dict(mse=1.), streets=[dict(mse=1.)]*4)
    frozen = {}
    for replay_seed, fit_seed in cfg['frozen_pairs']:
        frozen[replay_seed] = dict(complete=True, fit_seed=fit_seed,
            refits=[dict(mode='control',steps=512,metrics=copy.deepcopy(good_metrics))]
                   +[dict(mode=mode,steps=8192,metrics=copy.deepcopy(good_metrics)) for mode in cfg['candidate_modes']])
    verdict = assess(cfg, frozen, {})
    assert all(v['passed'] for v in verdict['methods']['budget']['frozen'])
    assert not verdict['six_player_ready']
    good_kuhn = dict(complete=True, verification={'checkpoint_verified': True},
                     tabular_reference={'exploitability': .001}, initial={'exploitability': .4},
                     curve=[dict(neural_average={'exploitability': .01}, collector_max_tv=.01,
                                 network_max_tv=.01, all_infosets_observed=True)])
    kuhn = {(mode, seed):copy.deepcopy(good_kuhn) for mode in cfg['candidate_modes'] for seed in cfg['kuhn']['seeds']}
    assert assess(cfg, frozen, kuhn)['six_player_ready']
    # Passing frozen evidence for one method and small-game evidence for the
    # other cannot be combined into an approval.
    frozen[42]['refits'][2]['metrics']['root']['aa']['policy'][0] = .25
    kuhn.pop(('budget', 71))
    assert not assess(cfg, frozen, kuhn)['six_player_ready']
    frozen[42]['refits'][1]['metrics']['root']['aa']['policy'][0] = .25
    assert not assess(cfg, frozen, {})['methods']['budget']['frozen'][0]['passed']
