"""Independent known-game and averaging checks for the benchmark adapter."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
pyspiel = pytest.importorskip('pyspiel')
from benchmark_leduc_reference import (ExactPolicySpace, normalized_legal, check_policy,
                                       current_policy, fit_average_for_evaluation)
from open_spiel.python.algorithms import expected_game_score


def test_exact_best_response_recovers_known_kuhn_equilibrium():
    space = ExactPolicySpace(pyspiel.load_game('kuhn_poker'))
    values = np.zeros_like(space.legal)
    for key, row in space.template.state_lookup.items():
        card, history = int(key[0]), key[1:]
        if history in ('', 'p'):
            values[row] = [(2/3, 1/3), (1, 0), (0, 1)][card]
        elif history == 'b':
            values[row] = [(1, 0), (2/3, 1/3), (0, 1)][card]
        elif history == 'pb':
            values[row] = [(1, 0), (1/3, 2/3), (0, 1)][card]
        else:
            raise AssertionError(key)
    result = space.metrics(values)
    assert result['exploitability'] == pytest.approx(0, abs=1e-12)
    assert result['policy_value'][0] == pytest.approx(-1/18, abs=1e-12)


def test_own_reach_average_matches_independent_per_hand_policy_mixture():
    game = pyspiel.load_game('kuhn_poker'); space = ExactPolicySpace(game)
    rng = np.random.default_rng(20261112)
    profiles = [normalized_legal(rng.random(space.legal.shape), space.legal) for _ in range(2)]
    weights = np.array([1., 3.])
    for values, weight in zip(profiles, weights):
        space.accumulate(values, weight)
    average = space.as_policy(space.average())
    realized = expected_game_score.policy_value(game.new_initial_state(), [average, average])
    expected = np.zeros(2)
    for first, w0 in zip(profiles, weights / weights.sum()):
        for second, w1 in zip(profiles, weights / weights.sum()):
            expected += w0 * w1 * expected_game_score.policy_value(
                game.new_initial_state(), [space.as_policy(first), space.as_policy(second)])
    np.testing.assert_allclose(realized, expected, atol=1e-12)


def test_legal_policy_adapter_removes_illegal_mass_and_rejects_nan():
    legal = np.array([[1., 1., 0.], [0., 1., 1.]])
    values = normalized_legal(np.array([[.1, .2, .7], [.8, .1, .1]]), legal)
    np.testing.assert_allclose(values, [[1/3, 2/3, 0], [0, .5, .5]])
    check_policy(values, legal)
    with pytest.raises(ValueError):
        normalized_legal(np.array([[float('nan'), 0, 1], [0, 1, 0]]), legal)


def test_full_leduc_information_states_are_complete_and_distinct():
    space = ExactPolicySpace(pyspiel.load_game('leduc_poker'))
    assert len(space.keys) == 936
    assert space.inputs.shape == (936, 30)
    assert len(np.unique(space.inputs, axis=0)) == 936
    assert all(path is not None for path in space.own_paths)
    # Serializing a candidate profile must not alter the uniform template.
    original = space.template.action_probability_array.copy()
    values = normalized_legal(np.random.default_rng(71).random(space.legal.shape), space.legal)
    space.as_policy(values)
    np.testing.assert_array_equal(space.template.action_probability_array, original)


def test_checkpoint_fit_does_not_change_official_advantage_learning_trajectory():
    import torch
    from open_spiel.python.pytorch import deep_cfr
    torch.set_num_threads(1)
    game = pyspiel.load_game('kuhn_poker'); space = ExactPolicySpace(game)
    def make():
        return deep_cfr.DeepCFRSolver(game, policy_network_layers=(8,),
            advantage_network_layers=(8,), num_iterations=2, num_traversals=16,
            batch_size_advantage=8, batch_size_strategy=8, memory_capacity=128,
            advantage_network_train_steps=4, policy_network_train_steps=8,
            learning_rate=.001, seed=71)
    reference = make(); reference.solve()
    observed = make()
    for iteration in (1, 2):
        for player in range(2):
            for _ in range(16):
                observed._traverse_game_tree(observed._root_node, player)
            observed._reinitialize_advantage_network(player)
            observed._learn_advantage_network(player)
            if player == 0:
                space.accumulate(current_policy(observed, space), iteration)
        observed._iteration += 1
        numpy_before = np.random.get_state(); torch_before = torch.get_rng_state()
        fitted, _, _ = fit_average_for_evaluation(observed, space)
        check_policy(fitted, space.legal)
        numpy_after = np.random.get_state()
        assert numpy_before[0] == numpy_after[0]
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        assert numpy_before[2:] == numpy_after[2:]
        assert torch.equal(torch_before, torch.get_rng_state())
    for first, second in zip(reference._advantage_networks, observed._advantage_networks):
        for key, value in first.state_dict().items():
            assert torch.equal(value, second.state_dict()[key])
    for first, second in zip(reference.advantage_buffers, observed.advantage_buffers):
        assert first.add_calls == second.add_calls
        for key in first.experience._fields:
            np.testing.assert_array_equal(getattr(first.experience, key)[:len(first)],
                                          getattr(second.experience, key)[:len(second)])
