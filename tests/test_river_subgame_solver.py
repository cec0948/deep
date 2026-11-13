"""Mathematical and chip-accounting checks for the independent river solver."""
from pathlib import Path
import sys

import eval7
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from river_subgame_solver import RiverGame, solve_river, terminal_value


BOARD = ('2c', '3d', '7h', '9s', 'Jc')


def test_exact_bluffing_game_value_and_best_response_gap():
    # Independent analytic oracle: value100/3, nuts always bet100, air
    # bets1/3. The bluffcatcher calls2/3. This is a whole-range decision;
    # solving each known hand as a separate perfect-information game fails it.
    game = RiverGame(BOARD, 200, (100, 100), bet_fractions=(.5,), max_raises=1)
    result = solve_river(game, [('As', 'Ad'), ('4s', '5s')], [.5, .5], [('Ks', 'Kd')], [1], iterations=10000)
    assert result.expected_values[0] == pytest.approx(100/3, abs=.15)
    assert sum(result.expected_values) == pytest.approx(0, abs=1e-9)
    assert result.probabilities[0, 1] > .99
    assert result.probabilities[1, 1] == pytest.approx(1/3, abs=.025)
    assert result.exploitability < .5
    assert result.best_response_values[0] >= result.expected_values[0]-1e-8
    assert result.best_response_values[1] >= result.expected_values[1]-1e-8


def test_joint_chance_excludes_collisions_without_conditioning_own_range():
    game = RiverGame(BOARD, 200, (100, 100), max_raises=0)
    own = [('Ac', 'Ad'), ('4h', '5h')]
    other = [('Ac', '2h'), ('Kd', 'Kc')]
    own_weights, other_weights = [.8, .2], [.25, .75]
    result = solve_river(game, own, own_weights, other, other_weights, iterations=0)
    numerator = mass = 0.
    board = [eval7.Card(c) for c in BOARD]
    for h, wh in zip(own, own_weights):
        for k, wk in zip(other, other_weights):
            if set(h).intersection(k):
                continue
            r0 = eval7.evaluate(board+[eval7.Card(c) for c in h])
            r1 = eval7.evaluate(board+[eval7.Card(c) for c in k])
            numerator += wh*wk*100*((r0 > r1)-(r0 < r1)); mass += wh*wk
    assert result.compatible_pairs == 3
    assert result.chance_normalizer == pytest.approx(.8)
    assert result.expected_values[0] == pytest.approx(numerator/mass)
    assert len(result.hands) == 2


def test_facing_observed_bet_call_and_fold_values_include_sunk_pot():
    game = RiverGame(BOARD, 300, (100, 0), street_bets=(0, 100), to_act=0, max_raises=0)
    result = solve_river(game, [('As', 'Ad')], [1], [('Ks', 'Kd')], [1], iterations=100)
    assert [a.kind for a in result.actions] == ['fold', 'call']
    assert result.action_values[0].tolist() == pytest.approx([-150, 150])
    assert result.probabilities[0, 1] > .99


def test_tied_board_call_is_better_than_fold_and_not_zero_continuation():
    board = ('As', 'Ks', 'Qs', 'Js', 'Ts')
    game = RiverGame(board, 300, (100, 0), street_bets=(0, 100), max_raises=0)
    result = solve_river(game, [('2c', '3c')], [1], [('4d', '5d')], [1], iterations=100)
    assert result.action_values[0].tolist() == pytest.approx([-150, -50])
    assert result.expected_values[0] == pytest.approx(-50, abs=.05)


def test_fold_refunds_uncalled_new_bet_implicitly_and_utilities_are_zero_sum():
    game = RiverGame(BOARD, 200, (1000, 1000))
    # P0 bets400 and P1 folds: P0 gains only the existing pot, not400
    # additional imaginary chips; centering subtracts100.
    assert terminal_value(game, (400, 0), 1, 0) == 100
    # If P1 previously put100 in and then folds, that100 is won as well.
    assert terminal_value(game, (400, 100), 1, 0) == 200
    assert terminal_value(game, (400, 400), None, 1) == 500
    assert terminal_value(game, (400, 400), None, -1) == -500


def test_effective_stack_caps_unmatched_future_all_in():
    game = RiverGame(BOARD, 200, (1000, 150), bet_fractions=(.5, 1.0))
    result = solve_river(game, [('As', 'Ad')], [1], [('Ks', 'Kd')], [1], iterations=2)
    assert all(a.to is None or a.to <= 150 for a in result.actions)
    assert {a.to for a in result.actions if a.kind == 'raise'} == {100, 150}


def test_root_player_one_and_prior_check_are_supported():
    game = RiverGame(BOARD, 200, (100, 100), to_act=1, max_raises=0, checks=1)
    result = solve_river(game, [('As', 'Ad')], [1], [('Ks', 'Kd')], [1], iterations=1)
    assert result.expected_values == pytest.approx((100, -100))
    assert result.distribution(('Kd', 'Ks')) == [{'action': {'kind': 'check'}, 'probability': 1.0}]
    assert result.terminal_nodes == 1


def test_zero_time_returns_valid_untrained_policy_and_small_deadline_is_bounded():
    game = RiverGame(BOARD, 200, (100, 100), bet_fractions=(.5,), max_raises=1)
    args = ([('As', 'Ad'), ('4s', '5s')], [.5, .5], [('Ks', 'Kd')], [1])
    zero = solve_river(game, *args, iterations=1000000, time_budget=0)
    assert zero.iterations == 0 and zero.stopped_by_time
    assert zero.expected_values is None
    assert np.allclose(zero.probabilities.sum(1), 1)
    timed = solve_river(game, *args, iterations=1000000, time_budget=.02)
    assert 0 < timed.iterations < 1000000 and timed.stopped_by_time
    assert timed.elapsed_seconds < .2
    assert np.isfinite(timed.probabilities).all() and (timed.probabilities >= 0).all()


@pytest.mark.parametrize('hands,weights', [([('2c', 'As')], [1]), ([('As', 'Ad'), ('Ad', 'As')], [1, 1]),
                                           ([('As', 'Ad')], [-1]), ([('As', 'Ad')], [float('nan')])])
def test_invalid_ranges_rejected(hands, weights):
    with pytest.raises(ValueError):
        solve_river(RiverGame(BOARD, 200, (100, 100)), hands, weights, [('Ks', 'Kd')], [1])


def test_no_compatible_hands_rejected_and_actual_hand_not_silently_added():
    game = RiverGame(BOARD, 200, (100, 100), max_raises=0)
    with pytest.raises(ValueError, match='no compatible'):
        solve_river(game, [('As', 'Ad')], [1], [('As', 'Ks')], [1])
    result = solve_river(game, [('As', 'Ad')], [1], [('Ks', 'Kd')], [1], iterations=0)
    with pytest.raises(ValueError, match='outside'):
        result.distribution(('4s', '5s'))


def test_unsupported_existing_uncalled_short_all_in_rejected():
    with pytest.raises(ValueError, match='unmatched'):
        RiverGame(BOARD, 400, (50, 200), street_bets=(0, 200), to_act=0)


def test_effective_short_raise_is_not_emitted_as_illegal_non_all_in_bet():
    with pytest.raises(ValueError, match='non-all-in'):
        RiverGame(BOARD, 200, (1000, 50), min_raise=100)
    # The same short bet is legal when it is the actor's own actual all-in.
    game = RiverGame(BOARD, 200, (50, 1000), min_raise=100)
    result = solve_river(game, [('As', 'Ad')], [1], [('Ks', 'Kd')], [1], iterations=2)
    assert any(a.kind == 'raise' and a.to == 50 for a in result.actions)
