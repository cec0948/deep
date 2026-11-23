from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_poker_flop_diagnostic import clustered_intervals, clustered_means


def test_shared_original_cards_are_not_independent_seeds_or_observations():
    values = np.array([[0, 1], [6, 7], [0, 1], [6, 7]], dtype=float)
    means, errors, by_seed = clustered_means(values, [342, 342, 343, 343], [10, 11, 10, 11], [342, 343])
    assert means == pytest.approx([3, 4])
    assert errors == pytest.approx([3, 3])
    assert by_seed[342] == pytest.approx(by_seed[343])
    # Further stochastic repeats of the same states do not create new card clusters.
    _, more_errors, _ = clustered_means(np.repeat(values, 3, axis=0),
                                       np.repeat([342, 342, 343, 343], 3),
                                       np.repeat([10, 11, 10, 11], 3), [342, 343])
    assert more_errors == pytest.approx(errors)


def test_disjoint_uneven_seed_strata_receive_equal_weight():
    means, errors, _ = clustered_means([[0], [6], [10], [10], [10]],
                                       [342, 342, 343, 343, 343], [11, 12, 13, 14, 15], [342, 343])
    assert means[0] == 6.5
    assert errors[0] == pytest.approx(1.5)


def test_paired_opponent_interaction_cancels_shared_card_effects():
    table = np.array([[1, 2], [4, 5], [-2, -1]], dtype=float)
    values = np.column_stack((table, table[:, 1] - table[:, 0]))
    results = clustered_intervals(values, [342] * 3, [10, 11, 12], [342], ['self', 'public', 'public-self'])
    assert results['public-self']['mean_bb'] == 1
    assert results['public-self']['simultaneous_95_bb'] == [1, 1]
    assert results['public']['simultaneous_95_bb'][0] < results['public']['paired_95_bb'][0]


def test_reject_missing_seeds_and_nonfinite_rollouts():
    with pytest.raises(ValueError, match='seed coverage'):
        clustered_means([[1], [2]], [342, 342], [1, 2], [342, 343])
    with pytest.raises(ValueError, match='finite'):
        clustered_means([[1], [np.inf]], [342, 342], [1, 2], [342])


def paired_sample():
    from score_poker_flop_diagnostic import LINEUPS

    states, rollouts = [], []
    for seed in (342, 343):
        for index in (0, 1, 2):
            deal = 10 * seed + index
            base = [1., 0., 0.] if index != 2 else [.5, .5, 0.]
            new = [0., 1., 0.] if index != 1 else base
            state = dict(seed=seed, deal_seed=deal, state_id=str(deal), hero=3,
                         legal_actions=[{'kind': 'fold'}, {'kind': 'call'}, {'kind': 'raise', 'to': 300}])
            policies = {'480': {'probabilities': base}}
            for label in ('486', '492', '498', '504'):
                policies[label] = {'probabilities': new}
            states.append(dict(state, policies=policies))
            for lineup, gap in zip(LINEUPS, (100, -100, 400)):
                pay = [[0, 200], [gap, gap + 200], None]
                if index == 1:
                    pay[1] = None
                rollouts.append(dict(state, repeats=2, lineup=lineup, action_payoffs_chips=pay))
    return states, rollouts


def test_full_scoring_and_most_likely_action_coverage():
    from score_poker_flop_diagnostic import score_flop_diagnostic

    states, rows = paired_sample()
    got = score_flop_diagnostic(states, rows, seeds=(342, 343), states_per_seed=3, repeats=2)
    assert got['states'] == got['distinct_original_deals'] == 6
    assert got['evaluated_terminal_branches'] == 60
    primary = got['primary_mixture_minus_480']
    assert primary['contrasts']['self480']['mean_bb'] == pytest.approx(.5)
    assert primary['contrasts']['public_minus_self480']['mean_bb'] == pytest.approx(-1.)
    assert primary['contrasts']['tight-value_minus_self480']['mean_bb'] == pytest.approx(1.5)
    assert primary['states_with_probability_change'] == 4
    assert primary['observed_signs_on_changed_states_self480_by_external']['public']['positive']['negative'] == 4
    switched = got['different_most_likely_root_actions']
    assert switched['ties_excluded'] == switched['same_unique_top'] == switched['different_top'] == 2
    assert switched['by_lineup']['self480']['new_top_minus_old_top_bb'] == 1.
    assert switched['by_lineup']['public']['negative'] == 2


def test_mixture_is_uniform_over_four_policy_distributions():
    from score_poker_flop_diagnostic import score_flop_diagnostic

    states, rows = paired_sample()
    for state in states:
        for snap in ('486', '492', '498'):
            state['policies'][snap] = state['policies']['480']
    got = score_flop_diagnostic(states, rows, seeds=(342, 343), states_per_seed=3, repeats=2)
    assert got['primary_mixture_minus_480']['contrasts']['self480']['mean_bb'] == pytest.approx(.125)
    assert got['secondary_minus_480']['486']['contrasts']['self480']['mean_bb'] == 0.
    assert got['secondary_minus_480']['504']['contrasts']['self480']['mean_bb'] == pytest.approx(.5)


def test_missing_positive_action_or_missing_lineup_is_not_a_complete_score():
    from score_poker_flop_diagnostic import score_flop_diagnostic

    states, rows = paired_sample()
    with pytest.raises(ValueError, match='missing'):
        score_flop_diagnostic(states, rows[1:], seeds=(342, 343), states_per_seed=3, repeats=2)
    rows[0]['action_payoffs_chips'][1] = None
    with pytest.raises(ValueError, match='positive-probability'):
        score_flop_diagnostic(states, rows, seeds=(342, 343), states_per_seed=3, repeats=2)
