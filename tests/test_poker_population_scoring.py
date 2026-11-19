from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_poker_population import score_panel


def group(family, *, role='primary', start=100):
    return dict(family=family, deal_seed=start, deals=4, role=role)


def pair(old, new, seeds=(342, 343, 344)):
    return {label: {s: blocks for s in seeds} for label, blocks in [('steps0', old), ('steps256', new)]}


def test_large_wins_do_not_hide_improvement_and_cap_applies_after_averaging():
    groups = {'public': group('public'), 'easy-a': group('rules'), 'easy-b': group('rules'),
              'mix': group('held', role='holdout', start=200)}
    panel = {'public': pair([-20] * 4, [-5] * 4),
             'easy-a': pair([0, 800, 0, 800], [300] * 4),
             'easy-b': pair([400] * 4, [300] * 4),
             'mix': pair([0] * 4, [-1000] * 4)}
    roles = score_panel(panel, groups, [342, 343, 344])['roles']
    main = roles['primary']
    assert main['field_weights'] == {'public': 0.5, 'easy-a': 0.25, 'easy-b': 0.25}
    assert main['uncapped_family_balanced_change_bb100']['mean'] == -42.5
    utility = main['family_balanced_capped_bb100']['100']
    assert utility['paired_change']['mean'] == 7.5
    assert utility['steps0'] == 40 and utility['steps256'] == 47.5
    assert main['breadth']['steps256']['family_balanced_positive_share'] == 0.5
    assert main['breadth']['steps256']['weakest_field_by_point_estimate'] == 'public'
    assert roles['holdout']['uncapped_family_balanced_change_bb100']['mean'] == -1000


def test_repeating_fixed_seeds_does_not_multiply_independent_cards():
    groups = {'public': group('public')}
    old, new = [1, -1, 3, -3], [2, 0, 4, -2]
    one = score_panel({'public': pair(old, new, (342,))}, groups, [342])
    three = score_panel({'public': pair(old, new)}, groups, [342, 343, 344])
    stats = three['fields']['public']
    assert stats == one['fields']['public']
    assert stats['steps0']['lower'] < 0 < stats['steps0']['upper']
    assert stats['paired_change'] == {'mean': 1, 'lower': 1, 'upper': 1}


def test_overlap_or_incomplete_pairs_rejected():
    groups = {'public': group('public'), 'held': group('rule', role='holdout', start=103)}
    panel = {field: pair([0] * 4, [1] * 4) for field in groups}
    with pytest.raises(ValueError, match='overlaps'):
        score_panel(panel, groups, [342, 343, 344])
    groups['held']['deal_seed'] = 104
    del panel['held']['steps256'][343]
    with pytest.raises(ValueError, match='coverage'):
        score_panel(panel, groups, [342, 343, 344])
    panel['held']['steps256'][343] = [1, 1, float('nan'), 1]
    with pytest.raises(ValueError, match='nonfinite'):
        score_panel(panel, groups, [342, 343, 344])
