"""Coverage of action aliases and first-reached sampling, including agreement states."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from poker_river_runtime import legal_menu
import sample_poker_flop_diagnostic as sampler


def test_concrete_menu_keeps_zero_mass_actions_and_collapses_duplicate_raise_sizes():
    state = dict(can_check=False, raise_bounds=[200, 230], amount_owed=90,
                 your_bet_this_street=0, pot=400)
    menu = list(legal_menu(state).values())
    assert menu == [{'kind':'call'}, {'kind':'fold'}, {'kind':'raise','to':230}]
    assert sampler.abstract_indices(state, menu) == [1, 0, 2, 2, 2]
    state.update(can_check=True, raise_bounds=None, amount_owed=0)
    assert sampler.abstract_indices(state, list(legal_menu(state).values())) == [None, 0, None, None, None]


@pytest.mark.skipif(not sampler.SOURCE_PLAN.is_file(), reason='Requires source campaign and audited frozen policies')
def test_first_decisions_from_historical_cards_are_reproducible_and_keep_agreements(tmp_path):
    # Exhausted historical evaluation cards, separate from the registered diagnostic.
    common = (342, 2038400000, 8, 300)
    a = sampler.sample_seed(*common, tmp_path/'a')
    b = sampler.sample_seed(*common, tmp_path/'b')
    assert a['states_sha256'] == b['states_sha256']
    assert a['attempts'] > a['accepted'] == 8
    assert 0 < a['same_sampled_root_choice_as_480']['504'] <= 8
    states = (tmp_path/'a/seed-342/states.jsonl').read_text().splitlines()
    rows = [sampler.json.loads(line) for line in states]
    assert len({r['deal_seed'] for r in rows}) == len({r['state_id'] for r in rows}) == 8
    for row in rows:
        root = row['root_observation']
        assert root['street'] == 'flop' and root['seat_to_act'] == row['hero']
        assert row['hero'] == (row['deal_seed'] - common[1]) % 6
        assert all(len(p['probabilities']) == len(row['legal_actions']) for p in row['policies'].values())
        assert 'utilities' not in root and 'chips' not in row
