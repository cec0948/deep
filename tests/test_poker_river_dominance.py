from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from poker_river_dominance import closing_call_floor, guard_fold


def facing_bet():
    # Seats four and five can win each layer. Two folded contribution tiers
    # have odd chip totals. Seat five ranks last for distributing odd chips.
    commits = [100, 101, 102, 103, 200, 100]
    return dict(terminal=False, street='river', can_check=False, seat_to_act=5, amount_owed=100,
                community_cards=['Ad','Kd','Qs','Jh','Tc'], your_cards=['2s','4s'], pot=sum(commits),
                players=[dict(seat=i, stack=10_000-c, bet_this_street=100 if i==4 else 0,
                              is_folded=i<4) for i,c in enumerate(commits)])


def test_if_board_always_splits_odd_chips_are_rounded_in_each_pot_tier():
    state = facing_bet()
    # 600/2 + floor(5/2) + 4/2 + floor(3/2) + 194/2, less call.
    assert closing_call_floor(state) == 302
    assert (state['pot'] + 100)//2 - 100 == 303


def test_ineligible_chips_multiway_and_incomplete_calls_are_excluded():
    state = facing_bet()
    for update in (dict(street='turn'), dict(can_check=True), dict(terminal=True)):
        assert closing_call_floor(dict(state, **update)) == 0
    for position, change in [(2, dict(is_folded=False)), (0, dict(stack=9500)), (5,dict(stack=99))]:
        invalid = deepcopy(state)
        invalid['players'][position].update(change)
        invalid['pot'] = sum(10000 - p['stack'] for p in invalid['players'])
        assert closing_call_floor(invalid) == 0


def test_one_possible_winning_opponent_disables_guard_and_unbeatable_nontie_wins_entire_pot():
    state = facing_bet()
    # One suited opponent can make a flush on the board.
    assert closing_call_floor(dict(state, community_cards=['Ad','Kd','Qd','Jh','Tc'])) == 0
    assert closing_call_floor(dict(state, community_cards=['Ac','Ad','Js','8d','3s'],
                                   your_cards=['As','Ah'])) == state['pot']


def test_redirects_positive_fold_only_preserving_existing_random_draw_and_raise_distribution():
    state=facing_bet()
    rows=[dict(action=dict(kind='fold'),probability=.25),dict(action=dict(kind='call'),probability=.5),
          dict(action=dict(kind='raise',to=300),probability=.25)]
    before=dict(distribution=deepcopy(rows),action=rows[0]['action'])
    after=guard_fold(state,before)
    assert after['action']=={'kind':'call'} and before['distribution']==rows
    assert after['distribution']==[dict(action={'kind':'call'},probability=.75),rows[-1]]
    assert guard_fold(state,dict(before,action=rows[-1]['action']))['action']==rows[-1]['action']
    safe=dict(distribution=[rows[-1]],action=rows[-1]['action'])
    assert guard_fold(state,safe) is safe
