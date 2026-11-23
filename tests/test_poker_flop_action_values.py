from contextlib import closing
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fullhouse_runtime import Bridge
from poker_flop_action_values import (LINEUPS, POLICIES, continuation_action, future_deck,
                                      rollout_state, source_cards)
from poker_multivalued_turn import Rules
from poker_river_runtime import legal_menu


class Passive:
    def decide(self, state, seed):
        assert 'deck' not in state and len(state['your_cards']) == 2
        assert all('hand_commit' not in row for row in state['players'])
        kind = 'check' if state['can_check'] else 'call'
        return {'distribution': [{'action': {'kind': kind}, 'probability': 1.0}]}


def sample(bridge, card):
    state = bridge.call(dict(op='reset', button=0, seed=card))
    prefix = []
    while state['street'] == 'preflop':
        action = dict(seat=state['seat_to_act'], kind='check' if state['can_check'] else 'call')
        prefix.append(action)
        state = bridge.call(dict(op='step', action=action))
    # Put the big blind in a spot facing a wager where fold is a known-value action.
    action = dict(seat=state['seat_to_act'], kind='raise', to=state['raise_bounds'][0])
    state = bridge.call(dict(op='step', action=action))
    prefix.append(action)
    actions = list(legal_menu(state).values())
    prob = [0.5 if a['kind'] in ('fold', 'call') else 0 for a in actions]
    return dict(state_id='fixture-1', seed=342, deal_seed=card, hero=state['seat_to_act'], prefix=prefix,
                root_observation=state, legal_actions=actions,
                policies={label:{'probabilities':list(prob)} for label in POLICIES})


def test_conditional_worlds_and_common_action_branches():
    with closing(Bridge()) as bridge, closing(Rules()) as rules:
        spot = sample(bridge, 124)
        fixed = source_cards(bridge, 124)
        one, two = [future_deck(fixed, 124, n) for n in range(2)]
        assert one[:15] == two[:15] == list(fixed)
        assert one[12:15] == spot['root_observation']['community_cards']
        assert one[2*spot['hero']:2*spot['hero']+2] == spot['root_observation']['your_cards']
        assert len(set(one)) == len(set(two)) == 52 and one[15:17] != two[15:17]
        passive = Passive()
        rows, traces = rollout_state(spot, bridge, rules, passive, dict.fromkeys(LINEUPS, passive), 2)
        assert len(traces) == 6 and len(rows) == 3
        supports = rows[0]['action_payoffs_chips']
        for i, action in enumerate(spot['legal_actions']):
            if action['kind'] == 'fold':
                assert supports[i] == [-100, -100]
            elif action['kind'] == 'call':
                assert len(supports[i]) == 2
            else:
                assert supports[i] is None
        assert all(r['action_payoffs_chips'] == supports for r in rows)
        for trace in traces:
            assert trace['deck'] == one and trace['prefix'] == spot['prefix']
            assert trace['utilities'][spot['hero']] == supports[trace['action_index']][0]
        spot['root_observation']['players'][5]['stack'] += 1
        with pytest.raises(ValueError, match='Sampled root changed'):
            rollout_state(spot, bridge, rules, passive, dict.fromkeys(LINEUPS, passive), 1)


def test_continuation_uses_order_independent_uniform_over_exact_distribution():
    class Mixed:
        def __init__(self, reverse):
            self.reverse = reverse

        def decide(self, state, seed):
            rows = [{'action':{'kind':'call'},'probability':.25},
                    {'action':{'kind':'fold'},'probability':.75}]
            # Ignore its already sampled action and use the common tape.
            return {'action':{'kind':'raise','to':9999},'distribution':rows[::-1] if self.reverse else rows}

    root = {'can_check':False,'raise_bounds':[200,700]}
    for nonce, action in [(0, 'fold'), (3 << 62, 'call')]:
        assert [continuation_action(Mixed(reverse), root, nonce)['kind'] for reverse in (False,True)] == [action]*2
