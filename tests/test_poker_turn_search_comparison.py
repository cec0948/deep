"""The paired pilot keeps fallbacks and follows a successful root until showdown."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_poker_multivalued_turn import branch, eligible, restore
from fullhouse_runtime import BRIDGE, Bridge


class Passive:
    def decide(self, state, seed):
        return {'action':{'kind':'check' if state['can_check'] else 'call'}}


class Search(Passive):
    def __init__(self, used):
        self.used = used
        self.calls = 0

    def decide(self, state, seed):
        assert 'deck' not in state and len(state['your_cards']) == 2
        assert all('your_cards' not in player for player in state['players'])
        assert len(state['community_cards']) == (4 if self.calls == 0 else 5)
        reply = super().decide(state, seed)
        reply['turn_search'] = {'used': self.used,
            'reason': 'committed_continuation' if self.calls else ('search_complete' if self.used else 'fallback')}
        self.calls += 1
        return reply


@pytest.mark.skipif(not Path(BRIDGE).is_file(), reason='requires compiled reference poker rules')
@pytest.mark.parametrize('used', [False, True])
def test_both_arms_keep_the_same_deal_and_private_continuation(used):
    engine = Bridge()
    try:
        row = {'card':2719, 'hero':1, 'nonce':17, 'history':[
            *({'seat':seat, 'kind':'fold'} for seat in (3,4,5,0)),
            {'seat':1,'kind':'call'}, {'seat':2,'kind':'check'},
            {'seat':1,'kind':'check'}, {'seat':2,'kind':'check'}]}
        assert eligible(restore(engine, row['card'], row['history']), row['hero'])
        base = branch(engine, row, Passive(), Passive())
        search = Search(used)
        after = branch(engine, row, Passive(), Passive(), search)
        assert (base['suffix'], base['utility_bb']) == (after['suffix'], after['utility_bb'])
        assert after['followup_continuation_decisions'] == int(used)
        assert search.calls == 1+int(used)
    finally:
        engine.close()
