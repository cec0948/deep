from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fullhouse_runtime import Bridge
from fullhouse_v3_common import atomic_json
from poker_postflop_snapshot_mixture import (
    ENDPOINTS, SEEDS, STUDY, PostflopSnapshotMixture, read, source_records, write_manifest,
)
from poker_preflop_specialist import evaluator_policy


def test_private_draw_is_hand_scoped_and_reproducible():
    class Policy:
        def __init__(self, name):
            self.name = name

        def decide(self, state, action_seed):
            return self.name

    policy = object.__new__(PostflopSnapshotMixture)
    policy.active = None
    policy.preflop = Policy('preflop')
    policy.postflop = [Policy('480'), Policy('504')]
    with pytest.raises(ValueError, match='begin_hand'):
        policy.decide({'street': 'flop'}, 0)
    with pytest.raises(ValueError, match='hand seed'):
        policy.begin_hand(-1)
    chosen = set()
    for hand in range(32):
        policy.begin_hand(hand)
        assert policy.decide({'street': 'preflop'}, hand) == 'preflop'
        selected = policy.decide({'street': 'flop'}, 9)
        for street in ('turn', 'river'):
            assert policy.decide({'street': street, 'hole_cards': [3, 17]}, 15) == selected
        policy.begin_hand(hand)
        assert policy.decide({'street': 'flop'}, 97) == selected
        chosen.add(selected)
    assert chosen == {'480', '504'}


@pytest.mark.skipif(not (STUDY/'summary.json').exists(), reason='Requires private frozen sources')
@pytest.mark.parametrize('seed', SEEDS)
def test_four_street_parity_with_each_actual_snapshot(tmp_path, seed):
    _, plan = source_records()
    manifest = write_manifest(tmp_path/'mixture.json', seed)
    candidate = PostflopSnapshotMixture(manifest, plan['source'])
    endpoints = [evaluator_policy(plan['policies'][label][str(seed)]['path'], plan['source'])
                 for label in ENDPOINTS]
    engine = Bridge()
    covered = set()
    try:
        for hand_seed in range(32):
            candidate.begin_hand(hand_seed)
            index = candidate.active
            if index in covered:
                continue
            state = engine.call(dict(op='reset', seed=44, button=0))
            streets = set()
            while not state['terminal']:
                street = state['street']
                streets.add(street)
                reply = candidate.decide(state, 811)
                assert reply == endpoints[index].decide(state, 811)
                if street == 'preflop':
                    assert reply == endpoints[1-index].decide(state, 811)
                forced = dict(seat=state['seat_to_act'], kind='check' if state['can_check'] else 'call')
                state = engine.call(dict(op='step', action=forced))
            assert streets == {'preflop', 'flop', 'turn', 'river'}
            covered.add(index)
            if len(covered) == len(ENDPOINTS):
                break
    finally:
        engine.close()
    assert covered == {0, 1}
    spec = read(manifest)
    spec['postflop'].reverse()
    atomic_json(manifest, spec)
    with pytest.raises(ValueError, match='identities'):
        PostflopSnapshotMixture(manifest, plan['source'])
