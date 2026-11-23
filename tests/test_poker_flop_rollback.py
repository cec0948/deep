from contextlib import closing
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

from fullhouse_runtime import Bridge
from fullhouse_v3_common import atomic_json, digest
from poker_capture_postflop_snapshots import ScheduledPostflopMixture, read
from poker_flop_rollback import FlopRollbackPolicy, ITERATIONS, write_policy
from poker_preflop_specialist import evaluator_policy


def test_spec_checks_seed_lineage_schedule_and_hash(tmp_path):
    def ref(path):
        return dict(path=str(path), sha256=digest(path))

    parent = tmp_path/'specialist.json'
    atomic_json(parent, dict(preflop={'sha256':'frozen'}))
    parent_ref = ref(parent)
    capture = tmp_path/'plan.json'
    atomic_json(capture, dict(parent_policy_sha256=parent_ref['sha256']))
    mixture = tmp_path/'mixture.json'
    spec = dict(plan_sha256=digest(capture), preflop={'sha256':'frozen'}, seed=342, arm='pool504',
                postflop=[dict(iteration=t) for t in ITERATIONS])
    atomic_json(mixture, spec)
    target = tmp_path/'policy.json'
    assert write_policy(target, 342, parent_ref, ref(mixture)) == target
    assert write_policy(target, 342, parent_ref, ref(mixture)) == target
    with pytest.raises(ValueError, match='seed'):
        write_policy(target, 343, parent_ref, ref(mixture))
    with pytest.raises(ValueError, match='reference changed'):
        write_policy(target, 342, parent_ref, dict(path=str(mixture), sha256='bad'))
    for altered in [dict(arm='control504'), dict(postflop=list(reversed(spec['postflop'])))]:
        atomic_json(mixture, dict(spec, **altered))
        with pytest.raises(ValueError, match='Rollback sources differ'):
            write_policy(target, 342, parent_ref, ref(mixture))
    atomic_json(mixture, spec)
    atomic_json(parent, dict(preflop={'sha256':'different'}))
    with pytest.raises(ValueError, match='Rollback sources differ'):
        write_policy(target, 342, ref(parent), ref(mixture))


STUDY = ROOT/'artifacts/preflop-followup/480-v1/opponent-diversity-20261123'
PLAN = STUDY/'population/evaluation-plan.json'


@pytest.mark.skipif(not PLAN.is_file(), reason='Requires frozen private production weights')
@pytest.mark.parametrize('seed', [342, 343, 344])
def test_four_actual_slots_match_reference_through_divergent_flop(tmp_path, seed):
    plan = read(PLAN)
    source = plan['source']
    parent_ref = plan['policies']['specialist480'][str(seed)]
    mixture_ref = plan['policies']['diverse504'][str(seed)]
    target = write_policy(tmp_path/'rollback.json', seed, parent_ref, mixture_ref)
    candidate = FlopRollbackPolicy(target, source)
    parent = evaluator_policy(parent_ref['path'], source)
    mixture = ScheduledPostflopMixture(mixture_ref['path'], source)
    assert all(digest(path) == sha for path, sha in candidate.dependency_hashes.items())
    assert candidate.snapshot_iteration is None
    with pytest.raises(ValueError, match='begin_hand'):
        candidate.decide({'street':'flop'}, 42)
    for invalid in (True, -1, 2**64):
        with pytest.raises(ValueError):
            candidate.begin_hand(invalid)

    checked = set()
    with closing(Bridge()) as engine:
        for hand_key in range(64):
            mixture.begin_hand(hand_key)
            slot = mixture.postflop.index(mixture.active)
            if slot in checked:
                continue
            candidate.begin_hand(hand_key)
            assert candidate.snapshot_iteration == ITERATIONS[slot]
            assert candidate.active.digest == mixture.active.digest
            state = engine.call(dict(op='reset', seed=44, button=0))
            seen = set()
            divergent = False
            while not state['terminal']:
                street = state['street']
                seen.add(street)
                action_seed = 498
                expected = parent if street in ('preflop', 'flop') else mixture
                reply = candidate.decide(state, action_seed)
                assert reply == expected.decide(state, action_seed)
                assert candidate.snapshot_iteration == ITERATIONS[slot]
                action = dict(kind='check' if state['can_check'] else 'call')
                if street == 'flop' and not divergent:
                    # Take a genuine parent action unlike the one the unmodified
                    # mixture would have chosen. Continue the same hand to the river.
                    for action_seed in range(48):
                        early = parent.decide(state, action_seed)
                        late = mixture.decide(state, action_seed)
                        first = early['action']
                        allows_later_streets = (first['kind'] == 'check'
                                                or (first['kind'] == 'raise'
                                                    and first['to'] < state['raise_bounds'][1]))
                        if first != late['action'] and allows_later_streets:
                            assert candidate.decide(state, action_seed) == early
                            action = early['action']
                            divergent = True
                            break
                state = engine.call(dict(op='step', action=dict(action,seat=state['seat_to_act'])))
            assert seen == {'preflop','flop','turn','river'} and divergent
            checked.add(slot)
            if len(checked) == len(ITERATIONS):
                break
    assert checked == set(range(len(ITERATIONS)))
