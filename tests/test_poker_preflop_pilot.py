from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_poker_480_decisions import first_difference
from fullhouse_v3_common import atomic_json
from poker_preflop_specialist import StreetPolicy, write_policy
import run_poker_preflop_pilot as pilot


def test_different_raise_sizes_and_opponent_drift():
    before = [dict(seat=2, kind='call'), dict(seat=3, kind='raise', to=500)]
    after = [before[0], dict(seat=3, kind='raise', to=700)]
    assert first_difference(before, after, 3) == (((2, 'call', None),), before[1], after[1])
    assert first_difference(before, before, 3) is None
    with pytest.raises(ValueError):
        first_difference(before, after, 2)
    with pytest.raises(ValueError):
        first_difference(before, after[:1], 3)


def test_one_card_block_is_the_unit_even_when_seeds_repeat():
    rows = {}
    for field in pilot.FIELDS:
        for seed in pilot.SEEDS:
            rows[seed, field, 'parent'] = [0., 0., 0., 0.]
            rows[seed, field, 'candidate'] = [0., 0., 0., 4.]
    change = pilot.paired(rows, 'parent', 'candidate', 4)['public']
    assert change['mean'] == 1.
    assert change['ci95_half_width'] == pytest.approx(1.96)
    rows[pilot.SEEDS[1], 'public', 'candidate'][0] = float('nan')
    with pytest.raises(ValueError):
        pilot.paired(rows, 'parent', 'candidate', 4)


def test_interrupted_owner_and_single_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, 'preflight', lambda: {'implementation': {}})
    lock = tmp_path/'test-runner.py'
    lock.touch()
    monkeypatch.setattr(pilot, 'SCRIPT', lock)
    attempt = tmp_path/'one'
    assert pilot.status(attempt)['state'] == 'not_started'
    attempt.mkdir()
    assert pilot.status(attempt)['state'] == 'claimed_without_status'
    atomic_json(attempt/'status.json', dict(state='running', pid=1, process_start='0', workers=[]))
    assert pilot.status(attempt)['state'] == 'interrupted'
    with pytest.raises(FileExistsError):
        pilot.supervise(attempt, False)


def test_actual_composed_actions_match_source_model_at_each_street(tmp_path):
    from fullhouse_runtime import Bridge
    from poker_average_runtime import policy_for
    if not (pilot.PARENT/'evaluation-plan.json').is_file():
        pytest.skip('Private seed-matched source weights required for integration check')
    specs = pilot.read(pilot.PARENT/'evaluation-plan.json')['policies']
    before = Path(specs['parent420']['342']['path'])
    after = Path(specs['current480']['342']['path'])
    path = write_policy(tmp_path/'swap.json', before, after)
    composed = StreetPolicy(path, pilot.SOURCE)
    policies = [policy_for(p, pilot.SOURCE) for p in (before, after)]
    engine = Bridge()
    streets = set()
    try:
        state = engine.call(dict(op='reset', seed=44, button=0))
        while not state['terminal']:
            street = state['street']
            streets.add(street)
            expected = policies[0 if street == 'preflop' else 1].decide(state, 811)
            assert composed.decide(state, 811) == expected
            action = {'seat':state['seat_to_act'], 'kind':'check' if state['can_check'] else 'call'}
            state = engine.call(dict(op='step', action=action))
    finally:
        engine.close()
    assert streets == {'preflop', 'flop', 'turn', 'river'}
    spec = pilot.read(path)
    spec['preflop']['sha256'] = '0'*64
    atomic_json(path, spec)
    with pytest.raises(ValueError, match='weights changed'):
        StreetPolicy(path, pilot.SOURCE)
