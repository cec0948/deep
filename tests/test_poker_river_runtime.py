import json
import multiprocessing
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import poker_river_runtime as runtime
from fullhouse_runtime import Bridge
from poker_public_state import PublicState, Range


def request(actions=()):
    prefix = [{'seat':3,'kind':'call'}]
    prefix += [{'seat':seat,'kind':'fold'} for seat in (4,5,0,1)]
    prefix += [{'seat':2,'kind':'check'}]
    prefix += [{'seat':seat,'kind':'check'} for _ in range(2) for seat in (2,3)]
    return dict(op='replay',hero=2,button=0,hole=['As','Kd'],
                board=['2c','7d','Jh','9c','Qs'],actions=prefix+list(actions))


@pytest.fixture
def engine():
    bridge = Bridge()
    yield bridge
    bridge.close()


class UniformBase:
    digest = 'fixture-policy'

    def decide(self, state, seed):
        menu = list(runtime.legal_menu(state).values())
        return dict(ok=True, action=menu[0],
                    distribution=[dict(action=a,probability=1/len(menu)) for a in menu],
                    strength_verified=False)


def policy(seconds=1.):
    instance = runtime.RiverSearchPolicy.__new__(runtime.RiverSearchPolicy)
    instance.base = instance.opponent = UniformBase()
    instance.seconds = seconds
    instance.tremble = .01
    instance.max_raises = 2
    instance._strategy_cache = {}
    instance._context = None
    return instance


def test_nonriver_returns_same_seeded_baseline_without_worker(engine):
    state = engine.call(dict(op='replay',hero=3,button=0,hole=['As','Kd'],board=[],actions=[]))
    candidate = policy()
    old = candidate.base.decide(state,17)
    actual = candidate.decide(state,17)
    assert actual['action'] == old['action']
    assert actual['distribution'] == old['distribution']
    assert not actual['river_search']['attempted']


def test_whole_decision_hard_timeout_preserves_baseline_and_reaps_child(engine):
    state = engine.call(request())
    candidate = policy(.3)
    def stalled(connection, state, seed, budget):
        time.sleep(2)
    candidate._worker = stalled
    children = {p.pid for p in multiprocessing.active_children()}
    start = time.monotonic()
    actual = candidate.decide(state,0)
    assert time.monotonic()-start < .65
    assert actual['river_search']['reason'] == 'hard_deadline'
    assert not actual['river_search']['used']
    assert actual['action'] == candidate.base.decide(state,0)['action']
    assert {p.pid for p in multiprocessing.active_children()} <= children


def test_prior_search_likelihood_replaces_blueprint_for_own_past_action(engine,monkeypatch):
    initial = engine.call(request())
    later = engine.call(request([{'seat':2,'kind':'check'},
                                 {'seat':3,'kind':'raise','to':125}]))
    candidate = policy()
    combos = (('As','Kd'),('Ac','Kc'),('Ah','Kh'))
    small = Range.from_weights(combos,[1,1,1],provenance='test')
    monkeypatch.setattr(runtime.Range,'uniform',lambda board:small)
    key = candidate._history_key(PublicState.from_bridge(initial))
    candidate._strategy_cache[key] = dict(hands=combos,
        actions=[{'kind':'check'},{'kind':'raise','to':125}],
        probabilities=np.array([[.9,.1],[.1,.9],[.5,.5]]))
    ranges, calls, metadata = candidate._ranges(later,[2,3],time.monotonic()+1)
    assert ranges[0].weights == pytest.approx(np.array([.9,.1,.5])/1.5)
    assert metadata[0]['previous_search_decisions_replayed'] == 1
    assert metadata[0]['conditioned_on_actual_hero_hole'] is False
    assert metadata[0]['tremble'] == 0
    candidate.begin_hand(2)
    assert not candidate._strategy_cache


def test_single_untrained_iteration_falls_back_and_prior_check_is_preserved(engine,monkeypatch):
    # Seat2checked, so seat3'scheckmustendtheriver.
    initial = request([{'seat':2,'kind':'check'}])
    initial.update(hero=3,hole=['As','Kd'])
    state = engine.call(initial)
    candidate = policy()
    combos = (('As','Kd'),('Ac','Kc'))
    marginal = Range.from_weights(combos,[1,1],provenance='test')
    candidate._ranges = lambda *args:([marginal,marginal],0,[{},{}])
    seen = []
    def solve(game,*args,**kwargs):
        seen.append(game)
        return SimpleNamespace(iterations=1)
    monkeypatch.setattr(runtime,'solve_river',solve)
    actual = candidate._decide_local(state,0,1.)
    assert seen[0].checks == 1
    assert seen[0].min_raise == PublicState.from_bridge(state).min_raise_increment
    assert not actual['river_search']['used']
    assert actual['river_search']['reason'] == 'insufficient_solver_iterations'


def test_unsupported_off_tree_history_returns_explicit_baseline(engine,monkeypatch):
    state = engine.call(request([{'seat':2,'kind':'raise','to':357},
                                 {'seat':3,'kind':'raise','to':1000}]))
    candidate = policy()
    small = Range.from_weights([('As','Kd'),('Ac','Kc')],[1,1],provenance='test')
    monkeypatch.setattr(runtime.Range,'uniform',lambda board:small)
    actual = candidate._decide_local(state,0,1.)
    assert not actual['river_search']['used']
    assert actual['river_search']['reason'] == 'off_tree_history_not_modeled'


def test_replay_is_bounded_before_search_starts():
    process = subprocess.Popen([sys.executable,'-c','import time;time.sleep(3)'],
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    engine = SimpleNamespace(p=process)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError,match='replay exceeded'):
            runtime.timed_replay(engine,{},started+.05)
        assert time.monotonic()-started < .4
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill();process.wait()
        process.stdin.close();process.stdout.close()


def test_protocol_requires_explicit_hand_boundary_and_rejects_private_fields():
    raw=request();raw.pop('op')
    with pytest.raises(ValueError,match='hand_id'):
        runtime.protocol_request(raw)
    parsed,key=runtime.protocol_request(dict(raw,hand_id='deal-7'))
    assert parsed==raw and key==('str','deal-7')
    with pytest.raises(ValueError,match='Unknown request'):
        runtime.protocol_request(dict(raw,hand_id='deal-7',opponent_holes=[['Ac','Ad']]))
