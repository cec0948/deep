import concurrent.futures
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import poker_capture_postflop_snapshots as capture
from fullhouse_v3_common import atomic_json, digest


def example(tmp):
    src = tmp/'source'
    (src/'bot').mkdir(parents=True)
    (src/'bot/features.py').write_text('# fixture\n')
    fixed = tmp/'fixed.npz'
    np.savez(fixed, training_revision='r1-cpu-v4', opening_trunk_w0=[1.], trunk_w0=[2.])
    parent = tmp/'policy.json'
    atomic_json(parent, dict(format='preflop-street-policy-v1', preflop=dict(path=str(fixed), sha256=digest(fixed))))
    worker = tmp/'worker'
    worker.mkdir()
    return dict(dest=tmp/'exports', worker=worker, seed=342, arm='fixed', parent_policy=parent,
                iterations=[486, 492], source=src)


def publish(reg, t):
    worker = reg['worker']
    fixed = capture.read(reg['parent_policy'])['preflop']
    start = dict(seed=reg['seed'], iteration=480, deployed_policy_sha256=digest(reg['parent_policy']),
                 street_weight_sha256={'preflop':fixed['sha256']}, weights={'preflop':'fixed-state'})
    atomic_json(worker/'start-verification.json', start)
    slot = worker/'checkpoints'/f'iter-{t:05d}'
    slot.mkdir(parents=True)
    shutil.copyfile(fixed['path'], slot/'current.npz')
    atomic_json(slot/'state.json', dict(t='dict', items=[
        [key, {'t':'value','value':val}] for key, val in [('seed',reg['seed']),('iteration',t)]]))
    atomic_json(worker/'status.json', dict(status='training', seed=reg['seed'], arm=reg['arm'], iteration=t,
                                         pid=595, process_start='888'))
    atomic_json(worker/'current.json', dict(directory=slot.name, iteration=t,
                 files={p.name:digest(p) for p in slot.iterdir()}))
    row = dict(iteration=t, seed=reg['seed'], checkpoint_iteration=t,
               preflop_steps=0, opening_steps=0, input_weights=start['weights'], output_weights=start['weights'],
               frozen_preflop_probes=1, frozen_preflop_max_probability_error=0.)
    with (worker/'diagnostics.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')


def test_registration_capture_retry_and_portable_final(tmp_path):
    reg = example(tmp_path)
    dest = reg['dest']
    config = capture.register(**reg)
    assert capture.register(**reg) == config
    with pytest.raises(ValueError, match='different plan'):
        capture.register(**dict(reg, arm='another'))
    with pytest.raises(ValueError, match='Unscheduled'):
        capture.capture(dest, 489)
    publish(reg, 486)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        a, b = pool.map(lambda _: capture.capture(dest, 486), range(2))
    assert a == b
    with pytest.raises(FileNotFoundError):
        capture.finalize(dest)
    publish(reg, 492)
    assert capture.capture(dest, 486) == a
    capture.capture(dest, 492)
    manifest = capture.finalize(dest)
    shutil.rmtree(reg['worker'])
    moved = dest.rename(tmp_path/'moved')
    assert capture.finalize(moved) == manifest
    assert [slot['iteration'] for slot in manifest['postflop']] == [486, 492]
    assert {p.name for p in moved.rglob('*.npz')} == {'current.npz','preflop.npz'}
    assert len(list(moved.rglob('*.npz'))) == 3
    (moved/'iter-00486/current.npz').write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest'):
        capture.finalize(moved)


def test_four_checkpoint_plan_has_bounded_export_count(tmp_path):
    reg = example(tmp_path)
    times = [486, 492, 498, 504]
    capture.register(**dict(reg, iterations=times))
    for t in times:
        publish(reg, t)
        capture.capture(reg['dest'], t)
    manifest = capture.finalize(reg['dest'])
    assert [row['iteration'] for row in manifest['postflop']] == times
    assert len(list(reg['dest'].rglob('*.npz'))) == len(times)+1
    assert not list(reg['dest'].rglob('arrays.npz'))


def test_mismatched_worker_and_quality_or_missed_schedule(tmp_path):
    reg = example(tmp_path)
    with pytest.raises(ValueError, match='four'):
        capture.register(**dict(reg, iterations=[486, 492, 498, 504, 510]))
    capture.register(**reg)
    publish(reg, 486)
    log = reg['worker']/'diagnostics.jsonl'
    row = capture.read(log)
    log.write_text(json.dumps(dict(row, preflop_steps=12))+'\n')
    with pytest.raises(ValueError, match='fixed preflop'):
        capture.capture(reg['dest'], 486)
    assert not (reg['dest']/'iter-00486').exists()
    with pytest.raises(ValueError, match='before'):
        capture.register(**dict(reg, dest=tmp_path/'late'))
    publish(reg, 492)
    with pytest.raises(ValueError, match='Wrong current'):
        capture.capture(reg['dest'], 486)


def test_actual_published_504_can_be_captured_without_touching_original(tmp_path):
    from poker_postflop_snapshot_mixture import STUDY, source_records
    from fullhouse_runtime import Bridge
    from poker_preflop_specialist import evaluator_policy
    if not (STUDY/'summary.json').exists():
        pytest.skip('Requires original private weights')
    _, plan = source_records()
    seed = 342
    base = Path(plan['policies']['specialist480'][str(seed)]['path'])
    actual = Path(plan['policies']['fixed504'][str(seed)]['path']).parent
    worker = tmp_path/'worker'
    worker.mkdir()
    dest = tmp_path/'bundle'
    reg = dict(dest=dest, worker=worker, seed=seed, arm='fixed504', parent_policy=base,
               iterations=[498,504], source=plan['source'])
    capture.register(**reg)
    # A deliberately synthetic first pointer tests completeness. The second pointer,
    # metadata, trained output, run identity and diagnostics come from the real 504.
    publish(reg,498)
    start = capture.read(actual/'start-verification.json')
    atomic_json(worker/'start-verification.json', start)
    row = capture.read(worker/'diagnostics.jsonl')
    row.update(input_weights=start['weights'],output_weights=start['weights'])
    (worker/'diagnostics.jsonl').write_text(json.dumps(row)+'\n')
    status = capture.read(actual/'status.json')
    atomic_json(worker/'status.json',dict(status,iteration=498))
    slot = worker/'checkpoints/iter-00498'
    base_after = capture.read(base)['postflop']
    shutil.copyfile(base_after['path'],slot/'current.npz')
    ptr = capture.read(worker/'current.json')
    ptr['files']['current.npz'] = base_after['sha256']
    atomic_json(worker/'current.json',ptr)
    capture.capture(dest,498)
    for name in ('status.json','diagnostics.jsonl','current.json'):
        shutil.copyfile(actual/name,worker/name)
    slot = worker/'checkpoints/iter-00504'
    slot.mkdir()
    for name in ('current.npz','state.json'):
        shutil.copyfile(actual/'checkpoints/iter-00504'/name,slot/name)
    receipt = capture.capture(dest,504)
    model = capture.read(actual/'policy.json')['postflop']
    assert receipt['export_sha256'] == model['sha256']
    assert receipt['export_bytes'] == Path(model['path']).stat().st_size
    capture.finalize(dest)
    mixture = capture.ScheduledPostflopMixture(dest/'mixture.json',plan['source'])
    endpoints = [evaluator_policy(p,plan['source']) for p in (base,actual/'policy.json')]
    engine = Bridge()
    covered = set()
    try:
        for hand in range(32):
            mixture.begin_hand(hand)
            i = mixture.postflop.index(mixture.active)
            if i in covered:
                continue
            state = engine.call(dict(op='reset',seed=44,button=0))
            streets = set()
            while not state['terminal']:
                streets.add(state['street'])
                assert mixture.decide(state,498) == endpoints[i].decide(state,498)
                kind = 'check' if state['can_check'] else 'call'
                state = engine.call(dict(op='step', action=dict(kind=kind,seat=state['seat_to_act'])))
            assert len(streets) == 4
            covered.add(i)
            if len(covered) == 2: break
    finally:
        engine.close()
    assert covered == {0,1}
