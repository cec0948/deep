"""The isolated continuation must preserve state and fail closed on corruption."""
import copy
import json
from pathlib import Path
import sys
import subprocess
import time

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_compact_learning_curve as runner


def write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))


def test_fixed_study_budget_does_not_change_the_learning_method():
    cfg = json.loads((ROOT/'configs/compact-learning-curve-20261112.json').read_text())
    parent = dict(seeds=[342,343,344], iterations=120, checkpoint_every=6)
    runner.validate_config(cfg, parent)
    for key, value in [('stages', [180,300]), ('evaluation_deals', 2000),
                       ('promotion', 'automatic'), ('max_seconds', 14401)]:
        bad = copy.deepcopy(cfg); bad[key] = value
        with pytest.raises(ValueError):
            runner.validate_config(bad, parent)
    fixture = dict(parent, verification_fixture=True)
    with pytest.raises(ValueError, match='production120'):
        runner.validate_config(cfg, fixture)


def test_only_stopping_and_evaluation_metadata_can_change():
    before = dict(iterations=120, learning_rate=.001, batch_size=512, method='compact-opening-expert')
    after = dict(before, iterations=240, evaluation_deals=20000, stages=[180,240])
    assert set(runner.metadata_only_config(before, after)) == {'iterations','evaluation_deals','stages'}
    for key, value in [('learning_rate', .002), ('batch_size', 1024), ('method','raw-objective')]:
        bad = dict(after); bad[key] = value
        with pytest.raises(ValueError, match='hyperparameters'):
            runner.metadata_only_config(before, bad)


def test_exact_payload_check_rejects_dtype_shape_rng_or_key_changes():
    state = dict(net={'w':torch.tensor([1.,2.])}, buffers=[np.arange(4,dtype=np.float32)],
                 rng=(np.array([1,2], dtype=np.uint32), 42), iteration=120)
    runner.assert_same(state, copy.deepcopy(state))
    cases = []
    bad = copy.deepcopy(state); bad['buffers'][0] = bad['buffers'][0].astype(np.float64); cases.append(bad)
    bad = copy.deepcopy(state); bad['net']['w'] = bad['net']['w'].reshape(2,1); cases.append(bad)
    bad = copy.deepcopy(state); bad['rng'][0][0] = 7; cases.append(bad)
    bad = copy.deepcopy(state); bad['new'] = 1; cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError):
            runner.assert_same(state, bad)


def test_checkpoint_reconfiguration_preserves_every_other_key(tmp_path):
    parent, campaign = tmp_path/'parent', tmp_path/'campaign'
    old = dict(seeds=[342], iterations=120, learning_rate=.001)
    prior = dict(old, iterations=240)
    wanted = dict(prior, evaluation_deals=20000, stages=[180,240])
    write(parent/'config.json', old); write(campaign/'config.json', prior)
    state = dict(config=old, seed=342, iteration=120, net={'x':torch.tensor([1.,2.])},
        buffers=[np.ones((2,2),np.float32)], rng={'state':np.array([3,4],np.uint32)},
        source_manifest={'x':'y'}, snapshots=[dict(weight=1)], extra_preserved_key='required')
    for folder, cfg in [(parent,old),(campaign,prior)]:
        (folder/'seed-342').mkdir(); torch.save(dict(state, config=cfg), folder/'seed-342/latest.pt')
    original_hash = runner.digest(parent/'seed-342/latest.pt')
    result = runner.replace_checkpoint_config(parent, campaign, wanted)
    copied = torch.load(campaign/'seed-342/latest.pt', weights_only=False)
    assert copied['config'] == wanted and result['342']['complete_payload_except_config_identical']
    runner.assert_same({k:v for k,v in state.items() if k!='config'}, {k:v for k,v in copied.items() if k!='config'})
    assert runner.digest(parent/'seed-342/latest.pt') == original_hash
    assert copied['extra_preserved_key'] == 'required'


def test_method_audit_must_pass_and_match_actual_audited_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    source = tmp_path/'source.py'; source.write_text('frozen implementation')
    parent = tmp_path/'parent'; path = tmp_path/'audit.json'
    good = dict(complete=True, parent=str(parent), no_blocking_implementation_faults=True,
                checks={'deterministic_resume':True}, source_hashes={str(source):runner.digest(source)})
    write(path, good); assert runner.validate_audit(path, parent) == good
    for changed in [dict(good, complete=False), dict(good, checks={'deterministic_resume':False}),
                    dict(good, source_hashes={})]:
        write(path, changed)
        with pytest.raises(ValueError):
            runner.validate_audit(path, parent)
    write(path, good); source.write_text('changed implementation')
    with pytest.raises(ValueError, match='changed'):
        runner.validate_audit(path, parent)


def test_diagnostics_ignore_partial_line_but_reject_nan_or_missing_iteration(tmp_path):
    path = tmp_path/'seed-342/diagnostics.jsonl'; path.parent.mkdir(); path.write_text('old\n')
    guard = runner.DiagnosticGuard(tmp_path, [342])
    first = json.dumps(dict(seed=342, iteration=121, main_loss=1.))
    with path.open('a') as stream: stream.write(first[:20])
    guard.check(); assert guard.state[342]['iteration'] == 120
    with path.open('a') as stream: stream.write(first[20:]+'\n')
    guard.finished(121)
    with path.open('a') as stream: stream.write(json.dumps(dict(seed=342, iteration=122, main_loss=float('nan')))+'\n')
    with pytest.raises(ValueError, match='Nonfinite'):
        guard.check()
    path.write_text('')
    missing = runner.DiagnosticGuard(tmp_path, [342])
    path.write_text(json.dumps(dict(seed=342, iteration=122, main_loss=1.))+'\n')
    with pytest.raises(ValueError, match='lost a round'):
        missing.check()


def stage_fixture(tmp_path):
    cfg = dict(seeds=[342], iterations=240, checkpoint_every=6, method='compact-opening-expert',
               root_repetitions_first=16, root_repetitions=4, max_opening_tv=.1)
    source = tmp_path/'tools/compact_regret.py'; source.parent.mkdir(); source.write_text("ARRAY_MAP={'w':'weight'}\n")
    manifest = {'files':{'tools/compact_regret.py':runner.digest(source)}}
    write(tmp_path/'source-manifest.json', manifest)
    path = tmp_path/'seed-342'; path.mkdir(); history = []
    for t in range(1,7):
        artifact = path/f'policy_{t:05d}.npz'; artifact.write_bytes(f'frozen {t}'.encode())
        history.append(dict(file=artifact.name,iteration=t,weight=t,sha256=runner.digest(artifact)))
    net = {'main.w':torch.tensor([1.,2.]),'opening.w':torch.tensor([2.,3.]),
           'main_scales':torch.tensor([2.,3.]),'opening_scale':torch.tensor(2.)}
    ck = dict(seed=342,iteration=6,config=cfg,source_manifest=manifest,net=net,snapshots=history,
        root_replay=dict(features=np.zeros((169,51)),count=np.full(169,36),mass=np.full(169,21.)))
    torch.save(ck,path/'latest.pt')
    write(path/'status.json',dict(status='complete',iteration=6,through=6,quality={'weighted_tv':.01}))
    write(path/'mixture_00006.json',dict(iteration=6,models=history))
    arrays = dict(weight=net['main.w'].numpy(),opening_weight=net['opening.w'].numpy(),
        main_scales=net['main_scales'].numpy(),opening_scale=net['opening_scale'].numpy(),
        training_revision=np.asarray('r1-cpu-v4'),regret_fit_method=np.asarray(cfg['method']))
    np.savez(path/'current_00006.npz',**arrays)
    return cfg,path,ck,arrays


def test_stage_requires_exact_checkpoint_and_export_parity(tmp_path):
    cfg,path,ck,arrays = stage_fixture(tmp_path)
    result = runner.verify_stage(tmp_path,6,cfg)
    assert result['passed'] and result['seeds'][0]['current_export_checkpoint_parity']
    arrays['weight'] = np.array([9.,9.],np.float32); np.savez(path/'current_00006.npz',**arrays)
    with pytest.raises(ValueError, match='export-parity'):
        runner.verify_stage(tmp_path,6,cfg)


def test_stage_rejects_nan_checkpoint_or_unfinished_worker(tmp_path):
    cfg,path,ck,_ = stage_fixture(tmp_path)
    ck['net']['main.w'][0] = float('nan'); torch.save(ck,path/'latest.pt')
    with pytest.raises(ValueError, match='Nonfinite'):
        runner.verify_stage(tmp_path,6,cfg)
    write(path/'status.json',dict(status='running',iteration=6,through=6))
    with pytest.raises(ValueError, match='finish the exact stage'):
        runner.verify_stage(tmp_path,6,cfg)


def test_evaluation_completion_cannot_accept_another_plan_or_card_budget(tmp_path):
    cfg = {'evaluation_deals':20000}
    plan = dict(config=cfg, study_plan_sha256='original-study-plan')
    write(tmp_path/'evaluation-plan.json',plan)
    summary = dict(complete=True,completed_evaluations=54,total_hands=6480000,
        raw_hands_verified=True,source_provenance_verified=True,config=cfg,
        evaluation_plan_sha256=runner.digest(tmp_path/'evaluation-plan.json'))
    write(tmp_path/'evaluation-summary.json',summary)
    assert runner.verify_evaluation_completion(tmp_path,cfg,'original-study-plan') == summary
    with pytest.raises(ValueError,match='matching plan'):
        runner.verify_evaluation_completion(tmp_path,cfg,'different-study-plan')
    wrong = dict(summary,config={'evaluation_deals':2000})
    write(tmp_path/'evaluation-summary.json',wrong)
    with pytest.raises(ValueError,match='matching plan'):
        runner.verify_evaluation_completion(tmp_path,cfg,'original-study-plan')


def test_cleanup_allows_evaluator_to_reap_separate_cell_after_five_seconds(tmp_path):
    # A real separate-session cell models the evaluator's worst-case cleanup:
    # the old five-second supervisor kill would leave this cell alive.
    source = tmp_path/'helper.py'; ready = tmp_path/'ready.json'
    source.write_text('''import json,os,signal,subprocess,sys,time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
def stop(*args):
 time.sleep(5.2)
 os.killpg(child.pid,signal.SIGTERM)
 child.wait(timeout=3)
 raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
Path(sys.argv[1]).write_text(json.dumps({'cell':child.pid}))
while True:time.sleep(.1)
''')
    helper = subprocess.Popen([sys.executable,str(source),str(ready)],start_new_session=True)
    child_pid = None
    try:
        deadline = time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        assert ready.exists()
        child_pid = json.loads(ready.read_text())['cell']
        token = runner.process_start(helper.pid)
        began = time.monotonic()
        runner.terminate_owned({'evaluation':dict(process=helper,start=token)},evaluator_grace=8)
        assert helper.returncode == 0
        assert time.monotonic()-began >= 5.0
        assert runner.process_start(child_pid) is None
    finally:
        import os,signal
        for pid in [helper.pid,child_pid]:
            if pid and runner.process_start(pid) is not None:
                try:os.killpg(pid,signal.SIGKILL)
                except ProcessLookupError:pass
        helper.wait(timeout=5)
