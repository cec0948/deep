"""Check current-process evidence and fail-closed raw-objective reporting."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fullhouse_v3_common import digest, process_start
from publish_refit_progress import CONDITIONS, collect, runner_alive
from run_raw_objective_probe import GATE_CHECKS


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path,complete=False):
    project=Path(__file__).resolve().parents[1]
    cfg=json.loads((project/'configs/raw-objective-probe-20261112.json').read_text())
    folder=tmp_path/'artifacts/refit-probe/20261112T070000000000Z'
    parent=tmp_path/'artifacts/compact-v4/20261112T032318825281Z'
    plan=dict(config=cfg,parent_iteration=120,parent=str(parent))
    save(folder/'config.json',cfg);save(folder/'plan.json',plan)
    status=dict(status='complete' if complete else 'running',phase='complete' if complete else 'objective_gate',
        updated_utc='2020-01-01T00:00:00+00:00',supervisor_pid=0,supervisor_start='0',
        plan_sha256=digest(folder/'plan.json'),completed_evaluations=54 if complete else 0,
        planned_evaluations=54,gate_passed=complete,running={})
    registry=dict(folder=str(folder),pid=0,process_start='0')
    save(folder/'status.json',status);save(tmp_path/'artifacts/refit-probe/current.json',registry)
    if complete:
        for seed in cfg['seeds']:
            candidate=folder/f'seed-{seed}/candidate.npz';candidate.parent.mkdir()
            candidate.write_bytes(f'policy-{seed}'.encode())
            fit=dict(complete=True,seed=seed,parent_iteration=120,parent=str(parent),
                resumable_selfplay=False,extra_steps=cfg['extra_steps'],
                files={'candidate.npz':dict(bytes=candidate.stat().st_size,sha256=digest(candidate))})
            save(candidate.parent/'results.json',fit)
        checks={key:True for key in GATE_CHECKS}
        gate=dict(complete=True,passed=True,config=cfg['gate_config'],method=cfg['method'],
                  gate=dict(passed=True,checks=checks))
        save(folder/'objective-gate/results.json',gate)
        conditions={key:True for key in CONDITIONS};conditions['public_lower_above_zero']=False
        summary=dict(complete=True,raw_hands_verified=True,source_provenance_verified=True,
            plan_sha256=digest(folder/'plan.json'),config=cfg,parent=str(parent),
            completed_evaluations=54,total_hands=648000,
            objective_gate=dict(passed=True,checks=checks,report_sha256=digest(folder/'objective-gate/results.json')),
            decisions={'raw-objective':dict(conditions=conditions,original_four_conditions_pass=False,
                                           deployment_authorized=False)})
        save(folder/'summary.json',summary)
    return folder,registry,status


def test_liveness_requires_real_pid_start_command_and_matching_output(tmp_path):
    folder,registry,status=fixture(tmp_path)
    script=tmp_path/'run_raw_objective_probe.py';script.write_text('import time\ntime.sleep(30)\n')
    child=subprocess.Popen([sys.executable,str(script),'--output',str(folder)])
    try:
        token=process_start(child.pid)
        registry.update(pid=child.pid,process_start=token)
        status.update(supervisor_pid=child.pid,supervisor_start=token)
        assert runner_alive(registry,status,folder)
        assert not runner_alive(registry,status,folder/'wrong-output')
        assert not runner_alive(dict(registry,process_start=str(int(token)+1)),status,folder)
        child.terminate();child.wait(timeout=5)
        assert not runner_alive(registry,status,folder)
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_old_running_record_never_proves_live_gate_or_fit(tmp_path):
    fixture(tmp_path)
    result=collect(tmp_path)
    assert result['status']=='interrupted_or_stale'
    assert result['phase']=='not_running'
    assert result['supervisor_alive'] is False
    assert result['objective_gate_passed'] is False
    assert result['completed_candidate_fits']==0
    assert result['planned_evaluations']==54
    assert result['status_age_seconds']>86400


def test_failed_gate_reports_failure_without_leaking_private_error(tmp_path):
    folder,_,status=fixture(tmp_path)
    private='https://drive.google.com/file/d/private-test-id/view'
    status.update(status='failed',phase='failed',failure_stage='objective_gate',error=private)
    save(folder/'status.json',status)
    result=collect(tmp_path)
    assert result['status']=='failed' and result['failure_stage']=='objective_gate'
    assert result['objective_gate_passed'] is False
    assert result['completed_candidate_fits']==0
    assert result['recorded_error_present'] is True
    assert 'do not automatically restart' in result['next_action']
    assert private not in json.dumps(result)


def test_complete_raw_probe_has_only_raw_decision_and_correct_hand_budget(tmp_path):
    fixture(tmp_path,complete=True)
    result=collect(tmp_path)
    assert result['status']=='complete' and result['summary_verified']
    assert result['supervisor_alive'] is False
    assert result['objective_gate_passed'] is True
    assert result['total_hands']==648000 and result['completed_evaluations']==54
    assert result['completed_candidate_fits']==3
    assert set(result['decisions'])=={'raw-objective'}
    assert result['decisions']['raw-objective']['deployment_authorized'] is False
    assert result['next_action'].startswith('paused_for_analysis')


@pytest.mark.parametrize('corruption',['status_gate_false','gate_report_false','gate_hash_changed'])
def test_complete_claim_without_consistent_gate_proof_is_unverified(tmp_path,corruption):
    folder,_,status=fixture(tmp_path,complete=True)
    if corruption=='status_gate_false':
        status['gate_passed']=False;save(folder/'status.json',status)
    else:
        path=folder/'objective-gate/results.json';gate=json.loads(path.read_text())
        if corruption=='gate_report_false':gate['passed']=False
        else:gate['unexpected_change']='after evaluation'
        save(path,gate)
    result=collect(tmp_path)
    assert result['status']=='unverified_complete'
    assert result['summary_verified'] is False
    assert result['decisions'] is None


def test_missing_candidate_fit_report_cannot_be_verified_complete(tmp_path):
    folder,_,_=fixture(tmp_path,complete=True)
    (folder/'seed-343/results.json').unlink()
    result=collect(tmp_path)
    assert result['status']=='unverified_complete'
    assert result['summary_verified'] is False
