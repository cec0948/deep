"""Current-process, completion-evidence and private-index publication checks."""
import copy
import datetime
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fullhouse_v3_common import digest, process_start
from publish_continuation_progress import collect, external_backup, runner_alive


def test_lightweight_backup_does_not_claim_legacy_resume_or_mixture():
    row=dict(kind='training-checkpoint',run='study',iterations={str(s):240 for s in [342,343,344]},
        bytes=100,status='complete',verified_remote_size=True,verified_owner_only=True,
        remote_checksum_available=False,recorded_at='2026-11-13T00:00:00+00:00',file_id='file',
        checkpoint_mode='compact-bounded-replay-recovery-v1',resumable_selfplay=False,
        exact_legacy_resume=False,historical_mixture_available=False,warm_start_state_verified=True)
    report=external_backup({'backups':[row]},'study',240)
    assert report['resumable_selfplay'] is False
    assert report['historical_mixture_available'] is False
    assert report['warm_start_state_verified'] is True
    row['exact_legacy_resume']=True
    with pytest.raises(ValueError,match='capabilities'):external_backup({'backups':[row]},'study',240)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def artifact(path, contents=b'local-evidence'):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(contents)
    return digest(path)


def fixture(root):
    project = Path(__file__).resolve().parents[1]
    cfg = json.loads((project/'configs/compact-learning-curve-20261112.json').read_text())
    folder = root/'artifacts/continuation-study/20261112T090000000000Z'
    campaign = folder/'campaign'/folder.name
    parent = root/'artifacts/compact-v4/20261112T032318825281Z'
    source = root/'scripts/audited_source.py'; artifact(source, b'frozen source')
    protected = {}
    for name in ('artifacts/compact-v4/current.json', 'artifacts/retraining-loop.json'):
        path=root/name;save(path, {'original':True});protected[str(path)]=digest(path)
    audit = dict(complete=True, no_blocking_implementation_faults=True, parent=str(parent),
        checks={'real_source_checked':True}, source_hashes={'scripts/audited_source.py':digest(source)})
    save(folder/'method-audit.json', audit)
    prep = dict(complete=True,parent=str(parent),campaign=str(campaign),original_registries_unchanged=True,
                method_audit_sha256=digest(folder/'method-audit.json'))
    save(folder/'preparation.json', prep);save(folder/'config.json',cfg)
    frozen=dict(protected);frozen[str(source)]=digest(source)
    for name in ('method-audit.json','preparation.json','config.json'):
        frozen[str(folder/name)]=digest(folder/name)
    plan=dict(config=cfg,parent_iteration=120,parent=str(parent),campaign=str(campaign),
        planned_evaluations=54,planned_hands=6480000,no_model_promotion=True,no_automatic_retry=True,
        frozen_files=frozen,protected_registry_files=protected,
        method_audit_sha256=digest(folder/'method-audit.json'),preparation_sha256=digest(folder/'preparation.json'))
    save(folder/'plan.json',plan)
    status=dict(status='running',phase='training',stage=180,updated_utc='2020-01-01T00:00:00+00:00',
        supervisor_pid=0,supervisor_start='0',plan_sha256=digest(folder/'plan.json'),parent_iteration=120,
        planned_iterations=240,completed_stages=[],campaign=str(campaign),running={},
        checkpoint_iterations={str(seed):120 for seed in cfg['seeds']},method_audit_passed=True,
        backups={},original_registries_unchanged=True)
    save(folder/'status.json',status)
    save(folder/'stage-ready.json',dict(completed_stages=[],checkpoints={'120':dict(iteration=120,parent_verified=True)}))
    registry=dict(folder=str(folder),pid=0,process_start='0');save(root/'artifacts/continuation-study/current.json',registry)
    for seed in cfg['seeds']:
        save(campaign/f'seed-{seed}/status.json',dict(iteration=120,through=120,status='complete'))
    return folder,campaign,cfg,plan,status,registry


def add_training_stage(root, context, stage):
    folder,campaign,cfg,plan,status,registry=context
    rows=[]
    for seed in cfg['seeds']:
        current=campaign/f'seed-{seed}/current_{stage:05d}.npz'
        mixture=campaign/f'seed-{seed}/mixture_{stage:05d}.json'
        current_sha=artifact(current,f'current{seed}/{stage}'.encode())
        save(mixture,dict(iteration=stage,models=[]))
        rows.append(dict(seed=seed,iteration=stage,snapshots=stage,current_sha256=current_sha,
            mixture_sha256=digest(mixture),finite_state=True,root_weighting_verified=True,
            pre_update_history_verified=True))
        save(campaign/f'seed-{seed}/status.json',dict(iteration=stage,through=stage,status='complete'))
    report=dict(complete=True,passed=True,stage=stage,seeds=rows)
    save(folder/f'stage-{stage}-verification.json',report)
    ready=json.loads((folder/'stage-ready.json').read_text());ready['completed_stages'].append(stage)
    ready['checkpoints'][str(stage)]=dict(iteration=stage,verification_sha256=digest(folder/f'stage-{stage}-verification.json'))
    save(folder/'stage-ready.json',ready)
    status['completed_stages'].append(stage);status['stage']=stage
    status['checkpoint_iterations']={str(seed):stage for seed in cfg['seeds']}
    save(folder/'status.json',status)
    archive=root/'artifacts/exports'/f'compact-v4-{folder.name}-snapshot-20261112T090000000000Z-stage{stage}.tar.gz'
    sha=artifact(archive,f'archive{stage}'.encode())
    save(folder/f'stage-{stage}-backup.json',dict(archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha,iterations={str(seed):stage for seed in cfg['seeds']},verified_files=12,resumable_selfplay=True))


def comparison(cfg, positive=True):
    mean=10. if positive else -10.
    return dict(mean=mean,lower=mean-1.,upper=mean+1.,ci95_half_width=1.,
        per_seed=[dict(seed=seed,mean=mean) for seed in cfg['seeds']])


def evaluation_summary(cfg, evaluation_sha, through):
    decisions={};paired={}
    for stage in cfg['stages']:
        if stage>through:continue
        for kind in cfg['variants']:
            name=f'{kind}-{stage}'
            for field in cfg['evaluation_fields']:
                paired[f'{field}/{name}-minus-current-120']=comparison(cfg)
            conditions=dict(public_lower_above_zero=True,at_least_two_public_seeds_improve=True,
                mixed_lower_above_minus_25=True,noregrets_lower_above_minus_50=True)
            decisions[name]=dict(conditions=conditions,original_four_conditions_pass=True,deployment_authorized=False)
    cells=36 if through==180 else 54
    return dict(complete=True,config=cfg,through_iteration=through,evaluation_plan_sha256=evaluation_sha,
        raw_hands_verified=True,source_provenance_verified=True,completed_evaluations=cells,
        total_hands=cells*20000*6,independent_card_blocks=20000,decisions=decisions,paired_changes=paired)


def add_complete_evaluation(context):
    folder,campaign,cfg,plan,status,registry=context
    jobs=[];proofs={}
    for stage in cfg['evaluation_iterations']:
        for kind in cfg['variants']:
            for seed in cfg['seeds']:
                model=campaign/f'seed-{seed}'/f'{kind}_{stage:05d}.{"json" if kind=="mixture" else "npz"}'
                if not model.exists():artifact(model,f'baseline{seed}/{kind}'.encode())
                for field in cfg['evaluation_fields']:
                    name=f'{field}-seed-{seed}-{kind}-{stage}'
                    job=dict(id=name,field=field,variant=kind,iteration=stage,seed=seed,model=str(model),source=str(campaign/'source'))
                    jobs.append(job)
                    expected=dict(candidate=digest(model),evaluator_sha256='evaluator',runtime_sha256='runtime',
                        engine_sha256='engine',source_features_sha256='features',opponents={'public':'publicmodel'})
                    proof_path=folder/'evaluation'/(name+'.provenance.json')
                    save(proof_path,dict(job=job,expected=expected,policy_files={str(model):digest(model)}))
                    row=dict(complete=True,field=field,hands=120000,deals=20000,deal_seed=cfg['evaluation_deal_seed'],
                        candidate=str(model),candidate_sha256=digest(model),**{k:v for k,v in expected.items() if k!='candidate'})
                    cell=folder/'evaluation'/name;save(cell/'results.json',row);artifact(cell/'hands.jsonl.gz',name.encode())
                    proofs[name]=dict(results_sha256=digest(cell/'results.json'),hands_sha256=digest(cell/'hands.jsonl.gz'),
                                      provenance_sha256=digest(proof_path))
    eval_plan=dict(config=cfg,study_plan_sha256=digest(folder/'plan.json'),jobs=jobs,frozen_files={})
    save(folder/'evaluation-plan.json',eval_plan);eval_sha=digest(folder/'evaluation-plan.json')
    save(folder/'evaluation-status.json',dict(status='complete',updated_utc='2026-11-12T09:00:00+00:00',
        evaluator_pid=0,evaluator_start='0',evaluation_plan_sha256=eval_sha,completed_evaluations=54,
        planned_evaluations=54,verified_hands=6480000,finished=list(proofs),pending=[],running={},
        completed_checkpoint_evaluations=[180,240],verified_result_files=proofs))
    for stage in cfg['stages']:
        save(folder/f'stage-{stage}-evaluation.json',evaluation_summary(cfg,eval_sha,stage))
    final=evaluation_summary(cfg,eval_sha,240);save(folder/'evaluation-summary.json',final)
    summary=dict(complete=True,plan_sha256=digest(folder/'plan.json'),config=cfg,parent=plan['parent'],
        campaign=str(campaign),completed_stages=[180,240],selfplay_iteration=240,parent_iteration=120,
        additional_selfplay_iterations=120,original_registries_unchanged=True,method_audit_passed=True,
        raw_hands_verified=True,source_provenance_verified=True,completed_evaluations=54,total_hands=6480000,
        evaluation_summary_sha256=digest(folder/'evaluation-summary.json'),no_model_promotion=True,no_automatic_retry=True)
    save(folder/'summary.json',summary)
    status.update(status='complete',phase='complete');save(folder/'status.json',status)


def complete_fixture(root):
    context=fixture(root)
    for stage in [180,240]:add_training_stage(root,context,stage)
    add_complete_evaluation(context)
    return context


def test_actual_pid_token_command_output_and_zombie_liveness(tmp_path):
    folder,_,_,_,status,registry=fixture(tmp_path)
    script=tmp_path/'scripts/run_compact_learning_curve.py'
    script.write_text('import time\ntime.sleep(30)\n')
    child=subprocess.Popen([sys.executable,str(script),'--output',str(folder)])
    try:
        start=process_start(child.pid);registry.update(pid=child.pid,process_start=start)
        status.update(supervisor_pid=child.pid,supervisor_start=start)
        assert runner_alive(tmp_path,registry,status,folder)
        assert not runner_alive(tmp_path,dict(registry,process_start=str(int(start)+1)),status,folder)
        assert not runner_alive(tmp_path,registry,status,folder/'wrong')
        child.terminate();child.wait(timeout=5)
        assert not runner_alive(tmp_path,registry,status,folder)
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


def test_old_saved_running_record_is_not_live_and_private_values_do_not_leak(tmp_path):
    context=fixture(tmp_path);folder,_,_,_,status,_=context
    status.update(error='https://drive.google.com/private-secret',failure_stage='training')
    save(folder/'status.json',status)
    result=collect(tmp_path)
    assert result['status']=='interrupted_or_stale' and not result['supervisor_alive']
    assert result['status_age_seconds']>86400
    assert not result['study_complete'] and not result['training_complete']
    assert result['planned_hands']==6480000 and result['independent_card_blocks']==20000
    assert 'private-secret' not in json.dumps(result) and str(tmp_path) not in json.dumps(result)


def test_completed_stage_is_not_completed_large_study_or_offsite_backup(tmp_path):
    context=fixture(tmp_path);add_training_stage(tmp_path,context,180)
    result=collect(tmp_path)
    assert result['completed_training_stages']==[180] and not result['study_complete']
    assert result['training_stages']['180']['checkpoint_verified']
    assert result['training_stages']['180']['local_snapshot_verified']
    assert not result['training_stages']['180']['evaluation_complete']
    assert result['external_checkpoint'] is None


def test_all_checkpoint_stages_without_final_eval_are_not_study_complete(tmp_path):
    context=fixture(tmp_path)
    for stage in [180,240]:add_training_stage(tmp_path,context,stage)
    folder,_,_,_,status,_=context;status.update(status='complete',phase='complete');save(folder/'status.json',status)
    result=collect(tmp_path)
    assert result['training_complete'] and not result['study_complete']
    assert result['status']=='unverified_complete'


def test_full_study_requires_all54_verified_results_and_adoption_decisions(tmp_path):
    complete_fixture(tmp_path);result=collect(tmp_path)
    assert result['status']=='complete' and result['study_complete']
    assert result['total_hands']==6480000 and result['completed_evaluations']==54
    assert not result['supervisor_alive'] and not result['original_model_replaced']
    assert set(result['adoption_decisions'])=={'current-180','mixture-180','current-240','mixture-240'}
    assert all(v['deployment_authorized'] is False for v in result['adoption_decisions'].values())


@pytest.mark.parametrize('corruption',['missing_result','changed_raw_hands','changed_proof','wrong_gate','changed_source'])
def test_completion_evidence_corruption_fails_closed(tmp_path,corruption):
    context=complete_fixture(tmp_path);folder=context[0]
    name='public-seed-342-current-120';cell=folder/'evaluation'/name
    if corruption=='missing_result':(cell/'results.json').unlink()
    elif corruption=='changed_raw_hands':(cell/'hands.jsonl.gz').write_bytes(b'changed after verifier')
    elif corruption=='changed_proof':(folder/'evaluation'/(name+'.provenance.json')).write_text('{}')
    elif corruption=='changed_source':(tmp_path/'scripts/audited_source.py').write_text('changed')
    else:
        path=folder/'stage-240-evaluation.json';row=json.loads(path.read_text())
        row['decisions']['current-240']['conditions']['public_lower_above_zero']=False;save(path,row)
    with pytest.raises((ValueError,FileNotFoundError)):collect(tmp_path)


def test_gate_comparison_values_cannot_override_original_conditions(tmp_path):
    folder,*_=complete_fixture(tmp_path)
    path=folder/'stage-180-evaluation.json';row=json.loads(path.read_text())
    row['paired_changes']['public/current-180-minus-current-120']['lower']=-100
    save(path,row)
    with pytest.raises(ValueError,match='gates disagree'):collect(tmp_path)


def test_backup_index_excludes_deleted_partial_wrongrun_and_private_fields():
    base=dict(kind='training-checkpoint',run='study',iterations={'342':180,'343':180,'344':180},
        status='complete',verified_remote_size=True,verified_owner_only=True,bytes=12,
        recorded_at='2026-11-12T08:00:00+00:00',remote_checksum_available=False,
        file_id='PRIVATE-FILE-ID',url='https://drive.google.com/PRIVATE-FILE-ID')
    partial=dict(base,parts=[dict(bytes=12,file_id='PRIVATE-PART',verified_remote_size=True,verified_owner_only=False)],
                 manifest_file_id='PRIVATE-MANIFEST',manifest_bytes=1)
    records=[dict(base,deleted_remote=True),dict(base,run='another'),dict(base,verified_remote_size=False),partial]
    assert external_backup({'backups':records},'study',180) is None
    records.append(base);result=external_backup({'backups':records},'study',180)
    assert result['iterations']['342']==180 and result['verified_owner_only']
    assert 'PRIVATE' not in json.dumps(result) and 'file_id' not in result
    assert 'no Drive request' in result['verification_scope']


def test_completed_claim_without_final_summary_is_unverified(tmp_path):
    folder,*_=complete_fixture(tmp_path);(folder/'summary.json').unlink()
    result=collect(tmp_path)
    assert result['status']=='unverified_complete' and not result['study_complete']
