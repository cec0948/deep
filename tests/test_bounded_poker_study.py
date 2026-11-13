import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from bounded_poker_study_support import check_learning_config,validate_config
from evaluate_bounded_poker_study import evaluation_parallelism,jobs_for,summarize
from train_compact_bounded import load_current,prune_checkpoints
from fullhouse_v3_common import digest
from publish_bounded_poker_progress import collect
from bounded_poker_study_support import snapshot
from compact_lightweight_backup import equal_state,load_state,save_state,verify_archive
from test_compact_lightweight_backup import fixture as state_fixture


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/poker-bounded-continuation-20261113.json').read_text())


def test_predeclared_design_has36_current_only_cells_and4320000_hands(tmp_path):
    cfg=config();validate_config(cfg);jobs=jobs_for(tmp_path,cfg)
    assert len(jobs)==36 and len({j['id'] for j in jobs})==36
    assert {j['iteration'] for j in jobs}=={120,240,300,360}
    assert all(j['variant']=='current' and j['model'].endswith('.npz') for j in jobs)
    assert len(jobs)*cfg['evaluation_deals']*6==4320000


def test_stale_complete_label_does_not_oversubscribe_live_training(monkeypatch):
    import evaluate_bounded_poker_study as evaluator
    launch=dict(status='complete',pids={'342':123},process_starts={'342':'456'})
    monkeypatch.setattr(evaluator,'process_matches',lambda *args:True)
    assert evaluation_parallelism(launch)==1
    monkeypatch.setattr(evaluator,'process_matches',lambda *args:False)
    assert evaluation_parallelism(launch)==3
    launch['status']='running';assert evaluation_parallelism(launch)==1


@pytest.mark.parametrize('field,value',[('stages',[300,420]),('variants',['current','mixture']),
    ('evaluation_deals',2000),('promotion','auto'),('max_seconds',14401)])
def test_unbounded_or_changed_design_is_rejected(field,value):
    cfg=config();cfg[field]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_metadata_changes_do_not_change_learning_but_sample_changes_are_rejected():
    before=dict(traversals=256,buffer_capacity=100000,learning_rate=.001,iterations=240)
    after=dict(before,iterations=360,initial_iteration=240,stages=[300,360])
    check_learning_config(before,after)
    after['traversals']=512
    with pytest.raises(ValueError):check_learning_config(before,after)


def test_checkpoint_pruning_keeps_parent_stages_and_latest_without_touching_unknown(tmp_path):
    parent=tmp_path/'checkpoints';parent.mkdir()
    for n in [240,246,252,300,306]:
        d=parent/f'iter-{n:05d}';d.mkdir();(d/'data').write_text('checkpoint')
    (parent/'unrelated').mkdir()
    prune_checkpoints(tmp_path,{240,300,306})
    assert {p.name for p in parent.iterdir()}=={'iter-00240','iter-00300','iter-00306','unrelated'}


def test_checkpoint_pointer_cannot_escape_its_directory(tmp_path):
    folder=tmp_path/'seed-342';folder.mkdir()
    (folder/'current.json').write_text(json.dumps({'directory':'../../escape','iteration':240,'files':{}}))
    with pytest.raises(ValueError,match='Unsafe'):load_current(tmp_path,342)


def test_statistics_use_card_blocks_and_original120_adoption_baseline():
    cfg=config();cfg['evaluation_deals']=3
    rows={}
    for iteration in cfg['evaluation_iterations']:
        for seed in cfg['seeds']:
            for field in cfg['evaluation_fields']:
                values=[0.,0.,0.] if iteration==120 else [0.,30.,60.] if iteration==240 else [30.,60.,90.]
                rows[f'{field}-seed-{seed}-current-{iteration}']={'deal_blocks':values}
    result=summarize(rows,cfg,360)
    delta=result['paired_changes']['public/current-360-minus-current-120']
    assert delta['mean']==60
    assert delta['ci95_half_width']==pytest.approx(1.96*30/np.sqrt(3))
    assert result['independent_card_blocks']==3
    assert result['decisions']['current-360']['original_four_conditions_pass'] is True
    assert result['decisions']['current-360']['deployment_authorized'] is False
    assert result['paired_changes']['public/current-360-minus-current-240']['mean']==30


def test_interim_summary_does_not_claim_final_checkpoint():
    cfg=config();cfg['evaluation_deals']=2;rows={}
    for iteration in [120,240,300]:
        for seed in cfg['seeds']:
            for field in cfg['evaluation_fields']:
                rows[f'{field}-seed-{seed}-current-{iteration}']={'deal_blocks':[float(iteration),float(iteration)]}
    result=summarize(rows,cfg,300)
    assert result['completed_evaluations']==27
    assert set(result['decisions'])=={'current-300'}
    assert not any('360' in key for key in result['paired_changes'])


def progress_fixture(root):
    cfg=config();study=root/'artifacts/bounded-continuation/20261113T000000000000Z';study.mkdir(parents=True)
    source=root/'source.txt';source.write_text('frozen')
    def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    save(study/'config.json',cfg)
    save(study/'plan.json',dict(config=cfg,frozen_files={str(source):digest(source)},no_model_promotion=True,
        no_automatic_retry=True,planned_evaluations=36,planned_hands=4320000))
    save(root/'artifacts/bounded-continuation/current.json',dict(folder=str(study),pid=0,process_start='0'))
    state=dict(status='running',phase='training',stage=300,updated_utc='2020-01-01T00:00:00+00:00',
        plan_sha256=digest(study/'plan.json'),supervisor_pid=0,supervisor_start='0',completed_stages=[],running={})
    save(study/'status.json',state);save(study/'stage-ready.json',dict(checkpoints={}))
    save(root/'artifacts/offsite-backups.json',dict(backups=[]))
    for seed in cfg['seeds']:
        save(study/f'campaign/seed-{seed}/current.json',dict(iteration=240))
        save(study/f'campaign/seed-{seed}/status.json',dict(status='complete'))
    return study,state,source


def test_stale_running_record_is_not_reported_as_live(tmp_path):
    progress_fixture(tmp_path);result=collect(tmp_path)
    assert result['status']=='interrupted_or_stale' and not result['supervisor_alive']
    assert result['study_complete'] is False and result['completed_evaluations']==0


def test_false_completion_and_frozen_source_corruption_are_rejected(tmp_path):
    study,state,source=progress_fixture(tmp_path);state['status']='complete'
    (study/'status.json').write_text(json.dumps(state))
    result=collect(tmp_path);assert result['status']=='unverified_complete' and not result['study_complete']
    source.write_text('changed')
    with pytest.raises(ValueError,match='Frozen'):collect(tmp_path)


def test_compact_stage_snapshot_restores_all_three_seeds_without_policy_history(tmp_path,monkeypatch):
    import bounded_poker_study_support as support
    from fullhouse_v3_common import atomic_json
    monkeypatch.setattr(support,'ROOT',tmp_path)
    exports=tmp_path/'artifacts/exports';exports.mkdir(parents=True)
    scripts=tmp_path/'scripts';scripts.mkdir();(scripts/'compact_lightweight_backup.py').write_text('restore helper')
    study=tmp_path/'artifacts/bounded-continuation/20261113T000000000000Z';campaign=study/'campaign'
    campaign.mkdir(parents=True);(campaign/'source').mkdir();(campaign/'tools').mkdir()
    atomic_json(campaign/'config.json',{'seeds':[342,343,344]});atomic_json(campaign/'source-manifest.json',{})
    for name in ['config.json','plan.json','preparation.json','stage-300-verification.json']:
        atomic_json(study/name,{'passed':True,'stage':300})
    for seed in [342,343,344]:
        folder=campaign/f'seed-{seed}';folder.mkdir();(folder/'worker.lock').touch()
        state=state_fixture(seed);state['iteration']=300
        commit=lambda p:np.savez(p,weight=np.ones(2,np.float32))
        from train_compact_bounded import commit_checkpoint
        commit_checkpoint(campaign,seed,state,commit,{300})
    result=snapshot(study,300)
    assert result['checkpoint_mode']=='compact-bounded-replay-recovery-v1' and result['warm_start_state_verified']
    assert result['iterations']=={'342':300,'343':300,'344':300}
    out=tmp_path/'restored';verify_archive(result['archive'],out)
    expected=state_fixture(343);expected['iteration']=300
    assert equal_state(expected,load_state(out/'seed-343'))
