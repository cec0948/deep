import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_rich_poker_study as study


def config():
    return dict(seeds=[342,343,344],stages=[12,48,96],iterations=96,parent_iteration=360,
        traversals=64,buffer_capacity=5000,main_steps=256,batch_size=128,
        checkpoint_every=6,learning_rate=.001,prerequisites_passed=True,
        evaluation_fields=['public','mixed','noregrets'],evaluation_deals=20000,
        evaluation_deal_seed=2035000000,source='/actual/source',
        teachers={str(seed):dict(path=f'/teacher-{seed}.npz',sha256='a'*64) for seed in study.SEEDS},
        baselines120={str(seed):dict(path=f'/baseline-{seed}.npz',sha256='b'*64) for seed in study.SEEDS})


def test_fixed_study_is_new96_family_not_legacy456():
    cfg=config();study.validate_config(cfg)
    jobs=study.evaluation_jobs(Path('/study'))
    assert len(jobs)==45 and len({j['id'] for j in jobs})==45
    assert {j['label'] for j in jobs}=={'original120','parent360','rich12','rich48','rich96'}
    assert len(jobs)*20000*6==5400000
    assert all('456' not in j['id'] for j in jobs)


@pytest.mark.parametrize('key,value',[('stages',[12,48,120]),('iterations',456),
    ('prerequisites_passed',False),('evaluation_deals',2000),('evaluation_deal_seed',2026200000),
    ('promotion','auto'),('automatic_retry',True),('max_seconds',21601),
    ('training_seconds_per_stage',5401),('evaluation_seconds_per_cell',2401)])
def test_changed_design_or_unbounded_work_is_rejected(key,value):
    cfg=config();cfg[key]=value
    with pytest.raises(ValueError):study.validate_config(cfg)


def test_shared_neural_gate_does_not_accept_tabular_success(tmp_path):
    path=tmp_path/'report.json'
    rows=[dict(seed=s,shared_neural_approximator=True,tabular_approximator_used=False)
          for s in [71,72,73]]
    report=dict(status='complete',all_three_neural_gates_passed=True,seed_results=rows)
    path.write_text(json.dumps(report));assert study.validate_gate(path)==report
    report['seed_results'][0]['tabular_approximator_used']=True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='shared-neural'):study.validate_gate(path)
    report['seed_results'][0]['tabular_approximator_used']=False
    report['all_three_neural_gates_passed']=False;path.write_text(json.dumps(report))
    with pytest.raises(ValueError):study.validate_gate(path)


def test_alive_workers_and_interstage_gaps_keep_one_evaluator(monkeypatch,tmp_path):
    child=SimpleNamespace(pid=123,poll=lambda:None)
    training={342:dict(process=child,start='44')}
    monkeypatch.setattr(study,'command_matches',lambda *args:True)
    assert study.evaluation_parallelism(training,tmp_path,True)==1
    monkeypatch.setattr(study,'command_matches',lambda *args:False)
    assert study.evaluation_parallelism(training,tmp_path,True)==3
    assert study.evaluation_parallelism(training,tmp_path,False)==1
    assert study.evaluation_parallelism({},tmp_path,False)==1


def test_status_checkpoint_commit_race_is_not_reported_as_failure(tmp_path):
    for seed in study.SEEDS:
        folder=tmp_path/f'seed-{seed}';folder.mkdir()
        (folder/'status.json').write_text(json.dumps(dict(status='running',iteration=5,
             replay_rows=[100]*6,fitting={'loss':1.})))
        (folder/'current.json').write_text(json.dumps(dict(iteration=6)))
        (folder/'diagnostics.jsonl').write_text(json.dumps(dict(iteration=6,fitting={'loss':.9}))+'\n')
    rows=study.seed_progress(tmp_path,{})
    assert all(v['new_family_iteration']==6 and v['checkpoint_iteration']==6 for v in rows.values())
    assert all(v['worker_alive'] is False for v in rows.values())


def test_failed_worker_and_nonfinite_diagnostics_stop_monitor(tmp_path):
    for seed in study.SEEDS:
        folder=tmp_path/f'seed-{seed}';folder.mkdir()
        (folder/'status.json').write_text(json.dumps(dict(status='failed',iteration=1)))
    with pytest.raises(RuntimeError,match='worker recorded failure'):study.seed_progress(tmp_path,{})
    assert all(v['recorded_status']=='failed' for v in study.seed_progress(tmp_path,{},strict=False).values())
    with pytest.raises(ValueError,match='Nonfinite'):study.finite({'nested':[float('nan')]})


def test_partial_in_progress_diagnostic_tail_waits_but_committed_corruption_fails(tmp_path):
    for seed in study.SEEDS:
        folder=tmp_path/f'seed-{seed}';folder.mkdir()
        (folder/'status.json').write_text(json.dumps(dict(status='running',iteration=5)))
        (folder/'diagnostics.jsonl').write_text('{"iteration":6')
    assert study.seed_progress(tmp_path,{})['342']['new_family_iteration']==5
    (tmp_path/'seed-342/diagnostics.jsonl').write_text('{"iteration":6\n')
    with pytest.raises(ValueError,match='malformed committed'):
        study.seed_progress(tmp_path,{})


def test_shared_card_statistics_use_original120_and_separate_parent360():
    rows={}
    values={'original120':[0.,0.,0.], 'parent360':[0.,30.,60.],
            'rich12':[10.,40.,70.], 'rich48':[20.,50.,80.], 'rich96':[30.,60.,90.]}
    for job in study.evaluation_jobs(Path('/study')):
        rows[job['id']]={'deal_blocks':values[job['label']]}
    result=study.summarize_evaluations(rows,deals=3)
    assert result['completed_evaluations']==45 and result['independent_card_blocks']==3
    assert result['total_hands']==810
    delta=result['paired_changes']['public/rich96-minus-original120']
    assert delta['mean']==60 and delta['ci95_half_width']==pytest.approx(1.96*30/3**.5)
    assert result['paired_changes']['public/rich96-minus-parent360']['mean']==30
    assert result['decisions']['rich96']['original_four_conditions_pass'] is True
    assert result['model_promoted'] is False
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError,match='All45'):study.summarize_evaluations(rows,deals=3)


def test_ready_policy_hash_is_required_and_unready_stage_waits(tmp_path):
    policy=tmp_path/'model.json';policy.write_text('{}')
    ready=dict(policies={'parent360':{'342':dict(path=str(policy),sha256=study.digest(policy))}})
    (tmp_path/'stage-ready.json').write_text(json.dumps(ready))
    assert study.policy_for_job(tmp_path,dict(label='rich12',seed=342)) is None
    assert study.policy_for_job(tmp_path,dict(label='parent360',seed=342))['path']==str(policy)
    policy.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='policy changed'):
        study.policy_for_job(tmp_path,dict(label='parent360',seed=342))
