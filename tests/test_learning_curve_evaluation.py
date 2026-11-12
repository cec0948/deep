"""Fixed-budget learning-curve uncertainty, adoption and artifact safety."""
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_compact_learning_curve as evaluator
from fullhouse_v3_common import digest, process_start


def config():
    return dict(seeds=[342,343,344], evaluation_iterations=[120,180,240],
        stages=[180,240],variants=['current','mixture'],
        evaluation_fields=['public','mixed','noregrets'], evaluation_deals=20000,
        evaluation_deal_seed=2026160000, evaluation_seconds_per_cell=2400,
        max_seconds=14400, promotion='none')


def test_fixed_tenfold_card_plan_has54_cells_and6480000_hands(tmp_path):
    cfg=config(); evaluator.validate_config(cfg)
    jobs=evaluator.evaluation_jobs(tmp_path,cfg)
    assert len(jobs)==54 and len({row['id'] for row in jobs})==54
    assert len(jobs)*cfg['evaluation_deals']*6==6480000
    for key,value in [('evaluation_deals',2000),('seeds',[342]),('promotion','automatic'),
                       ('variants',['current']),('stages',[240])]:
        changed=copy.deepcopy(cfg);changed[key]=value
        with pytest.raises(ValueError):evaluator.validate_config(changed)


def synthetic_rows(cfg):
    return {f'{field}-seed-{seed}-{kind}-{iteration}':dict(deal_blocks=[0.]*cfg['evaluation_deals'],hands=cfg['evaluation_deals']*6)
            for iteration in cfg['evaluation_iterations'] for kind in evaluator.KINDS
            for seed in cfg['seeds'] for field in cfg['evaluation_fields']}


def test_card_block_is_not_multiplied_by_seed_count_and_all_learning_curve_pairs_exist():
    cfg=config();cfg['evaluation_deals']=4;rows=synthetic_rows(cfg);change=[0.,100.,0.,100.]
    for seed in cfg['seeds']:
        rows[f'public-seed-{seed}-current-240']['deal_blocks']=change
    result=evaluator.summarize(rows,cfg,240)
    score=result['paired_changes']['public/current-240-minus-current-120']
    assert score['ci95_half_width']==pytest.approx(1.96*statistics.stdev(change)/2)
    assert 'public/current-240-minus-current-180' in result['paired_changes']
    assert 'public/mixture-180-minus-current-180' in result['paired_changes']
    assert result['independent_card_blocks']==4
    assert result['total_hands']==54*4*6


def test_mixture_adoption_uses_original_current_not_a_weaker_mixture_baseline():
    cfg=config();cfg['evaluation_deals']=4;rows=synthetic_rows(cfg)
    for seed in cfg['seeds']:
        rows[f'public-seed-{seed}-mixture-120']['deal_blocks']=[-100.]*4
        rows[f'public-seed-{seed}-mixture-180']['deal_blocks']=[-50.]*4
        rows[f'public-seed-{seed}-current-180']['deal_blocks']=[10.]*4
    result=evaluator.summarize(rows,cfg,180)
    assert result['paired_changes']['public/mixture-180-minus-mixture-120']['mean']==50
    assert result['decisions']['mixture-180']['original_four_conditions_pass'] is False
    assert result['decisions']['current-180']['original_four_conditions_pass'] is True
    assert all(row['deployment_authorized'] is False for row in result['decisions'].values())
    assert result['completed_evaluations']==36
    assert 'Fixed predeclared240' in result['continuation_decision']


def test_original_mixed_and_noregrets_nonregression_thresholds_are_not_weakened():
    cfg=config();cfg['evaluation_deals']=4;rows=synthetic_rows(cfg)
    for seed in cfg['seeds']:
        rows[f'public-seed-{seed}-current-240']['deal_blocks']=[10.]*4
        rows[f'mixed-seed-{seed}-current-240']['deal_blocks']=[-25.]*4
        rows[f'noregrets-seed-{seed}-current-240']['deal_blocks']=[-50.]*4
    assert evaluator.summarize(rows,cfg,240)['decisions']['current-240']['original_four_conditions_pass']
    for seed in cfg['seeds']:
        rows[f'mixed-seed-{seed}-current-240']['deal_blocks']=[-25.01]*4
    assert not evaluator.summarize(rows,cfg,240)['decisions']['current-240']['original_four_conditions_pass']


def test_policy_readiness_requires_a_committed_seed_status(tmp_path):
    candidate=tmp_path/'current_00180.npz';candidate.write_bytes(b'candidate')
    job=dict(model=str(candidate),iteration=180)
    (tmp_path/'status.json').write_text(json.dumps(dict(iteration=174,status='running')))
    assert not evaluator.policy_ready(job)
    (tmp_path/'status.json').write_text(json.dumps(dict(iteration=180,status='complete')))
    assert evaluator.policy_ready(job)
    candidate.unlink()
    with pytest.raises(ValueError,match='missing'):evaluator.policy_ready(job)


def mixture_fixture(tmp_path):
    models=[]
    for iteration in [1,2]:
        p=tmp_path/f'policy_{iteration:05d}.npz';p.write_bytes(f'policy{iteration}'.encode())
        models.append(dict(iteration=iteration,weight=iteration,file=p.name,sha256=digest(p)))
    path=tmp_path/'mixture_00002.json'
    spec=dict(model_type='compact_snapshot_mixture',training_revision='r1-cpu-v4',iteration=2,models=models)
    path.write_text(json.dumps(spec))
    return dict(model=str(path),iteration=2,variant='mixture'),spec


def test_all_mixture_snapshots_are_checked_before_sampling(tmp_path):
    job,spec=mixture_fixture(tmp_path)
    assert len(evaluator.freeze_policy(job))==3
    (tmp_path/spec['models'][1]['file']).write_bytes(b'corrupt unsampled snapshot')
    with pytest.raises(ValueError,match='snapshot changed'):evaluator.freeze_policy(job)


def test_mixture_weights_and_path_escape_cannot_change_policy_silently(tmp_path):
    job,spec=mixture_fixture(tmp_path);path=Path(job['model'])
    changed=copy.deepcopy(spec);changed['models'][1]['weight']=1;path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='weighted'):evaluator.freeze_policy(job)
    spec['models'][1]['file']='../outside.npz';path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='escapes'):evaluator.freeze_policy(job)


def test_evaluation_parallelism_does_not_trust_a_stale_complete_label(monkeypatch):
    monkeypatch.setattr(evaluator,'process_matches',lambda pid,start,script=None:pid==11)
    launch=dict(status='complete',pids={'342':11},process_starts={'342':'100'})
    assert evaluator.evaluation_parallelism(launch)==1
    launch['pids']={'342':12};assert evaluator.evaluation_parallelism(launch)==3
    launch['status']='running';assert evaluator.evaluation_parallelism(launch)==1
    launch['status']='failed'
    with pytest.raises(ValueError):evaluator.evaluation_parallelism(launch)


def test_liveness_requires_pid_start_token_and_script_name():
    pid=os.getpid();start=process_start(pid)
    assert evaluator.process_matches(pid,start)
    assert not evaluator.process_matches(pid,str(int(start)+1))
    assert not evaluator.process_matches(pid,start,'missing-learning-curve-runner.py')
    assert not evaluator.process_matches(pid,start,output='/a/different/study')


def test_new_policy_waits_for_the_full_verified_stage_and_pins_its_hash(tmp_path):
    ready=tmp_path/'stage-ready.json';ready.write_text(json.dumps(dict(checkpoints={})))
    job=dict(iteration=180,seed=342,variant='current',model=str(tmp_path/'current_00180.npz'))
    assert evaluator.verified_stage_evidence(tmp_path,tmp_path/'parent',job) is None
    report=dict(complete=True,passed=True,stage=180,seeds=[dict(seed=seed,iteration=180,
        current_sha256='current'+str(seed),mixture_sha256='mixture'+str(seed),checkpoint_sha256='checkpoint'+str(seed)) for seed in [342,343,344]])
    path=tmp_path/'stage-180-verification.json';path.write_text(json.dumps(report))
    ready.write_text(json.dumps(dict(checkpoints={'180':dict(iteration=180,verification_sha256=digest(path))})))
    proof=evaluator.verified_stage_evidence(tmp_path,tmp_path/'parent',job)
    assert proof['candidate_sha256']=='current342' and proof['checkpoint_sha256']=='checkpoint342'
    report['passed']=False;path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='verification changed'):
        evaluator.verified_stage_evidence(tmp_path,tmp_path/'parent',job)


def test_original120_baseline_evidence_is_pinned_to_parent_not_the_new_campaign(tmp_path):
    parent=tmp_path/'parent';seed=parent/'seed-342';seed.mkdir(parents=True)
    original=seed/'current_00120.npz';original.write_bytes(b'original baseline')
    (tmp_path/'stage-ready.json').write_text(json.dumps(dict(checkpoints={'120':dict(iteration=120,parent_verified=True)})))
    job=dict(iteration=120,seed=342,variant='current',model=str(tmp_path/'candidate/current_00120.npz'))
    proof=evaluator.verified_stage_evidence(tmp_path,parent,job)
    assert proof['candidate_sha256']==digest(original)


def raw_fixture(tmp_path):
    cfg=dict(evaluation_deals=2,evaluation_deal_seed=123)
    model=tmp_path/'current.npz';model.write_bytes(b'frozen model')
    job=dict(model=str(model),field='public',variant='current')
    proof=dict(candidate=digest(model),evaluator_sha256='e',runtime_sha256='r',engine_sha256='b',source_features_sha256='s',opponents={})
    hands=[];blocks=[]
    for deal in range(2):
        chips=[]
        for hero in range(6):
            payoff=(deal+1)*10+hero;utilities=[0]*6;utilities[hero]=payoff;utilities[(hero+1)%6]=-payoff
            hand_seed=int.from_bytes(hashlib.sha256(f'compact-mixture:123:{deal}:{hero}'.encode()).digest()[:8],'big')
            hands.append(dict(deal=deal,hero=hero,seed=123+deal,chips=payoff,utilities=utilities,hand_seed=hand_seed,policy_iteration=None))
            chips.append(payoff)
        blocks.append(statistics.mean(chips))
    result=dict(complete=True,field='public',deals=2,hands=12,deal_seed=123,candidate=str(model),candidate_sha256=digest(model),
        candidate_type='compact_advantage',deal_blocks=blocks,bb_per_100=statistics.mean(blocks),ci95_half_width=1.96*statistics.stdev(blocks)/2**.5,
        **{key:proof[key] for key in ['evaluator_sha256','runtime_sha256','engine_sha256','source_features_sha256','opponents']})
    (tmp_path/'results.json').write_text(json.dumps(result))
    return cfg,job,proof,hands


def test_completed_cell_recomputes_raw_payoffs_and_rejects_duplicate_seats(tmp_path):
    cfg,job,proof,hands=raw_fixture(tmp_path)
    def write(rows):
        with gzip.open(tmp_path/'hands.jsonl.gz','wt') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    write(hands)
    assert evaluator.verify_result(tmp_path,job,cfg,proof)['hands']==12
    write(hands+[hands[0]])
    with pytest.raises(ValueError,match='Duplicate'):evaluator.verify_result(tmp_path,job,cfg,proof)


def test_completed_cell_cannot_substitute_a_different_candidate(tmp_path):
    cfg,job,proof,hands=raw_fixture(tmp_path)
    Path(job['model']).write_bytes(b'changed model')
    with pytest.raises(ValueError,match='Candidate changed'):evaluator.verify_result(tmp_path,job,cfg,proof)
