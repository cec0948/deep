import io
import json
from pathlib import Path
import sys
import shutil
import tarfile

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from controlled_poker_refit import (selected_indices, views_from_states, LEGACY_ARMS,
                                    ARMS, BASELINES, SEEDS, FIELDS)
import run_controlled_poker_study as study
from rich_regret_model import RichRegretModel
from export_controlled_poker_probe import verify_probe, export_probe, MODE
from plan_controlled_backup_retention import plan_retention
from test_poker_backup_retention import backup
from test_rich_backup import record as checkpoint_record


def test_replay_subset_is_nested_reproducible_and_does_not_consume_fitting_rng():
    np.random.seed(31); before=np.random.get_state()
    small=selected_indices(100000,5000,342,1)
    assert len(small)==len(set(small))==5000
    assert np.array_equal(small,selected_indices(100000,5000,342,1))
    assert np.array_equal(selected_indices(100000,100000,342,1),np.arange(100000))
    after=np.random.get_state()
    assert np.array_equal(before[1],after[1]) and before[2:]==after[2:]


def test_mask_changes_only_extra_inputs_without_mutating_source_or_targets():
    rng=np.random.default_rng(5)
    states=[dict(count=11,features=rng.normal(size=(11,80)).astype('f'),
                 advantages=rng.normal(size=(11,5)).astype('f'),iterations=np.arange(1,12,dtype='f'))]
    source=states[0]['features'].copy()
    a,sa=views_from_states(states);b,sb=views_from_states(states,mask_extra=True)
    assert sa==sb and np.array_equal(a[0].features[:,:51],b[0].features[:,:51])
    assert not b[0].features[:,51:].any()
    assert np.array_equal(a[0].advantages,b[0].advantages)
    assert np.array_equal(a[0].iterations,b[0].iterations)
    assert np.array_equal(source,states[0]['features'])


def test_legacy_arms_really_change_one_setting_each():
    baseline=LEGACY_ARMS['legacy_control']
    for arm,row in LEGACY_ARMS.items():
        if arm!='legacy_control':assert sum(row[k]!=baseline[k] for k in row)==1


def test_fitting_pair_rejects_opening_and_initialization_confound():
    a=dict(initial_parameter_sha256='a',opening_sha256='o',selected_row_sha256=['s'],arm='legacy_control')
    b=dict(a,arm='legacy_fewer_steps')
    study.validate_pairs({'legacy_control-seed-342':a,'legacy_fewer_steps-seed-342':b})
    b['opening_sha256']='changed'
    with pytest.raises(ValueError,match='opening'):
        study.validate_pairs({'legacy_control-seed-342':a,'legacy_fewer_steps-seed-342':b})


def test_input_pair_rejects_different_random_training_draws():
    a=dict(initial_parameter_sha256='a',selected_row_sha256=['s'],final_rng_sha256='r',replay_rows=[5000]*6)
    b=dict(a,final_rng_sha256='other')
    with pytest.raises(ValueError,match='input-only'):
        study.validate_pairs({'rich_masked-seed-342':a,'rich_full-seed-342':b})


def test_mask_can_be_folded_into_weights_with_identical_full_input_predictions():
    teacher=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign/seed-342/current_00360.npz'
    source=teacher.parent.parent/'source'
    torch.set_num_threads(1);torch.manual_seed(8)
    model=RichRegretModel(teacher,64,source,width=16)
    with torch.no_grad():
        model.residual.layers[-1].weight.normal_(0,.01)
    model._sync_numpy()
    x=np.zeros(64,np.float32);x[17]=1;x[21]=1;x[27]=.2;x[45]=1;x[46:51]=1;x[51:]=.3
    mask=x.copy();mask[51:]=0;expected=model.strategy(mask,mask[46:51])
    with torch.no_grad():model.residual.layers[0].weight[:,51:]=0
    model._sync_numpy()
    assert np.array_equal(expected,model.strategy(x,x[46:51]))
    assert np.array_equal(expected,model.strategy(mask,mask[46:51]))


def synthetic_results():
    result={}
    for j in study.evaluation_jobs():
        effect=(10 if j['label']=='legacy_small_replay' else 0)
        result[j['id']]=dict(deal_blocks=[effect+1.,effect-2.,effect+4.])
    return result


def test_final_summary_requires_all_cells_and_keeps_card_blocks_as_units():
    rows=synthetic_results();assert len(rows)==90
    s=study.summarize(rows,deals=3)
    assert s['independent_card_blocks']==3 and s['total_hands']==90*3*6
    assert s['new_selfplay_iterations']==0 and s['fixed_replay']
    x=s['primary_one_factor_changes']['public/legacy_small_replay-minus-legacy_control']
    assert x['mean']==10 and x['family_comparisons']==15
    assert x['simultaneous_lower']==10
    rows.pop(next(iter(rows)))
    with pytest.raises(ValueError,match='All90'):study.summarize(rows,deals=3)


def test_simultaneous_interval_is_wider_than_nominal_for_variable_difference():
    rows=synthetic_results()
    for seed in SEEDS:rows[f'public-seed-{seed}-legacy_fewer_steps']['deal_blocks']=[-4,0,9]
    x=study.summarize(rows,deals=3)['primary_one_factor_changes']['public/legacy_fewer_steps-minus-legacy_control']
    assert x['simultaneous_lower']<x['lower']<x['upper']<x['simultaneous_upper']


def test_protocol_forbids_budget_changes_and_new_selfplay():
    cfg=json.loads((ROOT/'configs/poker-controlled-refit-20261116.json').read_text())
    study.validate_config(cfg)
    for key,value in [('new_selfplay_iterations',1),('evaluation_deals',2000),('automatic_retry',True)]:
        bad=dict(cfg);bad[key]=value
        with pytest.raises(ValueError):study.validate_config(bad)


def test_prepare_preserves_parents_and_freezes_real_source(tmp_path,monkeypatch):
    config=ROOT/'configs/poker-controlled-refit-20261116.json'
    historical_plan = ROOT / 'reports/controlled-refit/20261116T004818765266Z/plan.json'
    prior_card_ranges = json.loads(historical_plan.read_text())['prior_card_ranges']
    monkeypatch.setattr(study,'ROOT_FOLDER',tmp_path)
    monkeypatch.setattr(study, 'previous_ranges', lambda _: prior_card_ranges)
    output=tmp_path/'20261116T010000000001Z'
    parent=ROOT/'artifacts/rich-selfplay/current.json';before=parent.read_bytes()
    plan=study.prepare(output,config)
    assert plan['prior_card_ranges'] == prior_card_ranges
    assert parent.read_bytes()==before and plan['new_selfplay_iterations']==0
    assert plan['planned_fitting_jobs']==21 and plan['planned_evaluations']==90
    assert len(plan['parents']['legacy'])==3
    study.verify_files(plan['frozen_files'])
    target=output/'source'/'training'/'buffer.py';target.write_text(target.read_text()+'\n# tamper')
    with pytest.raises(ValueError,match='changed'):study.verify_files(plan['frozen_files'])


def test_unsafe_policy_archive_is_rejected(tmp_path):
    p=tmp_path/'bad.tar.gz'
    with tarfile.open(p,'w:gz') as tar:
        item=tarfile.TarInfo('../escape');item.size=1;tar.addfile(item,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='Unsafe'):verify_probe(p)


def test_export_actually_restores_all_policies_and_rejects_corruption(tmp_path):
    # Archive-format fixture from existing known policies; no fitting or strength claim.
    source=ROOT/'artifacts/bounded-continuation/20261113T014811152621Z/campaign'
    rich=ROOT/'artifacts/rich-selfplay/20261113T075620400993Z/campaign'
    folder=tmp_path/'20261116T010000000002Z';folder.mkdir()
    shutil.copytree(source/'source',folder/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (folder/'legacy-tools').mkdir()
    parents={'legacy':{str(seed):{'model':str(source/f'seed-{seed}/current_00360.npz')} for seed in SEEDS}}
    plan={'parents':parents,'frozen_files':{}}
    ready={'policies':{}}
    for arm in ARMS:
        ready['policies'][arm]={}
        for seed in SEEDS:
            path=(source/f'seed-{seed}/current_00360.npz' if arm in LEGACY_ARMS
                  else rich/f'seed-{seed}/policy-00096.json')
            ready['policies'][arm][str(seed)]={'path':str(path)}
            target=folder/'fitting'/f'{arm}-seed-{seed}';target.mkdir(parents=True)
            (target/'result.json').write_text(json.dumps({'fixture':True,'model':str(path)}))
    for name,value in [('plan.json',plan),('stage-ready.json',ready),('config.json',{}),
                       ('summary.json',{'complete':True,'completed_evaluations':90}),
                       ('evaluation-summary.json',{})]:
        (folder/name).write_text(json.dumps(value))
    record=export_probe(folder,tmp_path/'exports')
    assert record['verification']['policies_verified']==21 and record['restoration_verified']
    assert not record['resumable_selfplay'] and not record['replay_included']
    archive=Path(record['archive']);corrupt=tmp_path/'corrupt.tar.gz'
    with tarfile.open(archive,'r:gz') as original,tarfile.open(corrupt,'w:gz') as out:
        for item in original.getmembers():
            payload=original.extractfile(item).read()
            if item.name.endswith('policy.npz'):payload=b'corrupted'
            item.size=len(payload);out.addfile(item,io.BytesIO(payload))
    with pytest.raises(ValueError,match='bytes differ'):verify_probe(corrupt)


@pytest.fixture
def retention_index(tmp_path):
    full60 = backup(tmp_path, 1, rounds=60)
    full120 = backup(tmp_path, 2, rounds=120)
    parent360 = checkpoint_record(tmp_path, 3, 360)
    rich96 = checkpoint_record(tmp_path, 4, 96, rich=True)
    older_probe = backup(tmp_path, 5, kind='derived-policy-probe', parent=full120['manifest_file_id'])
    latest_legacy_probe = backup(tmp_path, 6, kind='derived-policy-probe', parent=full120['manifest_file_id'])
    controlled_probe = backup(tmp_path, 7, kind='evaluation-assets')
    controlled_probe.update(
        kind='derived-policy-probe', checkpoint_mode=MODE, run='20261116T010000000001Z',
        status='complete', format='single-archive-v1', resumable_selfplay=False,
        recorded_at='2026-11-16T01:00:00+00:00', new_selfplay_iterations=0,
        restoration_verified=True,
        restore_dependencies=[
            {'parent_checkpoint_sha256': parent['sha256']} for parent in (parent360, rich96)
        ],
    )
    return {
        'approved_destination': 'Private Google Drive',
        'backups': [full60, full120, parent360, rich96, older_probe, latest_legacy_probe, controlled_probe],
    }


def test_retention_counts_latest_two_probes_across_families_and_preserves_both_parents(retention_index, tmp_path):
    plan = plan_retention(retention_index, tmp_path)
    assert {row['backup_index'] for row in plan['retained']} == {0, 1, 2, 3, 5, 6}
    assert {row['backup_index'] for row in plan['candidates']} == {4}
    assert plan['within_budget']


    retention_index['backups'][-1]['verified_owner_only'] = False
    with pytest.raises(ValueError):
        plan_retention(retention_index, tmp_path)


@pytest.mark.parametrize('parent_index', [0, 1], ids=['parent360', 'rich96'])
def test_retention_refuses_missing_restore_dependency(retention_index, tmp_path, parent_index):
    dependencies = retention_index['backups'][-1]['restore_dependencies']
    dependencies[parent_index]['parent_checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='dependency|parent SHA'):
        plan_retention(retention_index, tmp_path)
