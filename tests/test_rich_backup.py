"""Rich recovery round counts, dependencies and real byte restoration."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compact_lightweight_backup import equal_state,load_state,save_state
from fullhouse_v3_common import atomic_json,digest
from plan_rich_backup_retention import plan_retention
from rich_regret_model import RichRegretModel
from snapshot_rich_selfplay import FAMILY,SEEDS,snapshot,verify_archive,verify_stage,prepare_upload
from train_rich_selfplay import checkpoint_state,load_current


def url(name):return f'https://drive.google.com/file/d/{name}/view'


def record(root,n,iteration,rich=False):
    run='20261113T000000000000Z';stamp=f'20261113T000000{n:06d}Z'
    name=(f'rich-selfplay-{run}-stage-{iteration:05d}-{stamp}.tar.gz' if rich else
          f'compact-v4-{run}-snapshot-{stamp}.tar.gz')
    path=root/'artifacts/exports'/name;path.parent.mkdir(parents=True,exist_ok=True)
    data=f'fixture-{n}'.encode();path.write_bytes(data)
    row=dict(kind='training-checkpoint',run=run,local_archive=str(path),bytes=len(data),
        sha256=digest(path),file_id=f'file_{n}',url=url(f'file_{n}'),format='single-archive-v1',
        status='complete',recorded_at=f'2026-11-13T00:00:{n:02d}+00:00',
        verified_remote_size=True,verified_owner_only=True,remote_checksum_available=False)
    if rich:
        from snapshot_rich_selfplay import FORMAT
        row.update(training_family=FAMILY,checkpoint_mode=FORMAT,parent_iteration=360,
            family_iterations={str(s):iteration for s in SEEDS},training_stage_completed=iteration in [12,48,96],
            resumable_selfplay=True,resumable_worker_family=FAMILY,exact_legacy_resume=False,
            warm_start_state_verified=True,historical_mixture_available=False,restoration_verified=True)
    else:
        row.update(iterations={str(s):iteration for s in SEEDS})
        if iteration==360:row.update(checkpoint_mode='compact-bounded-replay-recovery-v1',
            resumable_selfplay=False,exact_legacy_resume=False,warm_start_state_verified=True,
            historical_mixture_available=False)
    return row


@pytest.fixture
def index(tmp_path):
    return dict(approved_destination='Private Google Drive',backups=[
        record(tmp_path,1,60),record(tmp_path,2,120),record(tmp_path,3,360)])


def test_rich_rounds_are_not_legacy_rounds_and_parent360_survives(index,tmp_path):
    index['backups'] += [record(tmp_path,4,12,True),record(tmp_path,5,48,True),record(tmp_path,6,54,True)]
    before=copy.deepcopy(index);plan=plan_retention(index,tmp_path)
    assert index==before
    assert {x['backup_index'] for x in plan['retained']}=={0,1,2,4,5}
    assert {x['backup_index'] for x in plan['candidates']}=={3}
    assert plan['retained_latest_legacy_local_verification']['backup_index']==2
    rich=[x for x in plan['retained'] if x.get('training_family')==FAMILY]
    assert [x['family_iterations']['342'] for x in rich]==[48,54]
    assert all('iterations' not in x for x in rich)
    assert plan['remote_operations_performed'] is False


def test_rich_dependencies_and_budget_do_not_remove_required_parent(index,tmp_path):
    row=record(tmp_path,4,12,True);row['restore_dependencies']=[{'parent_file_id':'file_2'}]
    index['backups'].append(row);result=plan_retention(index,tmp_path,budget_bytes=1)
    assert {x['backup_index'] for x in result['retained']}=={0,1,2,3}
    assert result['within_budget'] is False and not result['candidates']
    assert 'do not delete' in result['budget_note']


@pytest.mark.parametrize('bad',['unverified','partial','fake_legacy_round','missing_parent','corrupt_latest','shared_id'])
def test_invalid_rich_record_cannot_replace_last_safe_recovery(index,tmp_path,bad):
    row=record(tmp_path,4,12,True);index['backups'].append(row)
    if bad=='unverified':row['restoration_verified']=False
    elif bad=='partial':row['status']='partial'
    elif bad=='fake_legacy_round':row['iterations']={str(s):372 for s in SEEDS}
    elif bad=='missing_parent':index['backups'][2]['deleted_remote']=True
    elif bad=='corrupt_latest':Path(row['local_archive']).write_bytes(b'corrupt')
    elif bad=='shared_id':row['file_id']='file_3';row['url']=url('file_3')
    with pytest.raises(ValueError):plan_retention(index,tmp_path)


def test_unknown_backup_is_preserved_with_unknown_storage_total(index,tmp_path):
    index['backups'].append(record(tmp_path,4,12,True))
    index['backups'].append(dict(kind='future-format',file_id='future_unknown'))
    plan=plan_retention(index,tmp_path)
    assert 4 in {x['backup_index'] for x in plan['retained']}
    assert plan['planned_retained_bytes'] is None and plan['within_budget'] is None


def test_newer_stage_only_replaces_rich_intermediates(index,tmp_path):
    index['backups'] += [record(tmp_path,4,12,True),record(tmp_path,5,48,True),record(tmp_path,6,96,True)]
    plan=plan_retention(index,tmp_path)
    assert [x['family_iterations']['342'] for x in plan['retained'] if x.get('training_family')]==[96]
    assert {x['backup_index'] for x in plan['candidates']}=={3,4}


def test_malicious_tar_cannot_escape_restore_destination(tmp_path):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        data=b'not trusted';member=tarfile.TarInfo('../escaped');member.size=len(data)
        tar.addfile(member,io.BytesIO(data))
    with pytest.raises(ValueError,match='Unsafe'):verify_archive(archive,tmp_path/'restore')
    assert not (tmp_path/'escaped').exists()


def test_split_local_upload_manifest_has_order_sizes_hashes_and_no_completion_claim(tmp_path):
    archive=tmp_path/'fixture.tar.gz';data=b'0123456789abcdefghi';archive.write_bytes(data)
    manifest=prepare_upload(archive,max_file_bytes=8,part_bytes=4)
    assert [p['order'] for p in manifest['parts']]==[1,2,3,4,5]
    assert [p['bytes'] for p in manifest['parts']]==[4,4,4,4,3]
    assert b''.join(Path(p['path']).read_bytes() for p in manifest['parts'])==data
    assert manifest['sha256']==manifest['recombined_local_sha256']==digest(archive)
    assert all(p['file_id'] is None and not p['verified_owner_only'] for p in manifest['parts'])
    assert not manifest['remote_verification_complete'] and not manifest['external_backup']
    with pytest.raises(FileExistsError):prepare_upload(archive,max_file_bytes=8,part_bytes=4)
    with pytest.raises(ValueError):prepare_upload(archive,part_bytes=65*1024**2)


def test_single_upload_stays_single_below_file_limit(tmp_path):
    archive=tmp_path/'fixture.tar.gz';archive.write_bytes(b'0123')
    manifest=prepare_upload(archive,max_file_bytes=8,part_bytes=4)
    assert manifest['format']=='single-archive-v1' and len(manifest['files'])==1
    assert not list(tmp_path.glob('*.part-*'))


def test_rich_split_remote_size_and_dependency_rules_are_enforced(index,tmp_path):
    rich=record(tmp_path,4,12,True);data=Path(rich['local_archive']).read_bytes()
    rich.pop('file_id');rich.pop('url');rich.update(format='split-checkpoint-v1',parts=[],
        manifest_file_id='rich_manifest',manifest_url=url('rich_manifest'),manifest_bytes=10)
    for number,chunk in enumerate([data[:3],data[3:]],1):
        name=Path(rich['local_archive']).name+f'.part-{number:03d}'
        rich['parts'].append(dict(name=name,path=str(Path(rich['local_archive']).with_name(name)),
            bytes=len(chunk),sha256=__import__('hashlib').sha256(chunk).hexdigest(),
            file_id=f'richpart{number}',url=url(f'richpart{number}'),verified_remote_size=True,verified_owner_only=True))
    index['backups'].append(rich)
    result=plan_retention(index,tmp_path)
    item=next(p for p in result['retained'] if p['backup_index']==3)
    assert item['remote_bytes']==rich['bytes']+10 and len(item['remote_files'])==3
    rich['parts'][0]['verified_owner_only']=False
    with pytest.raises(ValueError,match='part verification'):plan_retention(index,tmp_path)


@pytest.fixture
def campaign(tmp_path):
    registry=ROOT/'artifacts/bounded-continuation/current.json'
    if not registry.exists():pytest.skip('Requires restored frozen360 assets')
    parent=Path(json.loads(registry.read_text())['folder']);teacher=parent/'campaign/seed-342/current_00360.npz'
    source=parent/'campaign/source'
    if not teacher.exists():pytest.skip('Requires verified360 teacher')
    study=tmp_path/'20261113T000000000000Z';campaign=study/'campaign';campaign.mkdir(parents=True)
    frozen=tmp_path/'frozen-input.txt';frozen.write_text('unchanged source evidence')
    cfg=dict(verification_fixture=True,seeds=list(SEEDS),stages=[1,2],iterations=2,traversals=1,
        buffer_capacity=2,main_steps=1,batch_size=2,learning_rate=.001,checkpoint_every=1,
        seconds_per_training_worker=60,parent_iteration=360,feature_dim=2885,width=4,
        source=str(source),teachers={str(s):dict(path=str(teacher),sha256=digest(teacher)) for s in SEEDS},
        frozen_files={str(frozen):digest(frozen)},prerequisites_passed=True)
    atomic_json(campaign/'config.json',cfg)
    class Buffer:
        def state_dict(self):
            x=np.zeros((2,2885),np.float32);x[:,47]=1
            return dict(features=x,advantages=np.zeros((2,5),np.float32),
                iterations=np.ones(2,np.float32),count=2,total_seen=2,capacity=2)
    for seed in SEEDS:
        model=RichRegretModel(teacher,2885,source,width=4)
        state=checkpoint_state(model,[Buffer() for _ in range(6)],cfg,seed,1,np.random.default_rng(seed))
        path=campaign/f'seed-{seed}/checkpoints/iter-00001';save_state(state,path);model.export(path/'current.npz')
        atomic_json(campaign/f'seed-{seed}/warm-start-verification.json',
            dict(passed=True,teacher_probability_byte_identical=True,teacher_sha256=digest(teacher)))
    return study,cfg


def test_real_archive_restores_model_replay_rng_and_new_worker_paths(campaign,tmp_path):
    study,cfg=campaign
    before={s:load_state(study/f'campaign/seed-{s}/checkpoints/iter-00001') for s in SEEDS}
    proof=verify_stage(study/'campaign',1,cfg)
    assert proof['passed'] and all(x['all_rng_states_present'] for x in proof['seeds'])
    packed=snapshot(study,1,tmp_path/'exports')
    assert packed['resumable_selfplay'] and packed['exact_legacy_resume'] is False
    assert packed['family_iterations']=={str(s):1 for s in SEEDS} and 'iterations' not in packed
    destination=tmp_path/'restored';verified=verify_archive(packed['archive'],destination,relocate=True)
    assert verified['passed'] and verified['relocation_performed']
    newcfg=json.loads((destination/'campaign/config.json').read_text())
    assert Path(newcfg['source']).is_relative_to(destination)
    for seed in SEEDS:
        after,path=load_current(destination/f'campaign/seed-{seed}',newcfg,seed,2885)
        assert equal_state({k:v for k,v in before[seed].items() if k!='config'},
                           {k:v for k,v in after.items() if k!='config'})
        expected=np.random.default_rng();expected.bit_generator.state=before[seed]['rng']['generator']
        actual=np.random.default_rng();actual.bit_generator.state=after['rng']['generator']
        assert np.array_equal(expected.integers(0,2**31,32),actual.integers(0,2**31,32))
        assert digest(destination/f'campaign/teachers/seed-{seed}.npz')==cfg['teachers'][str(seed)]['sha256']
    with pytest.raises(FileExistsError):verify_archive(packed['archive'],destination)
