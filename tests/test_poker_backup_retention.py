"""Retention plans preserve recovery dependencies and fail closed on bad evidence."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plan_poker_backup_retention import plan_retention


def url(identifier):
    return f'https://drive.google.com/file/d/{identifier}/view?usp=drivesdk'


def backup(root, number, rounds=None, kind='training-checkpoint', parent=None):
    run = '20261112T000000000000Z'
    stamp = f'20261112T000000{number:06d}Z'
    name = (f'compact-v4-{run}-snapshot-{stamp}.tar.gz' if rounds is not None
            else f'{kind}-{stamp}.tar.gz')
    path = root / 'artifacts/exports' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = f'archive-{number}'.encode()
    path.write_bytes(data)
    record = dict(kind=kind, local_archive=str(path), bytes=len(data),
                  sha256=hashlib.sha256(data).hexdigest(), verified_remote_size=True,
                  verified_owner_only=True, remote_checksum_available=False,
                  recorded_at=f'2026-11-12T00:00:{number:02d}+00:00')
    if rounds is None:
        record.update(file_id=f'file_{number}', url=url(f'file_{number}'))
    else:
        record.update(run=run, iterations={str(seed): rounds for seed in (342, 343, 344)},
                      format='split-checkpoint-v1', manifest_file_id=f'manifest_{number}',
                      manifest_url=url(f'manifest_{number}'), manifest_bytes=3, parts=[])
        chunks = [data[:3], data[3:]]
        for part_number, chunk in enumerate(chunks, 1):
            part_name = name + f'.part-{part_number:03d}'
            identifier = f'part_{number}_{part_number}'
            record['parts'].append(dict(name=part_name, path=str(path.with_name(part_name)),
                bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest(), file_id=identifier,
                url=url(identifier), verified_remote_size=True, verified_owner_only=True,
                remote_checksum_available=False))
    if kind == 'derived-policy-probe':
        record.update(parent_manifest_file_id=parent, resumable_selfplay=False, run=stamp)
    return record


@pytest.fixture
def index(tmp_path):
    return dict(approved_destination='Private Google Drive', backups=[
        backup(tmp_path, 1, kind='evaluation-assets'),
        backup(tmp_path, 2, 6), backup(tmp_path, 3, 42), backup(tmp_path, 4, 60),
        backup(tmp_path, 5, 102), backup(tmp_path, 6, 120),
        backup(tmp_path, 7, kind='derived-policy-probe', parent='manifest_6')])


def test_keep_baseline_latest_probe_assets_and_exact_remote_bytes(index, tmp_path):
    result = plan_retention(index, tmp_path)
    assert [r['backup_index'] for r in result['retained']] == [0, 3, 5, 6]
    assert [r['iterations']['342'] for r in result['candidates']] == [6, 42, 102]
    assert result['expected_reclaimed_bytes'] == sum(index['backups'][n]['bytes'] + 3 for n in (1, 2, 4))
    assert result['planned_retained_bytes'] == sum(index['backups'][n]['bytes'] for n in (0, 3, 5, 6)) + 6
    assert result['within_budget'] is True
    assert result['remote_operations_performed'] is False
    assert result['candidates'][0]['remote_files'][0]['url'] == index['backups'][1]['parts'][0]['url']


def test_recursive_dependency_closure_retains_older_checkpoint(index, tmp_path):
    index['backups'][-1]['parent_manifest_file_id'] = 'manifest_5'
    index['backups'][4]['parent_manifest_file_id'] = 'manifest_3'
    result = plan_retention(index, tmp_path)
    assert [r['backup_index'] for r in result['candidates']] == [1]
    for n in (2, 4):
        assert 'required_parent_manifest_dependency' in next(r for r in result['retained'] if r['backup_index'] == n)['reasons']


@pytest.mark.parametrize('broken', ['partial', 'part_missing', 'private', 'size', 'order', 'url', 'sha', 'seed'])
def test_incomplete_or_invalid_records_block_cleanup(index, tmp_path, broken):
    record = index['backups'][2]
    if broken == 'partial':
        record['status'] = 'partial'
    elif broken == 'part_missing':
        record['parts'].pop()
    elif broken == 'private':
        record['parts'][0]['verified_owner_only'] = False
    elif broken == 'size':
        record['bytes'] += 1
    elif broken == 'order':
        record['parts'].reverse()
    elif broken == 'url':
        record['parts'][0]['url'] = url('unrelated')
    elif broken == 'sha':
        record['sha256'] = 'not-a-hash'
    else:
        record['iterations'].pop('344')
    with pytest.raises(ValueError):
        plan_retention(index, tmp_path)


@pytest.mark.parametrize('change', ['missing', 'same_size_corruption'])
def test_actual_latest_archive_must_match_local_sha(index, tmp_path, change):
    archive = Path(index['backups'][5]['local_archive'])
    if change == 'missing':
        archive.unlink()
    else:
        archive.write_bytes(b'x' * archive.stat().st_size)
    with pytest.raises(ValueError, match='local size/SHA-256'):
        plan_retention(index, tmp_path)


def test_unknown_kind_and_status_preserved_with_unknown_storage_total(index, tmp_path):
    index['backups'].extend([dict(kind='future-kind', file_id='future'),
                             dict(kind='training-checkpoint', status='future-status', file_id='future2')])
    result = plan_retention(index, tmp_path)
    assert all(r['backup_index'] < 7 for r in result['candidates'])
    assert result['planned_retained_bytes'] is None
    assert result['within_budget'] is None
    assert len(result['retained']) == 6


def test_shared_id_blocks_deletion_even_when_unknown_owner(index, tmp_path):
    index['backups'].append(dict(kind='unknown', file_id='part_2_1'))
    with pytest.raises(ValueError, match='shared'):
        plan_retention(index, tmp_path)


def test_missing_parent_fails_closed(index, tmp_path):
    index['backups'][5]['deleted_remote'] = True
    # A new complete stage cannot replace the missing explicit parent.
    index['backups'].append(backup(tmp_path, 8, 180))
    with pytest.raises(ValueError, match='missing/deleted parent'):
        plan_retention(index, tmp_path)


def test_already_deleted_entries_are_not_retained_or_counted(index, tmp_path):
    deleted = backup(tmp_path, 8, 180)
    deleted['deleted_remote'] = True
    index['backups'].append(deleted)
    result = plan_retention(index, tmp_path)
    assert result['ignored_deleted_records'] == [7]
    assert result['retained_latest_local_verification']['backup_index'] == 5


def test_future_repeated_backups_bounded_and_latest_recovery_preserved(index, tmp_path):
    for number, rounds in [(8, 126), (9, 132), (10, 180), (11, 186), (12, 192)]:
        index['backups'].append(backup(tmp_path, number, rounds))
    for number in (13, 14, 15):
        index['backups'].append(backup(tmp_path, number, kind='derived-policy-probe', parent='manifest_10'))
    result = plan_retention(index, tmp_path)
    retained = result['retained']
    assert sorted(r['iterations']['342'] for r in retained if r['kind'] == 'training-checkpoint') == [60, 180, 192]
    assert len([r for r in retained if r['kind'] == 'derived-policy-probe']) == 2
    assert len(retained) == 6


def test_budget_does_not_override_dependencies(index, tmp_path):
    normal = plan_retention(index, tmp_path)
    limited = plan_retention(index, tmp_path, budget_bytes=1)
    assert limited['within_budget'] is False
    assert limited['candidates'] == normal['candidates']
    assert 'do not delete additional files' in limited['budget_note']


def test_planning_does_not_mutate_index(index, tmp_path):
    before = copy.deepcopy(index)
    plan_retention(index, tmp_path)
    assert index == before
