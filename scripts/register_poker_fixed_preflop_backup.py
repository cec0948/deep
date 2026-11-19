"""Prepare the fixed-preflop 504 recovery and register verified private Drive receipts."""
import argparse
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from controlled_poker_refit import read, utc
from fullhouse_v3_common import ROOT, atomic_json, digest
from plan_average_extension_backup_retention import plan_retention as prior_retention
from plan_poker_backup_retention import DEFAULT_BUDGET, MAX_PART, remote_file, validate_record
from plan_rich_backup_retention import _dependencies, _remote_ids
from snapshot_poker_fixed_preflop_window import ARCHIVE, MODE, OUTPUT
from snapshot_poker_preflop_dose import FULL, SPECIALIST, source_archives
from snapshot_rich_selfplay import prepare_upload


INDEX = ROOT / 'artifacts/offsite-backups.json'
FILE_LIMIT = 100 * 1024**2


def local_file(path):
    return dict(path=str(path), name=path.name, bytes=path.stat().st_size, sha256=digest(path))


def load_ready():
    ready = read(OUTPUT/'backup-ready.json')
    proof = ready['verification']
    if (ready['kind'] != 'training-checkpoint' or ready['checkpoint_mode'] != MODE
            or ready['training_family'] != 'fixed-preflop-window' or ready['arms'] != ['fixed504']
            or ready['iterations'] != {s:504 for s in ('342','343','344')}
            or ready['new_selfplay_iterations_per_seed'] != 24
            or ready['training_stage_completed'] is not True or ready['restoration_verified'] is not True
            or ready['warm_start_state_verified'] is not True or ready['external_backup'] is not False
            or ready['archive_includes_final_evaluation_summary'] is not False
            or not all(proof[k] is True for k in ('actual_extraction_verified','lossless_state_verified','both_street_weights_verified','frozen_preflop_verified'))
            or set(proof['seeds']) != set(ready['iterations'])
            or _dependencies(ready['restore_dependencies'])[1] != {FULL, SPECIALIST}
            or ready['archive'] != str(ARCHIVE) or ready['bytes'] != ARCHIVE.stat().st_size
            or ready['sha256'] != digest(ARCHIVE)):
        raise ValueError('Stage, bounded restore or local archive verification failed')
    return ready


def prepare_objects():
    if ARCHIVE.stat().st_size <= FILE_LIMIT:
        return prepare_upload(ARCHIVE)
    # Existing matching parts can be reused after an interruption. The splitter
    # itself expects a clean directory and produces a preliminary local manifest.
    with tempfile.TemporaryDirectory(dir=ARCHIVE.parent) as temporary:
        staged = Path(temporary)/ARCHIVE.name
        os.link(ARCHIVE, staged)
        layout = prepare_upload(staged)
        for item in layout['parts']:
            src, dst = Path(item['path']), ARCHIVE.parent/item['name']
            try:
                os.link(src, dst)
            except FileExistsError:
                if dst.stat().st_size != item['bytes'] or digest(dst) != item['sha256']:
                    raise ValueError('Previously prepared part differs from this archive')
            item['path'] = str(dst)
        layout.pop('manifest_path')
        layout.update(format='split-checkpoint-v1', checkpoint_mode=MODE, archive=str(ARCHIVE))
    return layout


def expected_objects(layout, ready):
    split = ready['bytes'] > FILE_LIMIT
    if (layout['archive'] != str(ARCHIVE) or layout['bytes'] != ready['bytes']
            or layout['sha256'] != ready['sha256'] or split != ('parts' in layout)
            or layout['format'] != ('split-checkpoint-v1' if split else 'single-archive-v1')
            or (split and layout.get('checkpoint_mode') != MODE)
            or layout['ready_sha256'] != digest(OUTPUT/'backup-ready.json')):
        raise ValueError('Upload layout differs from the verified source')
    if not split:
        if layout['files'] != [local_file(ARCHIVE)]:
            raise ValueError('Single-file upload layout changed')
        return layout['files']
    joined = hashlib.sha256()
    total = 0
    for order, item in enumerate(layout['parts'], 1):
        path = ARCHIVE.with_name(ARCHIVE.name + f'.part-{order:03d}')
        if (item['order'] != order or item['name'] != path.name or item['path'] != str(path)
                or not 0 < item['bytes'] <= MAX_PART or path.stat().st_size != item['bytes']):
            raise ValueError('Invalid part size, path or sequence')
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                sha.update(block)
                joined.update(block)
        if sha.hexdigest() != item['sha256']:
            raise ValueError('Local part hash differs from upload layout')
        total += item['bytes']
    if (total != ready['bytes'] or joined.hexdigest() != ready['sha256']
            or layout['recombined_local_sha256'] != joined.hexdigest()):
        raise ValueError('Rejoined local parts differ from the source archive')
    return layout['parts']


def prepare():
    ready = load_ready()
    index_sha = digest(INDEX)
    index = read(INDEX)
    if index['approved_destination'] != 'Private Google Drive' or not index['folder_id']:
        raise ValueError('Unknown backup destination')
    if any(r.get('sha256') == ready['sha256'] for r in index['backups']):
        raise ValueError('Archive already appears in the private index; use check to verify the existing record')
    source_archives(index)
    layout_path = OUTPUT/'upload-layout.json'
    if layout_path.is_file():
        layout = read(layout_path)
    else:
        layout = prepare_objects()
        layout.update(folder_id=index['folder_id'], index_sha256=index_sha,
                      ready_sha256=digest(OUTPUT/'backup-ready.json'))
    items = expected_objects(layout, ready)
    if (index_sha != digest(INDEX) or layout['index_sha256'] != index_sha
            or layout['folder_id'] != index['folder_id']):
        raise ValueError('Private backup index changed during or since upload preparation')
    atomic_json(layout_path, layout)
    return dict(format=layout['format'], local_bytes=ready['bytes'], objects=len(items))


def verify_owner(meta, owner):
    grants = meta.get('permissions', [])
    if (meta.get('shared') is not False or meta.get('trashed') is True or len(grants) != 1
            or meta.get('source_visibility_status', 'permission_metadata_available') != 'permission_metadata_available'
            or grants[0].get('emailAddress') != owner or grants[0].get('role') != 'owner'
            or grants[0].get('type') != 'user'):
        raise ValueError('Drive did not confirm a single owner-only permission')


def verified_file(item, meta, folder, owner):
    verify_owner(meta, owner)
    fid = meta.get('id')
    if (not fid or fid == folder or meta.get('parent_ids', meta.get('parents')) != [folder]
            or meta.get('title', meta.get('name')) != item['name'] or int(meta.get('size', -1)) != item['bytes']):
        raise ValueError('Drive file identity or byte count differs from upload layout')
    url = meta.get('url') or meta.get('webViewLink')
    remote_file(fid, url, item['bytes'], item['name'])
    remote_sha = meta.get('sha256Checksum') or meta.get('sha256_checksum')
    if remote_sha is not None and remote_sha != item['sha256']:
        raise ValueError('Drive-provided SHA-256 differs from local data')
    return dict(item, file_id=fid, url=url, verified_remote_size=True, verified_owner_only=True,
                remote_sha256_verified=remote_sha is not None)


def read_receipts(layout, items, owner):
    receipt = read(OUTPUT/'drive-receipt.json')
    folder = receipt['folder']
    verify_owner(folder, owner)
    if (not owner or receipt['account'] != owner or folder['id'] != layout['folder_id']
            or folder.get('mime_type', folder.get('mimeType')) != 'application/vnd.google-apps.folder'):
        raise ValueError('Expected the approved account and destination folder')
    files = [verified_file(item, meta, folder['id'], owner) for item, meta in zip(items, receipt['files'], strict=True)]
    if len({f['file_id'] for f in files}) != len(files):
        raise ValueError('Duplicate remote file ID')
    return receipt, files


def restore_manifest(layout, files):
    if 'parts' not in layout:
        raise ValueError('The archive fits directly in one Drive file')
    path = ARCHIVE.with_name(ARCHIVE.name[:-7]+'.manifest.json')
    data = dict(format='split-checkpoint-v1', checkpoint_mode=MODE, archive_name=ARCHIVE.name,
                bytes=layout['bytes'], sha256=layout['sha256'], recombined_local_sha256=layout['recombined_local_sha256'],
                restoration='Verify downloaded sizes and local hashes; join parts in order and verify the archive hash.',
                parts=[{k:f[k] for k in ('order','name','bytes','sha256','file_id')} for f in files])
    if path.exists():
        if read(path) != data:
            raise ValueError('Previously prepared manifest differs from verified Drive files')
    else:
        atomic_json(path, data)
    ref = local_file(path)
    atomic_json(OUTPUT/'restore-manifest-local.json', ref)
    return ref


def plan_retention(index, root=ROOT):
    # The prior planner already implements all older family rules. Mask just
    # this validated fixed-window family as required assets for its traversal.
    shadow = copy.deepcopy(index)
    fixed = {}
    for number, record in enumerate(index['backups']):
        if record.get('checkpoint_mode') != MODE or record.get('deleted_remote'):
            continue
        if (record['status'] != 'complete' or record['kind'] != 'training-checkpoint'
                or record['iterations'] != {s:504 for s in ('342','343','344')}
                or record['training_family'] != 'fixed-preflop-window' or record['arms'] != ['fixed504']
                or _dependencies(record['restore_dependencies'])[1] != {FULL, SPECIALIST}
                or not all(record[k] is True for k in ('restoration_verified','training_stage_completed','warm_start_state_verified'))):
            raise ValueError('Incomplete fixed preflop recovery blocks retention planning')
        archive = Path(record['local_archive'])
        if archive.stat().st_size != record['bytes'] or digest(archive) != record['sha256']:
            raise ValueError('Indexed fixed preflop archive differs from verified local source')
        item = shadow['backups'][number]
        item['kind'] = 'evaluation-assets'
        validate_record(item, root)
        fixed[number] = record
    plan = prior_retention(shadow, root, DEFAULT_BUDGET)
    retained = {item['backup_index'] for item in plan['retained']}
    for item in plan['retained']:
        record = fixed.get(item['backup_index'])
        if record:
            item.update(kind=record['kind'], checkpoint_mode=MODE, reasons=['completed_fixed_preflop_window'])
    parents = source_archives(index)
    active = {n:r for n,r in enumerate(index['backups']) if not r.get('deleted_remote')}
    sha_owner = {r.get('sha256'):n for n,r in active.items()}
    id_owner = {fid:n for n,r in active.items() for fid in _remote_ids(r)}
    latest_probes = sorted((n for n,r in active.items() if r.get('kind') == 'derived-policy-probe'),
                           key=lambda n:datetime.fromisoformat(active[n]['recorded_at']), reverse=True)[:2]
    required = set(fixed) | {sha_owner[h] for h in parents} | set(latest_probes)
    for n in retained:
        ids, hashes = _dependencies(active[n])
        required.update(id_owner[fid] for fid in ids)
        required.update(sha_owner[h] for h in hashes)
    if not required <= retained:
        raise ValueError('New recovery, parent, recent probe or restore dependency excluded from retention')
    plan.update(format='private-fixed-preflop-window-retention-plan-v1',
                index_sha256=hashlib.sha256(json.dumps(index,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return plan


def register(action, owner):
    ready = load_ready()
    layout = read(OUTPUT/'upload-layout.json')
    receipt, files = read_receipts(layout, expected_objects(layout, ready), owner)
    record = dict(ready, local_archive=str(ARCHIVE), recorded_at=utc(), format=layout['format'],
                  status='complete', verified_remote_size=True, verified_owner_only=True, external_backup=True)
    if 'parts' in layout:
        manifest = restore_manifest(layout, files)
        if action == 'manifest':
            return dict(manifest_bytes=manifest['bytes'], path=manifest['path'], sha256=manifest['sha256'])
        remote = verified_file(manifest, receipt['manifest'], layout['folder_id'], owner)
        record.update(parts=files, manifest_file_id=remote['file_id'], manifest_url=remote['url'],
                      manifest_sha256=manifest['sha256'], manifest_bytes=manifest['bytes'],
                      recombined_local_sha256=layout['recombined_local_sha256'])
        files = [*files, remote]
    else:
        if action == 'manifest':
            raise ValueError('The verified archive fits directly in one Drive file')
        record.update(file_id=files[0]['file_id'], url=files[0]['url'])
    all_sha = all(f['remote_sha256_verified'] for f in files)
    record.update(remote_checksum_available=all_sha, remote_sha256_verified=all_sha)
    ids = {f['file_id'] for f in files}
    if len(files) != len(ids):
        raise ValueError('Manifest and archive components must have unique IDs')
    with INDEX.with_name('offsite-backups.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = read(INDEX)
        if index['folder_id'] != layout['folder_id'] or index['approved_destination'] != 'Private Google Drive':
            raise ValueError('Destination changed')
        matches = [r for r in index['backups'] if r.get('sha256') == record['sha256']]
        if matches:
            record['recorded_at'] = matches[0]['recorded_at']
            if matches != [record]:
                raise ValueError('Existing archive entry differs from Drive receipts')
            revised = index
        else:
            if (digest(INDEX) != layout['index_sha256'] or any(ids & _remote_ids(r) for r in index['backups'])):
                raise ValueError('Private index or Drive ID changed since preparation')
            revised = copy.deepcopy(index)
            revised['backups'].append(record)
        plan = plan_retention(revised)
        if action == 'check':
            return dict(receipts_verified=True, within_planned_budget=plan['within_budget'])
        if not matches:
            if digest(INDEX) != layout['index_sha256']:
                raise ValueError('Private index changed during retention planning')
            atomic_json(INDEX, revised)
        if read(INDEX) != revised:
            raise ValueError('The private index does not match the expected readback')
        active_size = sum(r['bytes']+r.get('manifest_bytes',0) for r in revised['backups'] if not r.get('deleted_remote'))
        proof = dict(verified_utc=utc(), kind=record['kind'], checkpoint_mode=MODE, stage_iteration=504,
                     bytes=record['bytes'], sha256=record['sha256'], restored_all_three_seeds=True,
                     restored_both_street_networks=True, backup_files=len(files), format=layout['format'],
                     restoration_verified=True, external_backup=True, verified_remote_size=True, verified_owner_only=True,
                     local_recombination_verified='parts' in layout, remote_checksum_available=all_sha,
                     remote_sha256_verified=all_sha, archive_includes_final_evaluation_summary=False,
                     private_index_readback_verified=True, active_managed_bytes=active_size,
                     within_2gib_target=active_size <= DEFAULT_BUDGET, retained_bytes=plan['planned_retained_bytes'],
                     policy_candidate_remote_files=plan['candidate_remote_files'], remote_deletions_performed=0)
        atomic_json(OUTPUT/'backup-retention-plan.json', plan)
        atomic_json(OUTPUT/'backup-verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','manifest','check','commit'))
    parser.add_argument('--owner', default=os.environ.get('POKER_BACKUP_OWNER'),
                        help='Approved private account (or set POKER_BACKUP_OWNER)')
    args = parser.parse_args()
    if args.action != 'prepare' and not args.owner:
        parser.error('Supply the independently approved owner with --owner or POKER_BACKUP_OWNER')
    print(json.dumps(prepare() if args.action == 'prepare' else register(args.action, args.owner), indent=2))
