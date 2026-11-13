"""Plan private poker backup retention. Never contacts Drive or deletes files.

The plan is evidence for a separate, explicitly authorized cleanup operation.
Recorded remote verification is not a fresh remote read or remote hash check.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from fullhouse_v3_common import ROOT, atomic_json, digest

KNOWN_KINDS = {'evaluation-assets', 'training-checkpoint', 'derived-policy-probe', 'research-validation'}
COMPLETE = {None, 'complete', 'verified', 'uploaded'}
INCOMPLETE = {'pending', 'uploading', 'partial', 'failed', 'interrupted'}
MAX_PART = 64 * 1024**2
DEFAULT_BUDGET = 2 * 1024**3
STAMP = r'\d{8}T\d{6,12}Z'


def positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError('Invalid positive byte/count field: ' + label)
    return value


def remote_file(file_id, url, size, name=None):
    if not isinstance(file_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', file_id):
        raise ValueError('Missing or invalid recorded Drive file ID')
    parsed = urlsplit(url if isinstance(url, str) else '')
    if (parsed.scheme != 'https' or parsed.netloc != 'drive.google.com'
            or parsed.path not in (f'/file/d/{file_id}/view', f'/file/d/{file_id}')):
        raise ValueError('Recorded Drive URL does not identify the file ID')
    return dict(file_id=file_id, url=url, expected_bytes=positive(size, 'remote bytes'),
                expected_name=name)


def references(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'parent_manifest_file_id':
                if not isinstance(item, str) or not item:
                    raise ValueError('Invalid parent manifest reference')
                found.add(item)
            else:
                found.update(references(item))
    elif isinstance(value, list):
        for item in value:
            found.update(references(item))
    return found


def validate_record(record, root):
    if record.get('verified_remote_size') is not True or record.get('verified_owner_only') is not True:
        raise ValueError('Incomplete remote size/private verification')
    if type(record.get('remote_checksum_available')) is not bool:
        raise ValueError('Remote checksum availability must be recorded explicitly')
    sha = record.get('sha256', '')
    if not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Missing local archive SHA-256')
    size = positive(record.get('bytes'), 'archive bytes')
    path = Path(record.get('local_archive', '')).resolve()
    if path.parent != (root / 'artifacts/exports').resolve() or not path.name.endswith('.tar.gz'):
        raise ValueError('Archive must be a named tar.gz directly in artifacts/exports')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', path.name):
        raise ValueError('Invalid archive filename')
    stamp = datetime.datetime.fromisoformat(record['recorded_at'])
    if stamp.tzinfo is None:
        raise ValueError('Backup time must include timezone')
    if record['kind'] == 'training-checkpoint':
        mode = record.get('checkpoint_mode')
        if mode not in (None, 'compact-bounded-replay-recovery-v1'):
            raise ValueError('Unknown training checkpoint mode blocks cleanup')
        if mode is not None and (record.get('resumable_selfplay') is not False
                or record.get('exact_legacy_resume') is not False
                or record.get('historical_mixture_available') is not False
                or record.get('warm_start_state_verified') is not True):
            raise ValueError('Lightweight checkpoint capability verification is incomplete')
        run = record.get('run', '')
        if not re.fullmatch(STAMP, run) or not re.fullmatch(
                r'compact-v4-' + re.escape(run) + r'(?:-snapshot-' + STAMP + r')?\.tar\.gz', path.name):
            raise ValueError('Training archive filename/run mismatch')
        iterations = record.get('iterations', {})
        if set(iterations) != {'342', '343', '344'}:
            raise ValueError('Incomplete training seed coverage')
        for value in iterations.values():
            positive(value, 'iteration')
    if record['kind'] == 'derived-policy-probe':
        if record.get('resumable_selfplay') is not False or not record.get('parent_manifest_file_id'):
            raise ValueError('Derived probe must identify its full parent and non-resumable status')
    if record['kind'] == 'research-validation' and record.get('resumable_selfplay') is not False:
        raise ValueError('Research validation is not a resumable six-player checkpoint')
    parts = record.get('parts')
    files = []
    if parts is not None:
        if record.get('format') != 'split-checkpoint-v1' or not isinstance(parts, list) or not parts:
            raise ValueError('Invalid split archive')
        if record.get('file_id'):
            raise ValueError('Ambiguous single/split archive')
        for number, part in enumerate(parts, 1):
            name = path.name + f'.part-{number:03d}'
            if part.get('name') != name or Path(part.get('path', '')).name != name:
                raise ValueError('Missing, out-of-order, or renamed archive part')
            if part.get('verified_remote_size') is not True or part.get('verified_owner_only') is not True:
                raise ValueError('Incomplete part verification')
            if not re.fullmatch('[0-9a-f]{64}', part.get('sha256', '')):
                raise ValueError('Missing part SHA-256')
            if positive(part.get('bytes'), 'part bytes') > MAX_PART:
                raise ValueError('Part exceeds the declared 64 MiB bound')
            files.append(remote_file(part.get('file_id'), part.get('url'), part['bytes'], name))
        if sum(p['expected_bytes'] for p in files) != size:
            raise ValueError('Split part sizes do not cover the full archive')
        files.append(remote_file(record.get('manifest_file_id'), record.get('manifest_url'),
                                 record.get('manifest_bytes'), path.name[:-7] + '.manifest.json'))
    else:
        if record.get('manifest_file_id'):
            raise ValueError('Manifest without archive parts')
        files.append(remote_file(record.get('file_id'), record.get('url'), size))
    if len({f['file_id'] for f in files}) != len(files):
        raise ValueError('Duplicate remote ID within backup')
    return files, stamp, path


def plan_retention(index, root=ROOT, budget_bytes=DEFAULT_BUDGET):
    root = Path(root).resolve()
    positive(budget_bytes, 'budget')
    if index.get('approved_destination') != 'Private Google Drive' or not isinstance(index.get('backups'), list):
        raise ValueError('Expected the private Drive backup index')
    records, files, times, paths, reasons, ignored, opaque = {}, {}, {}, {}, {}, [], []
    manifest_owners, id_owners = {}, {}
    for number, record in enumerate(index['backups']):
        if not isinstance(record, dict) or type(record.get('deleted_remote', False)) is not bool:
            raise ValueError('Malformed backup index record')
        if record.get('deleted_remote'):
            ignored.append(number)
            continue
        records[number] = record
        kind, status = record.get('kind'), record.get('status')
        if kind in KNOWN_KINDS and status in INCOMPLETE:
            raise ValueError('Partial/incomplete backup blocks cleanup planning')
        unknown = kind not in KNOWN_KINDS or status not in COMPLETE
        if unknown:
            reasons[number] = ['unknown_kind_or_status_preserved']
            opaque.append(number)
            # Unknown entries cannot be deleted, but their IDs must protect
            # shared files from cleanup of an otherwise eligible known entry.
            ids = [record.get('file_id'), record.get('manifest_file_id')]
            ids += [p.get('file_id') for p in record.get('parts', []) if isinstance(p, dict)]
            files[number] = []
        else:
            files[number], times[number], paths[number] = validate_record(record, root)
            ids = [f['file_id'] for f in files[number]]
        for file_id in filter(None, ids):
            if file_id in id_owners:
                raise ValueError('Remote ID shared by multiple active backup records')
            id_owners[file_id] = number
        if record.get('manifest_file_id'):
            manifest_owners[record['manifest_file_id']] = number

    training = [n for n in records if records[n].get('kind') == 'training-checkpoint' and n not in opaque]
    full = [n for n in training if len(set(records[n]['iterations'].values())) == 1
            and next(iter(records[n]['iterations'].values())) % 60 == 0]
    baseline = [n for n in full if set(records[n]['iterations'].values()) == {60}]
    if not baseline or not full:
        raise ValueError('A verified complete 60-round baseline and latest full checkpoint are required')
    baseline = min(baseline, key=lambda n: times[n])
    latest = max(full, key=lambda n: (min(records[n]['iterations'].values()), times[n]))
    if min(records[latest]['iterations'].values()) < 120:
        raise ValueError('Latest full checkpoint must include at least 120 rounds')
    reasons.setdefault(baseline, []).append('first_complete_60_round_baseline')
    reasons.setdefault(latest, []).append('latest_complete_training_stage')
    newer = [n for n in training if max(records[n]['iterations'].values()) > min(records[latest]['iterations'].values())]
    if newer:
        recovery = max(newer, key=lambda n: (min(records[n]['iterations'].values()),
                                           max(records[n]['iterations'].values()), times[n]))
        reasons.setdefault(recovery, []).append('latest_poststage_recovery')
    probes = sorted((n for n in records if records[n].get('kind') == 'derived-policy-probe' and n not in opaque),
                    key=lambda n: times[n], reverse=True)
    for n in probes[:2]:
        reasons.setdefault(n, []).append('one_of_latest_two_derived_probes')
    studies = [n for n in records if records[n].get('kind') == 'research-validation' and n not in opaque]
    latest_study = max(studies, key=lambda n: times[n]) if studies else None
    if latest_study is not None:
        reasons.setdefault(latest_study, []).append('latest_research_validation')
    for n, record in records.items():
        if record.get('kind') == 'evaluation-assets':
            reasons.setdefault(n, []).append('evaluation_assets')
    pending = list(reasons)
    while pending:
        n = pending.pop()
        for reference in references(records[n]):
            if reference not in manifest_owners:
                raise ValueError('Retained backup references a missing/deleted parent manifest')
            parent = manifest_owners[reference]
            if parent not in reasons:
                reasons[parent] = ['required_parent_manifest_dependency']
                pending.append(parent)

    # Read each latest retained archive once. No remote checksum
    # claim follows from this check; the executor must re-read remote metadata.
    latest_path = paths[latest]
    if (not latest_path.is_file() or latest_path.stat().st_size != records[latest]['bytes']
            or digest(latest_path) != records[latest]['sha256']):
        raise ValueError('Retained latest full checkpoint local size/SHA-256 mismatch')
    study_verification = None
    if latest_study is not None:
        study_path = paths[latest_study]
        study_record = records[latest_study]
        if (not study_path.is_file() or study_path.stat().st_size != study_record['bytes']
                or digest(study_path) != study_record['sha256']):
            raise ValueError('Retained latest research validation local size/SHA-256 mismatch')
        study_verification = dict(backup_index=latest_study, archive=str(study_path),
                                  bytes=study_record['bytes'], sha256=study_record['sha256'])
    retained, candidates = [], []
    for n, record in records.items():
        item = dict(backup_index=n, kind=record.get('kind'), run=record.get('run'),
                    iterations=record.get('iterations'), sha256=record.get('sha256'),
                    local_archive=record.get('local_archive'), remote_files=files[n],
                    remote_bytes=sum(f['expected_bytes'] for f in files[n]) if n not in opaque else None)
        if record.get('checkpoint_mode'):
            item.update(checkpoint_mode=record['checkpoint_mode'], exact_legacy_resume=False,
                        warm_start_state_verified=True, historical_mixture_available=False)
        if n in reasons:
            item['reasons'] = reasons[n]
            retained.append(item)
        else:
            item['reasons'] = ['superseded_verified_backup_without_retained_dependents']
            candidates.append(item)
    known_retained = sum(x['remote_bytes'] or 0 for x in retained)
    total_retained = None if opaque else known_retained
    return dict(format='private-poker-retention-plan-v1', dry_run=True, remote_operations_performed=False,
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                index_sha256=hashlib.sha256(json.dumps(index, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                policy=dict(baseline_round=60, latest_completed_stages=1, latest_derived_probes=2,
                            latest_research_validations=1,
                            keep_latest_poststage_recovery=True, preserve_dependency_closure=True,
                            budget_bytes=budget_bytes),
                retained=retained, candidates=candidates, ignored_deleted_records=ignored,
                candidate_remote_files=sum(len(x['remote_files']) for x in candidates),
                expected_reclaimed_bytes=sum(x['remote_bytes'] for x in candidates),
                planned_retained_bytes=total_retained, known_retained_bytes=known_retained,
                within_budget=total_retained <= budget_bytes if total_retained is not None else None,
                budget_note=('Unknown preserved records prevent an exact storage total' if opaque else
                             'Essential retained dependencies exceed budget; do not delete additional files' if total_retained > budget_bytes else
                             'Retained indexed files fit the budget'),
                retained_latest_local_verification=dict(backup_index=latest, archive=str(latest_path),
                    bytes=records[latest]['bytes'], sha256=records[latest]['sha256']),
                retained_latest_research_local_verification=study_verification,
                execution_requirements=['Re-read each retained/candidate remote ID and verify recorded size and owner-only permission before deletion',
                    'Recompute plan if the private index changes; record every successful remote deletion',
                    'Recorded URLs are used verbatim; no Drive URL is synthesized',
                    'No local archives are deleted by this planner; no remote hash verification is claimed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT/'artifacts/offsite-backups.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--budget-bytes', type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args()
    result = plan_retention(json.loads(args.index.read_text()), budget_bytes=args.budget_bytes)
    if args.output:
        destination = args.output.resolve()
        if (not destination.is_relative_to((ROOT/'artifacts').resolve())
                or destination.suffix != '.json' or destination == args.index.resolve()):
            raise ValueError('Plan output must be a distinct private file under artifacts/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(destination, result)
        print(json.dumps({k: result[k] for k in ('candidate_remote_files', 'expected_reclaimed_bytes', 'planned_retained_bytes', 'within_budget')}))
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
